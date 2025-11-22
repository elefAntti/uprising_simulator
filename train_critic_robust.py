
#!/usr/bin/env python3
# train_critic_robust.py
# - Optional: drop ties, downsample timesteps, class weighting, focal loss
# - Per-epoch sample caps, multi-worker block streaming, progress, checkpoints
import argparse, os, json, glob, math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Optional, Tuple

def human(n:int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.2f}k"
    return str(n)

class NPZBlocks(IterableDataset):
    def __init__(self, files: List[str], shuffle=True, samples_per_epoch: Optional[int]=None,
                 drop_ties=False, timestep_stride:int=1):
        self.files=list(files); self.shuffle=shuffle
        self.samples_per_epoch=samples_per_epoch
        self.drop_ties=drop_ties; self.timestep_stride=max(1,int(timestep_stride))
    def __iter__(self):
        rng = np.random.default_rng()
        files=self.files.copy()
        if self.shuffle: rng.shuffle(files)
        emitted=0
        for fp in files:
            with np.load(fp) as z:
                X=z["obs"].astype("float32")
                y=z["label"].astype("float32")
            # subsample timesteps to reduce heavy correlation
            if self.timestep_stride>1:
                X=X[::self.timestep_stride]; y=y[::self.timestep_stride]
            if self.drop_ties:
                m=(y==0.0)|(y==1.0)
                X=X[m]; y=y[m]
            if X.size==0: continue
            idx=np.arange(X.shape[0]); rng.shuffle(idx)
            yield torch.from_numpy(X[idx]), torch.from_numpy(y[idx])
            emitted += y.size
            if self.samples_per_epoch is not None and emitted>=self.samples_per_epoch: break

class MLP(nn.Module):
    def __init__(self, d:int, hidden:str="512,256,128", dropout:float=0.1):
        super().__init__()
        hs=[int(v) for v in hidden.split(',') if v.strip()]
        layers=[]; last=d
        for h in hs:
            layers += [nn.Linear(last,h), nn.LayerNorm(h), nn.GELU()]
            if dropout>0: layers+=[nn.Dropout(dropout)]
            last=h
        self.backbone=nn.Sequential(*layers)
        self.head=nn.Linear(last,1)
    def forward(self, x): return self.head(self.backbone(x)).squeeze(-1)

class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32))
    def forward(self, x): return (x - self.mean) / (self.std + 1e-6)

def try_load_norm(out_dir:str):
    npz=os.path.join(out_dir,"norm_stats.npz"); js=os.path.join(out_dir,"norm_stats.json")
    if os.path.exists(npz):
        z=np.load(npz); return torch.tensor(z["mean"]), torch.tensor(z["std"])
    if os.path.exists(js):
        d=json.load(open(js)); return torch.tensor(d["mean"]), torch.tensor(d["std"])
    return None,None

def compute_norm(files, batch=32768, max_files=20, max_batches_per_file=8):
    n=0; mean=None; M2=None; d=None
    scan=files[:max_files] if max_files>0 else files
    for fi,fp in enumerate(scan):
        with np.load(fp) as z: X=z["obs"].astype("float32")
        if d is None: d=int(X.shape[1]); mean=torch.zeros(d, dtype=torch.float64); M2=torch.zeros(d, dtype=torch.float64)
        rows=X.shape[0]; nb=math.ceil(rows/batch); nb=min(nb,max_batches_per_file)
        for bi in range(nb):
            lo=bi*batch; hi=min(rows, lo+batch); xb=torch.from_numpy(X[lo:hi]).double()
            bs=xb.shape[0]; b_mean=xb.mean(0); b_var=xb.var(0,unbiased=False)
            n_new=n+bs; delta=b_mean-mean; mean=mean+delta*(bs/max(1,n_new)); M2=M2+b_var*bs+(delta**2)*(n*bs/max(1,n_new)); n=n_new
        print(f"[norm] file {fi+1}/{len(scan)} rows={rows}")
    std=torch.sqrt(torch.clamp(M2/max(1,n), min=1e-8)).float()
    return mean.float(), std

def train(args):
    files=sorted(glob.glob(args.data))
    if not files: raise SystemExit("no shards matched")
    with np.load(files[0]) as z0: d=int(z0["obs"].shape[1])
    print(f"[info] shards={len(files)} dim={d}")
    mean,std=try_load_norm(args.out)
    if mean is None or mean.numel()!=d:
        mean,std=compute_norm(files, batch=args.batch, max_files=args.norm_files, max_batches_per_file=args.norm_batches_per_file)
        os.makedirs(args.out, exist_ok=True)
        np.savez(os.path.join(args.out,"norm_stats.npz"), mean=mean.numpy(), std=std.numpy())
    norm=Normalizer(mean,std)

    ds_tr=NPZBlocks(files, shuffle=True, samples_per_epoch=(args.samples_per_epoch or None),
                    drop_ties=args.drop_ties, timestep_stride=args.timestep_stride)
    ds_va=NPZBlocks(files, shuffle=True, samples_per_epoch=min(args.val_samples, args.samples_per_epoch or 100000),
                    drop_ties=args.drop_ties, timestep_stride=args.timestep_stride)

    dl_tr=DataLoader(ds_tr, batch_size=None, num_workers=args.workers, persistent_workers=(args.workers>0))
    dl_va=DataLoader(ds_va, batch_size=None, num_workers=max(1,args.workers//2), persistent_workers=(args.workers>0))

    device=torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    torch.set_num_threads(max(1,args.threads))

    model=MLP(d, hidden=args.hidden, dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Focal loss (optional) or weighted BCE
    def make_loss(pos_weight=None, focal_gamma=None):
        if focal_gamma is not None and focal_gamma>0:
            def focal(logits,target):
                p=torch.sigmoid(logits).clamp(1e-6,1-1e-6)
                ce=-(target*torch.log(p)+(1-target)*torch.log(1-p))
                mod=(1-p)**focal_gamma*target + p**focal_gamma*(1-target)
                return (mod*ce).mean()
            return focal
        if pos_weight is not None and pos_weight>0:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        return nn.BCEWithLogitsLoss()

    def loop_epoch(dl, train=True, ep=0):
        focal_gamma = math.pow(args.focal_gamma, 1.0/(ep))
        loss_fn=make_loss(pos_weight=args.pos_weight, focal_gamma=focal_gamma)
        model.train() if train else model.eval()
        tot=0.0; cnt=0; blocks=0; t0=time.time()
        for X_t,y_t in dl:
            X_t=X_t.to(device); y_t=y_t.to(device)
            N=X_t.shape[0]; mb=args.microbatch if args.microbatch>0 else N
            for lo in range(0,N,mb):
                hi=min(N,lo+mb); x=norm(X_t[lo:hi]); y=y_t[lo:hi]
                with torch.set_grad_enabled(train):
                    logits=model(x); loss=loss_fn(logits,y)
                    if train:
                        opt.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                tot += float(loss.item())*(hi-lo); cnt += (hi-lo)
            blocks+=1
            if blocks % args.log_every_blocks==0:
                dt=time.time()-t0; t0=time.time()
                print(f"[ep {ep:03d}] blocks={blocks} samples={human(cnt)} avg_loss={tot/max(1,cnt):.4f} (+{dt:.1f}s)")
            if args.blocks_per_epoch>0 and blocks>=args.blocks_per_epoch: break
        return tot/max(1,cnt)

    for ep in range(1,args.epochs+1):
        tr=loop_epoch(dl_tr, train=True, ep=ep)
        va=loop_epoch(dl_va, train=False, ep=ep)
        print(f"[ep {ep:03d}] train_bce={tr:.4f}  val_bce={va:.4f}")
        if ep % args.ckpt_every==0:
            torch.save({"epoch":ep,"model":model.state_dict(),"normalizer_mean":norm.mean.cpu().numpy(),"normalizer_std":norm.std.cpu().numpy()}, os.path.join(args.out,f"critic_ep{ep:03d}.pth"))
            print(f"[ckpt] saved {os.path.join(args.out,f'critic_ep{ep:03d}.pth')}")
    # export
    class Wrapped(nn.Module):
        def __init__(self,n,f): super().__init__(); self.n=n; self.f=f
        def forward(self,x): return torch.sigmoid(self.f(self.n(x)))
    ts=torch.jit.trace(Wrapped(norm, model).eval().cpu(), torch.zeros(1,d))
    torch.jit.save(ts, os.path.join(args.out,"critic.normed.script.pt"))
    print("[done]")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/critic_robust")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--norm-files", type=int, default=20)
    ap.add_argument("--norm-batches-per-file", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=400000)
    ap.add_argument("--val-samples", type=int, default=100000)
    ap.add_argument("--blocks-per-epoch", type=int, default=0)
    ap.add_argument("--log-every-blocks", type=int, default=1)
    ap.add_argument("--microbatch", type=int, default=4096)
    ap.add_argument("--hidden", default="512,256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--drop-ties", action="store_true")
    ap.add_argument("--timestep-stride", type=int, default=5)
    ap.add_argument("--pos-weight", type=float, default=0.0, help=">1 to upweight positive class")
    ap.add_argument("--focal-gamma", type=float, default=0.0, help=">0 to use focal loss")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=1, help="Save checkpoint every N epochs")
    args=ap.parse_args(); train(args)

if __name__=="__main__":
    main()
