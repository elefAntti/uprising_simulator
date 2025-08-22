
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, math, json, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from utils.distill_dataset import NPZDataset, collate_bc

class MLPPolicy(nn.Module):
    def __init__(self, input_dim:int, hidden:str="512,256,128", dropout:float=0.05):
        super().__init__(); dims=[int(x) for x in hidden.split(",") if x.strip()]; layers=[]; last=input_dim
        for h in dims:
            layers += [nn.Linear(last,h), nn.LayerNorm(h), nn.GELU()]
            if dropout and dropout>0: layers += [nn.Dropout(dropout)]
            last=h
        self.backbone=nn.Sequential(*layers); self.head=nn.Linear(last,2)
    def forward(self, x): return torch.tanh(self.head(self.backbone(x)))

def set_seed(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def compute_norm_stats(loader, max_batches:int=200):
    n=0; mean=None; sq=None
    for i,b in enumerate(loader):
        x=b["obs"]; bs,d=x.shape
        if mean is None: mean=torch.zeros(d, dtype=torch.float64); sq=torch.zeros(d, dtype=torch.float64)
        mean += x.double().sum(0); sq += (x.double()**2).sum(0); n += bs
        if i+1>=max_batches: break
    mean /= max(1,n); var = (sq/max(1,n)) - mean**2; std=torch.sqrt(torch.clamp(var, min=1e-8))
    return mean.float(), std.float()

class Normalizer(nn.Module):
    def __init__(self, mean, std): super().__init__(); self.register_buffer("mean",mean.clone()); self.register_buffer("std",std.clone())
    def forward(self, x): return (x - self.mean) / (self.std + 1e-6)

def save_all(out_dir:str, model:nn.Module, normalizer:nn.Module, input_dim:int):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, os.path.join(out_dir, "model.pt"))
    model.eval(); example=torch.zeros(1,input_dim,dtype=torch.float32)
    scripted=torch.jit.trace(model, example); torch.jit.save(scripted, os.path.join(out_dir, "model.script.pt"))
    class Wrapped(nn.Module):
        def __init__(self, norm, net): super().__init__(); self.norm=norm; self.net=net
        def forward(self, x): return self.net(self.norm(x))
    wrapped=Wrapped(normalizer, model).eval(); scripted2=torch.jit.trace(wrapped, example)
    torch.jit.save(scripted2, os.path.join(out_dir, "model.normed.script.pt"))
    try:
        import onnx  # noqa
        torch.onnx.export(model, example, os.path.join(out_dir,"model.onnx"),
                          input_names=["obs"], output_names=["action"], opset_version=13,
                          dynamic_axes={"obs":{0:"batch"},"action":{0:"batch"}})
    except Exception: pass

def train_bc(args):
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"); set_seed(args.seed)
    if args.npz_glob: ds_full=NPZDataset(args.npz_glob)
    else: raise SystemExit("Provide --npz-glob")
    input_dim=int(ds_full.obs_dim); print(f"Input dim: {input_dim}")
    n=len(ds_full); val_n=int(n*args.val_split); train_n=n-val_n
    if val_n>0: ds_train, ds_val = random_split(ds_full, [train_n,val_n], generator=torch.Generator().manual_seed(args.seed))
    else: ds_train, ds_val = ds_full, None
    loader_train=DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_bc)
    loader_val=DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_bc) if ds_val is not None else None
    tmp_loader=DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bc)
    mean,std=compute_norm_stats(tmp_loader, max_batches=200); normalizer=Normalizer(mean,std).to(device)
    model=MLPPolicy(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.95))
    total_steps=math.ceil(train_n/args.batch_size)*args.epochs; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,total_steps))
    scaler=torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and not args.no_amp))
    best_val=float("inf"); os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out,"norm_stats.json"),"w") as f: json.dump({"mean":mean.tolist(),"std":std.tolist()}, f, indent=2)
    def eval_loss(loader):
        model.eval(); total=0.0; count=0
        with torch.no_grad():
            for batch in loader:
                x=batch["obs"].to(device); y=batch["action"].to(device); x=normalizer(x)
                yhat=model(x); loss=F.mse_loss(yhat,y,reduction="sum"); total += loss.item(); count += y.shape[0]
        return total/max(1,count)
    step=0
    for ep in range(1, args.epochs+1):
        model.train(); running=0.0; count=0
        for batch in loader_train:
            x=batch["obs"].to(device, non_blocking=True); y=batch["action"].to(device, non_blocking=True); x=normalizer(x)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and not args.no_amp)):
                yhat=model(x); loss=F.mse_loss(yhat,y)
                if args.l1_weight>0: loss = loss + args.l1_weight * F.l1_loss(yhat,y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            if args.grad_clip>0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()*x.shape[0]; count += x.shape[0]; step += 1
        train_loss=running/max(1,count)
        if loader_val is not None:
            val_loss=eval_loss(loader_val); improved=val_loss<best_val
            if improved: best_val=val_loss; save_all(args.out, model, normalizer, input_dim)
            print(f"[Epoch {ep:3d}] train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")
        else:
            save_all(args.out, model, normalizer, input_dim)
            print(f"[Epoch {ep:3d}] train_mse={train_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")
    save_all(args.out, model, normalizer, input_dim); print("Done ->", args.out)

def main():
    ap=argparse.ArgumentParser(description="Behavior Cloning trainer (NPZ logs)")
    ap.add_argument("--npz-glob", type=str, required=True, help="Glob for NPZ shards, e.g. logs_npz/*.npz")
    ap.add_argument("--out", type=str, default="runs/bc_run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--hidden", type=str, default="512,256,128")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--l1-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--no-amp", action="store_true")
    args=ap.parse_args(); train_bc(args)

if __name__=="__main__": main()
