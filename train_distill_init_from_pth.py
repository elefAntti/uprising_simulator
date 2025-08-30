
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
def compute_norm_stats(loader, max_batches:int=200, progress_every:int=10):
    n=0; mean=None; sq=None
    for i,b in enumerate(loader):
        x=b["obs"]; bs,d=x.shape
        if mean is None: mean=torch.zeros(d, dtype=torch.float64); sq=torch.zeros(d, dtype=torch.float64)
        mean += x.double().sum(0); sq += (x.double()**2).sum(0); n += bs
        if (i+1) % max(1, progress_every) == 0:
            print(f"[norm] processed batches={i+1}, samples~{n}")
        if i+1>=max_batches: break
    mean /= max(1,n); var = (sq/max(1,n)) - mean**2; std=torch.sqrt(torch.clamp(var, min=1e-8))
    print(f"[norm] done. used_batches<= {min(i+1, max_batches)}, samples~{n}")
    return mean.float(), std.float()

class Normalizer(nn.Module):
    def __init__(self, mean, std): super().__init__(); self.register_buffer("mean",mean.clone()); self.register_buffer("std",std.clone())
    def forward(self, x): return (x - self.mean) / (self.std + 1e-6)

def save_exports(out_dir:str, model:nn.Module, normalizer:nn.Module, input_dim:int):
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

def load_or_compute_norm(tmp_loader, cache_path:str, max_batches:int):
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path, allow_pickle=True)
            mean = torch.tensor(arr["mean"], dtype=torch.float32)
            std  = torch.tensor(arr["std"], dtype=torch.float32)
            print(f"[norm] Loaded cached stats from {cache_path}")
            return mean, std
        except Exception as e:
            print(f"[norm] Failed to load cache ({e}), recomputing...")
    mean, std = compute_norm_stats(tmp_loader, max_batches=max_batches, progress_every=10)
    np.savez(cache_path, mean=mean.cpu().numpy(), std=std.cpu().numpy())
    print(f"[norm] Cached stats to {cache_path}")
    return mean, std

def maybe_load_initial_weights(path:str, model:nn.Module, normalizer:Normalizer|None, use_norm:bool, strict:bool, device:torch.device):
    if not path: return False
    print(f"[init] Loading initial weights from: {path}")
    ckpt = torch.load(path, map_location=device)
    state = None
    # Try common keys
    if isinstance(ckpt, dict):
        for key in ["model", "model_state", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        if state is None:
            # Might be raw state_dict
            state = {k:v for k,v in ckpt.items() if hasattr(v, "shape")}
        # Optionally load normalizer from checkpoint
        if use_norm and normalizer is not None and ("normalizer_mean" in ckpt) and ("normalizer_std" in ckpt):
            try:
                m = torch.tensor(ckpt["normalizer_mean"], dtype=torch.float32, device=device)
                s = torch.tensor(ckpt["normalizer_std"], dtype=torch.float32, device=device)
                normalizer.mean.data.copy_(m); normalizer.std.data.copy_(s)
                print("[init] Normalizer restored from checkpoint.")
            except Exception as e:
                print("[init] Warning: failed to restore normalizer:", e)
    else:
        # Unexpected format
        raise RuntimeError("Unsupported checkpoint format for --init-weights")

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[init] Missing keys: {len(missing)} (not fatal)")
    if unexpected:
        print(f"[init] Unexpected keys: {len(unexpected)} (not fatal)")
    print("[init] Initial weights loaded.")
    return True

def dump_checkpoint(path:str, epoch:int, model, opt, sched, scaler, normalizer, best_val:float, args_dict:dict):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "normalizer_mean": normalizer.mean.detach().cpu().numpy(),
        "normalizer_std": normalizer.std.detach().cpu().numpy(),
        "best_val": best_val,
        "args": args_dict,
    }
    torch.save(ckpt, path)

def load_checkpoint(path:str, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def train_bc(args):
    cuda_ok = torch.cuda.is_available() and not args.cpu
    device=torch.device("cuda" if cuda_ok else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type=="cuda":
        try:
            print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception as e:
            print("CUDA name check failed:", e)
    set_seed(args.seed)

    # Dataset
    ds_full=NPZDataset(args.npz_glob)
    input_dim=int(ds_full.obs_dim); print(f"Input dim: {input_dim}")
    n=len(ds_full); val_n=int(n*args.val_split); train_n=n-val_n
    if val_n>0: ds_train, ds_val = random_split(ds_full, [train_n,val_n], generator=torch.Generator().manual_seed(args.seed))
    else: ds_train, ds_val = ds_full, None

    # DataLoaders
    dl_kwargs = dict(batch_size=args.batch_size, pin_memory=(device.type=="cuda"),
                     num_workers=args.workers, collate_fn=collate_bc)
    if args.workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=args.prefetch))
    loader_train=DataLoader(ds_train, shuffle=True, **dl_kwargs)
    loader_val=DataLoader(ds_val, shuffle=False, **dl_kwargs) if ds_val is not None else None

    # Normalization (cached)
    tmp_loader=DataLoader(ds_train, shuffle=True, **dl_kwargs)
    os.makedirs(args.out, exist_ok=True)
    cache_path=os.path.join(args.out, "norm_stats.npz")
    mean,std=load_or_compute_norm(tmp_loader, cache_path=cache_path, max_batches=args.norm_batches)
    normalizer=Normalizer(mean,std).to(device)

    # Model / Opt / Sched
    model=MLPPolicy(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.95))
    total_steps=max(1, math.ceil(train_n/args.batch_size))*max(1, args.epochs)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler=torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and not args.no_amp))

    start_epoch=1
    best_val=float("inf")

    # Init vs Resume
    if args.resume:
        print(f"[resume] Loading checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, map_location=device)
        # model
        try:
            model.load_state_dict(ckpt["model"], strict=True); print("[resume] Model weights loaded.")
        except Exception as e:
            print("[resume] Warning: could not load model state:", e)
        # opt/sched/scaler
        try:
            if ckpt.get("optimizer") is not None:
                opt.load_state_dict(ckpt["optimizer"]); print("[resume] Optimizer loaded.")
        except Exception as e:
            print("[resume] Optimizer load failed:", e)
        try:
            if ckpt.get("scheduler") is not None:
                sched.load_state_dict(ckpt["scheduler"]); print("[resume] Scheduler loaded.")
        except Exception as e:
            print("[resume] Scheduler load failed:", e)
        try:
            if ckpt.get("scaler") is not None and device.type=="cuda":
                scaler.load_state_dict(ckpt["scaler"]); print("[resume] AMP scaler loaded.")
        except Exception as e:
            print("[resume] Scaler load failed:", e)
        # Normalizer
        if "normalizer_mean" in ckpt and "normalizer_std" in ckpt:
            normalizer.mean.data.copy_(torch.tensor(ckpt["normalizer_mean"], dtype=torch.float32, device=device))
            normalizer.std.data .copy_(torch.tensor(ckpt["normalizer_std"],  dtype=torch.float32, device=device))
            print("[resume] Normalizer restored from checkpoint.")
        # Epoch & best
        start_epoch = int(ckpt.get("epoch", 1))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"[resume] Resuming from epoch={start_epoch}, best_val={best_val:.6f}")
        if args.init_weights:
            print("[init] Note: --init-weights ignored because --resume is provided.")
    elif args.init_weights:
        maybe_load_initial_weights(args.init_weights, model, normalizer if args.init_use_norm else None, args.init_use_norm, args.init_strict, device)

    # Save args
    with open(os.path.join(args.out,"args.json"),"w") as f: json.dump(vars(args), f, indent=2)

    def eval_loss(loader):
        model.eval(); total=0.0; count=0
        with torch.no_grad():
            for batch in loader:
                x=batch["obs"].to(device); y=batch["action"].to(device); x=normalizer(x)
                yhat=model(x); loss=F.mse_loss(yhat,y,reduction="sum"); total += loss.item(); count += y.shape[0]
        return total/max(1,count)

    # Train
    for ep in range(start_epoch, args.epochs+1):
        model.train(); running=0.0; count=0
        for bi, batch in enumerate(loader_train, start=1):
            x=batch["obs"].to(device, non_blocking=True); y=batch["action"].to(device, non_blocking=True); x=normalizer(x)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and not args.no_amp)):
                yhat=model(x); loss=F.mse_loss(yhat,y)
                if args.l1_weight>0: loss = loss + args.l1_weight * F.l1_loss(yhat,y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            if args.grad_clip>0: scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()*x.shape[0]; count += x.shape[0]
            if args.print_every and (bi % args.print_every == 0):
                print(f"[ep {ep}] batch {bi}  mse={loss.item():.6f}")
        train_loss=running/max(1,count)
        if loader_val is not None:
            val_loss=eval_loss(loader_val); improved=val_loss<best_val
            if improved: best_val=val_loss; save_exports(args.out, model, normalizer, input_dim)
            print(f"[Epoch {ep:3d}] train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")
        else:
            save_exports(args.out, model, normalizer, input_dim)
            print(f"[Epoch {ep:3d}] train_mse={train_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")
        # checkpoint for resume
        ck_path = os.path.join(args.out, "checkpoint.pth")
        dump_checkpoint(ck_path, ep+1, model, opt, sched, scaler, normalizer, best_val, vars(args))

    save_exports(args.out, model, normalizer, input_dim); print("Done ->", args.out)

def main():
    ap=argparse.ArgumentParser(description="Behavior Cloning trainer (NPZ logs)")
    ap.add_argument("--npz-glob", type=str, required=True, help="Glob for NPZ shards, e.g. logs_npz/*.npz")
    ap.add_argument("--out", type=str, default="runs/bc_run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--prefetch", type=int, default=2, help="DataLoader prefetch factor (per worker)")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--hidden", type=str, default="512,256,128")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--l1-weight", type=float, default=0.0)
    ap.add_argument("--norm-batches", type=int, default=50, help="Batches to use for mean/std estimate cache")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--print-every", type=int, default=0, help="Per-batch progress print interval (0=off)")
    # Resume / init
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint.pth to resume training")
    ap.add_argument("--init-weights", type=str, default="", help="Init model from an existing .pth/.pt (state_dict/checkpoint) before training")
    ap.add_argument("--init-strict", action="store_true", help="Strict=True when loading --init-weights (default False)")
    ap.add_argument("--init-use-norm", action="store_true", help="If checkpoint has normalizer stats, use them to init the normalizer")
    args=ap.parse_args(); train_bc(args)

if __name__=="__main__": main()
