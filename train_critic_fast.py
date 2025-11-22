#!/usr/bin/env python3
import argparse, os, json, glob, math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
from typing import List, Tuple, Iterator, Optional
from contextlib import contextmanager

# ---------------- Utils ----------------

@contextmanager
def timer(msg: str):
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[time] {msg}: {t1 - t0:.2f}s")

def human(n:int) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.2f}k"
    return str(n)

# ---------------- Dataset ----------------

class NPZBlocks(IterableDataset):
    """
    Streams whole NPZ shards as big blocks to avoid per-sample np.load overhead.
    You can optionally cap samples per epoch.
    """
    def __init__(self, files: List[str], shuffle: bool = True, samples_per_epoch: Optional[int] = None):
        super().__init__()
        self.files = list(files)
        self.shuffle = shuffle
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng()
        files = self.files.copy()
        if self.shuffle:
            rng.shuffle(files)
        emitted = 0
        for fp in files:
            with np.load(fp) as z:
                X = z["obs"].astype("float32")
                y = z["label"].astype("float32")
            # shuffle inside the block
            idx = np.arange(X.shape[0])
            rng.shuffle(idx)
            X = X[idx]; y = y[idx]
            X_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
            yield X_t, y_t
            emitted += len(y)
            if self.samples_per_epoch is not None and emitted >= self.samples_per_epoch:
                break

def list_npz(glob_pat: str) -> List[str]:
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No npz files match {glob_pat}")
    return files

# ---------------- Model ----------------

class MLP(nn.Module):
    def __init__(self, d:int, hidden:str="256,128,64", dropout:float=0.05):
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

# ---------------- Normalization ----------------

def compute_norm_fast(files: List[str], max_files: int, max_batches_per_file: int, batch: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute feature-wise mean/std by scanning a subset of files and batches quickly.
    """
    n_total = 0
    mean = None
    M2 = None
    d = None
    files_scan = files[:max_files] if (max_files is not None and max_files > 0) else files
    for fi, fp in enumerate(files_scan):
        with np.load(fp) as z:
            X = z["obs"].astype("float32")
        if d is None:
            d = int(X.shape[1])
            mean = torch.zeros(d, dtype=torch.float64)
            M2   = torch.zeros(d, dtype=torch.float64)
        # iterate in batches
        n_rows = X.shape[0]
        n_batches = math.ceil(n_rows / batch)
        if max_batches_per_file is not None:
            n_batches = min(n_batches, max_batches_per_file)
        for bi in range(n_batches):
            lo = bi * batch
            hi = min(n_rows, lo + batch)
            xb = torch.from_numpy(X[lo:hi]).double()  # (bs, d)
            bs = xb.shape[0]
            b_mean = xb.mean(dim=0)                      # (d,)
            b_var  = xb.var(dim=0, unbiased=False)       # (d,)
            n_new = n_total + bs
            delta = b_mean - mean
            mean = mean + delta * (bs / max(1, n_new))
            M2   = M2 + b_var * bs + (delta**2) * (n_total * bs / max(1, n_new))
            n_total = n_new
        print(f"[norm] scanned file {fi+1}/{len(files_scan)} — rows={n_rows}")
    var = M2 / max(1, n_total)
    std = torch.sqrt(torch.clamp(var, min=1e-8)).float()
    return mean.float(), std, int(d)

# ---------------- Training ----------------

def save_checkpoint(out_dir: str, epoch: int, model: nn.Module, norm: Normalizer):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"critic_ep{epoch:03d}.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "normalizer_mean": norm.mean.cpu().numpy(),
        "normalizer_std":  norm.std.cpu().numpy(),
    }, path)
    print(f"[ckpt] saved {path}")

def save_exports(out_dir: str, model: nn.Module, norm: Normalizer, input_dim: int):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "normalizer_mean": norm.mean.cpu().numpy(),
        "normalizer_std":  norm.std.cpu().numpy(),
    }, os.path.join(out_dir, "critic.pth"))
    class Wrapped(nn.Module):
        def __init__(self, n, f): super().__init__(); self.n=n; self.f=f
        def forward(self, x): return torch.sigmoid(self.f(self.n(x)))
    ts = torch.jit.trace(Wrapped(norm, model).eval(), torch.zeros(1, input_dim))
    torch.jit.save(ts, os.path.join(out_dir, "critic.normed.script.pt"))
    print(f"[export] wrote TorchScript and .pth to {out_dir}")

def train(args):
    files = list_npz(args.data)
    # Peek input dim
    with np.load(files[0]) as z0:
        d = int(z0["obs"].shape[1])
    print(f"[info] files={len(files)}  input_dim={d}")

    # Fast normalization pass
    with timer("normalization"):
        mean, std, d2 = compute_norm_fast(files, max_files=args.norm_files, max_batches_per_file=args.norm_batches_per_file, batch=args.batch)
    assert d2 == d
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "norm_stats.npz"), mean=mean.numpy(), std=std.numpy())
    with open(os.path.join(args.out, "norm_stats.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    norm = Normalizer(mean, std)

    # Build datasets
    samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch > 0 else None
    ds_train = NPZBlocks(files, shuffle=True, samples_per_epoch=samples_per_epoch)
    ds_val   = NPZBlocks(files, shuffle=True, samples_per_epoch=min(args.val_samples, samples_per_epoch or 100_000))

    # DataLoaders – use many workers to utilize CPU
    train_loader = DataLoader(ds_train, batch_size=None, num_workers=args.workers, persistent_workers=(args.workers>0))
    val_loader   = DataLoader(ds_val,   batch_size=None, num_workers=max(1, args.workers//2), persistent_workers=(args.workers>0))

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    torch.set_num_threads(max(1, args.threads))

    model = MLP(d, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    def loop_epoch(loader, train=True, epoch=0):
        if train: model.train()
        else: model.eval()
        tot_loss = 0.0; tot_count = 0
        blocks = 0
        t0 = time.time()
        for X_t, y_t in loader:
            # X_t: (N_block, d), y_t: (N_block,)
            X_t = X_t.to(device); y_t = y_t.to(device)
            # chunk the block into microbatches (to manage memory and increase prints)
            N = X_t.shape[0]
            mb = args.microbatch if args.microbatch > 0 else N
            for lo in range(0, N, mb):
                hi = min(N, lo+mb)
                x = norm(X_t[lo:hi])
                y = y_t[lo:hi]
                with torch.set_grad_enabled(train):
                    logit = model(x)
                    loss = bce(logit, y)
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                tot_loss += float(loss.item()) * (hi-lo)
                tot_count += (hi-lo)
            blocks += 1
            if blocks % args.log_every_blocks == 0:
                dt = time.time() - t0; t0 = time.time()
                print(f"[ep {epoch:03d}] blocks={blocks} samples={human(tot_count)} avg_loss={tot_loss/max(1,tot_count):.4f} (+{dt:.1f}s)")
            # optional: limit blocks per epoch for speed
            if args.blocks_per_epoch > 0 and blocks >= args.blocks_per_epoch:
                break
        return tot_loss / max(1, tot_count)

    for ep in range(1, args.epochs+1):
        tr = loop_epoch(train_loader, train=True, epoch=ep)
        vl = loop_epoch(val_loader, train=False, epoch=ep)
        print(f"[ep {ep:03d}] train_bce={tr:.4f}  val_bce={vl:.4f}")
        if ep % args.ckpt_every == 0:
            save_checkpoint(args.out, ep, model, norm)

    save_exports(args.out, model.cpu().eval(), norm.cpu().eval(), d)
    print("[done]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help='Glob for shards, e.g. "logs_critic_npz/*.npz"')
    ap.add_argument("--out", type=str, default="runs/critic_fast")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32768, help="Normalization batch size")
    ap.add_argument("--norm-files", type=int, default=20, help="Max files to scan for normalization (subset)")
    ap.add_argument("--norm-batches-per-file", type=int, default=8, help="Max batches per file for normalization")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers (use CPU cores)")
    ap.add_argument("--threads", type=int, default=8, help="torch.set_num_threads; useful on CPU-only")
    ap.add_argument("--hidden", type=str, default="256,128,64")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1, help="Not used in streaming mode; keep for compatibility")
    ap.add_argument("--microbatch", type=int, default=4096, help="Split large blocks into microbatches")
    ap.add_argument("--samples-per-epoch", type=int, default=500_000, help="Cap how many samples each train epoch sees")
    ap.add_argument("--val-samples", type=int, default=100_000, help="Cap samples for val epoch")
    ap.add_argument("--blocks-per-epoch", type=int, default=0, help="Optional cap on number of shards per epoch")
    ap.add_argument("--log-every-blocks", type=int, default=1, help="Print progress every N blocks")
    ap.add_argument("--ckpt-every", type=int, default=1, help="Save checkpoint every N epochs")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
