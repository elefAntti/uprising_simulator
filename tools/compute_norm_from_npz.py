
#!/usr/bin/env python3
"""
compute_norm_from_npz.py
Compute per-feature mean/std from NPZ shards and save to norm_stats.npz/json.

Usage:
  python compute_norm_from_npz.py --glob "logs_npz/*.npz" --out runs/stats --batches 100 --batch-size 2048
"""
import argparse, os, json, numpy as np
from torch.utils.data import DataLoader
import torch

# minimal dataset reader (no extra deps)
def iter_npz(glob_pat, batch_size=2048, max_batches=100):
    import glob
    files = sorted(glob.glob(glob_pat))
    if not files: raise SystemExit(f"No files match: {glob_pat}")
    seen = 0
    for fp in files:
        w = np.load(fp, allow_pickle=True)
        X = w["obs"].astype("float32")
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i:i+batch_size]
            seen += 1
            if max_batches and seen >= max_batches: return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="NPZ glob, e.g. logs_npz/*.npz")
    ap.add_argument("--out", required=True, help="Output folder for norm_stats")
    ap.add_argument("--batches", type=int, default=100, help="Max batches to scan")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n=0; mean=None; sq=None
    for X in iter_npz(args.glob, batch_size=args.batch_size, max_batches=args.batches):
        X = torch.from_numpy(X).double()
        bs, d = X.shape
        if mean is None:
            mean = torch.zeros(d, dtype=torch.float64)
            sq = torch.zeros(d, dtype=torch.float64)
        mean += X.sum(0)
        sq += (X*X).sum(0)
        n += bs
    mean /= max(1,n)
    var = (sq/max(1,n)) - mean*mean
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    mean_np = mean.float().numpy(); std_np = std.float().numpy()

    np.savez(os.path.join(args.out, "norm_stats.npz"), mean=mean_np, std=std_np)
    with open(os.path.join(args.out, "norm_stats.json"), "w") as f:
        json.dump({"mean": mean_np.tolist(), "std": std_np.tolist()}, f, indent=2)
    print("Saved:", os.path.join(args.out, "norm_stats.npz"))
    print("Saved:", os.path.join(args.out, "norm_stats.json"))
    print("Input dim:", mean_np.shape[0])

if __name__ == "__main__":
    main()
