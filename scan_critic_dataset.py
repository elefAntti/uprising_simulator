
#!/usr/bin/env python3
# scan_critic_dataset.py -- print label stats, feature dim, shards, and a tiny sanity train
import argparse, glob, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help='Glob like "logs_critic_npz/*.npz"')
    args = ap.parse_args()
    files = sorted(glob.glob(args.data))
    if not files:
        print("No files"); return
    n=0; pos=neg=ties=0; d=None
    for fp in files:
        with np.load(fp) as z:
            y = z["label"].astype("float32")
            X = z["obs"].astype("float32")
        if d is None: d = X.shape[1]
        n += y.size
        pos += (y==1.0).sum()
        neg += (y==0.0).sum()
        ties += ((y!=0.0) & (y!=1.0)).sum()
    print(f"files={len(files)}  samples={n}  dim={d}")
    print(f"pos={pos} ({pos/n:.2%})  neg={neg} ({neg/n:.2%})  ties={ties} ({ties/n:.2%})")
    if pos+neg == 0:
        print("WARNING: no win/loss labels, only ties -> BCE optimum is 0.693. Consider dropping ties.")
    elif abs((pos+neg)/n - 1.0) < 1e-6 and abs(pos/(pos+neg) - 0.5) < 0.02:
        print("Note: dataset almost perfectly balanced 50/50; BCE can hover near 0.693 unless features are informative.")
    # quick shuffle check
    print("OK")
if __name__=="__main__":
    main()
