
#!/usr/bin/env python3
import argparse, os, json, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class NPZCritic(Dataset):
    def __init__(self, glob_pat:str):
        import glob as _g
        files = sorted(_g.glob(glob_pat))
        if not files: raise FileNotFoundError(f"No npz files match {glob_pat}")
        self.files = files; self._index = []
        for fp in files:
            with np.load(fp) as z:
                n = int(z['label'].shape[0])
                self._index.append((fp, n))
        self.n = sum(n for _,n in self._index)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        acc = 0
        for fp, n in self._index:
            if idx < acc + n:
                off = idx - acc
                with np.load(fp) as z:
                    x = z['obs'][off].astype('float32')
                    y = float(z['label'][off])
                return x, y
            acc += n
        raise IndexError

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

def compute_norm(ds, max_batches=200, batch=8192):
    """Compute feature-wise mean/std using a numerically stable batch update.
    Uses parallel Welford combining: maintain (n, mean, M2) across batches.
    """
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    n_total = 0
    mean = None
    M2 = None
    d = None
    for i,(x,_y) in enumerate(loader):
        x = x.double()  # (bs, d)
        bs, d_cur = x.shape
        if mean is None:
            d = int(d_cur)
            mean = torch.zeros(d, dtype=torch.float64)
            M2   = torch.zeros(d, dtype=torch.float64)
        # batch stats
        b_mean = x.mean(dim=0)                  # (d,)
        b_var  = x.var(dim=0, unbiased=False)   # (d,)
        # Combine
        n_new = n_total + bs
        delta = b_mean - mean                   # (d,)
        mean = mean + delta * (bs / max(1, n_new))
        # M2 = previous M2 + batch M2 + correction term
        M2 = M2 + b_var * bs + (delta**2) * (n_total * bs / max(1, n_new))
        n_total = n_new
        if i+1 >= max_batches: break
    var = M2 / max(1, n_total)                  # population variance
    std = torch.sqrt(torch.clamp(var, min=1e-8)).float()
    return mean.float(), std, int(d)

def save_exports(out, model, normalizer, input_dim):
    os.makedirs(out, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "normalizer_mean": normalizer.mean.cpu().numpy(),
                "normalizer_std":  normalizer.std.cpu().numpy()},
               os.path.join(out, "critic.pth"))
    class Wrapped(nn.Module):
        def __init__(self, n, f): super().__init__(); self.n=n; self.f=f
        def forward(self, x): return torch.sigmoid(self.f(self.n(x)))
    ts = torch.jit.trace(Wrapped(normalizer, model).eval(), torch.zeros(1, input_dim))
    torch.jit.save(ts, os.path.join(out, "critic.normed.script.pt"))

def train(args):
    ds_full = NPZCritic(args.data)
    print(f"[norm] scanning {len(ds_full)} samples (max_batches={args.norm_batches}, batch={args.batch})...")
    mean, std, d = compute_norm(ds_full, max_batches=args.norm_batches, batch=args.batch)
    import numpy as np
    np.savez(os.path.join(args.out, "norm_stats.npz"), mean=mean.numpy(), std=std.numpy())
    with open(os.path.join(args.out, "norm_stats.json"), "w") as f: json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    norm = Normalizer(mean, std)

    val_ratio = args.val_split
    if val_ratio > 0:
        val_n = int(len(ds_full) * val_ratio); train_n = len(ds_full) - val_n
        train_ds, val_ds = random_split(ds_full, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = ds_full, None

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    model = MLP(d, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    def step_epoch(loader, train=True):
        if train: model.train()
        else: model.eval()
        total=0.0; cnt=0
        for x,y in loader:
            x = norm(x.to(device)); y = y.to(device).float()
            with torch.set_grad_enabled(train):
                logit = model(x); loss = bce(logit, y)
                if train:
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item())*x.shape[0]; cnt += x.shape[0]
        return total/max(1,cnt)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers) if val_ds else None

    for ep in range(1, args.epochs+1):
        tr = step_epoch(train_loader, train=True)
        if val_loader is not None:
            vl = step_epoch(val_loader, train=False)
            print(f"[ep {ep:03d}] train_bce={tr:.4f}  val_bce={vl:.4f}")
        else:
            print(f"[ep {ep:03d}] train_bce={tr:.4f}")

    save_exports(args.out, model.cpu().eval(), norm.cpu().eval(), d)
    print("Saved ->", args.out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/critic")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--hidden", type=str, default="256,128,64")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--norm-batches", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(); train(args)

if __name__ == "__main__":
    main()
