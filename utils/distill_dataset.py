
from __future__ import annotations
from typing import List, Dict, Any
import glob, numpy as np

import torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, npz_glob: str):
        self.files = sorted(glob.glob(npz_glob))
        if not self.files: raise FileNotFoundError(f"No NPZ files matched: {npz_glob}")
        w0 = np.load(self.files[0], allow_pickle=True)
        self.obs_dim = w0["obs"].shape[1]; self.act_dim = w0["action"].shape[1]
        self._index = []
        self._lengths = []
        for fi, fp in enumerate(self.files):
            w = np.load(fp, allow_pickle=True); n = w["obs"].shape[0]
            self._index.append((fi, 0, n)); self._lengths.append(n)
        self.length = sum(self._lengths)

    def __len__(self): return self.length

    def __getitem__(self, idx: int):
        cum = 0
        for fi, off, n in self._index:
            if idx < cum + n:
                row = idx - cum + off
                w = np.load(self.files[fi], allow_pickle=True)
                obs = w["obs"][row].astype("float32")
                act = w["action"][row].astype("float32")
                done = bool(w["done"][row]); rew = float(w["reward"][row])
                nxt = w["next_obs"][row].astype("float32") if "next_obs" in w.files else None
                item = {"obs": torch.from_numpy(obs), "action": torch.from_numpy(act),
                        "reward": torch.tensor(rew, dtype=torch.float32), "done": torch.tensor(done, dtype=torch.bool)}
                if nxt is not None: item["next_obs"] = torch.from_numpy(nxt)
                return item
            cum += n
        raise IndexError(idx)

def collate_bc(batch: List[Dict[str, Any]]):
    obs = torch.stack([b["obs"] for b in batch], dim=0)
    act = torch.stack([b["action"] for b in batch], dim=0)
    out = {"obs": obs, "action": act}
    if "next_obs" in batch[0] and batch[0]["next_obs"] is not None:
        out["next_obs"] = torch.stack([b["next_obs"] for b in batch if b["next_obs"] is not None], dim=0)
    return out
