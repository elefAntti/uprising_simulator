
# bots/cloned_bot.py
# Normalization-aware neural bot that loads TorchScript or PyTorch checkpoints.
import os
import numpy as np
import torch
import torch.nn as nn
from bots import register_bot
from utils.obs_builder import EgoObsBuilder
from bots.utility_functions import get_base_coords, get_opponent_index

class MLPPolicy(nn.Module):
    def __init__(self, d:int, hidden:str="512,256,128", dropout:float=0.05):
        super().__init__()
        hs=[int(x) for x in hidden.split(",") if x.strip()]
        layers=[]; last=d
        for h in hs:
            layers += [nn.Linear(last,h), nn.LayerNorm(h), nn.GELU()]
            if dropout and dropout>0: layers += [nn.Dropout(dropout)]
            last=h
        self.backbone=nn.Sequential(*layers); self.head=nn.Linear(last,2)
    def forward(self, x): return torch.tanh(self.head(self.backbone(x)))

class IdentityNorm(nn.Module):
    def forward(self, x): return x

class Normalizer(nn.Module):
    def __init__(self, mean:np.ndarray, std:np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(std, dtype=torch.float32))
    def forward(self, x): return (x - self.mean) / (self.std + 1e-6)

class _NNBase:
    def __init__(self, agent_idx:int, model_path:str="bc_policy.pth",
                 mean_std_path:str="", hidden:str="512,256,128", dropout:float=0.05,
                 device:str="cpu", field_size:float=1.5, k_red:int=8, k_green:int=4,
                 no_norm:bool=False):
        """
        model_path: TorchScript (.pt/.script.pt) or PyTorch .pth/.pt
        mean_std_path: optional path to norm_stats.npz or norm_stats.json (for raw state_dict)
        no_norm: force no normalization (if your model expects raw obs)
        """
        self.idx = agent_idx
        self.builder = EgoObsBuilder(field_size=field_size, k_red=k_red, k_green=k_green)
        self.device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.net = self._load_model(model_path, mean_std_path, hidden, dropout, no_norm).to(self.device).eval()

    def _load_model(self, path:str, mean_std_path:str, hidden:str, dropout:float, no_norm:bool) -> nn.Module:
        # Try TorchScript (may already include normalizer)
        try:
            net = torch.jit.load(path, map_location=self.device).eval()
            # We don't know if normalizer is baked in; try a dry run to confirm IO dims later
            self._is_script = True
            return net
        except Exception:
            pass

        # PyTorch checkpoint or state_dict
        ckpt = torch.load(path, map_location=self.device)
        # Try to detect embedded normalizer
        mean=std=None; input_dim=None
        if isinstance(ckpt, dict) and ("normalizer_mean" in ckpt) and ("normalizer_std" in ckpt) and not no_norm:
            mean = np.asarray(ckpt["normalizer_mean"], dtype=np.float32)
            std  = np.asarray(ckpt["normalizer_std"], dtype=np.float32)
            input_dim = int(mean.shape[0])
            args = ckpt.get("args", {})
            hidden = args.get("hidden", hidden); dropout = float(args.get("dropout", dropout))

        # If we still don't have mean/std and user provided a sidecar
        if (mean is None or std is None) and (mean_std_path):
            if mean_std_path.endswith(".npz"):
                arr = np.load(mean_std_path, allow_pickle=True); mean=arr["mean"]; std=arr["std"]
            else:
                with open(mean_std_path, "r") as f: js=json.load(f); mean=np.asarray(js["mean"], dtype=np.float32); std=np.asarray(js["std"], dtype=np.float32)
            input_dim = int(mean.shape[0])

        # Build net & (maybe) normalizer
        # If we still don't know input_dim, infer from first Linear weight in state_dict
        state = None
        if isinstance(ckpt, dict):
            for key in ["model","model_state","state_dict"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]; break
            if state is None:
                # assume raw state dict
                state = {k:v for k,v in ckpt.items() if hasattr(v, "shape")}
        else:
            state = ckpt
        if input_dim is None:
            # find first 2D tensor (Linear weight [out,in])
            for k,v in state.items():
                if isinstance(v, torch.Tensor) and v.ndim==2:
                    input_dim = int(v.shape[1]); break
        if input_dim is None:
            raise RuntimeError("Could not infer input_dim; provide mean_std_path or a checkpoint with embedded stats")

        policy = MLPPolicy(input_dim, hidden=hidden, dropout=dropout)
        # load weights (non-strict to tolerate minor name differences)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[cloned_bot] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        norm = IdentityNorm() if (no_norm or mean is None or std is None) else Normalizer(mean, std)

        class Wrapped(nn.Module):
            def __init__(self, n, m): super().__init__(); self.n=n; self.m=m
            def forward(self, x): return self.m(self.n(x))
        self._is_script = False
        return Wrapped(norm, policy)

    def _bases(self):
        base_own=get_base_coords(self.idx)
        base_opp=get_base_coords(get_opponent_index(self.idx))
        return base_own, base_opp

    def get_controls(self, bot_coords, green_coords, red_coords):
        base_own, base_opp = self._bases()
        obs = self.builder.build(self.idx, bot_coords, red_coords, green_coords, base_own, base_opp)["flat"]
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.net(x)[0].detach().cpu().numpy().astype(np.float32)
        # Ensure valid action range
        a = np.clip(a, -1.0, 1.0)
        return float(a[0]), float(a[1])

#@register_bot
#class ClonedBot(_NNBase):
#    def __init__(self, agent_idx):
#        # Defaults assume TorchScript in working directory; adjust as needed
#        super().__init__(agent_idx, model_path="model.normed.script.pt", device="cpu")
#
#@register_bot
#class RLBot2(_NNBase):
#    def __init__(self, agent_idx):
#        super().__init__(agent_idx, model_path="rl_preview.pth", mean_std_path="norm_stats.npz", device="cpu")

@register_bot
class RLBot3(_NNBase):
    def __init__(self, agent_idx):
        super().__init__(agent_idx, model_path="weights/rl_ng.pth", no_norm=True, device="cpu")

@register_bot
class Dagger2(_NNBase):
    def __init__(self, agent_idx):
        super().__init__(agent_idx, model_path="weights/bc_policy_norm.pth", mean_std_path="weights/norm_stats.npz", device="cpu")

