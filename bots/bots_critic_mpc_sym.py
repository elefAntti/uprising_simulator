
# bots/critic_mpc.py — dimension-safe inference
import os, math, numpy as np, torch, torch.nn as nn
from bots import register_bot
from utils.obs_builder import EgoObsBuilder
from utils.obs_canonical import canonicalize_balls, reflect_world
from bots.utility_functions import get_base_coords, get_opponent_index

class _CriticLoader:
    def __init__(self, path:str, device:str="cpu"):
        self.device = torch.device(device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        self.in_dim = None
        try:
            self.net = torch.jit.load(path, map_location=self.device).eval()
            # Try to detect the expected input dim from the normalizer buffer inside the scripted module
            # Our Wrapped module has attribute 'n' with buffers 'mean' and 'std'
            try:
                # TorchScript exposes _c for internal; but we can try to getattr
                n = getattr(self.net, 'n')
                mean = getattr(n, 'mean', None)
                if mean is not None:
                    self.in_dim = int(mean.numel())
            except Exception:
                pass
            self.is_script = True
        except Exception:
            ckpt = torch.load(path, map_location=self.device)
            mean = ckpt.get("normalizer_mean", None); std = ckpt.get("normalizer_std", None)
            state = ckpt.get("model", ckpt)
            class Normalizer(nn.Module):
                def __init__(self, m,s): super().__init__(); self.register_buffer("mean", torch.tensor(m, dtype=torch.float32)); self.register_buffer("std", torch.tensor(s, dtype=torch.float32))
                def forward(self,x): return (x-self.mean)/(self.std+1e-6)
            class MLP(nn.Module):
                def __init__(self,d): super().__init__(); self.f=nn.Sequential(nn.Linear(d,256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128,64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64,1))
                def forward(self,x): return self.f(x).squeeze(-1)
            # infer input dim from state dict
            d = next(v for k,v in state.items() if hasattr(v, 'ndim') and v.ndim==2).shape[1]
            self.in_dim = int(d)
            norm = Normalizer(mean,std) if (mean is not None and std is not None) else nn.Identity()
            mlp = MLP(d); mlp.load_state_dict(state, strict=False)
            class Wrapped(nn.Module):
                def __init__(self,n,m): super().__init__(); self.n=n; self.m=m
                def forward(self,x): return torch.sigmoid(self.m(self.n(x)))
            self.net = Wrapped(norm, mlp).to(self.device).eval()
            self.is_script = False

        if self.in_dim is None:
            # Fallback: probe with a few candidate sizes
            for D in (135, 136, 137, 138, 139, 140):
                try:
                    _ = self.net(torch.zeros(1, D))
                    self.in_dim = D; break
                except Exception:
                    continue
            if self.in_dim is None:
                raise RuntimeError("Could not infer critic input dimension")

    def predict(self, x_np:np.ndarray)->float:
        x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(self.device)
        # pad or truncate to expected dim
        D = x.shape[1]
        if self.in_dim is not None and self.in_dim != D:
            if self.in_dim > D:
                pad = torch.zeros(1, self.in_dim - D, device=self.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :self.in_dim]
        with torch.no_grad():
            y = self.net(x).item()
        return float(y)

@register_bot
class CriticMPC:
    def __init__(self, agent_idx:int,
                 critic_path:str="runs/critic_robust_longer/critic.normed.script.pt",
                 device:str="cpu",
                 field_size:float=1.5, k_red:int=8, k_green:int=4,
                 horizon_steps:int=12, dt:float=0.05,
                 candidates:int=32, iters:int=1,
                 acc_limit:float=1.0,
                 v_scale:float=0.7, w_scale:float=2.0,
                 control_cost:float=0.01,
                 move_bonus:float=0.0,
                 ensemble_reflect: bool=False,
                 pot_weight: float=0.2,
                 min_velocity: float=0.05,
                 max_steps:int=1200):
        self.idx = agent_idx
        self.builder = EgoObsBuilder(field_size=field_size, k_red=k_red, k_green=k_green)
        self.device = device
        self.critic = _CriticLoader(critic_path, device=device)
        self.hz = int(horizon_steps); self.dt=float(dt)
        self.K = int(candidates); self.iters=int(iters)
        self.acc_lim = float(acc_limit)
        self.v_scale=float(v_scale); self.w_scale=float(w_scale)
        self.control_cost=float(control_cost)
        self.move_bonus=float(move_bonus)
        self.ensemble_reflect = bool(ensemble_reflect)
        self.prev_u = np.zeros(2, dtype=np.float32)
        self.t = 0; self.max_steps = int(max_steps)
        self.field_size = float(field_size)
        self.pot_weight = float(pot_weight)
        self.min_velocity = float(min_velocity)

    def _bases(self):
        b=get_base_coords(self.idx); ob=get_base_coords(get_opponent_index(self.idx)); return b,ob

    def _obs_with_time(self, bot_coords, red_coords, green_coords):
        base_own, base_opp = self._bases()
        greens_c = canonicalize_balls(list(green_coords), base_opp)
        reds_c   = canonicalize_balls(list(red_coords),   base_opp)
        d = self.builder.build(self.idx, bot_coords, reds_c, greens_c, base_own, base_opp, time_now=float(self.t))
        x = d["flat"].astype(np.float32)
        tleft = np.array([(self.max_steps - self.t)/self.max_steps], dtype=np.float32)
        vec = np.concatenate([x, tleft], axis=0)
        # If critic expects more dims (e.g., score_diff_norm + mean_dists), we'll pad zeros in predict().
        return vec

    def _rollout_end_state(self, pose, u_seq):
        x,y,theta = pose
        v_last = 0.0
        for u in u_seq:
            vl = float(np.clip(u[0], -1, 1))*self.v_scale
            vr = float(np.clip(u[1], -1, 1))*self.v_scale
            v  = 0.5*(vl+vr); w = self.w_scale*(vr - vl)
            x += v*math.cos(theta)*self.dt
            y += v*math.sin(theta)*self.dt
            theta += w*self.dt
            v_last = abs(v)
        return (x,y,theta), v_last

    def _predict(self, bot_coords, red_coords, green_coords):
        x0 = self._obs_with_time(bot_coords, red_coords, green_coords)
        return self.critic.predict(x0)

    def get_controls(self, bot_coords, green_coords, red_coords):
        self.t += 1
        xy, ang = bot_coords[self.idx]
        pose = (float(xy[0]), float(xy[1]), float(ang))
        base_own, base_opp = self._bases()

        U = np.random.uniform(-1,1,size=(self.K,2)).astype(np.float32)
        bestJ = -1e9; bestu = self.prev_u.copy()

        for k in range(self.K):
            u_seq = [U[k]] * self.hz
            (x_end,y_end,th_end), vmag = self._rollout_end_state(pose, u_seq)
            bots2 = []
            for i,(p,a) in enumerate(bot_coords):
                if i==self.idx: bots2.append(((x_end,y_end), th_end))
                else:           bots2.append((p,a))
            vhat = self._predict(bots2, red_coords, green_coords)
            J = vhat - self.control_cost*float(np.dot(U[k],U[k])) + 0.01 * max(0.0, vmag - self.min_velocity)
            if J > bestJ: bestJ = J; bestu = U[k]

        self.prev_u = 0.7*self.prev_u + 0.3*bestu
        u = np.clip(self.prev_u, -1.0, 1.0).astype(np.float32)
        return float(u[0]), float(u[1])
