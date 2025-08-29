
#!/usr/bin/env python3
from __future__ import annotations
import argparse, random, math, os, numpy as np, json
from typing import Optional, Tuple

from bots import load_all_bots, get_bot_registry
import bots.param_alias as PA

from simulator import Simulator
from utils.obs_builder import EgoObsBuilder
from utils.distill_logging import NPZShardLogger, ParquetLogger




def wrap_with_sensor_noise(controllers, rng, pos_sigma=0.02, angle_sigma_rad=math.radians(2.0)):
    wrapped=[]
    for ctrl in controllers:
        if not hasattr(ctrl, 'get_controls'): wrapped.append(ctrl); continue
        _orig=ctrl.get_controls
        def noisy_get_controls(bot_coords, green_coords, red_coords, _o=_orig, _r=rng, _ps=pos_sigma, _as=angle_sigma_rad):
            def jxy(p): return (p[0]+_r.gauss(0.0,_ps), p[1]+_r.gauss(0.0,_ps))
            n_bot=[(jxy(p), ang + _r.gauss(0.0,_as)) for (p,ang) in bot_coords]
            n_green=[jxy(p) for p in green_coords]; n_red=[jxy(p) for p in red_coords]
            return _o(n_bot, n_green, n_red)
        ctrl.get_controls = noisy_get_controls; wrapped.append(ctrl)
    return wrapped

def _build_mlp_and_norm(input_dim:int, hidden:str, dropout:float, mean:np.ndarray=None, std:np.ndarray=None):
    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, d:int, hidden_layers:str, dropout:float):
            super().__init__()
            hs=[int(x) for x in hidden_layers.split(",") if x.strip()]
            layers=[]; last=d
            for h in hs:
                layers += [nn.Linear(last,h), nn.LayerNorm(h), nn.GELU()]
                if dropout and dropout>0: layers += [nn.Dropout(dropout)]
                last=h
            self.backbone=nn.Sequential(*layers); self.head=nn.Linear(last,2)
        def forward(self, x): return torch.tanh(self.head(self.backbone(x)))
    net = MLP(input_dim, hidden, dropout)
    if mean is None or std is None:
        # identity normalizer
        class Identity(nn.Module):
            def forward(self, x): return x
        norm = Identity()
    else:
        class Normalizer(nn.Module):
            def __init__(self, m, s):
                super().__init__()
                self.register_buffer("mean", torch.tensor(m, dtype=torch.float32))
                self.register_buffer("std", torch.tensor(s, dtype=torch.float32))
            def forward(self, x): return (x - self.mean) / (self.std + 1e-6)
        norm = Normalizer(mean, std)
    class Wrapped(nn.Module):
        def __init__(self, n, m): super().__init__(); self.n=n; self.m=m
        def forward(self, x): return self.m(self.n(x))
    return Wrapped(norm, net), net

def _infer_input_dim_from_state(state_dict) -> int:
    # Try to infer input dimension from first Linear weight
    for k, v in state_dict.items():
        try:
            import torch
            t = v if isinstance(v, torch.Tensor) else None
            if t is not None and t.ndim == 2:
                # weight shape [out, in]
                return int(t.shape[1])
        except Exception:
            continue
    raise RuntimeError("Could not infer input_dim from checkpoint. Please pass --student-input-dim.")

class StudentPolicy:
    def __init__(self, path:str, device:str="cpu", mean_std_path:str="", input_dim:int=0, hidden:str="512,256,128", dropout:float=0.05, no_norm:bool=False):
        import torch
        self.torch = torch
        self.device = self.torch.device(device)
        self.net = None
        self._load(path, mean_std_path, input_dim, hidden, dropout, no_norm)

    def _load(self, path:str, mean_std_path:str, input_dim:int, hidden:str, dropout:float, no_norm:bool):
        # 1) TorchScript
        try:
            self.net = self.torch.jit.load(path, map_location=self.device).eval()
            print(f"[student] Loaded TorchScript: {path}")
            return
        except Exception as e:
            print(f"[student] TorchScript load failed ({e}); trying torch.load...")
        # 2) torch.load
        ckpt = self.torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and ("normalizer_mean" in ckpt) and ("normalizer_std" in ckpt) and not no_norm:
            mean = np.asarray(ckpt["normalizer_mean"], dtype=np.float32)
            std  = np.asarray(ckpt["normalizer_std"], dtype=np.float32)
            inferred_input_dim = int(mean.shape[0])
            cfg = ckpt.get("args", {})
            hidden = cfg.get("hidden", hidden); dropout = float(cfg.get("dropout", dropout))
            wrapped, bare = _build_mlp_and_norm(inferred_input_dim, hidden, dropout, mean, std)
            state = ckpt.get("model") or ckpt.get("model_state") or ckpt
            bare.load_state_dict(state, strict=False)
            self.net = wrapped.to(self.device).eval()
            print(f"[student] Loaded checkpoint with embedded stats. input_dim={inferred_input_dim}, hidden='{hidden}'")
            return
        # 3) state_dict only / or user requests no_norm
        mean = std = None
        if not no_norm:
            if mean_std_path:
                if mean_std_path.endswith(".npz"):
                    arr = np.load(mean_std_path, allow_pickle=True); mean=arr["mean"]; std=arr["std"]
                else:
                    with open(mean_std_path, "r") as f: js=json.load(f); mean=np.asarray(js["mean"], dtype=np.float32); std=np.asarray(js["std"], dtype=np.float32)
        if input_dim <= 0:
            # try infer from state dict
            state = ckpt.get("model") or ckpt.get("model_state") or ckpt if isinstance(ckpt, dict) else ckpt
            input_dim = _infer_input_dim_from_state(state)
        wrapped, bare = _build_mlp_and_norm(input_dim, hidden, dropout, mean, std)
        state = ckpt.get("model") or ckpt.get("model_state") or ckpt if isinstance(ckpt, dict) else ckpt
        bare.load_state_dict(state, strict=False)
        self.net = wrapped.to(self.device).eval()
        norm_note = "NO normalization" if (mean is None or std is None) else "with normalization"
        print(f"[student] Loaded state_dict (input_dim={input_dim}, hidden='{hidden}', {norm_note}).")

    def act(self, obs_flat: np.ndarray) -> Tuple[float,float]:
        with self.torch.no_grad():
            x = self.torch.from_numpy(obs_flat.astype(np.float32)).to(self.device).unsqueeze(0)
            a = self.net(x)[0].detach().cpu().numpy().astype(np.float32)
            return float(a[0]), float(a[1])

class DAggerWrapper:
    def __init__(self, teacher_bot, index:int, builder:EgoObsBuilder, step_logger, episode_id:int, teacher_name:str,
                 student_policy: Optional[StudentPolicy], student_frac: float, rng: random.Random):
        self.teacher = teacher_bot; self.i = index; self.builder = builder; self.logger = step_logger
        self.episode_id = episode_id; self.teacher_name = teacher_name
        self.student = student_policy; self.student_frac = float(student_frac); self.rng = rng
        self.t = 0; self.prev_obs = None; self.prev_teacher_act = None; self._apply_student_this_step = False

    def _bases(self):
        try:
            from bots.utility_functions import get_base_coords, get_opponent_index
            base_own=get_base_coords(self.i); base_opp=get_base_coords(get_opponent_index(self.i))
        except Exception:
            S = self.builder.field_size; base_own, base_opp = (0.0,0.0), (S,S)
        return base_own, base_opp

    def get_controls(self, bot_coords, green_coords, red_coords):
        base_own, base_opp = self._bases()
        obs = self.builder.build(self.i, bot_coords, red_coords, green_coords, base_own, base_opp, time_now=self.t or 0.0)["flat"]
        t_left, t_right = self.teacher.get_controls(bot_coords, green_coords, red_coords)
        teacher_act = (float(t_left), float(t_right))
        if self.prev_obs is not None and self.prev_teacher_act is not None:
            who = "student" if self._apply_student_this_step else "teacher"
            self.logger(self.episode_id, self.t-1, self.prev_obs, self.prev_teacher_act, done=False, reward=0.0, teacher=f"{self.teacher_name}|{who}", next_obs_flat=obs)
        self._apply_student_this_step = (self.student is not None) and (self.rng.random() < self.student_frac)
        if self._apply_student_this_step:
            try:
                s_left, s_right = self.student.act(np.asarray(obs, dtype=np.float32)); applied = (float(s_left), float(s_right))
            except Exception as e:
                print("[student] act() failed; fallback teacher:", e); applied = teacher_act
        else:
            applied = teacher_act
        self.prev_obs = obs; self.prev_teacher_act = teacher_act; self.t += 1
        return applied

    def flush_end(self):
        if self.prev_obs is not None and self.prev_teacher_act is not None:
            who = "student" if self._apply_student_this_step else "teacher"
            self.logger(self.episode_id, self.t, self.prev_obs, self.prev_teacher_act, done=True, reward=0.0, teacher=f"{self.teacher_name}|{who}", next_obs_flat=None)
        self.prev_obs = None; self.prev_teacher_act = None

def run_logging(args):
    rng = random.Random(args.seed)
    load_all_bots()
    PA.autoload("zoo/**/*.json")
    REG = get_bot_registry()
    if args.autoload_zoo: PA.autoload("zoo/**/*.json")
    opponents = args.opponents or ["SimpleBot","SimpleBot"]

    if args.format == "npz":
        os.makedirs(args.out, exist_ok=True); shard_logger = NPZShardLogger(args.out, shard_size=args.shard, prefix="dagger_run")
        log_step = shard_logger.log_step; closer = shard_logger.close
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); plog = ParquetLogger(args.out)
        log_step = plog.log_step; closer = plog.close

    physics_noise = {"core_radius":0.02,"core_density":0.15,"core_friction":0.25,"core_restitution":0.20,"core_linear_damping":0.10,"core_angular_damping":0.20,"robot_density":0.10,"robot_friction":0.25,"robot_ang_damp":0.10,"robot_speed_scale":0.05}

    student = None
    if args.student_script:
        device = "cuda" if (args.use_cuda and _cuda_available()) else "cpu"
        student = StudentPolicy(args.student_script, device=device, mean_std_path=args.student_mean_std, input_dim=args.student_input_dim, hidden=args.student_hidden, dropout=args.student_dropout, no_norm=args.student_no_norm)

    for ep in range(1, args.episodes + 1):
        frac = _anneal_linear(args.student_frac, args.student_frac_final, ep, args.student_anneal_episodes) if args.student_anneal_episodes>0 else args.student_frac
        teacher_name = rng.choice(args.teachers); teammate_name = args.teammate or teacher_name
        agent_team = "A" if rng.random() < 0.5 else "B"; agent_idx = 0 if agent_team == "A" else 2

        def make(name, idx):
            if name not in REG: raise SystemExit(f"Unknown bot '{name}'. Available: {', '.join(sorted(REG.keys()))}")
            return REG[name](idx)

        teacher_bot = make(teacher_name, agent_idx)
        builder = EgoObsBuilder(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green)
        wrapper = DAggerWrapper(teacher_bot, agent_idx, builder, log_step, episode_id=ep, teacher_name=teacher_name, student_policy=student, student_frac=frac, rng=rng)

        if agent_team == "A":
            controllers = [wrapper, make(teammate_name, 1), make(opponents[0], 2), make(opponents[1], 3)]
        else:
            controllers = [make(opponents[0], 0), make(opponents[1], 1), wrapper, make(teammate_name, 3)]

        controllers = wrap_with_sensor_noise(controllers, rng, pos_sigma=args.sensor_pos_sigma, angle_sigma_rad=math.radians(args.sensor_angle_sigma_deg))
        sim = Simulator(); sim_seed = rng.randrange(1_000_000_000); sim.init(controllers, True, noise=physics_noise, seed=sim_seed)
        steps = 0
        while not sim.is_game_over() and steps < args.max_steps:
            sim.update(); steps += 1
        wrapper.flush_end()
        if ep % max(1, args.log_every) == 0:
            print(f"[dagger] episode {ep}/{args.episodes}  teacher={teacher_name}  student_frac={frac:.2f}  team={agent_team}  steps={steps}")
    closer(); print("Done. DAgger logs saved to:", args.out)

def _anneal_linear(start: float, end: float, ep: int, total: int) -> float:
    if total <= 0: return start
    if ep >= total: return end
    t = max(0.0, min(1.0, ep / float(total))); return (1.0 - t) * start + t * end

def _cuda_available() -> bool:
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

def main():
    ap = argparse.ArgumentParser(description="DAgger (mixed-control) distillation logger")
    ap.add_argument("--episodes", type=int, default=200); ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--teachers", nargs="+", required=True); ap.add_argument("--teammate", type=str, default=None)
    ap.add_argument("--opponents", nargs="*", default=None); ap.add_argument("--field-size", type=float, default=1.5)
    ap.add_argument("--k-red", type=int, default=8); ap.add_argument("--k-green", type=int, default=4)
    ap.add_argument("--sensor-pos-sigma", type=float, default=0.02); ap.add_argument("--sensor-angle-sigma-deg", type=float, default=2.0)

    # Student loading options
    ap.add_argument("--student-script", type=str, default="", help="TorchScript OR raw checkpoint path")
    ap.add_argument("--student-mean-std", type=str, default="", help="Path to norm_stats.npz or norm_stats.json when using raw state_dict")
    ap.add_argument("--student-input-dim", type=int, default=0, help="Override input dim for raw state_dict (auto-infer if 0)")
    ap.add_argument("--student-hidden", type=str, default="512,256,128")
    ap.add_argument("--student-dropout", type=float, default=0.05)
    ap.add_argument("--student-no-norm", action="store_true", help="Assume model expects raw inputs; skip normalization wrapper")
    ap.add_argument("--student-frac", type=float, default=0.3, help="Probability student controls each step")
    ap.add_argument("--student-frac-final", type=float, default=None, help="Final probability for linear anneal")
    ap.add_argument("--student-anneal-episodes", type=int, default=0, help="Episodes over which to anneal (0=off)")
    ap.add_argument("--use-cuda", action="store_true", help="Run student on CUDA if available")

    ap.add_argument("--format", choices=["npz","parquet"], default="npz"); ap.add_argument("--out", type=str, default="logs_dagger_npz")
    ap.add_argument("--shard", type=int, default=10000); ap.add_argument("--autoload-zoo", action="store_true")
    ap.add_argument("--seed", type=int, default=123); ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    if args.student_frac_final is None: args.student_frac_final = args.student_frac
    run_logging(args)

if __name__ == "__main__":
    main()
