
#!/usr/bin/env python3
"""
log_distill.py
--------------
Run games in your Simulator and log **teacher** actions + ego-centric observations
for Behavior Cloning (distillation). Works with multiple teachers (sampled per episode).

Outputs:
  - NPZ shards in an output directory (default), or
  - a single Parquet file (requires pyarrow).

Examples
--------
# Sample between multiple teachers, log 300 episodes to NPZ shards
python log_distill.py --episodes 300 \
  --teachers ConeKeeper@best FieldMarshal AuctionStrider \
  --opponents SimpleBot2 SimpleBot2 \
  --out logs_npz --autoload-zoo

# Parquet output
python log_distill.py --episodes 100 --teachers FieldMarshal --format parquet --out logs_parquet/distill.parquet
"""
from __future__ import annotations
import argparse, random, math, os
from typing import List, Optional, Dict, Any, Tuple

from bots import load_all_bots, get_bot_registry
import bots.param_alias as PA

from simulator import Simulator
from utils.obs_builder import EgoObsBuilder
from utils.distill_logging import NPZShardLogger, ParquetLogger

# --- noise wrapper (mirrors eval harness) ---
def wrap_with_sensor_noise(controllers, rng, pos_sigma=0.02, angle_sigma_rad=math.radians(2.0)):
    wrapped=[]
    for ctrl in controllers:
        if not hasattr(ctrl, 'get_controls'): 
            wrapped.append(ctrl); 
            continue
        _orig=ctrl.get_controls
        def noisy_get_controls(bot_coords, green_coords, red_coords, _o=_orig, _r=rng, _ps=pos_sigma, _as=angle_sigma_rad):
            def jxy(p): return (p[0]+_r.gauss(0.0,_ps), p[1]+_r.gauss(0.0,_ps))
            n_bot=[(jxy(p), ang + _r.gauss(0.0,_as)) for (p,ang) in bot_coords]
            n_green=[jxy(p) for p in green_coords]; n_red=[jxy(p) for p in red_coords]
            return _o(n_bot, n_green, n_red)
        ctrl.get_controls = noisy_get_controls
        wrapped.append(ctrl)
    return wrapped

# --- Logging wrapper around the AGENT teacher ---
class LoggingTeacherWrapper:
    """
    Intercepts get_controls to (1) build ego-centric obs, (2) record obs/action pairs,
    and (3) attach next_obs on the following tick for training convenience.
    """
    def __init__(self, base_bot, index:int, builder:EgoObsBuilder, step_logger, episode_id:int, teacher_name:str):
        self.base = base_bot
        self.i = index
        self.builder = builder
        self.logger = step_logger
        self.episode_id = episode_id
        self.teacher_name = teacher_name
        self.t = 0
        self.prev_obs = None
        self.prev_act = None

    def _bases(self):
        try:
            from bots.utility_functions import get_base_coords, get_opponent_index
            base_own=get_base_coords(self.i); base_opp=get_base_coords(get_opponent_index(self.i))
        except Exception:
            S = self.builder.field_size
            base_own, base_opp = (0.0,0.0), (S,S)
        return base_own, base_opp

    def get_controls(self, bot_coords, green_coords, red_coords):
        # Build obs BEFORE action
        base_own, base_opp = self._bases()
        obs = self.builder.build(self.i, bot_coords, red_coords, green_coords, base_own, base_opp, time_now=self.t or 0.0)["flat"]

        # If we have a pending prev sample, log with next_obs = current obs
        if self.prev_obs is not None and self.prev_act is not None:
            self.logger(self.episode_id, self.t-1, self.prev_obs, self.prev_act, done=False, reward=0.0, teacher=self.teacher_name, next_obs_flat=obs)

        # Get teacher action
        a_left, a_right = self.base.get_controls(bot_coords, green_coords, red_coords)
        act = (float(a_left), float(a_right))

        # Store for next tick logging
        self.prev_obs = obs
        self.prev_act = act
        self.t += 1
        return a_left, a_right

    def flush_end(self):
        # At episode end, log the final step (without next_obs).
        if self.prev_obs is not None and self.prev_act is not None:
            self.logger(self.episode_id, self.t, self.prev_obs, self.prev_act, done=True, reward=0.0, teacher=self.teacher_name, next_obs_flat=None)
        self.prev_obs = None; self.prev_act = None

def run_logging(args):
    rng = random.Random(args.seed)
    load_all_bots()
    # Optionally autoload param aliases from "zoo/**/*.json"
    if args.autoload_zoo:
        PA.autoload("zoo/**/*.json")
    REG = get_bot_registry()
    bot_list = list(REG.keys())


    opponents = args.opponents or ["SimpleBot2","SimpleBot2"]
    specified_opponents = (args.opponents != None)

    # Choose logger impl
    if args.format == "npz":
        os.makedirs(args.out, exist_ok=True)
        shard_logger = NPZShardLogger(args.out, shard_size=args.shard, prefix="distill_run")
        log_step = shard_logger.log_step
        closer = shard_logger.close
    else:
        # single parquet file
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plog = ParquetLogger(args.out)
        log_step = plog.log_step
        closer = plog.close

    # Physics jitter like in eval harness
    physics_noise = {
        "core_radius": 0.02, "core_density": 0.15, "core_friction": 0.25,
        "core_restitution": 0.20, "core_linear_damping": 0.10, "core_angular_damping": 0.20,
        "robot_density": 0.10, "robot_friction": 0.25, "robot_ang_damp": 0.10, "robot_speed_scale": 0.05,
    }

    for ep in range(1, args.episodes + 1):
        # Sample teacher for the agent side
        teacher_name = rng.choice(args.teachers)
        teammate_name = args.teammate or teacher_name
        agent_team = "A" if rng.random() < 0.5 else "B"
        if not specified_opponents:
                opponent_name = random.choice(bot_list)
                opponents = [opponent_name for x in range(2)]
        # Build controllers
        agent_idx = 0 if agent_team == "A" else 2

        def make(name, idx):
            if name not in REG:
                raise SystemExit(f"Unknown bot '{name}'. Available: {', '.join(sorted(REG.keys()))}")
            return REG[name](idx)

        # Agent controller is wrapped teacher
        agent_teacher = make(teacher_name, agent_idx)
        builder = EgoObsBuilder(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green)
        logging_wrapper = LoggingTeacherWrapper(agent_teacher, agent_idx, builder, log_step, episode_id=ep, teacher_name=teacher_name)

        if agent_team == "A":
            controllers = [logging_wrapper, make(teammate_name, 1), make(opponents[0], 2), make(opponents[1], 3)]
        else:
            controllers = [make(opponents[0], 0), make(opponents[1], 1), logging_wrapper, make(teammate_name, 3)]

        # Sensor noise
        controllers = wrap_with_sensor_noise(controllers, rng, pos_sigma=args.sensor_pos_sigma, angle_sigma_rad=math.radians(args.sensor_angle_sigma_deg))

        # Init sim & run
        sim = Simulator()
        sim_seed = rng.randrange(1_000_000_000)
        sim.init(controllers, True, noise=physics_noise, seed=sim_seed)

        # Step to termination (or max_steps)
        steps = 0
        while not sim.is_game_over() and steps < args.max_steps:
            sim.update()
            steps += 1

        # Flush last
        logging_wrapper.flush_end()

        # Console feedback
        if ep % max(1, args.log_every) == 0:
            print(f"[log] episode {ep}/{args.episodes}  teacher={teacher_name}  team={agent_team}  steps={steps}")

    closer()
    print("Done. Logs saved to:", args.out)

def main():
    ap = argparse.ArgumentParser(description="Log distillation (BC) data from teachers")
    ap.add_argument("--episodes", type=int, default=200, help="Number of episodes to log")
    ap.add_argument("--max-steps", type=int, default=1200, help="Safety cap per episode")
    ap.add_argument("--teachers", nargs="+", required=True, help="Teacher bot names (registry/aliases) sampled per episode")
    ap.add_argument("--teammate", type=str, default=None, help="Fixed teammate name (default: same as sampled teacher)")
    ap.add_argument("--opponents", nargs="*", default=None, help="Two opponent names (default: SimpleBot2 SimpleBot2)")
    ap.add_argument("--field-size", type=float, default=1.5)
    ap.add_argument("--k-red", type=int, default=8)
    ap.add_argument("--k-green", type=int, default=4)
    ap.add_argument("--sensor-pos-sigma", type=float, default=0.02)
    ap.add_argument("--sensor-angle-sigma-deg", type=float, default=2.0)
    ap.add_argument("--format", choices=["npz","parquet"], default="npz")
    ap.add_argument("--out", type=str, default="logs_npz", help="Output directory (NPZ) or Parquet file path")
    ap.add_argument("--shard", type=int, default=10000, help="Rows per NPZ shard")
    ap.add_argument("--autoload-zoo", action="store_true", help="Autoload 'zoo/**/*.json' as param aliases")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    run_logging(args)

if __name__ == "__main__":
    main()
