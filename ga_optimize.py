
#!/usr/bin/env python3
"""
ga_optimize.py
---------------
Genetic algorithm to optimize PotentialWinner* parameters against a pool of opponents.
- Noisy evaluation using your Simulator (sensor noise + physics randomness).
- Works with decorator+registry discovery (bots.BOT_REGISTRY).
- Saves checkpoints and a JSON of best parameters.

Usage
-----
# Optimize PotentialWinner vs discovered opponents
python ga_optimize.py PotentialWinner --generations 20 --pop 24 --games-per-opponent 12 --seed 123

# Resume from a checkpoint
python ga_optimize.py PotentialWinner --resume ga_checkpoint.json

Notes
-----
- Draws count as 0.5 wins.
- We run both sides by swapping teams half the time to reduce side bias.
- Parameter keys exposed here must match PotentialWinner(..., **params).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --- Project imports ---
from simulator import Simulator
try:
    from bots import load_all_bots, get_bot_registry
except Exception as e:
    raise SystemExit("Expected decorator-based registry in bots/. Please provide load_all_bots() and get_bot_registry().") from e

# --------------- Sensor noise wrapper (local copy) ---------------------------
def wrap_controllers_with_sensor_noise(controllers, pos_sigma=0.02, angle_sigma_rad=math.radians(2.0), rng=None):
    rng = rng or random
    wrapped = []
    for ctrl in controllers:
        if not hasattr(ctrl, 'get_controls'):
            wrapped.append(ctrl); continue
        _orig = ctrl.get_controls
        def noisy_get_controls(bot_coords, green_coords, red_coords, _o=_orig, _r=rng, _ps=pos_sigma, _as=angle_sigma_rad):
            def jxy(p):
                return (p[0] + _r.gauss(0.0, _ps), p[1] + _r.gauss(0.0, _ps))
            n_bot = [(jxy(p), ang + _r.gauss(0.0, _as)) for (p, ang) in bot_coords]
            n_green = [jxy(p) for p in green_coords]
            n_red = [jxy(p) for p in red_coords]
            return _o(n_bot, n_green, n_red)
        ctrl.get_controls = noisy_get_controls
        wrapped.append(ctrl)
    return wrapped

# --------------- Parameter space for PotentialWinner ------------------------
# Reasonable ranges; adjust to your field scale & feel.
PARAM_SPEC = {
    "partner_repel_w":  (0.0, 2.0),
    "pair_repel_w":     (0.0, 4.0),
    "red_attr_w":       (0.2, 3.0),
    "green_attr_w":     (0.0, 1.5),
    "wall_power":       (2.0, 8.0),
    "wall_scale":       (0.05, 0.25),
    "sample_ahead":     (0.01, 0.15),
    "fd_step":          (0.002, 0.02),
}

def random_genome(rng: random.Random) -> Dict[str, float]:
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in PARAM_SPEC.items()}

def clamp_genome(g: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in g.items():
        lo, hi = PARAM_SPEC[k]
        out[k] = min(hi, max(lo, v))
    return out

def blend(p1: Dict[str, float], p2: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    """Arithmetic crossover with slight extrapolation."""
    alpha = rng.uniform(-0.2, 1.2)
    return clamp_genome({k: p1[k] * alpha + p2[k] * (1 - alpha) for k in PARAM_SPEC})

def mutate(g: Dict[str, float], rng: random.Random, rate=0.2, scale=0.1) -> Dict[str, float]:
    out = dict(g)
    for k, (lo, hi) in PARAM_SPEC.items():
        if rng.random() < rate:
            span = hi - lo
            out[k] += rng.gauss(0.0, scale * span)
    return clamp_genome(out)

# --------------- Evaluation --------------------------------------------------
def eval_genome(candidate_name: str, genome: Dict[str, float], opponents: List[str], games_per_opponent: int, seed: int, pos_sigma: float, angle_sigma_deg: float, swap_prob: float) -> float:
    """Return average win rate of candidate vs opponents (draw=0.5)."""
    rng = random.Random(seed)
    load_all_bots()
    REG = get_bot_registry()
    if candidate_name not in REG:
        raise RuntimeError(f"Unknown bot: {candidate_name}")
    cand_cls = REG[candidate_name]

    total_wins = 0.0
    total_games = 0

    for opp_name in opponents:
        opp_cls = REG[opp_name]

        for g in range(games_per_opponent):
            # Alternate sides to cancel bias
            swap = (g % 2 == 1)
            # Team1: candidate; Team2: opponent  (or swapped)
            if not swap:
                controllers = [cand_cls(0, **genome), cand_cls(1, **genome), opp_cls(2), opp_cls(3)]
            else:
                controllers = [opp_cls(0), opp_cls(1), cand_cls(2, **genome), cand_cls(3, **genome)]

            # Sensor noise wrapper
            controllers = wrap_controllers_with_sensor_noise(controllers, pos_sigma=pos_sigma, angle_sigma_rad=math.radians(angle_sigma_deg), rng=rng)

            # Physics noise
            noise = {
                "core_radius": 0.02, "core_density": 0.15, "core_friction": 0.25,
                "core_restitution": 0.20, "core_linear_damping": 0.10, "core_angular_damping": 0.20,
                "robot_density": 0.10, "robot_friction": 0.25, "robot_ang_damp": 0.10, "robot_speed_scale": 0.05,
            }
            sim = Simulator()
            sim_seed = rng.randrange(1_000_000_000)
            sim.init(controllers, True, noise=noise, seed=sim_seed)
            while not sim.is_game_over():
                sim.update()
            winner = sim.get_winner()

            # Count wins for candidate irrespective of side
            if winner == 0:
                total_wins += 0.5
            elif (winner == 1 and not swap) or (winner == 2 and swap):
                total_wins += 1.0

            total_games += 1

    return total_wins / total_games if total_games > 0 else 0.0

# --------------- GA loop -----------------------------------------------------
@dataclass
class GAConfig:
    pop: int = 24
    gens: int = 20
    elite: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mutation_scale: float = 0.1
    games_per_opponent: int = 10
    seed: int | None = None
    pos_sigma: float = 0.02
    angle_sigma_deg: float = 2.0
    swap_prob: float = 0.5

def run_ga(candidate_name: str, opponents: List[str], cfg: GAConfig, resume_state=None):
    rng = random.Random(cfg.seed)
    # Initialize population
    pop = []
    if resume_state is not None:
        pop = resume_state["population"]
    else:
        pop = [random_genome(rng) for _ in range(cfg.pop)]

    history = [] if resume_state is None else resume_state.get("history", [])
    best = None if resume_state is None else resume_state.get("best")

    for gen in range(len(history), cfg.gens):
        # Evaluate
        scores = []
        it = pop
        iterator = it
        if tqdm is not None:
            iterator = tqdm(it, desc=f"Gen {gen+1}/{cfg.gens} eval", dynamic_ncols=True)
        for genome in iterator:
            s = eval_genome(candidate_name, genome, opponents, cfg.games_per_opponent, seed=rng.randrange(1_000_000_000), pos_sigma=cfg.pos_sigma, angle_sigma_deg=cfg.angle_sigma_deg, swap_prob=cfg.swap_prob)
            scores.append(s)

        # Select elites
        ranked = sorted(zip(pop, scores), key=lambda t: t[1], reverse=True)
        elites = [g for g, s in ranked[:cfg.elite]]
        best = {"genome": ranked[0][0], "score": ranked[0][1], "gen": gen}
        history.append(best)

        # Print generation summary
        print(f"[Gen {gen+1}] best={best['score']:.3f}, median={sorted(scores)[len(scores)//2]:.3f}")

        # Next generation
        new_pop = elites[:]
        while len(new_pop) < cfg.pop:
            if rng.random() < cfg.crossover_rate and len(ranked) >= 2:
                p1 = ranked[rng.randrange(len(ranked)//2)][0]  # pick from top half
                p2 = ranked[rng.randrange(len(ranked)//2)][0]
                child = blend(p1, p2, rng)
            else:
                child = dict(ranked[rng.randrange(len(ranked))][0])
            child = mutate(child, rng, rate=cfg.mutation_rate, scale=cfg.mutation_scale)
            new_pop.append(child)
        pop = new_pop

        # Save checkpoint
        ckpt = {"population": pop, "history": history, "best": best, "candidate": candidate_name, "opponents": opponents, "config": cfg.__dict__}
        with open("ga_checkpoint.json", "w") as f:
            json.dump(ckpt, f, indent=2)

    # Save best params
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Best params saved to best_params.json")
    return best, history

# --------------- CLI ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GA optimizer for PotentialWinner parameters")
    ap.add_argument("candidate", help="Candidate bot class name to optimize (e.g., PotentialWinner)")
    ap.add_argument("--opponents", nargs="*", default=None, help="Specific opponents; default: all discovered bots except candidate")
    ap.add_argument("--pop", type=int, default=24, help="Population size")
    ap.add_argument("--generations", type=int, default=20, help="Number of generations")
    ap.add_argument("--elite", type=int, default=2, help="Elites carried over each generation")
    ap.add_argument("--games-per-opponent", type=int, default=10, help="Games per opponent per genome (higher = more stable fitness)")
    ap.add_argument("--mutation-rate", type=float, default=0.2)
    ap.add_argument("--mutation-scale", type=float, default=0.1, help="Stddev as fraction of param range")
    ap.add_argument("--crossover-rate", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    # Noise
    ap.add_argument("--sensor-pos-sigma", type=float, default=0.02)
    ap.add_argument("--sensor-angle-sigma-deg", type=float, default=2.0)
    ap.add_argument("--swap-sides-prob", type=float, default=0.5)
    # Resume
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint JSON to resume from")
    args = ap.parse_args()

    # Discover bots
    load_all_bots()
    REG = get_bot_registry()
    if args.candidate not in REG:
        raise SystemExit(f"Unknown candidate bot: {args.candidate}. Available: {', '.join(sorted(REG.keys()))}")
    opponents = args.opponents or [b for b in REG.keys() if b != args.candidate]
    print("Candidate:", args.candidate)
    print("Opponents:", ", ".join(opponents))

    resume_state = None
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            resume_state = json.load(f)
        print(f"Resuming from {args.resume}")

    cfg = GAConfig(
        pop=args.pop,
        gens=args.generations,
        elite=args.elite,
        games_per_opponent=args.games_per_opponent,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
        pos_sigma=args.sensor_pos_sigma,
        angle_sigma_deg=args.sensor_angle_sigma_deg,
        swap_prob=args.swap_sides_prob,
    )
    run_ga(args.candidate, opponents, cfg, resume_state=resume_state)

if __name__ == "__main__":
    main()
