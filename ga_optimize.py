#!/usr/bin/env python3
"""
ga_optimize.py
---------------
Genetic algorithm to optimize a parameterized bot (e.g., AegisPilot) against a pool of opponents.

Features
- Noisy evaluation (sensor + physics)
- Hall of Fame (HOF): keep top genomes this run as opponents
- Zoo: save/load frozen genomes across runs
- Checkpointing and final best params JSON

Usage
-----
# Optimize AegisPilot vs discovered opponents (all other bots)
python ga_optimize.py AegisPilot --generations 20 --pop 24 --games-per-opponent 12 --seed 123

# Resume
python ga_optimize.py AegisPilot --resume ga_checkpoint.json

# Include previously saved frozen genomes as opponents
python ga_optimize.py AegisPilot --include-zoo --zoo-path zoo/aegis
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# --- Project imports (expect these in your repo) ---
# - simulator.Simulator with .init(controllers, randomize, noise, seed), .update(), .is_game_over(), .get_winner()
# - decorator registry in bots: load_all_bots(), get_bot_registry()
from simulator import Simulator
from bots import load_all_bots, get_bot_registry


# ---------------- Sensor noise wrapper ----------------
def wrap_controllers_with_sensor_noise(
    controllers,
    pos_sigma: float = 0.02,
    angle_sigma_rad: float = math.radians(2.0),
    rng: Optional[random.Random] = None,
):
    """Wrap controller.get_controls to add Gaussian noise to poses/ball positions."""
    rng = rng or random
    wrapped = []
    for ctrl in controllers:
        if not hasattr(ctrl, "get_controls"):
            wrapped.append(ctrl)
            continue
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


# ---------------- Parameter space (tune freely) ----------------
# Keys should match your parameterized bot's **params (e.g., AegisPilot).
PARAM_SPEC: Dict[str, Tuple[float, float]] = {
    "partner_repel_w": (0.0, 2.0),
    "pair_repel_w":    (0.0, 4.0),
    "red_attr_w":      (0.2, 3.0),
    "green_attr_w":    (0.0, 1.5),
    "wall_power":      (2.0, 8.0),
    "wall_scale":      (0.05, 0.25),
    "sample_ahead":    (0.01, 0.15),
    "fd_step":         (0.002, 0.02),
    # If your bot supports this flag:
    # "use_prediction": (0.0, 1.0),  # interpret >0.5 as True
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


def mutate(g: Dict[str, float], rng: random.Random, rate: float = 0.2, scale: float = 0.1) -> Dict[str, float]:
    """Gaussian mutation; scale is fraction of parameter range."""
    out = dict(g)
    for k, (lo, hi) in PARAM_SPEC.items():
        if rng.random() < rate:
            span = hi - lo
            out[k] += rng.gauss(0.0, scale * span)
    return clamp_genome(out)


# ---------------- Evaluation ----------------
OpponentSpec = Union[str, Dict[str, Any]]  # registry name OR {"_genome": {...}, "name": "...", "class": "AegisPilot"}

def eval_genome(
    candidate_name: str,
    genome: Dict[str, float],
    opponents: List[OpponentSpec],
    games_per_opponent: int,
    seed: int,
    pos_sigma: float,
    angle_sigma_deg: float,
    swap_prob: float,  # kept for future use; sides alternate below
) -> float:
    """Return average win rate of candidate vs opponents (draw=0.5)."""
    rng = random.Random(seed)
    load_all_bots()
    REG = get_bot_registry()
    if candidate_name not in REG:
        raise RuntimeError(f"Unknown bot: {candidate_name}")
    cand_cls = REG[candidate_name]

    total_wins = 0.0
    total_games = 0

    for opp in opponents:
        # Map opponent spec → constructor
        if isinstance(opp, str):
            opp_cls = REG[opp]
            def opp_make(idx, _cls=opp_cls):
                return _cls(idx)
            opp_label = opp
        elif isinstance(opp, dict) and "_genome" in opp:
            opp_cls = REG.get(opp.get("class", candidate_name), cand_cls)
            gparams = opp["_genome"]
            def opp_make(idx, _cls=opp_cls, _gp=gparams):
                return _cls(idx, **_gp)
            opp_label = opp.get("name", f"{candidate_name}@frozen")
        else:
            raise RuntimeError(f"Unsupported opponent spec: {opp!r}")

        for g in range(games_per_opponent):
            swap = (g % 2 == 1)  # alternate sides
            if not swap:
                controllers = [cand_cls(0, **genome), cand_cls(1, **genome), opp_make(2), opp_make(3)]
            else:
                controllers = [opp_make(0), opp_make(1), cand_cls(2, **genome), cand_cls(3, **genome)]

            # Sensor noise
            controllers = wrap_controllers_with_sensor_noise(
                controllers, pos_sigma=pos_sigma, angle_sigma_rad=math.radians(angle_sigma_deg), rng=rng
            )

            # Physics noise (same defaults we used elsewhere)
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


# ---------------- Hall of Fame / Zoo helpers ----------------
def _genome_id(genome: dict) -> str:
    s = ",".join(f"{k}={genome[k]:.6g}" for k in sorted(genome.keys()))
    return hashlib.sha1(s.encode()).hexdigest()[:8]


@dataclass
class GAConfig:
    pop: int = 24
    gens: int = 20
    elite: int = 2
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mutation_scale: float = 0.1
    games_per_opponent: int = 10
    seed: Optional[int] = None
    pos_sigma: float = 0.02
    angle_sigma_deg: float = 2.0
    swap_prob: float = 0.5
    # Hall of Fame / Zoo
    hof_size: int = 5
    zoo_path: str = "zoo/aegis"
    include_zoo: bool = True
    save_zoo: bool = True


def _save_to_zoo(gen: int, genome: dict, score: float, cfg: GAConfig, candidate_name: str) -> Optional[str]:
    if not cfg.save_zoo:
        return None
    os.makedirs(cfg.zoo_path, exist_ok=True)
    gid = _genome_id(genome)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {
        "candidate": candidate_name,
        "gen": gen + 1,
        "score": score,
        "datetime_utc": ts,
        "genome": genome,
        "class": candidate_name,
        "name": f"{candidate_name}@gen{gen+1}_{gid}",
    }
    fname = f"{candidate_name}_gen{gen+1}_{gid}_{score:.3f}.json"
    path = os.path.join(cfg.zoo_path, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def _load_zoo(cfg: GAConfig, candidate_name: str, max_entries: Optional[int] = None) -> List[OpponentSpec]:
    if not cfg.include_zoo or not os.path.isdir(cfg.zoo_path):
        return []
    files = sorted(glob.glob(os.path.join(cfg.zoo_path, f"{candidate_name}_*.json")))
    if max_entries is not None:
        files = files[-max_entries:]
    entries: List[OpponentSpec] = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            g = data.get("genome")
            nm = data.get("name") or os.path.basename(fp).replace(".json", "")
            entries.append({"_genome": g, "name": nm, "class": data.get("class", candidate_name)})
        except Exception:
            continue
    return entries


# ---------------- GA loop ----------------
def run_ga(candidate_name: str, opponents: List[str], cfg: GAConfig, resume_state: Optional[dict] = None):
    rng = random.Random(cfg.seed)
    # Init population
    if resume_state is not None:
        pop = resume_state["population"]
        history = resume_state.get("history", [])
        best = resume_state.get("best")
        hof = resume_state.get("hof", [])
    else:
        pop = [random_genome(rng) for _ in range(cfg.pop)]
        history = []
        best = None
        hof: List[Tuple[dict, float]] = []

    # Load persisted zoo entries
    zoo_opponents = _load_zoo(cfg, candidate_name, max_entries=cfg.hof_size)

    for gen in range(len(history), cfg.gens):
        # Composite opponents: registry names + HOF (frozen genomes) + Zoo
        composite_opps: List[OpponentSpec] = list(opponents)
        composite_opps.extend([{"_genome": g, "name": f"{candidate_name}@HOF"} for (g, s) in hof])
        composite_opps.extend(zoo_opponents)

        # Evaluate population
        scores: List[float] = []
        iterator = tqdm(pop, desc=f"Gen {gen+1}/{cfg.gens} eval", dynamic_ncols=True) if tqdm else pop
        for genome in iterator:
            s = eval_genome(
                candidate_name,
                genome,
                composite_opps,
                cfg.games_per_opponent,
                seed=rng.randrange(1_000_000_000),
                pos_sigma=cfg.pos_sigma,
                angle_sigma_deg=cfg.angle_sigma_deg,
                swap_prob=cfg.swap_prob,
            )
            scores.append(s)

        # Rank & elites
        ranked = sorted(zip(pop, scores), key=lambda t: t[1], reverse=True)
        elites = [g for g, s in ranked[: cfg.elite]]
        best = {"genome": ranked[0][0], "score": ranked[0][1], "gen": gen}
        history.append(best)

        # Update HOF and trim
        hof.extend([(g, s) for g, s in ranked[: max(3, cfg.elite)]])
        hof = sorted(hof, key=lambda t: t[1], reverse=True)[: cfg.hof_size]

        # Save best to zoo
        zoo_path = _save_to_zoo(gen, best["genome"], best["score"], cfg, candidate_name)
        if zoo_path:
            print(f"[Gen {gen+1}] saved best to {zoo_path}")

        # Summary
        median = sorted(scores)[len(scores) // 2]
        print(f"[Gen {gen+1}] best={best['score']:.3f}, median={median:.3f} | HOF={len(hof)} | opps={len(composite_opps)}")

        # Next generation
        new_pop = elites[:]
        while len(new_pop) < cfg.pop:
            if rng.random() < cfg.crossover_rate and len(ranked) >= 2:
                p1 = ranked[rng.randrange(len(ranked) // 2)][0]  # choose from top half
                p2 = ranked[rng.randrange(len(ranked) // 2)][0]
                child = blend(p1, p2, rng)
            else:
                child = dict(ranked[rng.randrange(len(ranked))][0])
            child = mutate(child, rng, rate=cfg.mutation_rate, scale=cfg.mutation_scale)
            new_pop.append(child)
        pop = new_pop

        # Checkpoint
        ckpt = {
            "population": pop,
            "history": history,
            "best": best,
            "hof": hof,
            "candidate": candidate_name,
            "opponents": opponents,
            "config": vars(cfg),
        }
        with open("ga_checkpoint.json", "w") as f:
            json.dump(ckpt, f, indent=2)

    # Save best params
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Best params saved to best_params.json")
    return best, history


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="GA optimizer for parameterized bots (e.g., AegisPilot)")
    ap.add_argument("candidate", help="Candidate bot class name to optimize (e.g., AegisPilot)")
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
    # HOF / Zoo
    ap.add_argument("--hof", type=int, default=5, help="Hall-of-Fame size to retain as opponents")
    ap.add_argument("--zoo-path", type=str, default="zoo/aegis", help="Directory to save/load frozen genomes")
    ap.add_argument("--include-zoo", action="store_true", help="Include previously saved genomes from --zoo-path as opponents")
    ap.add_argument("--no-save-zoo", action="store_true", help="Do not save best-of-generation into the zoo")
    # Resume
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint JSON to resume from")
    args = ap.parse_args()

    # Discover bots
    load_all_bots()
    REG = get_bot_registry()
    if args.candidate not in REG:
        raise SystemExit(f"Unknown candidate bot: {args.candidate}. Available: {', '.join(sorted(REG.keys()))}")

    opponents = args.opponents or [b for b in REG.keys() if b != args.candidate]

    resume_state = None
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            resume_state = json.load(f)
        print(f"Resuming from {args.resume}")
        opponents=resume_state["opponents"]
    print("Candidate:", args.candidate)
    print("Opponents:", ", ".join(opponents) if opponents else "(none)")

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
        hof_size=args.hof,
        zoo_path=args.zoo_path,
        include_zoo=args.include_zoo,
        save_zoo=(not args.no_save_zoo),
    )
    run_ga(args.candidate, opponents, cfg, resume_state=resume_state)


if __name__ == "__main__":
    main()

