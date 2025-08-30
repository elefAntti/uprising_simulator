
#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from simulator import Simulator
from bots import load_all_bots, get_bot_registry
import bots.param_alias as PA

from utils.param_spec import get_param_spec_for_class, params_to_vec01, vec01_to_params, spec_keys

import glob, datetime, hashlib as _h

# -------------------- Utils --------------------
def _genome_id(genome: dict) -> str:
    s = ",".join(f"{k}={genome[k]:.6g}" for k in sorted(genome.keys()))
    return _h.sha1(s.encode()).hexdigest()[:8]

def _save_to_zoo(gen: int, genome: dict, score: float, zoo_path: str, candidate_name: str):
    if not zoo_path:
        return None
    os.makedirs(zoo_path, exist_ok=True)
    gid = _genome_id(genome)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {
        "candidate": candidate_name,
        "gen": gen,
        "score": score,
        "datetime_utc": ts,
        "genome": genome,
        "class": candidate_name,
        "name": f"{candidate_name}@gen{gen}_{gid}",
    }
    fname = f"{candidate_name}_gen{gen}_{gid}_{score:.3f}.json"
    path = os.path.join(zoo_path, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path

def _load_zoo(zoo_path: str, candidate_name: str, max_entries: int | None = None):
    if not zoo_path or not os.path.isdir(zoo_path):
        return []
    files = sorted(glob.glob(os.path.join(zoo_path, f"{candidate_name}_*.json")))
    if max_entries is not None:
        files = files[-max_entries:]
    entries = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            g = data.get("genome")
            nm = data.get("name") or os.path.basename(fp).replace(".json","")
            entries.append({"_genome": g, "name": nm, "class": data.get("class", candidate_name)})
        except Exception:
            continue
    return entries

# -------------------- Noise wrap --------------------
import math as _m
def wrap_controllers_with_sensor_noise(controllers, pos_sigma=0.02, angle_sigma_rad=_m.radians(2.0), rng=None):
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

# -------------------- Evaluation --------------------
def eval_params(candidate_name: str, params: Dict[str, float], opponents, games_per_opponent: int, seed: int, pos_sigma: float, angle_sigma_deg: float, swap_prob: float) -> float:
    rng = random.Random(seed)
    load_all_bots()
    REG = get_bot_registry()
    cand_cls = REG[candidate_name]
    total_wins = 0.0
    total_games = 0
    for opp in opponents:
        if isinstance(opp, str):
            opp_cls = REG[opp]
            opp_make = lambda idx: opp_cls(idx)
        elif isinstance(opp, dict) and "_genome" in opp:
            opp_cls = REG.get(opp.get("class", candidate_name), cand_cls)
            gparams = opp["_genome"]
            opp_make = lambda idx, gp=gparams: opp_cls(idx, **gp)
        else:
            raise RuntimeError(f"Unsupported opponent spec: {opp!r}")
        for g in range(games_per_opponent):
            swap = (g % 2 == 1)
            if not swap:
                controllers = [cand_cls(0, **params), cand_cls(1, **params), opp_make(2), opp_make(3)]
            else:
                controllers = [opp_make(0), opp_make(1), cand_cls(2, **params), cand_cls(3, **params)]
            controllers = wrap_controllers_with_sensor_noise(controllers, pos_sigma=pos_sigma, angle_sigma_rad=_m.radians(angle_sigma_deg), rng=rng)
            # physics jitter (Monte Carlo)
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
            if winner == 0:
                total_wins += 0.5
            elif (winner == 1 and not swap) or (winner == 2 and swap):
                total_wins += 1.0
            total_games += 1
    return total_wins / total_games if total_games > 0 else 0.0

# -------------------- GA core --------------------
@dataclass
class GAConfig:
    pop: int = 24
    generations: int = 20
    cx_prob: float = 0.8
    mut_prob: float = 0.3
    mut_sigma: float = 0.15   # stddev in [0,1] space
    elite_k: int = 2
    seed: int | None = None
    games_per_opponent: int = 8
    pos_sigma: float = 0.02
    angle_sigma_deg: float = 2.0
    swap_prob: float = 0.5
    hof_size: int = 6
    zoo_path: str = "zoo/aegis"
    include_zoo: bool = False
    save_zoo: bool = True

def _cx_blend(x, y, alpha=0.5):
    return [min(1.0, max(0.0, (1.0 - alpha)*xi + alpha*yi)) for xi, yi in zip(x, y)]

def _mut_gauss(x, sigma=0.1, rng=None):
    rng = rng or random
    return [min(1.0, max(0.0, xi + rng.gauss(0.0, sigma))) for xi in x]

def _sel_tournament(pop, fitness, k, tourn_size=3, rng=None):
    rng = rng or random
    out = []
    idxs = list(range(len(pop)))
    for _ in range(k):
        cand = rng.sample(idxs, min(tourn_size, len(idxs)))
        best = max(cand, key=lambda i: fitness[i])
        out.append(best)
    return out

def run_ga(candidate_name: str, opponents: List[str], cfg: GAConfig, init_params: Dict[str, float] | None = None):
    load_all_bots()
    REG = get_bot_registry()
    cand_cls = REG[candidate_name]
    SPEC = get_param_spec_for_class(cand_cls)
    names = spec_keys(SPEC)
    n = len(names)
    rng = random.Random(cfg.seed)

    # initial population in [0,1]^n
    pop = []
    if init_params is not None:
        pop.append(params_to_vec01(SPEC, init_params))
    while len(pop) < cfg.pop:
        pop.append([rng.random() for _ in range(n)])

    hof: list[tuple[dict, float]] = []
    zoo_opponents = _load_zoo(cfg.zoo_path, candidate_name, max_entries=cfg.hof_size) if cfg.include_zoo else []

    best = None
    for gen in range(1, cfg.generations + 1):
        # Evaluate
        fitness = [0.0]*len(pop)
        iterator = enumerate(pop) if tqdm is None else tqdm(list(enumerate(pop)), desc=f"GA gen {gen}/{cfg.generations} eval", dynamic_ncols=True)
        # Compose opponent set: named + HOF + Zoo
        composite = list(opponents)
        composite.extend([{"_genome": g, "name": f"{candidate_name}@HOF"} for (g, s) in hof])
        composite.extend(zoo_opponents)

        for idx, x in iterator:
            params = vec01_to_params(SPEC, x)
            s = eval_params(candidate_name, params, composite, cfg.games_per_opponent, seed=rng.randrange(1_000_000_000), pos_sigma=cfg.pos_sigma, angle_sigma_deg=cfg.angle_sigma_deg, swap_prob=cfg.swap_prob)
            fitness[idx] = s

        # Sort by fitness
        order = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
        best_idx = order[0]
        best_s = fitness[best_idx]
        best_genome = vec01_to_params(SPEC, pop[best_idx])
        best = {"genome": best_genome, "score": best_s, "gen": gen, "names": names}

        # Update HOF
        hof.extend([(vec01_to_params(SPEC, pop[i]), fitness[i]) for i in order[:max(3, cfg.hof_size//2)]])
        hof = sorted(hof, key=lambda t: t[1], reverse=True)[:cfg.hof_size]

        # Save to Zoo
        if cfg.save_zoo:
            _save_to_zoo(gen, best_genome, best_s, cfg.zoo_path, candidate_name)

        # Report
        median = sorted(fitness)[len(fitness)//2]
        print(f"[GA {gen}] best={best_s:.3f} median={median:.3f} pop={len(pop)} opps={len(composite)}")

        # Elitism
        new_pop = [pop[i] for i in order[:cfg.elite_k]]

        # Selection
        parents_idx = _sel_tournament(pop, fitness, k=max(2, len(pop)//2), tourn_size=3, rng=rng)

        # Variation
        while len(new_pop) < cfg.pop:
            if rng.random() < cfg.cx_prob and len(parents_idx) >= 2:
                i, j = rng.sample(parents_idx, 2)
                child = _cx_blend(pop[i], pop[j], alpha=0.5)
            else:
                i = rng.choice(parents_idx)
                child = list(pop[i])
            if rng.random() < cfg.mut_prob:
                child = _mut_gauss(child, sigma=cfg.mut_sigma, rng=rng)
            new_pop.append(child)

        pop = new_pop

    # Save best
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Best params saved to best_params.json")
    return best

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="GA optimizer for any registered bot (dynamic param spec)")
    ap.add_argument("candidate", help="Candidate bot class to optimize (e.g., AegisPilot, ConeKeeper, ...)")
    ap.add_argument("--opponents", nargs="*", default=None, help="Specific registry names; default: all except candidate")
    ap.add_argument("--games-per-opponent", type=int, default=8, help="Games per opponent per evaluation")
    ap.add_argument("--pop", type=int, default=24, help="Population size")
    ap.add_argument("--generations", type=int, default=20, help="Number of generations")
    ap.add_argument("--cx-prob", type=float, default=0.8, help="Crossover probability")
    ap.add_argument("--mut-prob", type=float, default=0.3, help="Mutation probability")
    ap.add_argument("--mut-sigma", type=float, default=0.15, help="Mutation stddev in [0,1] space")
    ap.add_argument("--elite-k", type=int, default=2, help="Elites copied each generation")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    ap.add_argument("--sensor-pos-sigma", type=float, default=0.02)
    ap.add_argument("--sensor-angle-sigma-deg", type=float, default=2.0)
    ap.add_argument("--swap-sides-prob", type=float, default=0.5)
    ap.add_argument("--hof", type=int, default=6, help="Hall-of-Fame size (also opponents)")
    ap.add_argument("--zoo-path", type=str, default="zoo/aegis", help="Dir to save/load zoo genomes")
    ap.add_argument("--include-zoo", action="store_true", help="Include previously saved genomes from --zoo-path")
    ap.add_argument("--no-save-zoo", action="store_true", help="Do not save best-of-generation to zoo")
    ap.add_argument("--init-params", type=str, default=None, help="JSON with initial params (or CMA/GA best_params.json)")
    args = ap.parse_args()

    load_all_bots()
    PA.autoload("zoo/**/*.json")
    REG = get_bot_registry()
    if args.candidate not in REG:
        raise SystemExit(f"Unknown candidate: {args.candidate}. Available: {', '.join(sorted(REG.keys()))}")
    opponents = args.opponents or [b for b in REG.keys() if b != args.candidate]
    print("Candidate:", args.candidate)
    print("Opponents:", ", ".join(opponents))

    init_params = None
    if args.init_params and os.path.exists(args.init_params):
        with open(args.init_params) as f:
            data = json.load(f)
        init_params = data.get("genome", data)

    cfg = GAConfig(
        pop=args.pop,
        generations=args.generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        mut_sigma=args.mut_sigma,
        elite_k=args.elite_k,
        seed=args.seed,
        games_per_opponent=args.games_per_opponent,
        pos_sigma=args.sensor_pos_sigma,
        angle_sigma_deg=args.sensor_angle_sigma_deg,
        swap_prob=args.swap_sides_prob,
        hof_size=args.hof,
        zoo_path=args.zoo_path,
        include_zoo=args.include_zoo,
        save_zoo=(not args.no_save_zoo),
    )

    # Pass HOF/Zoo config through eval by composing opponents internally
    run_ga(args.candidate, opponents, cfg, init_params=init_params)

if __name__ == "__main__":
    main()
