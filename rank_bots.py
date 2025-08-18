
#!/usr/bin/env python3
"""
rank_bots.py (improved UX)
--------------------------
- Echoes configuration up front.
- Uses ActiveDuelRanker progress bar with live pair + undecided count.
- Writes pairwise CSV (requires csv import).
"""
from __future__ import annotations

import argparse
import csv
import random
from typing import Tuple

try:
    from tqdm import tqdm  # noqa: F401 (used in underlying ranker)
except Exception:
    tqdm = None

from active_duel_ranker import ActiveDuelRanker

from bots import load_all_bots, get_bot_registry
import bots.param_alias as PA
from win_probabilities import simulate_game, set_bot_types

load_all_bots()

PA.autoload("zoo/**/*.json")
set_bot_types(get_bot_registry())

def make_duel_by_name(items, args):
    game_counter = {"k": 0}

    def duel(i: int, j: int, n: int) -> Tuple[int, int, int]:
        wi = wj = dr = 0
        for _ in range(n):
            player_names = [items[i], items[i], items[j], items[j]]
            seed = (args.seed + game_counter["k"]) if args.seed is not None else None
            rng = random.Random(seed) if seed is not None else random.Random()
            result = simulate_game(
                player_names,
                pos_sigma=args.sensor_pos_sigma,
                angle_sigma_deg=args.sensor_angle_sigma_deg,
                swap_sides_prob=args.swap_sides_prob,
                rng=rng,
                seed=seed,
            )
            if result == 1:
                wi += 1
            elif result == 2:
                wj += 1
            else:
                dr += 1
            game_counter["k"] += 1
        return wi, wj, dr

    return duel

def write_csv(items, result, out_csv):
    n = len(items)
    p = result["matrix"]["p"]
    lo = result["matrix"]["lo"]
    hi = result["matrix"]["hi"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "bot_i", "bot_j", "p_i_beats_j", "ci_low", "ci_high"])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w.writerow([i, j, items[i], items[j], f"{p[i][j]:.4f}", f"{lo[i][j]:.4f}", f"{hi[i][j]:.4f}"])

def main():
    ap = argparse.ArgumentParser(prog="rank_bots", description="Active dueling ranker over bot names")
    ap.add_argument("bots", nargs="+", help="Bot names registered in your project (e.g., SimpleBot)")
    ap.add_argument("--games-per-batch", type=int, default=3, help="Games per scheduled pair (batch size)")
    ap.add_argument("--max-matches", type=int, default=1500, help="Max total games to schedule")
    ap.add_argument("--z", type=float, default=2.0, help="Z-score for Wilson intervals (e.g., 1.96, 2.58)")
    ap.add_argument("--strategy", choices=["copeland-ucb", "uncertainty"], default="copeland-ucb", help="Pair selection heuristic")
    ap.add_argument("--topk", type=int, default=None, help="Stop when top-k is statistically certain")
    # Sensor & simulation noise
    ap.add_argument("--sensor-pos-sigma", type=float, default=0.02, help="Std dev of position noise (m)")
    ap.add_argument("--sensor-angle-sigma-deg", type=float, default=2.0, help="Std dev of angle noise (deg)")
    ap.add_argument("--swap-sides-prob", type=float, default=0.5, help="Probability to swap sides each game")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    # Outputs
    ap.add_argument("--out-csv", default="duel_matrix.csv", help="Where to write pairwise p/CI CSV")
    ap.add_argument("--no-stop-when-total-order", action="store_true", help="Do not stop when a total order is proven")
    args = ap.parse_args()

    print("=== Active dueling: configuration ===")
    print("Bots:", ", ".join(args.bots))
    print(f"Strategy={args.strategy} | z={args.z} | batch={args.games_per_batch} | max_matches={args.max_matches}")
    print(f"Noise: pos_sigma={args.sensor_pos_sigma} m, angle_sigma={args.sensor_angle_sigma_deg} deg, swap_prob={args.swap_sides_prob}")
    if args.seed is not None:
        print(f"Base seed: {args.seed}")
    print("Selecting and simulating pairs adaptively... (progress bar will show current pair and ETA)")

    items = args.bots
    duel = make_duel_by_name(items, args)
    ranker = ActiveDuelRanker(items, duel, batch_size=args.games_per_batch, z=args.z, strategy=args.strategy)
    result = ranker.run(
        max_matches=args.max_matches,
        stop_when_total_order=not args.no_stop_when_total_order,
        topk=args.topk,
        progress=True,
    )

    print("\n=== Summary ===")
    print(f"Games played (scheduled): {result['played']}")
    if result["order"] is not None:
        print("Total order (provably correct at chosen z):")
        print("  " + " > ".join(result["order"]))
    else:
        print("Partial order; cycles detected or undecided pairs remain.")
    if result["cycles"]:
        print("Non-transitive cycles (SCCs):")
        for comp in result["cycles"]:
            print("  {" + ", ".join(comp) + "}")
    else:
        print("No non-transitive cycles detected among decided edges.")

    out_csv = args.out_csv
    write_csv(items, result, out_csv)
    print(f"Pairwise estimates written to: {out_csv}")

if __name__ == "__main__":
    main()
