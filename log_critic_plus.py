
#!/usr/bin/env python3
# log_critic_plus_fixed.py
# - Correct diagonal reflection for angles and bases
# - Side-swap keeps tie as 0.5 (no flip)
# - Prints class balance at the end
import argparse, os, json, numpy as np, random, math
from simulator import Simulator
from utils.obs_builder import EgoObsBuilder
from utils.obs_canonical import canonicalize_balls, reflect_world, reflect_bases
from bots import load_all_bots, get_bot_registry

def goal_dists(balls, goal_xy):
    if not balls: return 0.0, 0.0
    ds = [math.hypot(b[0]-goal_xy[0], b[1]-goal_xy[1]) for b in balls]
    return float(np.mean(ds)), float(np.std(ds))

def build_obs(builder:EgoObsBuilder, agent_idx:int, sim, t:int, max_steps:int, k_green:int,
              canonical:bool=True, augment_reflect:bool=False):
    bots = [(tuple(b.position), b.angle) for b in sim.robots]
    reds = [tuple(c.position) for c in sim.red_cores]
    greens = [tuple(c.position) for c in sim.green_cores]
    field = builder.field_size
    base_own = (0.0,0.0) if agent_idx in (0,1) else (field, field)
    base_opp = (field, field) if agent_idx in (0,1) else (0.0, 0.0)

    if canonical:
        greens = canonicalize_balls(greens, base_opp)
        reds   = canonicalize_balls(reds,   base_opp)

    d = builder.build(agent_idx, bots, reds, greens, base_own, base_opp, time_now=float(t))
    x = d["flat"].astype("float32")

    time_left_norm = np.array([(max_steps - t)/max_steps], dtype=np.float32)
    sA, sB = sim.scores
    own = 0 if agent_idx in (0,1) else 1
    opp = 1 - own
    score_diff = (sA - sB) if own==0 else (sB - sA)
    score_diff_norm = np.array([score_diff / max(1,k_green)], dtype=np.float32)

    mg = mr = 0.0
    if len(greens) > 0 or len(reds) > 0:
        mg, _ = goal_dists(greens, base_opp)
        mr, _ = goal_dists(reds, base_opp)
    agg = np.array([mg, mr], dtype=np.float32)

    base = np.concatenate([x, time_left_norm, score_diff_norm, agg], axis=0)

    if not augment_reflect:
        return [base]

    # reflect bots/balls AND bases
    bots_r, reds_r, greens_r = reflect_world(bots, reds, greens, field)
    base_own_r, base_opp_r = reflect_bases(base_own, base_opp, field)
    d2 = builder.build(agent_idx, bots_r, reds_r, greens_r, base_own_r, base_opp_r, time_now=float(t))
    x2 = d2["flat"].astype("float32")
    base2 = np.concatenate([x2, time_left_norm, score_diff_norm, agg], axis=0)
    return [base, base2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="logs_critic_npz")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--field-size", type=float, default=1.5)
    ap.add_argument("--k-red", type=int, default=8)
    ap.add_argument("--k-green", type=int, default=4)
    ap.add_argument("--agent-idx", type=int, default=0, choices=[0,2])
    ap.add_argument("--ally", type=str, default="AegisPilot")
    ap.add_argument("--opponents", nargs="*", default=["AegisPilot"])
    ap.add_argument("--augment-reflect", action="store_true")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--side-swap", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    load_all_bots(); REG = get_bot_registry()
    rng = random.Random(args.seed)
    builder = EgoObsBuilder(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green)

    shard_id = 0; total_obs = 0; pos=0; neg=0; ties=0
    X_buf=[]; y_buf=[]

    def flush():
        nonlocal shard_id, X_buf, y_buf, total_obs, pos, neg, ties
        if not X_buf: return
        X = np.stack(X_buf).astype("float32")
        y = np.asarray(y_buf, dtype=np.float32)
        meta = dict(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green,
                    max_steps=args.max_steps, canonical=args.canonical,
                    augment_reflect=args.augment_reflect, side_swap=args.side_swap,
                    extra_features=["time_left_norm","score_diff_norm","mean_dist_green_to_opp_goal","mean_dist_red_to_opp_goal"])
        path = os.path.join(args.out, f"critic_{shard_id:05d}.npz")
        np.savez(path, obs=X, label=y, meta=json.dumps(meta))
        print(f"[write] {path} obs={X.shape} pos={pos} neg={neg} ties={ties}")
        shard_id += 1; total_obs += len(y); X_buf.clear(); y_buf.clear(); pos=neg=ties=0

    for ep in range(args.episodes):
        agent_idx = args.agent_idx if (ep % 2 == 0) else (2 if args.agent_idx==0 else 0)
        def pick(name, i): return REG[name](i)
        if agent_idx in (0,1):
            ally = pick(args.ally, 1 if agent_idx==0 else 0)
            opp1 = pick(rng.choice(args.opponents), 2)
            opp2 = pick(rng.choice(args.opponents), 3)
            controllers = [None, ally, opp1, opp2]
            controllers[agent_idx] = pick(args.ally, agent_idx)
        else:
            ally = pick(args.ally, 3 if agent_idx==2 else 2)
            opp1 = pick(rng.choice(args.opponents), 0)
            opp2 = pick(rng.choice(args.opponents), 1)
            controllers = [opp1, opp2, None, ally]
            controllers[agent_idx] = pick(args.ally, agent_idx)

        sim = Simulator(); sim.init(controllers, randomize=True, seed=rng.randrange(1_000_000))
        t=0
        while not sim.is_game_over() and t < args.max_steps: sim.update(); t+=1
        sA,sB = sim.scores
        win = 1.0 if ((sA > sB) if agent_idx in (0,1) else (sB > sA)) else (0.0 if (sA!=sB) else 0.5)

        sim2 = Simulator(); sim2.init(controllers, randomize=False, seed=0)
        t=0
        while not sim2.is_game_over() and t < args.max_steps:
            xs = build_obs(builder, agent_idx, sim2, t, args.max_steps, args.k_green, canonical=args.canonical, augment_reflect=args.augment_reflect)
            for x in xs:
                X_buf.append(x); y_buf.append(win)
                if win == 1.0: pos += 1
                elif win == 0.0: neg += 1
                else: ties += 1
            if args.side_swap:
                agent_idx_swapped = 2 if agent_idx==0 else 0
                xs_swapped = build_obs(builder, agent_idx_swapped, sim2, t, args.max_steps, args.k_green, canonical=args.canonical, augment_reflect=args.augment_reflect)
                for x in xs_swapped:
                    y = (1.0 - win) if win in (0.0,1.0) else 0.5
                    X_buf.append(x); y_buf.append(y)
                    if y == 1.0: pos += 1
                    elif y == 0.0: neg += 1
                    else: ties += 1
            sim2.update(); t+=1

        if (ep+1) % args.flush_every == 0: flush()

    flush()
    print("[done]")

if __name__=="__main__":
    main()
