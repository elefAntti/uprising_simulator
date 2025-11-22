
#!/usr/bin/env python3
import argparse, os, json, time, random, numpy as np, traceback, sys
from simulator import Simulator
from utils.obs_builder import EgoObsBuilder
from utils.obs_canonical import canonicalize_balls, reflect_world
from bots import load_all_bots, get_bot_registry

def build_obs(builder:EgoObsBuilder, agent_idx:int, sim, t:int, max_steps:int, canonical:bool=True, augment_reflect:bool=False):
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
    tl = np.array([(max_steps - t)/max_steps], dtype=np.float32)
    base = np.concatenate([x, tl], axis=0)
    if not augment_reflect:
        return [base]
    bots_r, reds_r, greens_r = reflect_world(bots, reds, greens, field)
    d2 = builder.build(agent_idx, bots_r, reds_r, greens_r, base_own, base_opp, time_now=float(t))
    x2 = d2["flat"].astype("float32")
    base2 = np.concatenate([x2, tl], axis=0)
    return [base, base2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="logs_critic_npz")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--field-size", type=float, default=1.5)
    ap.add_argument("--k-red", type=int, default=8)
    ap.add_argument("--k-green", type=int, default=4)
    ap.add_argument("--agent-idx", type=int, default=0, choices=[0,2])
    ap.add_argument("--ally", type=str, default="AegisPilot")
    ap.add_argument("--opponents", nargs="*", default=["SimpleBot2","ConeKeeper","TerritoryDash","AuctionStrider"])
    ap.add_argument("--augment-reflect", action="store_true")
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--side-swap", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1, help="Write a shard every N episodes")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    load_all_bots(); REG = get_bot_registry()

    if args.ally not in REG:
        print(f"[error] Unknown ally '{args.ally}'. Available: {', '.join(sorted(REG.keys()))}")
        sys.exit(2)
    for n in args.opponents:
        if n not in REG:
            print(f"[error] Unknown opponent '{n}'. Available: {', '.join(sorted(REG.keys()))}")
            sys.exit(2)

    print(f"[start] episodes={args.episodes} ally={args.ally} opps={args.opponents} agent_idx={args.agent_idx}")
    print(f"[start] canonical={args.canonical} reflect_aug={args.augment_reflect} side_swap={args.side_swap}")
    builder = EgoObsBuilder(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green)
    rng = random.Random(args.seed)

    shard_id = 0; total_obs = 0
    X_buf=[]; y_buf=[]

    def flush():
        nonlocal shard_id, X_buf, y_buf, total_obs
        if not X_buf: return
        X = np.stack(X_buf).astype("float32")
        y = np.asarray(y_buf, dtype=np.uint8)
        meta = dict(field_size=args.field_size, k_red=args.k_red, k_green=args.k_green,
                    max_steps=args.max_steps, canonical=args.canonical,
                    augment_reflect=args.augment_reflect, side_swap=args.side_swap)
        path = os.path.join(args.out, f"critic_{shard_id:05d}.npz")
        np.savez(path, obs=X, label=y, meta=json.dumps(meta))
        print(f"[write] {path} obs={X.shape} wins={int(y.sum())}/{len(y)}")
        shard_id += 1; total_obs += len(y); X_buf.clear(); y_buf.clear()

    try:
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

            # Episode 1-line status
            if args.debug or ep % max(1, args.flush_every) == 0:
                print(f"[ep {ep+1}/{args.episodes}] building match...")

            # Run to outcome
            sim = Simulator(); sim.init(controllers, randomize=True, seed=rng.randrange(1_000_000))
            t=0
            while not sim.is_game_over() and t < args.max_steps:
                sim.update(); t+=1
            sA,sB = sim.scores
            win = int( (sA > sB) if agent_idx in (0,1) else (sB > sA) )
            if args.debug:
                print(f"  outcome: scoreA={sA} scoreB={sB} win={win} steps={t}")

            # Deterministic re-run for stepwise logging
            sim2 = Simulator(); sim2.init(controllers, randomize=False, seed=0)
            t=0
            while not sim2.is_game_over() and t < args.max_steps:
                xs = build_obs(builder, agent_idx, sim2, t, args.max_steps, canonical=args.canonical, augment_reflect=args.augment_reflect)
                for x in xs:
                    X_buf.append(x); y_buf.append(win)
                if args.side_swap:
                    agent_idx_swapped = 2 if agent_idx==0 else 0
                    xs_swapped = build_obs(builder, agent_idx_swapped, sim2, t, args.max_steps, canonical=args.canonical, augment_reflect=args.augment_reflect)
                    for x in xs_swapped:
                        X_buf.append(x); y_buf.append(1 - win)
                sim2.update(); t+=1

            if (ep+1) % args.flush_every == 0:
                flush()

        flush()
        print(f"[done] episodes={args.episodes}, total_obs={total_obs}")

    except Exception as e:
        print("[fatal] exception during logging:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
