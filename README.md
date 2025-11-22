# uprising_simulator

<img src="https://github.com/elefAntti/uprising_simulator/blob/main/images/screenshot.png" alt="Screenshot" width="600px">

Simulate the game played at RobotUprising hackathon.

Two teams of robots (differential drive with tracks) compete in an arena; their positions and the positions of energy cores (balls) are tracked from above using machine vision. There are two bots per team, the objective is to push the **red** balls to the opponent corner and **green** balls to your own corner. If a player gets three green balls in their corner, they lose. Otherwise the game ends as all the balls have been scored or after a time limit (30 s here; 2 m 30 s IRL). Check the official [rules](https://github.com/robot-uprising-hq/ai-rules-mi2020).

---

## What’s new

- **Physics & sensor randomness for Monte Carlo**  
  - Physics jitter (ball/robot friction, restitution, damping, etc.) when `randomize=True` in the simulator.
  - Optional **sensor noise** (position σ in meters; angle σ in degrees) injected into bot observations.
  - **Side randomization**: per game, teams can be swapped to cancel side bias.
- **Decorator-based bot registry**  
  - Add `@register_bot` to any bot class and call `load_all_bots()` to auto‑discover. No more hardcoding.
- **New parameterized bot: _AegisPilot_** (`bots/aegis_pilot.py`)  
  - Potential‑field controller with `**params` overrides (e.g., `red_attr_w`, `wall_power`, `sample_ahead`, …).
- **Active dueling ranker** (`active_duel_ranker.py`, `rank_bots.py`)  
  - Actively chooses the next matchup, handles noise, detects non‑transitive cycles, and stops early when rankings are statistically certain.
- **Parameter optimizers**  
  - **Genetic Algorithm** (`ga_optimize.py`) with **Hall‑of‑Fame** and **Zoo** opponents.
  - **CMA‑ES** (`cma_optimize.py`) for faster convergence on continuous params; also supports HOF/Zoo.

---

## Controls (simulator)

- **ESC** Quit
- **SPACE** Pause
- **ENTER** Reset
- **R** Toggle random core starting positions

You can also select **Human** to control a bot with the arrow keys. It's in the retired_bots now, to prevent selection by optimizer algorithms. So move it to bots first.

---

## Install

Base requirements:
- [pybox2d](https://github.com/pybox2d/pybox2d)
- [pygame](https://www.pygame.org/)
- [tqdm](https://pypi.org/project/tqdm/) (progress bars)
- [tabulate](https://pypi.org/project/tabulate/) (pretty tables; used by some scripts)

For optimization (optional):
- `pip install cma` (CMA‑ES; this pulls in `numpy`)

---

## Bot registry (auto‑discovery)

Add a decorator in `bots/__init__.py`:

```python
# bots/__init__.py
import importlib, pkgutil
BOT_REGISTRY = {}

def register_bot(cls=None, *, name=None):
    def _wrap(c):
        BOT_REGISTRY[name or c.__name__] = c
        return c
    return _wrap(cls) if cls is not None else _wrap

def load_all_bots():
    import bots as _pkg
    for _, modname, ispkg in pkgutil.iter_modules(_pkg.__path__):
        if ispkg or modname.startswith('_'):
            continue
        importlib.import_module(f"bots.{modname}")

def get_bot_registry():
    return dict(BOT_REGISTRY)
```

Usage:

```python
from bots import load_all_bots, get_bot_registry
load_all_bots()
REG = get_bot_registry()
print(sorted(REG.keys()))  # all discovered bots
```

---

## Monte‑Carlo simulation with noise

`win_probabilities.py` now supports sensor noise, side swapping, and seeding.

```bash
# Two bots, 1000 games, default noise, ~50% side swaps
python win_probabilities.py --games 1000 SimpleBot PotentialWinner

# Heavier noise, always swap sides, reproducible
python win_probabilities.py --games 500   --sensor-pos-sigma 0.03 --sensor-angle-sigma-deg 3.0   --swap-sides-prob 1.0 --seed 1234   SimpleBot PotentialWinner
```

**Flags**
- `--sensor-pos-sigma` (m), `--sensor-angle-sigma-deg` (°)
- `--swap-sides-prob` (0–1)
- `--seed` (base integer seed)

> Progress bars now show _“Simulating games”_ with dynamic width, and the legacy “Team 1/Team 2” dump is suppressed in favor of clearer summaries.

---

## Active ranking (minimal simulations)

Use `rank_bots.py` (backed by `active_duel_ranker.py`) to actively schedule the most informative matches and stop when the order (or top‑k) is statistically certain.

```bash
# Rank four bots
python rank_bots.py SimpleBot AegisPilot SimpleBot2 SimpleBot3

# Tighter CIs, reproducible
python rank_bots.py --games-per-batch 5 --max-matches 3000 --z 2.58   --seed 1234 --sensor-pos-sigma 0.02 --sensor-angle-sigma-deg 2.0   SimpleBot AegisPilot SimpleBot2 SimpleBot3
```

What you’ll see:
- Live progress: `BotA vs BotB | undecided=<count>`
- Summary: total/partial order, **non‑transitive cycles (SCCs)**, and a CSV of pairwise probabilities + Wilson CIs.

---

## AegisPilot (parameterized potential‑field bot)

File: `bots/aegis_pilot.py`

Constructor accepts `**params` to override weights/shape (examples):
- `partner_repel_w`, `pair_repel_w`, `red_attr_w`, `green_attr_w`
- `wall_power`, `wall_scale`
- `sample_ahead`, `fd_step`
- `use_prediction=True` (if `utils.velocity_estimate.Predictor` is available)

Example:

```python
from bots import load_all_bots, get_bot_registry
load_all_bots(); REG = get_bot_registry()

bot = REG["AegisPilot"](0)  # defaults
tuned = REG["AegisPilot"](1, red_attr_w=1.5, wall_power=5.0, sample_ahead=0.06, use_prediction=True)
```

---

## Optimizing parameters

### Genetic Algorithm

```bash
# Optimize AegisPilot vs. discovered opponents
python ga_optimize.py AegisPilot --generations 20 --pop 24 --games-per-opponent 12 --seed 123   --include-zoo --zoo-path zoo/aegis
```

Features:
- **Hall‑of‑Fame (HOF)**: keeps top genomes as extra opponents.
- **Zoo**: saves best of each generation to JSON; future runs can include them with `--include-zoo`.
- Outputs: `ga_checkpoint.json`, **`best_params.json`**.

### CMA‑ES

```bash
# Install once: pip install cma
python cma_optimize.py AegisPilot --iters 60 --games-per-opponent 8   --include-zoo --zoo-path zoo/aegis

# Warm start from GA/CMA best
python cma_optimize.py AegisPilot --init-params best_params.json --iters 40
```

Outputs: **`best_params.json`**, optional Zoo JSONs per iteration.

---

## Images

<img src="https://github.com/elefAntti/uprising_simulator/blob/main/images/win_probabilities.png" alt="One against one" width="400px">
<img src="https://github.com/elefAntti/uprising_simulator/blob/main/images/one_against_many.png" alt="One against many" width="600px">
<img src="https://github.com/elefAntti/uprising_simulator/blob/main/images/all_against_all.png" alt="All against all" width="100%">

---

## Tips & reproducibility

- Use `--seed` (or `--init-params`) for reproducible experiments.
- Keep **Zoo/HOF** opponents enabled to avoid forgetting and improve robustness.
- Start with modest evaluation budgets (games per opponent) and increase when refining.

---
## Behavior cloning

* `--resume <checkpoint.pth>`
  Full resume (model + optimizer + scheduler + AMP scaler + normalizer + epoch counter).
* `--init-weights <weights.pth>`
  Load **just the model weights** from a checkpoint or raw state dict, then start a **fresh run** (new optimizer/schedule).

  * Works with dicts containing keys like `model`, `model_state`, or a raw state\_dict.
  * Optional: `--init-use-norm` will also load `normalizer_mean/std` from the checkpoint if present.
  * Optional: `--init-strict` to enforce exact key match (defaults to non-strict).

> If you pass both `--resume` and `--init-weights`, `--resume` wins and `--init-weights` is ignored.

### Examples

**1) Resume a previous run (exact continuation):**

```bash
python train_distill_init_from_pth.py \
  --npz-glob "logs_npz/*.npz" \
  --out runs/bc_run \
  --resume runs/bc_run/checkpoint.pth \
  --workers 4 --batch-size 1024
```

**2) Start a new run but initialize from existing weights:**

```bash
python train_distill_init_from_pth.py \
  --npz-glob "logs_npz/*.npz" \
  --out runs/bc_finetune \
  --init-weights pretrained/bc_policy_200.pth \
  --init-use-norm \
  --epochs 20 --batch-size 1024 --workers 4
```

**3) Initialize non-strictly (ignore a few missing/extra keys) and keep our cached normalization:**

```bash
python train_distill_init_from_pth.py \
  --npz-glob "/content/logs_npz_local/*.npz" \
  --out runs/bc_adapt \
  --init-weights external/model.pth \
  --init-strict false \
  --epochs 10 --batch-size 2048 --workers 4
```

## DAgger

After behavior cloning logs, it might be useful to let the student control the bot part of the time and the teacher log what it would have done in those situations. For these purposes there is the following script:

No problem—you’ve got a few solid paths even if you trained with a different script and don’t have normalization stats.

## Option A — Rebuild mean/std quickly from your NPZ logs

Use this tiny helper to compute and cache `norm_stats` from your existing shards (fast, scans a subset):


```bash
python compute_norm_from_npz.py --glob "logs_npz/*.npz" --out runs/stats --batches 100 --batch-size 2048
# Produces: runs/stats/norm_stats.npz and norm_stats.json
```

Then run DAgger with your plain `.pth`:

```bash
python log_dagger.py --episodes 200 \
  --teachers AegisPilot@AegisPilot_gen15_afb0c584_0.777 \
  --student-frac 0.3 \
  --student-script bc_policy_200.pth \
  --student-mean-std runs/stats/norm_stats.npz \
  --out logs_dagger_npz
```

## Option B — Skip normalization (if your model expects raw inputs)

I patched DAgger to support a **no-normalize** mode and to **infer input\_dim from the checkpoint** automatically:

* **[log\_dagger.py](sandbox:/mnt/data/log_dagger.py)** (also bundled: [dagger\_no\_stats\_bundle.zip](sandbox:/mnt/data/dagger_no_stats_bundle.zip))

Run:

```bash
python log_dagger.py --episodes 200 \
  --teachers AegisPilot@AegisPilot_gen15_afb0c584_0.777 \
  --student-frac 0.3 \
  --student-script bc_policy_200.pth \
  --student-no-norm \
  --out logs_dagger_npz
```

Notes:

* The loader will try TorchScript first.
* If it’s a raw state\_dict, it will **infer** `input_dim` from the first Linear layer’s weight shape.
* You can override the architecture if needed:

  ```
  --student-hidden "512,256,128" --student-dropout 0.05
  ```

## Option C — Our resume checkpoint (if available)

If you have a `checkpoint.pth` saved by my patched trainer, it already includes `normalizer_mean/std` and `args`, so DAgger will reconstruct everything automatically:

```bash
python log_dagger.py --episodes 200 \
  --teachers AegisPilot@AegisPilot_gen15_afb0c584_0.777 \
  --student-frac 0.3 \
  --student-script runs/bc_policy/checkpoint.pth \
  --out logs_dagger_npz
```

## What if we only trained a critic and thepolicy would be a dynamic window optimization using it?

Files
- utils_obs_canonical.py
- log_critic_aug.py
- train_critic_augmented.py
- bots_critic_mpc_sym.py

### Example, logging
> python log_critic.py \
  --out logs_critic_npz \
  --episodes 500 \
  --ally AegisPilot \
  --opponents SimpleBot2 ConeKeeper TerritoryDash AuctionStrider \
  --agent-idx 0 \
  --canonical \
  --augment-reflect \
  --side-swap

### Example, training
> python train_critic.py \
  --data "logs_critic_npz/*.npz" \
  --out runs/critic_sym \
  --epochs 12 --batch 4096


---
## License

See repository license.
