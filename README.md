# MARL Fleet Rebalancing Under Demand Shift

**Research question:** How does a frozen fleet-rebalancing policy degrade when
passenger demand at deployment differs from the demand it was trained on — and do
independent (IPPO) and centralized-training (MAPPO) agents degrade differently,
relative to a tuned non-RL baseline?

This is a **measurement study**, not an algorithm paper: train on one demand
distribution, freeze the policy, stress-test on a parameterized family of
shifted distributions (S0–S7), plot degradation curves. Scope, hypotheses
experiments; agents must refuse work that contradicts it.

## Getting Started

See `GETTING_STARTED.md` for detailed setup instructions.

## Dependencies

Install `mise` to manage tools (Python, just, uv):
```bash
curl https://mise.jdx.dev/install.sh | sh
```

Then in WSL:
```bash
cd /home/luffy/marl-fleet-rebalancing
mise install  # installs Python 3.14, just, uv
mise exec -- uv sync  # creates venv and installs dependencies
```

## Quick Start

```bash
# Create synthetic network for testing (when OSM data unavailable)
mise exec -- python scripts/create_synthetic_network.py

# Run training (interactive menu)
mise exec -- python scripts/run_training.py

# Or train directly
mise exec -- python -m marl_fleet_rebalancing.experiments.train \
    --env nyc --algo mappo --seed 0 --iterations 200

# Run tests
mise exec -- pytest
```

Note: `main.py` is a visualization smoke test only; experiments never
import from it.

## Training & evaluation

```bash
# Install dependencies (first time only)
uv sync

# Train (NYC, validation-based best-checkpoint tracking every 4 iters)
uv run python -m marl_fleet_rebalancing.experiments.train \
    --env nyc --algo ippo --seed 0 --iterations 120 \
    --nyc-window-limit 500 --eval-interval 4 --eval-episodes 3

# Train with just (alternative)
just train-ippo env=nyc iters=120 name=my-run

# Run all improvements script (when uv is installed)
bash scripts/run_training.sh  # or scripts/run_training.bat on Windows
```

See `IMPROVEMENTS.md` for the complete list of improvements and their commands.

## Degradation Curves

Stress-testing frozen policies under the S0–S7 shift taxonomy on NYC16,
5 seeds, deterministic argmax actions: MAPPO-200 and IPPO-200 vs greedy and
the re-tuned LP baseline. Figures regenerate with `plot_shifts_v2` after a
shift eval run.

![Service rate, empty distance and wait across shifts](docs/figures/degradation_metrics.png)
![Retention across shifts](docs/figures/retention_by_shift.png)
![S0 service vs empty distance](docs/figures/pareto_service_vs_empty.png)

Caveat: these curves use greedy (argmax) frozen actions. With sampled actions
the same checkpoints score higher on S0; the argmax-vs-sampled gap is itself
a measured result (see D89).

## Repo map

| Path | What it is |
|---|---|
| `docs/HANDOFF.md` | Current state, what's broken, what's next |
| `docs/demand-shift.md` | S0–S7 shift taxonomy (scenario IDs) |
| `docs/architecture.md` | Module map, KPI definitions |
| `src/marl_fleet_rebalancing/` | `env/` (PettingZoo wrapper), `agents/` (IPPO/MAPPO), `baselines/` (greedy, LP), `experiments/` (train, shift_eval, plotting), `demand/`, `config.py` |
| `results/` | Regenerated eval artifacts (git-ignored; figures copied to `docs/figures/`) |
| `docs/figures/` | Committed degradation curves used in this README |
| `runs/` | Training checkpoints + metrics (git-ignored) |

## Experiment rules (short version)

Every reported result: ≥ 5 seeds, mean ± std. Policies are frozen after
training — no fine-tuning on shifted scenarios. Checkpoint selection runs on
the held-out validation split (`--eval-split val`, default), never on the
training split, and `best_checkpoint.json` records how each checkpoint was
picked. Shifted test scenarios are touched only for final evaluation. Full
rules: `AGENTS.md`.

## Current status

- Study environment is NYC16 (16 real FHVHV taxi zones, 26 drivers, 130
  orders/hour). The synthetic Bandra toy env is retained only for fast unit
  tests and visualization; it is not a study environment.
- Baselines tuned on NYC16 training windows: the LP rebalancer leads the
  service/deadhead frontier. LP at S0: 69.7% service at 290 empty km.
- MAPPO-200 (env 3.4, 5 seeds, deterministic argmax) serves 64.2% at 397 km
  on S0. IPPO-200 is trained on the same protocol. Both degrade most under
  S4 cold-zone concentration; the LP is the most robust arm there.
- Checkpoint selection uses the held-out validation split only. The 200-iter
  arm plateaus around iteration 100; validation is the load-bearing protocol
  and is regression-locked in `tests/test_training_validation.py`.
- H1–H4 remain open; the measured result so far is the degradation curve and
  the frontier gap, not a difference between IPPO and MAPPO.
