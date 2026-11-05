# MARL Fleet Rebalancing Under Demand Shift

**Research question:** How does a frozen fleet-rebalancing policy degrade when
passenger demand at deployment differs from the demand it was trained on — and do
independent (IPPO) and centralized-training (MAPPO) agents degrade differently,
relative to a tuned non-RL baseline?

This is a **measurement study**, not an algorithm paper: train on one demand
distribution, freeze the policy, stress-test on a parameterized family of
shifted distributions (S0–S7), plot degradation curves. Scope, hypotheses
experiments; agents must refuse work that contradicts it.

## Setup

```bash
curl https://mise.jdx.dev/install.sh | sh   # if mise is not already installed
mise install                                 # Python 3.14, just, uv
mise exec -- uv sync                         # creates .venv and installs deps
```

If Python 3.14 and uv are already on the path, `uv sync` alone is enough.
Run the tests with `uv run pytest`. `main.py` is a visualization smoke test
only; experiments never import from it.

## Data (NYC16)

Two files are downloaded manually from NYC TLC. Both are git-ignored and live
under `dataset/`.

1. Taxi zone shapefile: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip`
   unpacked into `dataset/taxi_zones/`. The archive nests a read-only
   `taxi_zones/` folder; the unpack recipe is in `docs/nyc-pipeline.md`.
2. HVFHV trip records, April 2026:
   `https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2026-04.parquet`
   (510 MB) saved as `dataset/fhvhv_tripdata_2026-04.parquet`.

Build the road network and the split windows, then cut the 16-zone study build:

```bash
uv run python -m ride_gym.data_tools.nyc.zone_centroids
uv run python -m ride_gym.data_tools.nyc.build_nyc_network
uv run python -m ride_gym.data_tools.nyc.build_splits
uv run python scripts/build_zone_subset.py --num-zones 16
```

Stage 2 (`preprocess_orders`) is optional: nothing in the study reads its
output. Full stage-by-stage detail, expected file layout and the degree-vs-metre
gotcha are in `docs/nyc-pipeline.md`.

## Training and evaluation

```bash
# Train one seed (use --algo ippo for the independent arm)
uv run python -m marl_fleet_rebalancing.experiments.train \
    --env nyc --algo mappo --seed 0 --iterations 200

# Evaluate a checkpoint on 12 held-out validation windows (argmax by default)
uv run python -m marl_fleet_rebalancing.experiments.evaluate \
    --env nyc --checkpoint runs/<run>/checkpoint_best.pt \
    --episodes 12 --base-seed 12000

# S0-S7 shift evaluation for every run, plus the greedy and LP baselines
uv run python -m marl_fleet_rebalancing.experiments.shift_eval \
    --all-runs --include-baselines --episodes 5

# Re-tune the LP on training windows; aggregate results across seeds
uv run python scripts/tune_lp_objective.py
uv run python scripts/aggregate_seeds.py --results results/shift_eval_nyc
```

A 200-iteration NYC16 run takes about 75 minutes on a 12-core CPU; the env
rollout, not the PPO update, dominates. The `just` wrappers are optional, e.g.
`just train-mappo env=nyc iters=200 rollouts=6` (the recipe defaults to three
rollout episodes, not the paper's six). Every environment and reward constant
lives in `NYCConfig` in `src/marl_fleet_rebalancing/config.py`; nothing numeric
is set inside the training loop.

## Reproducibility archive

The checkpoints and raw evaluation outputs behind the paper's tables are
packaged as `results/nyc16-marl-fleet-rebalancing-repro-v1.0.0.tar.gz`
(49 MB, 113 files, SHA-256
`3c3f9500d26738d99085a3dd3cb1dd47cc94252bcdd3751915aeb056c0c2bfb2`).
It is not in git, because checkpoints are git-ignored; it is available from the
authors on request. After unpacking, verify it with:

```bash
sha256sum -c SHA256SUMS
```

`MANIFEST.json` records the two training commits (MAPPO
`f570f22a11ad26cbb18d0c96f10531fffdec23e0`, IPPO
`819f7a5fa7c4ab8b2a6aa17fc7a642a6d07b169b`), the packaging commit, the
observation/state/action dimensions, the evaluation protocol and per-file
hashes. Re-evaluating the tabulated values from this archive is exact;
bit-exact retraining is not claimed.

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
  orders/hour). A small synthetic toy env is used only for fast unit tests; it
  is not a study environment and nothing from it is plotted or reported.
- Baselines tuned on NYC16 training windows: the LP rebalancer leads the
  service/deadhead frontier. LP at S0: 69.7% service at 290 empty km.
- MAPPO-200 (5 seeds, deterministic argmax) serves 65.2% ± 1.2 at 436 ± 26
  empty km on S0; IPPO-200 serves 63.7% ± 1.9 at 478 ± 19. Both degrade most
  under S4 cold-zone concentration (MAPPO retains 50.8%, IPPO 49.2%); the LP
  is the most robust arm there (67.3%).
- Checkpoint selection uses the held-out validation split only. The 200-iter
  arm plateaus around iteration 100; validation is the load-bearing protocol
  and is regression-locked in `tests/test_training_validation.py`.
- H1–H4 remain open; the measured result so far is the degradation curve and
  the frontier gap, not a difference between IPPO and MAPPO.
