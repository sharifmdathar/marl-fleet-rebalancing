# AGENTS.md — rules for any agent (human or AI) working in this repo

## The project in one paragraph

We measure how MARL rebalancing policies degrade under passenger-demand
distribution shift. Train on one demand distribution, FREEZE the policy, then
stress-test on a parameterized family of shifted distributions and plot
degradation curves. Main comparison: IPPO vs MAPPO. Control: a tuned
non-RL rebalancing baseline. We are NOT proposing a new algorithm and NOT
claiming anyone "solved" robust rebalancing — this is a measurement study.

## Commands

- Run smoke test:        `uv run python main.py`
- Run interactive:       `uv run python main.py -i`
- Run MP4 output:        `uv run python main.py -o episode.mp4`
- Test PettingZoo env:   `uv run python -m marl_fleet_rebalancing.env.test_wrapper`
- Install deps:          `uv sync`
- Run tests:             `uv run pytest`            <TODO: confirm once tests exist>
- Lint/format:           `uv run ruff check . && uv run ruff format .`  <TODO: confirm>

## Repo layout

See `docs/architecture.md`. Key rule: environment code, agent code, and
experiment/eval code live in separate modules. Experiments never import from
`main.py`; `main.py` is a smoke test only.

## Coding conventions

- Python ≥ 3.9, type hints on all public functions (match existing style).
- All experiment parameters live in a config object/dataclass — never as
  magic numbers inside training loops. `main.py`'s constants are the template.
- Every training/eval run must log: config, seed(s), git commit hash, demand
  scenario ID, and output KPIs.

## Experiment rules (non-negotiable)

1. **Seeds:** every reported result is ≥ 5 seeds, reported as mean ± std.
   Single-seed numbers are debugging output, never results.
2. **Frozen-policy protocol:** after training, policies are frozen. No
   fine-tuning, no adaptation, no parameter updates on shifted scenarios.
3. **No test-set tuning:** hyperparameters and reward shaping are tuned on
   training/validation demand only. Shifted test scenarios are touched only
   for final evaluation.
4. **Baseline fairness:** the non-RL baseline must be tuned with comparable
   effort to the RL agents. Beating a lazy baseline is not a result.
5. **Scenario versioning:** demand scenarios are defined in
   `docs/demand-shift.md` and referenced by ID (e.g., S2-magnitude-1.5x).
   New scenarios require a new entry there FIRST.

## Git rules

- Branch per experiment/feature: `exp/<scenario>-<algo>` or `feat/<thing>`.
- Never commit: `data/*.gpickle`, `*.mp4`, checkpoints, raw TLC/OSM downloads.
- Commit messages: conventional commits (`feat:`, `fix:`, `exp:`, `docs:`).

## Forbidden actions

- Do NOT change env dynamics, reward function, or observation/action space
  without: (1) an entry in `docs/decisions.md`, (2) a config version bump.
- Do NOT "improve" the baseline downward to make RL look better.
- Do NOT report in-distribution performance as the headline result — the
  headline is always the degradation curve under shift.
- Do NOT add a new dependency without a one-line justification in the PR.

## Scope and hypotheses: see `INTENT.md`

The four outcomes (H1–H4) and non-goals are defined in `INTENT.md`. Do not
steer experiments toward a preferred answer; any scope change must be logged in
`docs/decisions.md` and approved by the advisor.

## Research guardrails: see `INTENT.md`

Related work claims must match `docs/literature.md`. When unsure whether a
change alters the scientific claim, stop and ask. Refer to `INTENT.md` for
the full list of non-goals and the definition of done.