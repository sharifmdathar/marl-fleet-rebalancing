# AGENTS.md — how to work in this repo

General rules for any agent or human working here. These are about *how* to
work, not about what the project currently is. The research targets live in
`../GOAL.md`.

## Measure before claiming

- Never state a result you have not run. No estimated numbers, no
  "should be around", no numbers carried over from another config.
- If a run failed, say it failed. Do not report the best of several attempts
  without saying so.
- Distinguish measured, inferred, and assumed. Label them.

## Do not fabricate process

- No backdating commits, no invented authorship, no synthetic "prior work"
  notes to make a history look busier or older.
- Do not delete or rewrite git history to hide what happened.

## Keep the record honest

- When a measurement is invalidated (environment, protocol, or code changed
  underneath it), delete the artifact. A stale number is worse than a missing
  one, because it gets cited.
- A fix that only looks good because it was measured wrong is not a fix.
  Re-measure under the corrected conditions before reporting an effect size.
- Log negative results. They are the cheapest thing in the repo and the most
  often re-derived.

## Environment and code changes

- Any change to environment dynamics, reward, or observation/action space is
  a version bump plus one entry recording what changed and why. No exceptions
  for "small" changes.
- Every experiment parameter lives in a config object. No magic numbers inside
  training loops.
- Every run logs: config, seed(s), scenario ID, and its KPIs.

## Baselines and comparison fairness

- Never weaken a baseline to make a method look better.
- A baseline must be tuned on training/validation demand only, with effort
  comparable to the method it is compared against. Beating a lazily-tuned
  baseline is not a result.
- Test scenarios are touched only for final evaluation. No tuning on them.

## Reporting

- Every reported result is ≥ 5 seeds, mean ± std. Single-seed numbers are
  debugging output.
- The headline is never in-distribution performance.
- Frame the contribution around the experimental question. Do not claim a new
  algorithm unless one was actually built.
- State limitations and external-validity limits explicitly.

## Repo hygiene

- Never commit: datasets, checkpoints, `*.pt`, `*.mp4`, large generated
  artifacts. They are git-ignored; keep it that way.
- Commits are conventional (`feat:`, `fix:`, `exp:`, `docs:`), short, and
  describe what changed — not the process of changing it.
- Code carries few comments. Explain *why* where it is not obvious; do not
  narrate what the next line does.
- No AI-generated filler in docs. If a document does not change a decision,
  it does not need to exist.
- Do not put working notes, transcripts, or scratch analysis in the repo.

## Asking

- Scope changes and anything touching the research targets are the user's
  call, not the agent's. Ask.
- If a premise looks wrong, say so with the evidence and stop, rather than
  building on it.
