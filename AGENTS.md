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

## Mistakes already made here — do not repeat

Concrete failures from this project. Each cost days.

- **Check reward constants against reward arithmetic before setting them.**
  `relocator_share = 0.35` sat below the break-even `beta × km_per_relocation`
  = 0.374, so relocation was net-negative even when it worked perfectly, and
  the agent learned to churn. Worked example: env 2.8.
- **Never report a metric that can be bought.** Service rate alone is
  purchasable with deadhead distance; a policy scored +8.8pp over LP while
  being 2.4x worse per empty km. Always pair a gain with the resource it cost.
- **A short run cannot grade a long-horizon question.** 40 iterations said
  nothing about a 200-iteration plateau. Check the full-length trend before
  concluding, and say which you have.
- **Apply a fix at train AND eval, or nowhere.** The relocation-claim fix was
  evaluated on a checkpoint trained without it, giving a fake +7pp.
- **Do not instrument by mutating the state you are measuring.** Tracing the
  completion reward consumed the `_seen_completed_ids` delta before the real
  call, reporting "0 credits" twice when the credit fired on 64% of
  completions. Score first, restore, then call through.
- **Verify a diagnostic before believing its output.** A probe read stale
  generator state and claimed all demand sat in one zone; the generator was
  correct.
- **A view that crops can hide the thing being measured.** Zooming to the zone
  extent cut 29% of road nodes off-screen, so cars leaving the frame were
  invisible and "cars moved north-east" was an artifact.
- **Compare paired, never two means.** Per-window demand varies 36-69%; paired
  on identical windows the same comparison has half the spread. See
  `docs/eval-noise.md`.
- **Validate an agent's targeting directly, not only its score.** Validation
  service rate hid a policy targeting at chance. Relocation precision is the
  acceptance test.
- **Check what the evaluation set actually contains.** Every seeded reset
  rewound the held-out window cursor, so every validation number up to D70 came
  from one window out of twelve, and the ranking of LP against RL changed once
  the full split was used. Print the window index before trusting a held-out
  number.
- **Tune the baseline on the environment you compare on.** The LP kept defaults
  from an older build; re-tuned, it went from 54% to 70% service.
- **Sum the zone rewards and compare with the objective.** A relocator bonus
  paid on top of gamma made the shared policy's objective R plus 0.65 per moved
  car. Training raised that and lowered R.
- **Do not diagnose from an undertrained checkpoint.** An iteration-30 policy
  was analysed for hours before anyone checked it had converged.

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
- When an earlier claim turns out to be wrong, record the correction where the
  original was made. Do not quietly supersede it.
- Correct a public claim in the same breath as making a new one.

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
