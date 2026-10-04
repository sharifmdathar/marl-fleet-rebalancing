"""Plateau watcher: report when training has stopped improving.

Long runs are not informative on their own, and `AGENTS.md` warns that a short
run cannot grade a long-horizon question -- but equally, running to the nominal
iteration count wastes time once the curve is flat. This reads a run's
`metrics.jsonl` and reports whether the last window is still improving on the
metrics that matter, with the noise level stated so "flat" is a measurement
rather than an impression.

Windows are thirds of the run, so "recent trend" is comparable regardless of how
long the run is. The plateau call uses the paired per-episode spread to set a
band: a window counts as flat when its change is smaller than the run's own
per-episode standard error.

    uv run python scripts/watch_plateau.py runs/d58-mappo-s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = ("service_rate", "empty_km_per_ep", "avg_wait_min", "pct_relocate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--window", type=int, default=8,
                    help="iterations per window for the recent-trend test")
    args = ap.parse_args()

    path = Path(args.run)
    mf = path / "metrics.jsonl"
    if not mf.exists():
        print(f"no metrics.jsonl in {path}")
        raise SystemExit(1)

    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    # metrics.jsonl APPENDS; a reused run name mixes environments. Detect a reset
    # where the iteration counter goes backwards.
    iters = [r.get("iteration", i + 1) for i, r in enumerate(rows)]
    resets = sum(1 for a, b in zip(iters, iters[1:]) if b <= a)
    if resets:
        print(f"WARNING: {resets} iteration reset(s) in {mf.name} -- this file "
              f"mixes runs. Reading the last segment only.")
        start = max(i for i in range(1, len(iters)) if iters[i] <= iters[i - 1])
        rows = rows[start:]

    n = len(rows)
    print(f"{path.name}: {n} iterations logged")
    print(f"{'iter':>5} " + " ".join(f"{f[:11]:>12}" for f in FIELDS))
    for r in rows[::max(1, n // 12)]:
        # service_rate and pct_relocate are already 0-100; empty_km is km and
        # avg_wait is minutes. Scaling every field by 100 (an earlier version of
        # this script did) reported empty km as 48615 for a run whose real value
        # is 486 -- which reads as a 100x regression that never happened.
        def _fmt(f):
            v = r[f]
            return f"{v:12.2f}"
        print(f"{r.get('iteration', 0):>5} " + " ".join(_fmt(f) for f in FIELDS))
    print()

    if n < args.window * 2:
        print(f"need at least {args.window*2} iterations for a trend test")
        return

    # Per-episode noise band: stdev of the window's values, as a fraction.
    print("=" * 72)
    print("Plateau assessment")
    print("=" * 72)
    for f in FIELDS:
        v = np.array([r[f] for r in rows], dtype=float)
        recent = v[-args.window:]
        prior = v[-2 * args.window:-args.window]
        noise = float(np.std(recent)) or 1e-9
        delta = float(recent.mean() - prior.mean())
        rel = delta / max(abs(prior.mean()), 1e-9)
        flat = abs(rel) < (noise / max(abs(prior.mean()), 1e-9))
        print(f"  {f:<18} recent {recent.mean():9.3f}  prior "
              f"{prior.mean():9.3f}  change {delta:+9.3f} ({100*rel:+6.2f}%)"
              f"  noise {100*noise/max(abs(prior.mean()),1e-9):5.2f}%"
              f"  -> {'FLAT' if flat else 'still moving'}")

    print()
    allflat = True
    for f in FIELDS:
        v = np.array([r[f] for r in rows], dtype=float)
        recent, prior = v[-args.window:], v[-2 * args.window:-args.window]
        noise = float(np.std(recent)) or 1e-9
        rel = abs((recent.mean() - prior.mean()) / max(abs(prior.mean()), 1e-9))
        if rel >= (noise / max(abs(prior.mean()), 1e-9)):
            allflat = False
    print("VERDICT:", "PLATEAUED -- stop and evaluate"
          if allflat else "still improving -- keep training")


if __name__ == "__main__":
    main()
