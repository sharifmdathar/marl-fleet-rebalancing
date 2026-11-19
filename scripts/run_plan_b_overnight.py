#!/usr/bin/env python
"""Sequential overnight training queue for the Plan B (D100) study.

Runs ONE multi-worker training job at a time on purpose: two concurrent
4-worker runs exhausted RAM on this box before (see scripts/run_queue.sh).
Resumable: a run whose ``final_validation.json`` already exists is skipped,
because train.py writes that file only after the full iteration loop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
OUT_ROOT = ROOT / "runs" / "plan_b"
LOGS = OUT_ROOT / "_logs"
SEEDS = range(5)

ARMS = (
    ("attention-mappo", "Attention-MAPPO", ["--arch", "attention"]),
    ("dr-attention-mappo", "DR-Attention-MAPPO",
     ["--arch", "attention", "--domain-randomize"]),
    ("dr-mlp-mappo", "DR-MLP-MAPPO", ["--arch", "mlp", "--domain-randomize"]),
)
IPPO_ARM = ("attention-ippo", "Attention-IPPO", ["--algo", "ippo", "--arch", "attention"])


def build_jobs(include_ippo: bool) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    for tag, _label, extra in ARMS + ((IPPO_ARM,) if include_ippo else ()):
        algo = "ippo" if "--algo" in extra else "mappo"
        for seed in SEEDS:
            name = f"{tag}-s{seed}"
            cmd = [
                str(PY), "-u", "-m", "marl_fleet_rebalancing.experiments.train",
                "--env", "nyc", "--algo", algo,
                "--seed", str(seed), "--iterations", "200",
                "--eval-interval", "50", "--eval-episodes", "12",
                "--device", "cpu", "--num-workers", "4",
                "--out-root", str(OUT_ROOT), "--name", name,
                *extra,
            ]
            jobs.append((name, cmd))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-ippo", action="store_true",
                    help="append the conditional Attention-IPPO arm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    queue_log = LOGS / "queue.log"
    jobs = build_jobs(args.include_ippo)
    done = skipped = failed = 0
    started = time.time()
    print(f"[queue] {len(jobs)} jobs, one at a time; out={OUT_ROOT}", flush=True)

    for i, (name, cmd) in enumerate(jobs, 1):
        run_dir = OUT_ROOT / name
        marker = run_dir / "final_validation.json"
        if marker.exists():
            skipped += 1
            line = f"[queue] SKIP  {i}/{len(jobs)} {name} ({datetime.now().isoformat(timespec='seconds')})"
            print(line, flush=True)
            with open(queue_log, "a") as f:
                f.write(line + "\n")
            continue
        if args.dry_run:
            print(" ".join(cmd))
            continue

        log_path = LOGS / f"{name}.log"
        t0 = time.time()
        header = f"[queue] START {i}/{len(jobs)} {name} ({datetime.now().isoformat(timespec='seconds')})"
        print(header, flush=True)
        with open(queue_log, "a") as qlog, open(log_path, "w") as log:
            qlog.write(header + "\n")
            qlog.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0
        if rc == 0 and marker.exists():
            done += 1
            status = "DONE"
        else:
            failed += 1
            status = f"FAILED rc={rc}"
        line = (f"[queue] {status:12s} {name} {elapsed / 60:.1f} min "
                f"({datetime.now().isoformat(timespec='seconds')})")
        print(line, flush=True)
        with open(queue_log, "a") as f:
            f.write(line + "\n")

    total = (time.time() - started) / 60
    print(f"[queue] finished: {done} trained, {skipped} skipped, {failed} failed "
          f"in {total:.1f} min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
