#!/usr/bin/env python3
"""Aggregate D28 seed-replica shift evals into cross-seed head-to-head tables.

Merges the run of record (results/shift_eval/, training seed 0) with the
replica evals (results/shift_eval_seeds/, training seeds from
scripts/run_d28_seed_replicas.sh), pairs the D28 IPPO/MAPPO rows across
training seeds, and prints service-rate mean +- std over training seeds per
shift, plus a cross-seed retention delta with a paired per-shift sign count.

The two algos were trained once per seed and evaluated on the SAME eval seeds
(1000-1004), so the comparison is paired per shift: the delta
MAPPO - IPPO is computed per shift per training seed, then summarized.

Usage: uv run python scripts/aggregate_seed_replicas.py \
           [--record results/shift_eval] [--replicas results/shift_eval_seeds]
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

SHIFT_ORDER = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6"]

# run-dir name of a replica stage2 -> its (algo, seed)
REPLICA_RE = re.compile(r"^(ippo|mappo)-nyc-d28-s(\d+)-stage2$")

# policies from the run of record that are part of the paired comparison
RECORD_MAP = {
    "ippo-nyc-d28-stage2": ("ippo", 0),
    "mappo-nyc-d28-stage2": ("mappo", 0),
}


def read_service_rates(csv_dir):
    """Return {(algo, seed): {shift: (mean, std)}} for D28 stage2 policies."""
    out = {}
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        mapping = RECORD_MAP.get(name)
        if mapping is None:
            m = REPLICA_RE.match(name)
            if m:
                mapping = (m.group(1), int(m.group(2)))
        if mapping is None:
            continue
        table = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("metric") == "service_rate":
                    # CSV stores fractions; convert to % once, at the source
                    table[row["shift"]] = (
                        float(row["mean"]) * 100,
                        float(row["std"]) * 100,
                    )
        if table:
            out[mapping] = table
    return out


def retention(table):
    """Mean over non-S0 shifts of SR(shift)/SR(S0)."""
    s0 = table.get("S0")
    if not s0:
        return None
    ratios = [table[s][0] / s0[0] for s in table if s != "S0"]
    return sum(ratios) / len(ratios) * 100 if ratios else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="results/shift_eval")
    ap.add_argument("--replicas", default="results/shift_eval_seeds")
    args = ap.parse_args()

    data = read_service_rates(args.record)
    data.update(read_service_rates(args.replicas))

    algos = sorted({a for a, _ in data})
    seeds = sorted({s for _, s in data})
    if not seeds:
        raise SystemExit("No D28 stage2 CSVs found in either directory.")

    print(f"Training seeds found: {seeds} (algos: {', '.join(algos)})\n")

    # Head-to-head table: per-algo, per-shift mean over training seeds
    # (eval-seed mean and std are already aggregated inside each CSV; here we
    # report the mean of those means and the spread across training seeds).
    header = "| Shift | " + " | ".join(
        f"{a.upper()} (n={len([s for s in seeds if (a, s) in data])})" for a in algos
    ) + " | Delta (MAPPO-IPPO) |"
    print("### Service rate % by training seed (mean over eval seeds 1000-1004)\n")
    print(header)
    print("|:---|" + ":---:|" * (len(algos) + 1))

    deltas = defaultdict(list)  # shift -> [mappo - ippo per seed]
    for shift in SHIFT_ORDER:
        row, per_algo = [], {}
        for algo in algos:
            vals = [data[(algo, s)][shift][0] for s in seeds if (algo, s) in data and shift in data[(algo, s)]]
            per_algo[algo] = sum(vals) / len(vals) if vals else None
            row.append(f"{sum(vals)/len(vals):.1f} ± {spread(vals)}" if len(vals) > 1 else (f"{vals[0]:.1f}" if vals else "—"))
        d = None
        if per_algo.get("ippo") is not None and per_algo.get("mappo") is not None:
            d = per_algo["mappo"] - per_algo["ippo"]
            row.append(f"{d:+.1f}")
            for s in seeds:
                if ("ippo", s) in data and ("mappo", s) in data:
                    deltas[shift].append(
                        data[("mappo", s)][shift][0] - data[("ippo", s)][shift][0]
                    )
        else:
            row.append("—")
        print(f"| {shift} | " + " | ".join(row) + " |")

    # Paired per-seed delta summary
    print("\n### Paired delta (MAPPO - IPPO), per training seed\n")
    n_seeds = min(
        len([s for s in seeds if ("ippo", s) in data]),
        len([s for s in seeds if ("mappo", s) in data]),
    )
    if n_seeds == 0:
        print("No paired seeds available.")
        return
    rows = []
    for shift in SHIFT_ORDER:
        ds = deltas.get(shift, [])[:n_seeds]
        if not ds:
            continue
        m = sum(ds) / len(ds)
        rows.append((shift, m, spread(ds), sum(1 for x in ds if x > 0)))
    print("| Shift | Mean delta | Spread across seeds | MAPPO wins n/n |")
    print("|:---|:---:|:---:|:---:|")
    for shift, m, sp, wins in rows:
        print(f"| {shift} | {m:+.1f} | ± {sp} | {wins}/{n_seeds} |")

    ret = {}
    for algo in algos:
        r = [retention(data[(algo, s)]) for s in seeds if (algo, s) in data]
        r = [x for x in r if x is not None]
        if r:
            ret[algo] = (sum(r) / len(r), spread(r))
    if "ippo" in ret and "mappo" in ret:
        print(
            f"\nMean retention: MAPPO {ret['mappo'][0]:.1f}% (± {ret['mappo'][1]}"
            f" across seeds) vs IPPO {ret['ippo'][0]:.1f}% (± {ret['ippo'][1]})"
        )


def spread(vals):
    """Population-ish spread across training seeds (max-min / 2)."""
    if len(vals) < 2:
        return "0.0"
    return f"{(max(vals) - min(vals)) / 2:.1f}"


if __name__ == "__main__":
    main()
