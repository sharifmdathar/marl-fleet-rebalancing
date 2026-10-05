#!/usr/bin/env python3
r"""Cross-seed aggregation for shift-eval CSVs.

Every CSV in a results dir is named after its run dir. Pass a regex with
named groups `algo` and `seed` so runs map to (algo, training seed):

    uv run python scripts/aggregate_seeds.py \
        --results results/shift_eval --pattern '(?P<algo>ippo|mappo)-.*-s(?P<seed>\d+)'

Prints per-shift mean+-std over training seeds, and, when two algos are
present, the paired delta (b - a) per shift with a sign count.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

SHIFT_ORDER = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6"]


def read_dir(d: Path, pattern: re.Pattern, metric: str) -> dict:
    out: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for path in sorted(d.glob("*.csv")):
        m = pattern.fullmatch(path.stem)
        if not m:
            continue
        key = (m["algo"], int(m["seed"]))
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row["metric"] == metric:
                    out[key][row["shift"]] = float(row["mean"]) * 100
    return dict(out)


def fmt(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = mean(vals)
    if len(vals) < 2:
        return f"{m:.1f}"
    return f"{m:.1f}+-{pstdev(vals):.1f}"


def retention(table: dict[str, float]) -> float | None:
    s0 = table.get("S0")
    if not s0:
        return None
    ratios = [v / s0 for s, v in table.items() if s != "S0"]
    return mean(ratios) * 100 if ratios else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/shift_eval")
    ap.add_argument("--pattern",
                    default=r"(?P<algo>ippo|mappo|greedy|lp).*?-s(?P<seed>\d+)$")
    ap.add_argument("--metric", default="service_rate")
    args = ap.parse_args()

    data = read_dir(Path(args.results), re.compile(args.pattern), args.metric)
    if not data:
        raise SystemExit(f"no runs matched {args.pattern!r} in {args.results}")

    algos = sorted({a for a, _ in data})
    seeds = sorted({s for _, s in data})
    shifts = [s for s in SHIFT_ORDER if any(s in t for t in data.values())]
    shifts += sorted({s for t in data.values() for s in t} - set(shifts))

    width = max(len(a) for a in algos) + 2
    print(f"n_seeds per algo: " + ", ".join(
        f"{a}={len({s for x, s in data if x == a})}" for a in algos))
    print()
    print("metric: " + args.metric + " (%)   mean+-std over training seeds")
    header = "shift".ljust(7) + "".join(a.ljust(width) for a in algos)
    print(header)
    for shift in shifts:
        row = shift.ljust(7)
        for a in algos:
            vals = [data[(a, s)][shift] for s in seeds
                    if (a, s) in data and shift in data[(a, s)]]
            row += fmt(vals).ljust(width)
        print(row)
    print()

    print("retention (mean SR(shift)/SR(S0) over non-S0 shifts, %)")
    for a in algos:
        rets = [r for s in seeds if (r := retention(data.get((a, s), {}))) is not None]
        print(f"  {a.ljust(width)}{fmt(rets)}")

    if len(algos) == 2:
        a, b = algos
        print()
        print(f"paired delta {b} - {a} (pp), per shift")
        print("shift".ljust(7) + "delta".ljust(10) + "n>0")
        for shift in shifts:
            deltas = [data[(b, s)][shift] - data[(a, s)][shift]
                      for s in seeds
                      if (a, s) in data and (b, s) in data
                      and shift in data[(a, s)] and shift in data[(b, s)]]
            if not deltas:
                continue
            wins = sum(d > 0 for d in deltas)
            print(shift.ljust(7) + fmt(deltas).ljust(10) + f"{wins}/{len(deltas)}")
        pair = [retention(data[(b, s)]) - retention(data[(a, s)])
                for s in seeds
                if (a, s) in data and (b, s) in data
                and retention(data[(a, s)]) is not None
                and retention(data[(b, s)]) is not None]
        if pair:
            print()
            print(f"retention delta {b} - {a} (pp): {fmt(pair)}  "
                  f"n>0: {sum(d > 0 for d in pair)}/{len(pair)}")


if __name__ == "__main__":
    main()
