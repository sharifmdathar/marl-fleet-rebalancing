#!/usr/bin/env python3
"""Rebuild head-to-head degradation tables from per-policy shift_eval CSVs.

Reads results/shift_eval/*.csv (long format: shift,metric,mean,std,min,max),
computes service-rate mean ± std per shift, mean retention relative to S0,
and prints markdown tables for service rate, empty km, and avg wait.

Usage: uv run python scripts/summarize_shift_eval.py [--csv-dir results/shift_eval]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

METRICS = ("service_rate", "empty_km", "avg_wait_min")


def load(csv_dir):
    """Return {policy: {shift: {metric: (mean, std)}}}."""
    data = {}
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        table = defaultdict(dict)
        with open(path) as fh:
            for row in csv.DictReader(fh):
                m = row.get("metric")
                if m in METRICS:
                    table[row["shift"]][m] = (float(row["mean"]), float(row["std"]))
        if table:
            data[name] = table
    return data


def retention(table):
    """Mean over shifts of SR(shift)/SR(S0), S0 excluded from the mean."""
    s0 = table.get("S0", {}).get("service_rate")
    if not s0:
        return None
    ratios = [
        table[s]["service_rate"][0] / s0[0]
        for s in table
        if s != "S0" and "service_rate" in table[s]
    ]
    return sum(ratios) / len(ratios) * 100 if ratios else None


def fmt(v):
    return f"{v[0] * 100:.1f} ± {v[1] * 100:.1f}%" if v else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="results/shift_eval")
    args = ap.parse_args()
    data = load(args.csv_dir)
    shifts = sorted(
        {s for t in data.values() for s in t},
        key=lambda x: (x[0], x[1:]),
    )

    print("### Demand Shift Degradation Summary (Service Rate % ± Std over 5 seeds)\n")
    header = "| Policy | " + " | ".join(shifts) + " | Mean Retention |"
    print(header)
    print("|:---|" + ":---:|" * (len(shifts) + 1))
    for name in sorted(data):
        t = data[name]
        row = [fmt(t[s].get("service_rate")) if s in t else "—" for s in shifts]
        r = retention(t)
        print(f"| **{name}** | " + " | ".join(row) + f" | {r:.1f}% |" if r else
              f"| **{name}** | " + " | ".join(row) + " | — |")

    for metric, label, unit in (
        ("empty_km", "Empty Repositioning Distance", "km / episode"),
        ("avg_wait_min", "Average Passenger Waiting Time", "min"),
    ):
        print(f"\n### {label} ({unit})\n")
        print("| Policy | " + " | ".join(shifts) + " |")
        print("|:---|" + ":---:|" * len(shifts))
        for name in sorted(data):
            t = data[name]
            row = [
                f"{t[s][metric][0]:.0f}" if s in t and metric in t[s] else "—"
                for s in shifts
            ]
            print(f"| **{name}** | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
