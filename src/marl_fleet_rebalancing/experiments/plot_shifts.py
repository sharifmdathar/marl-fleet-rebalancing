"""Plot frozen-policy degradation curves from shift_eval summary JSON.

Usage:
    uv run python -m marl_fleet_rebalancing.experiments.plot_shifts \
        [summary.json] [--outdir results/figures]

Reads results/shift_eval/summary.json (policy -> shift -> metric stats)
and writes one PNG per KPI: service_rate, empty_km, avg_wait_min.
Shifts render in taxonomy order (S0..S6); missing keys are skipped so
partial summaries (e.g. pre-S3/S5/S6) still plot.

mappo-proper-seed1 is drawn dashed: docs/shift-results.md finding 4 flags
it as a centralized-critic exploration failure, excluded from
IPPO-vs-MAPPO comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Taxonomy order (S2c omitted: same-config duplicate of S0)
SHIFT_ORDER = ["S0", "S1a_z0", "S1b_z30", "S2a", "S2b", "S2d", "S2e",
               "S3", "S4a_z60", "S5_z0", "S6"]
SHIFT_TICKS = ["S0\nbase (train)", "S1a\n70%→z0", "S1b\n70%→z30",
               "S2a\n×0.5 orders", "S2b\n×0.75 orders", "S2d\n×1.5 orders",
               "S2e\n×2.0 orders", "S3\nval split", "S4a\n60%→z60",
               "S5\n20% burst z0", "S6\ntest split"]

POLICIES = ["ippo-proper-seed0", "ippo-proper-seed1",
            "mappo-proper-seed0", "mappo-proper-seed1"]

# mappo seed1 flagged degenerate (finding 4) -> dashed
DEGENERATE = {"mappo-proper-seed1"}

KPI_SPECS = {
    "service_rate": {"ylabel": "Service rate", "scale": 100.0, "fmt": "{:.0f}%"},
    "empty_km": {"ylabel": "Empty distance (km/ep)", "scale": 1.0, "fmt": "{:.0f}"},
    "avg_wait_min": {"ylabel": "Avg wait (min)", "scale": 1.0, "fmt": "{:.2f}"},
}


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_kpi(summary: dict, kpi: str, ylabel: str, scale: float,
             outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs: list[int] = []
    tick_labels: list[str] = []
    for i, (key, tick) in enumerate(zip(SHIFT_ORDER, SHIFT_TICKS)):
        if any(key in summary.get(p, {}) for p in POLICIES):
            xs.append(i)
            tick_labels.append(tick)
    for policy in POLICIES:
        pdata = summary.get(policy, {})
        x, y, err = [], [], []
        for i, key in enumerate(SHIFT_ORDER):
            if key not in pdata or kpi not in pdata[key]:
                continue
            stat = pdata[key][kpi]
            x.append(i)
            y.append(stat["mean"] * scale)
            err.append(stat["std"] * scale)
        if not x:
            continue
        style = "--" if policy in DEGENERATE else "-o"
        label = policy + (" (degenerate, cf. finding 4)"
                          if policy in DEGENERATE else "")
        ax.errorbar(x, y, yerr=err, fmt=style, capsize=3,
                    label=label, markersize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=18, ha="right")
    ax.set_xlabel("Demand shift scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Frozen-policy degradation: {ylabel} vs shift severity")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?",
                    default="results/shift_eval/summary.json")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    summary = load_summary(Path(args.summary))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for kpi, spec in KPI_SPECS.items():
        plot_kpi(summary, kpi, spec["ylabel"], spec["scale"],
                 outdir / f"degradation_{kpi}.png")


if __name__ == "__main__":
    main()
