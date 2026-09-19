"""Plot frozen-policy degradation curves from shift_eval summary JSON.

Usage:
    uv run python -m marl_fleet_rebalancing.experiments.plot_shifts \\
        [summary.json] [--outdir results/figures]

Reads results/shift_eval/summary.json (policy -> shift -> metric stats)
and writes one PNG per KPI: service_rate, empty_km, avg_wait_min.
Shifts render in taxonomy order (S0..S6).
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

# Taxonomy order
SHIFT_ORDER = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6"]
SHIFT_TICKS = [
    "S0\nBase (train)",
    "S1\nPermute",
    "S2a\n×0.5 orders",
    "S2b\n×0.75 orders",
    "S2d\n×1.5 orders",
    "S2e\n×2.0 orders",
    "S3\nVal split",
    "S4\nCold zone z1",
    "S5\nBurst z0",
    "S6\nTest split",
]

KPI_SPECS = {
    "service_rate": {"ylabel": "Service Rate (%)", "scale": 100.0, "fmt": "{:.0f}%"},
    "empty_km": {"ylabel": "Empty Distance (km/ep)", "scale": 1.0, "fmt": "{:.0f}"},
    "avg_wait_min": {"ylabel": "Avg Wait Time (min)", "scale": 1.0, "fmt": "{:.2f}"},
}


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_kpi(summary: dict, kpi: str, ylabel: str, scale: float, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs: list[int] = []
    tick_labels: list[str] = []

    policies = list(summary.keys())

    for i, (key, tick) in enumerate(zip(SHIFT_ORDER, SHIFT_TICKS)):
        if any(key in summary.get(p, {}) for p in policies):
            xs.append(i)
            tick_labels.append(tick)

    # Color palette
    colors = plt.cm.tab10(range(len(policies)))

    for idx, policy in enumerate(policies):
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

        style = "--s" if "baseline" in policy.lower() or "greedy" in policy.lower() or "lp" in policy.lower() else "-o"
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt=style,
            capsize=3,
            label=policy,
            color=colors[idx],
            markersize=5,
            linewidth=1.8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Demand Shift Scenario", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"Frozen-Policy Degradation: {ylabel} across Demand Shifts", fontsize=12, fontweight="bold")
    ax.legend(fontsize="small", loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?", default="results/shift_eval/summary.json")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    sum_path = Path(args.summary)
    if not sum_path.exists():
        print(f"Error: {sum_path} not found.")
        sys.exit(1)

    summary = load_summary(sum_path)
    outdir = Path(args.outdir)

    for kpi, spec in KPI_SPECS.items():
        outpath = outdir / f"degradation_{kpi}.png"
        plot_kpi(summary, kpi, spec["ylabel"], spec["scale"], outpath)


if __name__ == "__main__":
    main()
