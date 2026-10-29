"""Plot frozen-policy degradation curves from shift_eval summary JSON.

Usage:
    uv run python -m marl_fleet_rebalancing.experiments.plot_shifts \\
        [summary.json] [--outdir results/figures]

Reads results/shift_eval/summary.json (policy -> shift -> metric stats)
and writes one PNG per KPI: service_rate, empty_km, avg_wait_min.

The summary mixes two environments: Bandra RL checkpoints and NYC16
baselines. They are not comparable, so each figure has two panels.
Bands are +/- 1 sample std (ddof=1) across seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHIFT_ORDER = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6", "S7"]
SHIFT_TICKS = [
    "S0\nBase",
    "S1\nPermute",
    "S2a\n×0.5",
    "S2b\n×0.75",
    "S2d\n×1.5",
    "S2e\n×2.0",
    "S3\nVal",
    "S4\nCold z1",
    "S5\nBurst z0",
    "S6\nHeld-out",
    "S7\nDrift",
]

KPI_SPECS = {
    "service_rate": {"ylabel": "Service Rate (%)", "scale": 100.0},
    "empty_km": {"ylabel": "Empty Distance (km/ep)", "scale": 1.0},
    "avg_wait_min": {"ylabel": "Avg Wait Time (min)", "scale": 1.0},
}

RL_RE = re.compile(r"^(ippo|mappo)-bandra-(\d+)-s\d+$")
BASELINE_KEYS = {"Greedy-Baseline": ("Greedy", "#2ca02c", "D"),
                 "LP-Rebalancer": ("LP", "#d62728", "v")}

ALGO_COLORS = {"IPPO": "#1f77b4", "MAPPO": "#ff7f0e"}
ITER_STYLE = {40: (0, (4, 4)), 80: (0, (2, 2)), 200: (0, ())}
ITER_MARKER = {40: "o", 80: "s", 200: "^"}


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_rl(summary: dict) -> dict[tuple[str, int], list[str]]:
    groups: dict[tuple[str, int], list[str]] = {}
    for name in summary:
        m = RL_RE.match(name)
        if m:
            groups.setdefault((m.group(1).upper(), int(m.group(2))), []).append(name)
    return groups


def series(policies: list[str], summary: dict, shift: str, kpi: str, scale: float):
    vals = []
    for p in policies:
        d = summary.get(p, {}).get(shift, {}).get(kpi)
        if d is not None:
            vals.append(d["mean"] * scale)
    if not vals:
        return None
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def draw(ax, summary, groups, kpi, scale, env: str) -> None:
    present = [i for i, s in enumerate(SHIFT_ORDER)
               if any(s in summary.get(p, {}) for p in summary)]
    if env == "bandra":
        for (algo, iters), policies in sorted(groups.items()):
            xs, ys, es = [], [], []
            for i in present:
                out = series(policies, summary, SHIFT_ORDER[i], kpi, scale)
                if out is None:
                    continue
                xs.append(i); ys.append(out[0]); es.append(out[1])
            if not xs:
                continue
            ax.plot(xs, ys, color=ALGO_COLORS[algo], ls=ITER_STYLE[iters],
                    marker=ITER_MARKER[iters], ms=5, lw=1.8,
                    label=f"{algo}-{iters}")
            lo = np.array(ys) - np.array(es)
            if kpi == "service_rate":
                lo = np.maximum(lo, 0.0)
            ax.fill_between(xs, lo, np.array(ys) + np.array(es),
                            color=ALGO_COLORS[algo], alpha=0.15, lw=0)
    else:
        for key, (label, color, marker) in BASELINE_KEYS.items():
            if key not in summary:
                continue
            xs, ys, es = [], [], []
            for i in present:
                out = series([key], summary, SHIFT_ORDER[i], kpi, scale)
                if out is None:
                    continue
                xs.append(i); ys.append(out[0]); es.append(out[1])
            if not xs:
                continue
            ax.plot(xs, ys, color=color, ls=(0, ()), marker=marker, ms=5,
                    lw=1.8, label=label)
            lo = np.array(ys) - np.array(es)
            if kpi == "service_rate":
                lo = np.maximum(lo, 0.0)
            ax.fill_between(xs, lo, np.array(ys) + np.array(es),
                            color=color, alpha=0.15, lw=0)

    ax.set_xticks(present)
    ax.set_xticklabels([SHIFT_TICKS[i] for i in present], rotation=30,
                       ha="right", fontsize=8)
    if kpi == "service_rate":
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=True, framealpha=0.9)


def plot_kpi(summary: dict, groups: dict, kpi: str, ylabel: str, scale: float,
             outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    draw(axes[0], summary, groups, kpi, scale, "bandra")
    draw(axes[1], summary, groups, kpi, scale, "nyc")
    axes[0].set_title("Bandra RL arms (synthetic, 5 seeds)", fontsize=11,
                      fontweight="bold")
    axes[1].set_title("NYC16 baselines (real demand)", fontsize=11,
                      fontweight="bold")
    axes[0].set_ylabel(ylabel, fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Demand shift", fontsize=10)
    axes[1].set_xlabel("Demand shift", fontsize=10)
    fig.suptitle(f"Frozen-policy degradation: {ylabel}", fontsize=12,
                 fontweight="bold")
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.18)
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
    groups = group_rl(summary)
    outdir = Path(args.outdir)

    for kpi, spec in KPI_SPECS.items():
        plot_kpi(summary, groups, kpi, spec["ylabel"], spec["scale"],
                 outdir / f"degradation_{kpi}.png")


if __name__ == "__main__":
    main()