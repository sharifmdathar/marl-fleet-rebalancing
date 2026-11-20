#!/usr/bin/env python3
"""Publication figures for the D100 paper, read directly from episode CSVs.

Usage:
    .venv/bin/python scripts/plot_paper_v2.py

Writes PDF + 300 dpi PNG into docs/paper_v2/figs/ and prints file sizes and
dimensions.  No result artifact is written or modified.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FULL = pd.read_csv(ROOT / "results/full_study_v2/episodes.csv")
PLANB = pd.read_csv(ROOT / "results/plan_b_v1/episodes.csv")
OUT = ROOT / "docs/paper_v2/figs"
OUT.mkdir(parents=True, exist_ok=True)
PNG_DPI = 300

# Okabe-Ito colour-blind-safe palette (plus neutral grey for the heuristic).
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#4D4D4D",
}
COLORS = {
    "Greedy": OI["grey"],
    "LP": OI["blue"],
    "IPPO-it200": OI["sky"],
    "MLP-MAPPO": OI["green"],
    "Attention-MAPPO": OI["orange"],
    "DR-MLP-MAPPO": OI["purple"],
    "DR-Attention-MAPPO": OI["vermillion"],
}
MARKERS = {
    "Greedy": "s",
    "LP": "o",
    "IPPO-it200": "^",
    "MLP-MAPPO": "D",
    "Attention-MAPPO": "v",
    "DR-MLP-MAPPO": "P",
    "DR-Attention-MAPPO": "X",
}
MAIN_ARMS = [
    "Greedy", "LP", "IPPO-it200", "MLP-MAPPO", "Attention-MAPPO",
    "DR-MLP-MAPPO", "DR-Attention-MAPPO",
]
# display name -> (data frame, arm column value)
ARM_SOURCE = {
    "Greedy": (FULL, "Greedy"),
    "LP": (FULL, "LP"),
    "IPPO-it200": (FULL, "IPPO-it200"),
    "MLP-MAPPO": (FULL, "MAPPO-it200"),
    "Attention-MAPPO": (PLANB, "Attention-MAPPO"),
    "DR-MLP-MAPPO": (PLANB, "DR-MLP-MAPPO"),
    "DR-Attention-MAPPO": (PLANB, "DR-Attention-MAPPO"),
    "MAPPO-sampled": (FULL, "MAPPO-it200-sampled"),
    "BC-LP-rw10": (FULL, "BC-LP-rw10"),
}
SCENARIO_ORDER = ["S0", "S1", "S2a", "S2b", "S2d", "S2e",
                  "S3", "S4", "S5", "S6", "S7"]
FAMILIES = [
    ("Temporal", ["S0", "S3", "S6"]),
    ("Volume/Burst", ["S2a", "S2b", "S2d", "S2e", "S5"]),
    ("Spatial", ["S1", "S4", "S7"]),
]
# The sweep endpoints are aliases of the main-grid scenarios.
S1_SWEEP = [("S1@p0.0", 0.0), ("S1@p0.25", 0.25), ("S1@p0.5", 0.5),
            ("S1@p0.75", 0.75), ("S1", 1.0)]
S4_SWEEP = [("S4@z1@10", 10), ("S4@z1@20", 20), ("S4@z1@40", 40),
            ("S4", 60), ("S4@z1@80", 80)]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def agg(df: pd.DataFrame, arm: str, scenario: str, column: str):
    """Mean ± 1 sd of the evaluation unit.

    RL arms: mean across windows per model seed, then mean ± sample sd over the
    five model seeds.  Deterministic baselines: mean ± sample sd over the 12
    paired windows.
    """
    g = df[(df.arm == arm) & (df.scenario == scenario)]
    if g.empty:
        raise KeyError(f"no rows for {arm!r} / {scenario!r}")
    if g.model_seed.max() >= 0:
        per = g.groupby("model_seed")[column].mean()
        return float(per.mean()), float(per.std(ddof=1)), int(len(per))
    return float(g[column].mean()), float(g[column].std(ddof=1)), int(len(g))


def arm_stats(display: str, scenario: str, column: str):
    df, arm = ARM_SOURCE[display]
    return agg(df, arm, scenario, column)


def total_variation_vs_s0(df: pd.DataFrame) -> dict[str, float]:
    """Mean per-window TV divergence of origin-zone demand vs S0."""
    zcols = [f"orig_z{i}" for i in range(16)]
    ref = df[df.arm == "Greedy"] if "Greedy" in set(df.arm) else df
    counts = ref.groupby(["scenario", "seed"])[zcols].sum()
    norm = counts.div(counts.sum(axis=1), axis=0)
    p0 = norm.loc["S0"]
    out = {}
    for sc in counts.index.get_level_values("scenario").unique():
        ps = norm.loc[sc]
        tv = []
        for seed in ps.index:
            if seed not in p0.index:
                continue
            a = ps.loc[seed].to_numpy(float)
            b = p0.loc[seed].to_numpy(float)
            tv.append(0.5 * np.abs(a - b).sum())
        out[sc] = float(np.mean(tv))
    return out


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=PNG_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def fig1_pareto_and_decomp() -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.15), gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(wspace=0.34)

    # Panel (a): S0 Pareto plane.  Closed symbols are deterministic; the two
    # open symbols are the sampled ablation and the BC-LP control.
    handles = []
    for name in MAIN_ARMS:
        sr, sr_sd, _ = arm_stats(name, "S0", "service_rate")
        km, km_sd, _ = arm_stats(name, "S0", "empty_km")
        ax_a.errorbar(
            km, 100 * sr, xerr=km_sd, yerr=100 * sr_sd, fmt="none",
            ecolor=to_rgba(COLORS[name], 0.7), elinewidth=0.8,
            capsize=2.0, capthick=0.7, zorder=2,
        )
        ax_a.plot(
            km, 100 * sr, marker=MARKERS[name], linestyle="none",
            color=COLORS[name], mfc=COLORS[name], mec=COLORS[name],
            ms=5.2, mew=1.0, zorder=4,
        )
        handles.append(Line2D(
            [0], [0], marker=MARKERS[name], linestyle="none", ms=4.6,
            mfc=COLORS[name], mec=COLORS[name], label=name))
    open_arms = [
        ("MAPPO-sampled", FULL, "MAPPO-it200-sampled", OI["green"], "s"),
        ("BC-LP-rw10", FULL, "BC-LP-rw10", OI["purple"], "D"),
    ]
    for label, df, arm, color, marker in open_arms:
        sr, sr_sd, _ = agg(df, arm, "S0", "service_rate")
        km, km_sd, _ = agg(df, arm, "S0", "empty_km")
        ax_a.errorbar(
            km, 100 * sr, xerr=km_sd, yerr=100 * sr_sd, fmt="none",
            ecolor=to_rgba(color, 0.7), elinewidth=0.8, capsize=2.0,
            capthick=0.7, zorder=2,
        )
        ax_a.plot(
            km, 100 * sr, marker=marker, linestyle="none", color=color,
            mfc="white", mec=color, ms=6.0, mew=1.2, zorder=6,
        )
        handles.append(Line2D(
            [0], [0], marker=marker, linestyle="none", ms=5.2,
            mfc="white", mec=color, label=label))
    ax_a.set_xlim(0, 530)
    ax_a.set_ylim(35, 80)
    ax_a.set_xlabel("Empty distance (km)")
    ax_a.set_ylabel("Service rate (%)")
    ax_a.set_title("(a) S0 Pareto plane", loc="left")
    style_axis(ax_a, "both")
    ax_a.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.5,
                frameon=False, handletextpad=0.3, columnspacing=0.6,
                borderpad=0.0, labelspacing=0.3)

    # Panel (b): relocation vs pickup attribution.
    x = np.arange(len(MAIN_ARMS))
    reloc = np.array([arm_stats(n, "S0", "reloc_leg_km")[0]
                      for n in MAIN_ARMS])
    pickup = np.array([arm_stats(n, "S0", "pickup_km")[0]
                       for n in MAIN_ARMS])
    ax_b.bar(x, reloc, width=0.68, color=OI["blue"],
             edgecolor="white", linewidth=0.6, label="Relocation-leg km",
             zorder=3)
    ax_b.bar(x, pickup, width=0.68, bottom=reloc, color=OI["orange"],
             edgecolor="white", linewidth=0.6, label="Pickup-leg km",
             zorder=3)
    for xi, total in zip(x, reloc + pickup):
        ax_b.text(xi, total + 5, f"{total:.0f}", ha="center", va="bottom",
                  fontsize=7.2, color="#333333")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Greedy", "LP", "IPPO", "MLP", "Attn",
                          "DR-MLP", "DR-Attn"], rotation=32, ha="right",
                         rotation_mode="anchor")
    ax_b.set_ylabel("Empty distance (km)")
    ax_b.set_ylim(0, max(reloc + pickup) * 1.16)
    ax_b.set_title("(b) S0 empty-km decomposition", loc="left")
    style_axis(ax_b, "y")
    ax_b.legend(loc="upper left", frameon=False, handlelength=1.2,
                handletextpad=0.4, borderpad=0.2)
    save(fig, "fig1_pareto_and_decomp")


def fig2_shift_scenarios() -> None:
    arms = ["Greedy", "LP", "IPPO-it200", "MLP-MAPPO",
            "Attention-MAPPO", "DR-Attention-MAPPO"]
    fig, (ax_sr, ax_km) = plt.subplots(2, 1, figsize=(7.0, 5.75),
                                       sharex=True)
    fig.subplots_adjust(hspace=0.38, top=0.85, bottom=0.15)

    scenario_pos: dict[str, float] = {}
    family_centres = {}
    cursor = 0.0
    for fam, scs in FAMILIES:
        if scenario_pos:
            cursor += 0.95
        start = cursor
        for sc in scs:
            scenario_pos[sc] = cursor
            cursor += 1.0
        family_centres[fam] = 0.5 * (start + scenario_pos[scs[-1]])
    boundaries = []
    prev_last = None
    for fam, scs in FAMILIES:
        if prev_last is not None:
            boundaries.append(0.5 * (prev_last + scenario_pos[scs[0]]))
        prev_last = scenario_pos[scs[-1]]

    offsets = np.linspace(-0.30, 0.30, len(arms))
    for name in arms:
        xs = np.array([scenario_pos[sc] for sc in SCENARIO_ORDER])
        sr = np.array([100 * arm_stats(name, sc, "service_rate")[0]
                       for sc in SCENARIO_ORDER])
        sr_sd = np.array([100 * arm_stats(name, sc, "service_rate")[1]
                          for sc in SCENARIO_ORDER])
        km = np.array([arm_stats(name, sc, "empty_km")[0]
                       for sc in SCENARIO_ORDER])
        km_sd = np.array([arm_stats(name, sc, "empty_km")[1]
                          for sc in SCENARIO_ORDER])
        off = offsets[arms.index(name)]
        ax_sr.errorbar(xs + off, sr, yerr=sr_sd, fmt=MARKERS[name],
                       color=COLORS[name], mfc=COLORS[name],
                       mec=COLORS[name], ms=4.2, mew=0.9,
                       elinewidth=1.4, ecolor=to_rgba(COLORS[name], 0.9),
                       capsize=1.5, capthick=0.6, zorder=4, label=name)
        # The S4 IPPO upper sd is 648 km, above the 560 km panel limit.
        # Cap the whisker at the top edge and label the full sd on the plot.
        # The top cap is omitted for clipped points so no horizontal rule is
        # drawn at the axis limit.
        km_top = 560.0
        upper = np.minimum(km_sd, np.maximum(km_top - km, 0.0))
        clipped = km + km_sd > km_top
        ax_km.errorbar(xs + off, km, yerr=np.vstack([km_sd, upper]),
                       fmt=MARKERS[name], color=COLORS[name],
                       mfc=COLORS[name], mec=COLORS[name], ms=4.2,
                       mew=0.9, elinewidth=1.4,
                       ecolor=to_rgba(COLORS[name], 0.9), capsize=0.0,
                       zorder=4)
        cap_half = 0.055
        for xi, mean, sd, is_clipped in zip(xs + off, km, km_sd, clipped):
            ax_km.hlines(mean - sd, xi - cap_half, xi + cap_half,
                         colors=COLORS[name], linewidth=0.6, alpha=0.7,
                         zorder=3)
            if not is_clipped:
                ax_km.hlines(mean + sd, xi - cap_half, xi + cap_half,
                             colors=COLORS[name], linewidth=0.6,
                             alpha=0.7, zorder=3)
            else:
                ax_km.text(xi + 0.10, km_top - 4, f"\u00b1{sd:.0f}",
                           ha="left", va="top", fontsize=6.8,
                           color=COLORS[name], zorder=6)

    tick_pos = [scenario_pos[sc] for sc in SCENARIO_ORDER]
    for ax in (ax_sr, ax_km):
        for b in boundaries:
            ax.axvline(b, color="#777777", linewidth=0.7,
                       linestyle=(0, (2, 2)), zorder=1)
        ax.set_xlim(min(tick_pos) - 0.62, max(tick_pos) + 0.62)
        style_axis(ax, "y")
    ax_km.set_xticks(tick_pos)
    ax_km.set_xticklabels(SCENARIO_ORDER)
    ax_km.set_xlabel("Scenario")

    ax_sr.set_ylim(0, 100)
    ax_sr.set_ylabel("Service rate (%)")
    ax_sr.set_title("(a) Service rate across demand-shift scenarios", loc="left")
    ax_km.set_ylim(0, 560)
    ax_km.set_ylabel("Empty distance (km)")
    ax_km.set_title("(b) Empty distance across demand-shift scenarios", loc="left")

    for fam, scs in FAMILIES:
        ax_sr.text(family_centres[fam], 96, fam, ha="center", va="top",
                   fontsize=8.2, fontweight="bold", color="#333333",
                   bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                             edgecolor="none", alpha=0.85), zorder=6)

    handles = [
        Line2D([0], [0], marker=MARKERS[n], linestyle="none", ms=4.8,
               mfc=COLORS[n], mec=COLORS[n], label=n)
        for n in arms
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False,
               handletextpad=0.25, columnspacing=1.0)
    save(fig, "fig2_shift_scenarios")


def fig3_parametric_curves() -> None:
    tv = total_variation_vs_s0(FULL)
    arms = ["LP", "IPPO-it200", "MLP-MAPPO", "Attention-MAPPO",
            "DR-MLP-MAPPO", "DR-Attention-MAPPO"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.55))
    fig.subplots_adjust(wspace=0.32, bottom=0.20, top=0.86)

    def draw_sweep(ax, sweep, xlabel, title):
        xs = np.array([p for _, p in sweep], dtype=float)
        tvs = [tv[sc] for sc, _ in sweep]
        for name in arms:
            mean = np.array([100 * arm_stats(name, sc, "service_rate")[0]
                             for sc, _ in sweep])
            sd = np.array([100 * arm_stats(name, sc, "service_rate")[1]
                           for sc, _ in sweep])
            ax.plot(xs, mean, color=COLORS[name], marker=MARKERS[name],
                    ms=4.0, mew=0.8, linewidth=1.3, zorder=4, label=name)
            ax.fill_between(xs, mean - sd, mean + sd, color=COLORS[name],
                            alpha=0.14, linewidth=0, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Service rate (%)")
        ax.set_title(title, loc="left", pad=31)
        style_axis(ax, "both")
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        top.set_xticks(xs)
        top.set_xticklabels([f"{v:.2f}" for v in tvs], fontsize=7.0,
                            color="#555555")
        top.tick_params(axis="x", length=0, pad=2)
        top.spines["top"].set_color("#999999")
        top.spines["top"].set_linewidth(0.6)
        top.set_xlabel("TV divergence vs S0", fontsize=7.0,
                       color="#555555", labelpad=3)
        return xs

    xs_a = draw_sweep(ax_a, S1_SWEEP, "Permutation strength $p$",
                      "(a) S1 permutation sweep")
    ax_a.set_xlim(-0.06, 1.06)
    ax_a.set_ylim(35, 78)
    ax_a.set_xticks(xs_a)

    xs_b = draw_sweep(ax_b, S4_SWEEP,
                      "Demand concentrated on zone 1 (%)",
                      "(b) S4 single-zone concentration")
    ax_b.set_xlim(5, 85)
    ax_b.set_ylim(0, 78)
    ax_b.set_xticks(xs_b)
    ax_b.axvspan(55, 85, color=OI["vermillion"], alpha=0.055, zorder=0)
    ax_b.annotate(
        "DR-Attention > LP\nat 60\u201380%\nconcentration",
        xy=(60, 100 * arm_stats("DR-Attention-MAPPO", "S4",
                                "service_rate")[0]),
        xytext=(68, 67), textcoords="data", fontsize=7.2,
        color=OI["vermillion"], ha="center", va="top",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.5),
        arrowprops=dict(arrowstyle="->", color=OI["vermillion"],
                        lw=0.7, shrinkA=2, shrinkB=3),
        zorder=7,
    )

    handles = [
        Line2D([0], [0], marker=MARKERS[n], linestyle="-", ms=4.0,
               color=COLORS[n], label=n)
        for n in arms
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), ncol=6, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig3_parametric_curves")


def fig4_retention_2x2() -> None:
    arms = ["LP", "IPPO-it200", "MLP-MAPPO", "Attention-MAPPO",
            "DR-MLP-MAPPO", "DR-Attention-MAPPO"]
    family_scenarios = [
        ("Spatial\n(S1, S4, S7)", ["S1", "S4", "S7"]),
        ("Volume/Burst\n(S2a, S2b, S2d, S2e, S5)",
         ["S2a", "S2b", "S2d", "S2e", "S5"]),
        ("Temporal\n(S3, S6)", ["S3", "S6"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    fig.subplots_adjust(bottom=0.24, top=0.84)

    x = np.arange(len(family_scenarios), dtype=float)
    width = 0.12
    offsets = np.linspace(-(len(arms) - 1) * width / 2,
                          (len(arms) - 1) * width / 2, len(arms))
    for j, name in enumerate(arms):
        s0 = arm_stats(name, "S0", "service_rate")[0]
        means, sds = [], []
        for _, scs in family_scenarios:
            ret = np.array([100 * arm_stats(name, sc, "service_rate")[0] / s0
                            for sc in scs])
            means.append(float(ret.mean()))
            sds.append(float(ret.std(ddof=1)) if len(ret) > 1 else 0.0)
        ax.bar(x + offsets[j], means, width=width * 0.92,
               color=COLORS[name], edgecolor="white", linewidth=0.5,
               yerr=sds, ecolor="#555555", capsize=1.5,
               error_kw=dict(elinewidth=0.6, capthick=0.6), zorder=3,
               label=name)
    ax.axhline(100, color="#222222", linestyle=(0, (3, 2)),
               linewidth=0.9, zorder=5)
    ax.text(-0.52, 101.5, "100% of S0", ha="left", va="bottom",
            fontsize=7.5, color="#222222",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none",
                      pad=1.0), zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in family_scenarios])
    ax.set_ylabel("Shift-family retention (% of S0 service rate)")
    ax.set_ylim(0, 132)
    ax.set_xlim(-0.55, 2.55)
    style_axis(ax, "y")

    def family_mean(name: str, family_idx: int) -> float:
        s0 = arm_stats(name, "S0", "service_rate")[0]
        _, scs = family_scenarios[family_idx]
        return float(np.mean([
            100 * arm_stats(name, sc, "service_rate")[0] / s0 for sc in scs
        ]))

    # Use the one-decimal family means printed in the summary table so this
    # annotation is reproducible from the table: 83.8-73.9=+9.9, 76.2-76.9=-0.7.
    dr_gain_attn = (round(family_mean("DR-Attention-MAPPO", 0), 1)
                    - round(family_mean("Attention-MAPPO", 0), 1))
    dr_gain_mlp = (round(family_mean("DR-MLP-MAPPO", 0), 1)
                   - round(family_mean("MLP-MAPPO", 0), 1))
    ax.text(0.0, 119.5,
            f"Spatial DR gain: {dr_gain_attn:+.1f} pp\n"
            f"with attention, {dr_gain_mlp:+.1f} pp with MLP",
            ha="center", va="center", fontsize=7.0, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9), zorder=6)

    handles = [
        Patch(facecolor=COLORS[n], edgecolor="white", label=n)
        for n in arms
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig4_retention_2x2")


def print_artifacts() -> None:
    print("\nGenerated figures (PDF vector + 300 dpi PNG preview)")
    print(f"{'file':<36s} {'bytes':>10s}  dimensions")
    for path in sorted(OUT.glob("fig*")):
        if path.suffix == ".png":
            with Image.open(path) as im:
                px = f"{im.width} x {im.height} px at {PNG_DPI} dpi"
            print(f"{path.name:<36s} {path.stat().st_size:>10,d}  {px}")
        elif path.suffix == ".pdf":
            raw = path.read_bytes()
            m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+"
                          rb"([\d.]+)\s+([\d.]+)\s*\]", raw)
            if m:
                w_pt = float(m.group(3)) - float(m.group(1))
                h_pt = float(m.group(4)) - float(m.group(2))
                dim = (f"{w_pt:.1f} x {h_pt:.1f} pt "
                       f"({w_pt / 72:.2f} x {h_pt / 72:.2f} in)")
            else:
                dim = "MediaBox not parsed"
            print(f"{path.name:<36s} {path.stat().st_size:>10,d}  {dim}")


def main() -> None:
    fig1_pareto_and_decomp()
    fig2_shift_scenarios()
    fig3_parametric_curves()
    fig4_retention_2x2()
    print_artifacts()


if __name__ == "__main__":
    main()
