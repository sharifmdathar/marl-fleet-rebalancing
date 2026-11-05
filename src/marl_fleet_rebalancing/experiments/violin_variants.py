"""Violin variant requested: IPPO vs MAPPO violin plots; LP and Greedy as a
stats block only (not drawn). Axes: SR 61-67, ED 410-500.

The stats block uses the same palette as the rest of the figures, with the
method token colored to match its violins.

Run with:

    uv run python src/marl_fleet_rebalancing/experiments/violin_variants.py

Output is written to results/figures/pareto_alternatives/violin_e_arms_only.{pdf,png}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marl_fleet_rebalancing.experiments.plot_shifts_v2 import (
    COLOR,
    load_baselines,
    load_rl,
)

KM = "empty_km"
SR = "service_rate"

ARMS = ["MAPPO-200", "IPPO-200"]

ROOT = Path(__file__).resolve().parents[3]
OUT_RESULTS = ROOT / "results" / "figures" / "pareto_alternatives"
OUT_DOCS = ROOT / "docs" / "figures"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_DOCS.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "MAPPO-200": "#ff7f0e",
    "IPPO-200": "#1f77b4",
    "LP": "#d62728",
    "Greedy": "#2ca02c",
}

SPECS = [
    (SR, "Service rate (%)", 100.0, (61.0, 67.0)),
    (KM, "Empty distance (km / episode)", 1.0, (410.0, 500.0)),
]


def _violins(ax, rl, metric: str, scale: float, bw: float, width: float) -> None:
    data = [np.asarray(rl[m]["S0"][metric], dtype=float) * scale for m in ARMS]
    parts = ax.violinplot(
        data,
        positions=[0, 1],
        widths=width,
        points=200,
        bw_method=bw,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE[ARMS[i]])
        body.set_edgecolor(PALETTE[ARMS[i]])
        body.set_alpha(0.32)
        body.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for i, m in enumerate(ARMS):
        vals = data[i]
        ax.scatter(
            np.full(vals.size, i) + rng.uniform(-0.07, 0.07, vals.size),
            vals,
            s=26,
            color=PALETTE[m],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.scatter(
            [i],
            [vals.mean()],
            marker="_",
            s=460,
            linewidths=2.4,
            color=PALETTE[m],
            zorder=5,
        )


def _frame(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, stem: str) -> Path:
    pdf_results = OUT_RESULTS / f"{stem}.pdf"
    png_results = OUT_RESULTS / f"{stem}.png"
    fig.savefig(pdf_results, bbox_inches="tight")
    fig.savefig(png_results, dpi=160, bbox_inches="tight")
    plt.close(fig)

    pdf_docs = OUT_DOCS / f"{stem}.pdf"
    png_docs = OUT_DOCS / f"{stem}.png"
    pdf_docs.write_bytes(pdf_results.read_bytes())
    png_docs.write_bytes(png_results.read_bytes())

    print(f"wrote {pdf_results}")
    print(f"wrote {png_results}")
    print(f"wrote {pdf_docs}")
    print(f"wrote {png_docs}")
    return png_results


def _stat_lines_colors(rl, base):
    lines: list[str] = []
    colors: list[str] = []
    for m in ARMS:
        sr = np.asarray(rl[m]["S0"][SR]) * 100.0
        km = np.asarray(rl[m]["S0"][KM])
        lines.append(
            f"{m:<12}"
            f" SR {sr.mean():.1f} \u00b1 {sr.std(ddof=1):.1f}"
            f" (n={sr.size}, {sr.min():.1f}-{sr.max():.1f})"
            f"   ED {km.mean():.0f} \u00b1 {km.std(ddof=1):.0f}"
            f" (n={km.size}, {km.min():.0f}-{km.max():.0f})"
        )
        colors.append(PALETTE[m])
    for m in ("LP", "Greedy"):
        entry = base.get(m, {}).get("S0", {})
        sr_mu, sr_sd = entry.get(SR, (0.0, 0.0))
        km_mu, km_sd = entry.get(KM, (0.0, 0.0))
        lines.append(
            f"{m:<12}"
            f" SR {sr_mu*100:.1f} \u00b1 {sr_sd*100:.1f}"
            f"   ED {km_mu:.0f} \u00b1 {km_sd:.0f}"
        )
        colors.append(PALETTE[m])
    return lines, colors


def figure_arms_only() -> Path:
    rl = load_rl()
    base = load_baselines()

    fig = plt.figure(figsize=(10.8, 6.8))
    gs = fig.add_gridspec(
        1, 2, left=0.08, right=0.97, top=0.90, bottom=0.34, wspace=0.22,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, (metric, label, scale, (ymin, ymax)) in zip(axes, SPECS):
        _violins(ax, rl, metric, scale, bw=0.35, width=1.30)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.75, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ARMS, fontsize=12)
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(10.5)
        _frame(ax)

        for i, m in enumerate(ARMS):
            vals = np.asarray(rl[m]["S0"][metric], dtype=float) * scale
            mean_txt = (
                f"{vals.mean():.1f}" if metric == SR
                else f"{vals.mean():.0f}"
            )
            ax.annotate(
                mean_txt,
                (i, vals.mean()),
                textcoords="offset points",
                xytext=(0, -22),
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=PALETTE[m],
            )

    fig.suptitle(
        "S0 spread: IPPO vs MAPPO",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    lines, colors = _stat_lines_colors(rl, base)
    y_text = 0.165
    y_step = 0.024
    for color, line in zip(colors, lines):
        fig.text(
            x=0.5,
            y=y_text,
            s=line,
            ha="center",
            va="bottom",
            fontsize=9.5,
            family="monospace",
            linespacing=1.6,
            color=color,
        )
        y_text -= y_step

    fig.subplots_adjust(bottom=0.345)
    return _save(fig, "violin_e_arms_only")


def main() -> None:
    figure_arms_only()

    rl = load_rl()
    base = load_baselines()

    print("\nreference values")
    for m in ARMS:
        sr = np.asarray(rl[m]["S0"][SR]) * 100
        km = np.asarray(rl[m]["S0"][KM])
        print(
            f"  {m:<10} SR {sr.mean():.1f} \u00b1 {sr.std(ddof=1):.1f}"
            f" (range {sr.min():.1f}-{sr.max():.1f})"
            f"   ED {km.mean():.1f} \u00b1 {km.std(ddof=1):.1f}"
            f" (range {km.min():.0f}-{km.max():.0f})"
        )
    for m in ("LP", "Greedy"):
        sr_mu, sr_sd = base[m]["S0"][SR]
        km_mu, km_sd = base[m]["S0"][KM]
        print(
            f"  {m:<10} SR {sr_mu*100:.1f} \u00b1 {sr_sd*100:.1f}"
            f"   ED {km_mu:.1f} \u00b1 {km_sd:.1f}"
        )


def copy_to_docs(stem: str) -> None:
    src_pdf = OUT_RESULTS / f"{stem}.pdf"
    src_png = OUT_RESULTS / f"{stem}.png"
    dst_pdf = OUT_DOCS / f"{stem}.pdf"
    dst_png = OUT_DOCS / f"{stem}.png"
    dst_pdf.write_bytes(src_pdf.read_bytes())
    dst_png.write_bytes(src_png.read_bytes())
    print(f"copied {stem} to docs/figures/{stem}.pdf")
    print(f"copied {stem} to docs/figures/{stem}.png")


def _publish_to_docs(stem: str) -> None:
    src_pdf = OUT_RESULTS / f"{stem}.pdf"
    src_png = OUT_RESULTS / f"{stem}.png"
    dst_pdf = OUT_DOCS / f"{stem}.pdf"
    dst_png = OUT_DOCS / f"{stem}.png"
    dst_pdf.write_bytes(src_pdf.read_bytes())
    dst_png.write_bytes(src_png.read_bytes())
    print(f"copied {stem} to docs/figures/{stem}.pdf")
    print(f"copied {stem} to docs/figures/{stem}.png")


if __name__ == "__main__":
    main()
    _publish_to_docs("violin_e_arms_only")
