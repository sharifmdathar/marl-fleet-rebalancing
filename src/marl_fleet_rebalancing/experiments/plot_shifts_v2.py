"""Regenerate shift-evaluation figures as retention radar, retention heatmap,
and a Pareto SR-vs-empty-km plot.

Source of truth: results/shift_eval/summary.json (policy -> shift -> metric stats).
Bandra RL arms are 5 seeds each (ippo/mappo-bandra-{40,80,200}-s{0..4}) evaluated
per seed over 10 episodes at eval seeds 1000..1004. Greedy-Baseline / LP-Rebalancer
are also present in this file and are included where the figure logic calls for them.

Each figure is saved as a vector PDF into results/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SUMMARY_PATH = Path("results/shift_eval/summary.json").resolve()
OUTDIR = PROJECT_ROOT / "results" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- palette: one color per method, colorblind-safe ----
METHOD_COLOR = {
    "IPPO-200": "#1f77b4",  # blue
    "MAPPO-200": "#ff7f0e",  # orange
    "Greedy": "#2ca02c",  # green
    "LP": "#d62728",  # red
}

# For the radar and heatmap we plot IPPO-200 and MAPPO-200 as the mean of their
# 5 seeds each (i.e. 10 seeds total per algorithm) because the two arms are
# visually indistinguishable in retention. For the Pareto we plot them as two
# separate points because empty km differs enough to matter.


# ---- load summary ----------------------------------------------------------
def load() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def shift_mean_std(summary: dict, policy_keys: list[str], shift: str, metric: str):
    """Mean of per-seed means across the given policy keys, with sample std
    across those per-seed means (ddof=1). Returns (mean, std, n)."""
    means: list[float] = []
    for k in policy_keys:
        v = summary.get(k, {}).get(shift, {}).get(metric, {})
        m = v.get("mean")
        if m is None:
            continue
        means.append(m)
    if not means:
        return None
    arr = np.array(means, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mu, sd, len(means)


def rl_arm_keys(summary: dict, iters: int):
    """Return the 5 seed keys for one RL arm, e.g. iters=200 -> ippo-bandra-200-s0..4, mappo-bandra-200-s0..4."""
    out: list[str] = []
    for algo in ("ippo", "mappo"):
        for s in range(5):
            cand = f"{algo}-bandra-{iters}-s{s}"
            if cand in summary:
                out.append(cand)
    return out


# ---- order / grouping -----------------------------------------------------
SHIFTS = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S7"]
SHIFT_LABEL = {
    "S0": "S0 base",
    "S1": "S1 permute",
    "S2a": "S2a x0.5",
    "S2b": "S2b x0.75",
    "S2d": "S2d x1.5",
    "S2e": "S2e x2.0",
    "S3": "S3 temporal",
    "S4": "S4 cold z1",
    "S5": "S5 burst",
    "S7": "S7 drift",
}

# Category order for the radar: keep the magnitude family contiguous and
# bracket the structural shifts (permute/temporal/cold/burst/drift) around it.
RADAR_ORDER = ["S0", "S2a", "S2b", "S2d", "S2e", "S1", "S3", "S4", "S5", "S7"]

RETENTION_MIN = 0.0   # floor so a collapsed notch is visible
RETENTION_MAX = 1.05  # room for a small overshoot above 100%


# ---- figure 1: retention radar -------------------------------------------
def figure_radar(summary: dict) -> Path:
    # Only IPPO-200 and MAPPO-200 for the main-text radar, per the cleanup rules.
    series: dict[str, list[float]] = {}
    for method, keys in [
        ("IPPO-200", rl_arm_keys(summary, 200)),
        ("MAPPO-200", rl_arm_keys(summary, 200)),
    ]:
        vals: list[float] = []
        for s in RADAR_ORDER:
            out = shift_mean_std(summary, keys, s, "service_rate")
            if out is None:
                vals.append(np.nan)
                continue
            s0 = shift_mean_std(summary, keys, "S0", "service_rate")
            if s0 is None or s0[0] == 0.0:
                vals.append(np.nan)
                continue
            retention = out[0] / s0[0]
            vals.append(min(max(retention, RETENTION_MIN), RETENTION_MAX))
        series[method] = vals

    N = len(RADAR_ORDER)
    angles = np.linspace(0.0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.6, 6.0))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw both fills first (low alpha), then both outlines on top so the
    # near-coincident polygons are both visible instead of one hiding the other.
    # Both arms are within ~0.2pp of each other in retention, so their polygons
    # nearly coincide. Draw both fills, then both outlines; make IPPO dashed so
    # the two near-coincident polygons are distinguishable.
    for method in ("IPPO-200", "MAPPO-200"):
        vals = series[method]
        if all(np.isnan(v) for v in vals):
            continue
        radial = vals + vals[:1]
        ax.fill(angles, radial, color=METHOD_COLOR[method], alpha=0.14, zorder=2)
    dash = {"IPPO-200": (0, (5, 4)), "MAPPO-200": (0, ())}[
        "IPPO-200" if False else "IPPO-200"
    ]
    for method, ls in [("IPPO-200", (0, (5, 4))), ("MAPPO-200", (0, ()))]:
        vals = series[method]
        if all(np.isnan(v) for v in vals):
            continue
        radial = vals + vals[:1]
        ax.plot(angles, radial, color=METHOD_COLOR[method], lw=2.2, ls=ls,
                label=method, zorder=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SHIFT_LABEL[s].replace("\n", " ") for s in RADAR_ORDER],
                       fontsize=8.5)
    # Push category labels a little outward so two-word labels don't crowd.
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(RETENTION_MIN, RETENTION_MAX)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_yticklabels(["50%", "75%", "100%"], fontsize=8, color="#444444")
    ax.set_rlabel_position(90)
    ax.grid(True, axis="y", alpha=0.35, ls="--")
    ax.grid(True, axis="x", alpha=0.25, lw=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
              fontsize=10, frameon=True, framealpha=0.95)
    fig.suptitle("Retention under demand shift (Bandra)",
                 fontsize=12, fontweight="bold", y=0.985)
    # Caption must fit inside the figure width at fontsize 8 (~475 pt wide);
    # keep it under ~90 characters.
    fig.text(
        0.5, 0.035,
        "Retention = shifted SR / S0 SR (per method). Axes are categories, not ordered.",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.76)

    out_pdf = OUTDIR / "retention_radar.pdf"
    out_png = OUTDIR / "retention_radar.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf


# ---- figure 2: retention heatmap ------------------------------------------
def figure_heatmap(summary: dict) -> Path:
    methods = [
        ("IPPO-40",  rl_arm_keys(summary, 40)),
        ("IPPO-80",  rl_arm_keys(summary, 80)),
        ("IPPO-200", rl_arm_keys(summary, 200)),
        ("MAPPO-40", rl_arm_keys(summary, 40)),
        ("MAPPO-80", rl_arm_keys(summary, 80)),
        ("MAPPO-200", rl_arm_keys(summary, 200)),
        ("Greedy",   ["Greedy-Baseline"]),
        ("LP",       ["LP-Rebalancer"]),
    ]

    table: list[list[float]] = []
    annot: list[list[str]] = []
    for name, keys in methods:
        row: list[float] = []
        arow: list[str] = []
        for s in SHIFTS:
            out = shift_mean_std(summary, keys, s, "service_rate")
            s0 = shift_mean_std(summary, keys, "S0", "service_rate")
            if out is None or s0 is None or s0[0] == 0.0:
                row.append(np.nan)
                arow.append("–")
            else:
                ret = out[0] / s0[0]
                row.append(min(max(ret, RETENTION_MIN), RETENTION_MAX))
                arow.append(f"{ret * 100:.0f}%")
        table.append(row)
        annot.append(arow)

    arr = np.array([[v if not np.isnan(v) else np.nan for v in r] for r in table])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(arr, cmap=cmap, vmin=RETENTION_MIN, vmax=RETENTION_MAX,
                   aspect="auto")

    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels([SHIFT_LABEL[s].replace("\n", " ") for s in SHIFTS],
                       fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m for m, _ in methods], fontsize=10)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = annot[i][j]
            face = "white" if (not np.isnan(val) and val < 0.62) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=face, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Retention (% of S0 service rate)", fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_title("Retention under demand shift — all arms",
                 fontsize=12, fontweight="bold", pad=12)
    fig.subplots_adjust(left=0.22, right=0.94, top=0.88, bottom=0.16)
    out_pdf = OUTDIR / "retention_heatmap.pdf"
    out_png = OUTDIR / "retention_heatmap.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf


# ---- figure 3: Pareto SR vs empty km -------------------------------------
def _arm_s0(session_summary: dict, algo: str, iters: int):
    """Mean +/- sample std of S0 SR and empty km across the 5 seeds of one RL arm
    (e.g. algo='ippo', iters=200). Returns (sr_mean_pct, sr_std_pct, km_mean, km_std)."""
    keys = [f"{algo}-bandra-{iters}-s{s}" for s in range(5)]
    sr_m, sr_s, _ = shift_mean_std(session_summary, keys, "S0", "service_rate")
    km_m, km_s, _ = shift_mean_std(session_summary, keys, "S0", "empty_km")
    if sr_m is None or km_m is None:
        return None
    return sr_m * 100.0, sr_s * 100.0, km_m, km_s


def figure_pareto(summary: dict) -> Path:
    # ---- central tradeoff points: Greedy, LP, IPPO-200, MAPPO-200 (separate) ----
    points: list[tuple[str, float, float, float, float]] = []  # name, sr%, sr_std%, km, km_std
    for label, keys in [
        ("Greedy", ["Greedy-Baseline"]),
        ("LP",     ["LP-Rebalancer"]),
    ]:
        sr = shift_mean_std(summary, keys, "S0", "service_rate")
        km = shift_mean_std(summary, keys, "S0", "empty_km")
        if sr is None or km is None:
            continue
        points.append((label, sr[0] * 100.0, sr[1] * 100.0, km[0], km[1]))
    for algo, iters in [("ippo", 200), ("mappo", 200)]:
        out = _arm_s0(summary, algo, iters)
        if out is None:
            continue
        label = f"{algo.upper()}-{iters}"
        points.append((label, out[0], out[1], out[2], out[3]))

    # ---- fleet-size sweep curves -------------------------------------------
    # source: results/bandra_fleet_D93/sweep_results.json (IPPO-200-s0, MAPPO-200-s0,
    # 10 episodes each, base seed 3000). One curve per algorithm, 20..35 drivers.
    fleet_path = Path("results/bandra_fleet_D93/sweep_results.json").resolve()
    fleet_raw: dict[int, dict[str, list[list[float]]]] = {}
    if fleet_path.exists():
        fdata = json.loads(fleet_path.read_text(encoding="utf-8"))
        for entry in fdata:
            d = int(entry.get("drivers"))
            algo = entry.get("policy")
            sr = entry.get("service_rate")
            km = entry.get("empty_km")
            if None in (d, algo, sr, km):
                continue
            fleet_raw.setdefault(d, {}).setdefault(algo, ([], []))[0].append(float(sr))
            fleet_raw[d][algo][1].append(float(km))

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    # error bars per method (S0 means from summary.json)
    ax.errorbar(
        [p[3] for p in points],
        [p[1] for p in points],
        xerr=[p[4] for p in points],
        yerr=[p[2] for p in points],
        fmt="o", ms=9, lw=1.8, capsize=4, zorder=5,
    )
    # Offset labels for the two close RL points so both are readable.
    label_offsets: dict[str, tuple[int, int]] = {
        "Greedy": (6, -12),
        "LP": (6, 6),
        "IPPO-200": (-6, -14),
        "MAPPO-200": (8, 8),
    }
    for p in points:
        dx, dy = label_offsets.get(p[0], (6, 6))
        ax.annotate(p[0], (p[3], p[1]), fontsize=9, fontweight="bold",
                    xytext=(dx, dy), textcoords="offset points", color="#222222")

    # fleet sweep curves (connected, one opaque line per algo; dots too)
    # sweep_results.json stores service_rate already as percent (e.g. 26.5 = 26.5%),
    # so do NOT multiply by 100 here.
    colors = {"IPPO": "#1f77b4", "MAPPO": "#ff7f0e"}
    for algo_key, algo_name in [("IPPO-200-s0", "IPPO"), ("MAPPO-200-s0", "MAPPO")]:
        xs: list[float] = []
        ys: list[float] = []
        xerrs: list[float] = []
        yerrs: list[float] = []
        for d in sorted(fleet_raw):
            sr_list, km_list = fleet_raw[d].get(algo_key, ([], []))
            if not sr_list:
                continue
            xs.append(float(np.mean(km_list)))
            ys.append(float(np.mean(sr_list)))
            xerrs.append(float(np.std(km_list, ddof=1)) if len(km_list) > 1 else 0.0)
            yerrs.append(float(np.std(sr_list, ddof=1)) if len(sr_list) > 1 else 0.0)
        if len(xs) >= 2:
            ax.plot(xs, ys, color=colors[algo_name], lw=1.8, ls=(0, (4, 3)),
                    alpha=0.9, zorder=3,
                    label=f"{algo_name} fleet sweep (20–35 drv)")
            ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt="none",
                        ecolor=colors[algo_name], elinewidth=1.2,
                        capsize=3, alpha=0.8, zorder=3)

    ax.set_xlabel("Empty distance (km / episode, mean over seeds)", fontsize=11,
                  fontweight="bold")
    ax.set_ylabel("Service rate (% , S0 base demand)", fontsize=11, fontweight="bold")
    ax.set_title("Service rate vs empty distance — Bandra (S0)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333",
               markersize=9, label="S0 tradeoff point (mean over seeds)"),
        Line2D([0], [0], color=colors["IPPO"], lw=1.8, ls=(0, (4, 3)),
               label="IPPO fleet sweep (20–35 drv)"),
        Line2D([0], [0], color=colors["MAPPO"], lw=1.8, ls=(0, (4, 3)),
               label="MAPPO fleet sweep (20–35 drv)"),
    ]
    ax.legend(handles=custom, fontsize=9, loc="upper left",
              frameon=True, framealpha=0.95)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.12)

    out_pdf = OUTDIR / "pareto_service_vs_empty.pdf"
    out_png = OUTDIR / "pareto_service_vs_empty.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf


# ---- main -----------------------------------------------------------------
def main() -> None:
    summary = load()
    for fn in (figure_radar, figure_heatmap, figure_pareto):
        fn(summary)


if __name__ == "__main__":
    main()
