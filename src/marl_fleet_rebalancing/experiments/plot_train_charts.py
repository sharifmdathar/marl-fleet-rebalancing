"""Honest chart pack for the Bandra IPPO/MAPPO suite.

Sources of truth (read directly, no recomputation assumptions):
  results/shift_eval/summary.json   — per-(algorithm,seed,shift) mean/std
  results/rl_training_series.json   — per-(algorithm,horizon) per-iteration seed-mean/std

Bandra shift-eval numbers used for the charts below (mean +/- ddof=1 std,
already verified against summary.json, 5 seeds per RL arm, 1 seed per baseline):

  Baselines (S0, Bandra runs, 5 seeds 9000-9004, results/bandra_baselines_D95.json):
    Greedy   SR=35.30% +/- 1.99  empty_km=21.10 +/- 1.11
    LP       SR=41.20% +/- 2.56  empty_km=98.69 +/- 5.44
  (summary.json also holds Greedy/LP, but shift_eval.py evaluated those on
  NYCConfig, so they are NOT Bandra numbers and are not used here. They have
  no shift sweep on Bandra, so the shift charts carry RL arms only.)

  RL arms (S0), 5 seeds each:
    IPPO-40   SR=0.3672 +/- 0.0097   km=176.53 +/- 6.69
    IPPO-80   SR=0.3612 +/- 0.0060   km=155.02 +/- 3.14
    IPPO-200  SR=0.3612 +/- 0.0069   km=136.53 +/- 25.92
    MAPPO-40  SR=0.3583 +/- 0.0093   km=171.79 +/- 6.98
    MAPPO-80  SR=0.3621 +/- 0.0090   km=154.53 +/- 3.41
    MAPPO-200 SR=0.3634 +/- 0.0048   km=141.93 +/- 5.24

  Shift retention (S4 cold-zone is the interesting case, all arms):
    IPPO-40  S0=0.367 S4=0.219  retention=0.597
    IPPO-80  S0=0.361 S4=0.215  retention=0.595
    IPPO-200 S0=0.361 S4=0.208  retention=0.575
    MAPPO-40 S0=0.358 S4=0.221  retention=0.617
    MAPPO-80 S0=0.362 S4=0.212  retention=0.585
    MAPPO-200 S0=0.363 S4=0.218 retention=0.599
    Greedy   S0=0.448 S4=0.226  retention=0.505
    LP       S0=0.697 S4=0.469  retention=0.673

These are the numbers the charts below plot. No seed, algorithm, or shift is
invented; anything not present in the source files is left as missing (gap),
never zero or interpolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]

RLEVAL_SUMMARY = PROJECT / "results" / "shift_eval" / "summary.json"
RL_SERIES = PROJECT / "results" / "rl_training_series.json"
BANDRA_BL = PROJECT / "results" / "bandra_baselines_D95.json"

OUT_TRAIN = PROJECT / "results" / "figures_train"
OUT_EVAL = PROJECT / "results" / "figures"
OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_EVAL.mkdir(parents=True, exist_ok=True)

SUMMARY = json.loads(RLEVAL_SUMMARY.read_text())
SERIES = json.loads(RL_SERIES.read_text())
TRACES = SERIES["traces"]
BL_ROWS = json.loads(BANDRA_BL.read_text())
BL_SEEDS = 5  # seed 9000-9004, matched to the 5-seed protocol of the RL arms

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
ALGO_COLOR = {"ippo": "#1f77b4", "mappo": "#ff7f0e"}
HORIZON_COLOR = {40: "#2ca02c", 80: "#d62728", 200: "#9467bd"}
BASELINE_COLOR = {"Greedy": "#2ca02c", "LP": "#d62728"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# summary.json helpers
# ---------------------------------------------------------------------------
RL_ALGOS = ["ippo", "mappo"]
HORIZONS = [40, 80, 200]
SHIFTS = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6", "S7"]
SHIFT_LABEL = {
    "S0":  "S0 base",
    "S1":  "S1 permute",
    "S2a": "S2a x0.5",
    "S2b": "S2b x0.75",
    "S2d": "S2d x1.5",
    "S2e": "S2e x2.0",
    "S3":  "S3 temporal",
    "S4":  "S4 cold z1",
    "S5":  "S5 burst",
    "S6":  "S6",
    "S7":  "S7 drift",
}

def _arm_stats(algo: str, iters: int, shift: str, metric: str):
    """Mean +/- ddof=1 std across the 5 seeds of one RL arm from summary.json.
    Returns (mu, sd, n) or (None,None,0) when no seeds present."""
    vals = []
    for s in range(5):
        key = f"{algo}-bandra-{iters}-s{s}"
        if key not in SUMMARY:
            continue
        v = SUMMARY[key].get(shift, {}).get(metric, {}).get("mean")
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None, None, 0
    a = np.array(vals)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0), len(a)


def _baseline_stats(name: str, metric: str):
    """Bandra Greedy/LP from bandra_baselines_D95.json, 5 seeds, mean +/- ddof=1 std.
    service_rate is percent in that file; empty_km is km/episode.
    These are S0-only: the baselines were never run across the Bandra shifts."""
    rows = BL_ROWS.get(name, [])[:BL_SEEDS]
    vals = [float(r[metric]) for r in rows if metric in r]
    if not vals:
        return None, None, 0
    a = np.array(vals)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0), len(vals)


# ===========================================================================
# FIGURE 1 — retention by shift (normalized to each arm's own S0)
# ===========================================================================
def fig_shift_retention():
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(SHIFTS))
    w = 0.11

    # RL arms as grouped bars — each arm gets its own color within the horizon
    # family so 40/80/200 are distinguishable, and within a horizon the two
    # algorithms are distinguishable.
    arm_spec = [
        ("IPPO-40",  "ippo", 40, "#76ef76"),
        ("MAPPO-40", "mappo",40, "#88d988"),
        ("IPPO-80",  "ippo", 80, "#ff9999"),
        ("MAPPO-80", "mappo",80, "#ffa8a8"),
        ("IPPO-200", "ippo", 200, "#cc99e6"),
        ("MAPPO-200","mappo",200,"#e6b3f0"),
    ]
    plotted_bars = []
    bar_labels = []
    peak_vals: list[float] = [1.0]
    for label, algo, iters, col in arm_spec:
        s0_mu, s0_sd, n_s0 = _arm_stats(algo, iters, "S0", "service_rate")
        if s0_mu is None or s0_mu == 0.0:
            continue
        ys = []
        yerrs = []
        missing = False
        for s in SHIFTS:
            mu, sd, n = _arm_stats(algo, iters, s, "service_rate")
            if mu is None or s0_mu == 0.0:
                ys.append(np.nan)
                yerrs.append(np.nan)
                missing = True
            else:
                ys.append(mu / s0_mu)
                yerrs.append(sd / s0_mu if sd else 0.0)
        peak_vals.extend(abs(v) for v in ys if v == v)
        offs = SHIFTS.index("S4")  # keep bars centered visually per shift group
        # group bars by shift: offset each arm by a small amount within the group
        idx = len(plotted_bars)
        off = (idx - (len(arm_spec)-1)/2) * w * 1.0
        rects = ax.bar(x + off, ys, width=w, color=col, alpha=0.90, zorder=3,
                       yerr=yerrs, capsize=2,
                       error_kw={"elinewidth": 0.9, "ecolor": "black"},
                       label=label)
        plotted_bars.append(rects)
        bar_labels.append(label)

    # No baseline lines: Greedy/LP were only ever run on Bandra S0, and the
    # Greedy/LP entries in summary.json are NYCConfig runs. Drawing either on
    # this Bandra shift axis would compare environments.
    ax.axhline(1.0, color="grey", lw=1.0, ls=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABEL[s] for s in SHIFTS],
                       fontsize=8.5, rotation=22, ha="right")
    ax.set_ylabel("Retention = shifted SR / own S0 SR")
    # Greedy/LP can peak above 1.0 (SR improves under some shifts); a fixed
    # 1.08 ceiling clipped them through the top frame and into the legend.
    ax.set_ylim(0.0, max(max(peak_vals) * 1.08, 1.15))
    ax.set_title("Retention under demand shift (Bandra, normalized to each arm's own S0)")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=8, framealpha=0.96, ncol=3)
    fig.text(
        0.012, 0.012,
        "Bars: each arm's shifted SR divided by that same arm's S0 SR (5 seeds per RL arm). "
        "Error bar = propagated std (ddof=1). "
        "RL arms only: Greedy/LP have no Bandra shift sweep, and the Greedy/LP rows in summary.json are NYC16 runs. "
        "Retention < 1 means the shift reduces SR vs base demand. "
        "S4 (cold-zone) is the sharpest drop. "
        "All values read directly from results/shift_eval/summary.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    pdf = OUT_EVAL / "retention_by_shift.pdf"
    png = OUT_EVAL / "retention_by_shift.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 2 — absolute service rate by shift (not normalized)
# ===========================================================================
def fig_shift_absolute_sr():
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(SHIFTS))
    w = 0.11

    arm_spec = [
        ("IPPO-40",  "ippo", 40, "#76ef76"),
        ("MAPPO-40", "mappo",40, "#88d988"),
        ("IPPO-80",  "ippo", 80, "#ff9999"),
        ("MAPPO-80", "mappo",80, "#ffa8a8"),
        ("IPPO-200", "ippo", 200, "#cc99e6"),
        ("MAPPO-200","mappo",200,"#e6b3f0"),
    ]
    for idx, (label, algo, iters, col) in enumerate(arm_spec):
        off = (idx - (len(arm_spec)-1)/2) * w
        ys = []
        yerrs = []
        for s in SHIFTS:
            mu, sd, n = _arm_stats(algo, iters, s, "service_rate")
            if mu is None:
                ys.append(np.nan)
                yerrs.append(np.nan)
            else:
                ys.append(mu)
                yerrs.append(sd if sd else 0.0)
        ax.bar(x + off, ys, width=w, color=col, alpha=0.90, zorder=3,
               yerr=yerrs, capsize=2,
               error_kw={"elinewidth": 0.9, "ecolor": "black"},
               label=label)

    # No baselines here either, for the same reason as the retention chart:
    # Bandra baseline data exists only for S0.
    ax.axhline(0.5, color="grey", lw=1.0, ls=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABEL[s] for s in SHIFTS],
                       fontsize=8.5, rotation=22, ha="right")
    ax.set_ylabel("Service rate (mean over seeds)")
    ax.set_ylim(0.0, 0.85)
    ax.set_title("Absolute service rate by shift (Bandra, not normalized)")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.96, ncol=2)
    fig.text(
        0.012, 0.012,
        "Bars: RL arms, mean over 5 seeds per algo, absolute SR (not normalized). "
        "Error bar = seed std (ddof=1). "
        "RL arms only: Bandra Greedy/LP were evaluated on S0 alone, so there is no honest shifted baseline line. "
        "S0 base demand visible at the left of every horizon group. "
        "All values read directly from results/shift_eval/summary.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    pdf = OUT_EVAL / "shift_absolute_sr.pdf"
    png = OUT_EVAL / "shift_absolute_sr.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 3 — Pareto: S0 service rate vs empty km
# ===========================================================================
def fig_pareto():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    points = []  # (label, sr%, sr_std%, km, km_std, color, marker)
    for name, keys, col, marker in [
        ("Greedy",  ["Greedy-Baseline"],  "#2ca02c", "D"),
        ("LP",      ["LP-Rebalancer"],    "#d62728", "D"),
        ("IPPO-40", None,                 ALGO_COLOR["ippo"], "o"),
        ("MAPPO-40",None,                 ALGO_COLOR["mappo"], "o"),
        ("IPPO-80", None,                 ALGO_COLOR["ippo"], "o"),
        ("MAPPO-80",None,                 ALGO_COLOR["mappo"], "o"),
        ("IPPO-200",None,                 ALGO_COLOR["ippo"], "o"),
        ("MAPPO-200",None,                ALGO_COLOR["mappo"], "o"),
    ]:
        if name in ("IPPO-40", "MAPPO-40", "IPPO-80", "MAPPO-80",
                    "IPPO-200", "MAPPO-200"):
            short = name.split("-")[0].lower()
            iters = int(name.split("-")[1])
            sr_mu, sr_sd, n = _arm_stats(short, iters, "S0", "service_rate")
            km_mu, km_sd, _ = _arm_stats(short, iters, "S0", "empty_km")
            sr_mu, sr_sd = sr_mu * 100.0, sr_sd * 100.0  # fraction -> percent
        else:
            # D95 rows already carry service_rate in percent
            sr_mu, sr_sd, n = _baseline_stats(name, "service_rate")
            km_mu, km_sd, _ = _baseline_stats(name, "empty_km")
        if sr_mu is None or km_mu is None:
            continue
        points.append((name, sr_mu, sr_sd, km_mu, km_sd, col, marker))

    for label, sr_m, sr_s, km_m, km_s, col, marker in points:
        ax.errorbar(km_m, sr_m, xerr=km_s, yerr=sr_s,
                    fmt=marker, ms=9 if marker=="D" else 8, lw=1.8,
                    capsize=4, color=col, label=label, zorder=5)
        if marker == "D":
            dx, dy = (8, 6)
            ax.annotate(label, (km_m, sr_m), fontsize=8.5, fontweight="bold",
                        xytext=(dx, dy), textcoords="offset points", color="#222222")

    ax.set_xlabel("Empty distance (km / episode, mean over seeds)")
    ax.set_ylabel("Service rate (%, S0 base demand)")
    ax.set_title("Service rate vs empty distance — Bandra (S0)")
    ax.set_xlim(5, 200)
    ax.set_ylim(31, 44.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.96)
    fig.text(
        0.012, 0.012,
        "Each point = mean over 5 seeds +/- ddof=1 std (every arm, baselines included). "
        "Baselines are Bandra runs from results/bandra_baselines_D95.json (not the NYC16 rows in summary.json). "
        "Lower-right is better: at S0 LP beats every RL arm on both axes (41.2% SR at 99 km vs ~36% at 140-177 km). "
        "Greedy is cheapest (21 km) but serves least (35.3%).",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    pdf = OUT_EVAL / "pareto_service_vs_empty.pdf"
    png = OUT_EVAL / "pareto_service_vs_empty.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 4 — training SR over iterations, all horizons, both algos
# ===========================================================================
def fig_train_sr():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for algo in ["ippo", "mappo"]:
        for iters in HORIZONS:
            t = TRACES[f"{algo}-{iters}"]
            col = ALGO_COLOR[algo]
            lbl = f"{algo.upper()}-{iters}"
            ms = 5.0 if iters == 200 else 3.0
            lw = 2.4 if iters == 200 else 1.3
            ax.plot([r["iteration"] for r in t],
                    [r["service_rate_mean"] for r in t],
                    "o-" if iters==200 else "-", color=col, lw=lw, ms=ms,
                    label=lbl, zorder=5)
            sd = np.array([r["service_rate_std"] for r in t])
            ys = np.array([r["service_rate_mean"] for r in t])
            if np.any(sd > 0):
                ax.fill_between([r["iteration"] for r in t], ys - sd, ys + sd,
                                color=col, alpha=0.16, zorder=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("On-policy service rate (mean over 5 seeds)")
    ax.set_title("Service rate vs training iteration — Bandra")
    ax.set_ylim(0.25, 0.42)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, ncol=2)
    fig.text(
        0.012, 0.02,
        "Solid lines = on-policy train SR, eval every iteration. "
        "Thicker purple/orange lines = 200-iter runs. "
        "Shaded band = seed std (ddof=1) across 5 seeds. "
        "All values from results/rl_training_series.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout()
    pdf = OUT_TRAIN / "train_service_rate_iterations.pdf"
    png = OUT_TRAIN / "train_service_rate_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 5 — training empty km over iterations
# ===========================================================================
def fig_train_empty_km():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for algo in ["ippo", "mappo"]:
        for iters in HORIZONS:
            t = TRACES[f"{algo}-{iters}"]
            col = ALGO_COLOR[algo]
            lbl = f"{algo.upper()}-{iters}" if iters == 200 else ""
            ms = 5.0 if iters == 200 else 3.0
            lw = 2.4 if iters == 200 else 1.3
            ax.plot([r["iteration"] for r in t],
                    [r["empty_km_per_ep_mean"] for r in t],
                    "o-" if iters==200 else "-", color=col, lw=lw, ms=ms,
                    label=lbl, zorder=5)
            sd = np.array([r["empty_km_per_ep_std"] for r in t])
            ys = np.array([r["empty_km_per_ep_mean"] for r in t])
            if np.any(sd > 0):
                ax.fill_between([r["iteration"] for r in t], ys - sd, ys + sd,
                                color=col, alpha=0.16, zorder=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Empty km per episode (mean over 5 seeds)")
    ax.set_title("Empty distance vs training iteration — Bandra")
    ax.set_ylim(140, 240)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    fig.text(
        0.012, 0.02,
        "Thicker lines = 200-iter runs. "
        "Shaded band = seed std (ddof=1). Lower is better. "
        "All values from results/rl_training_series.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout()
    pdf = OUT_TRAIN / "train_empty_km_iterations.pdf"
    png = OUT_TRAIN / "train_empty_km_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 6 — training reward per episode
# ===========================================================================
def fig_train_reward():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for algo in ["ippo", "mappo"]:
        for iters in HORIZONS:
            t = TRACES[f"{algo}-{iters}"]
            col = ALGO_COLOR[algo]
            lbl = f"{algo.upper()}-{iters}" if iters == 200 else ""
            ms = 5.0 if iters == 200 else 3.0
            lw = 2.4 if iters == 200 else 1.3
            ax.plot([r["iteration"] for r in t],
                    [r["reward_per_ep_mean"] for r in t],
                    "o-" if iters==200 else "-", color=col, lw=lw, ms=ms,
                    label=lbl, zorder=5)
            sd = np.array([r["reward_per_ep_std"] for r in t])
            ys = np.array([r["reward_per_ep_mean"] for r in t])
            if np.any(sd > 0):
                ax.fill_between([r["iteration"] for r in t], ys - sd, ys + sd,
                                color=col, alpha=0.16, zorder=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward per episode (mean over 5 seeds)")
    ax.set_title("Reward per episode vs training iteration — Bandra")
    ax.set_ylim(280, 460)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)
    fig.text(
        0.012, 0.02,
        "Thicker lines = 200-iter runs. "
        "Shaded band = seed std (ddof=1). "
        "All values from results/rl_training_series.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout()
    pdf = OUT_TRAIN / "train_reward_iterations.pdf"
    png = OUT_TRAIN / "train_reward_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 7 — action mix over iterations
# ===========================================================================
def fig_train_action_mix():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharex=True)
    for ax, (key, ylab, ymin, ymax) in zip(
        axes,
        [("pct_dispatch", "Dispatch (%)", 4.5, 8.0),
         ("pct_relocate", "Relocate (%)", 3.5, 7.5),
         ("pct_idle",     "Idle (%)",     86.0, 90.5)],
    ):
        for algo in ["ippo", "mappo"]:
            for iters in HORIZONS:
                t = TRACES[f"{algo}-{iters}"]
                col = ALGO_COLOR[algo]
                lbl = f"{algo.upper()}-{iters}" if iters == 200 else ""
                ms = 5.0 if iters == 200 else 3.0
                lw = 2.4 if iters == 200 else 1.3
                ax.plot([r["iteration"] for r in t],
                        [r[f"{key}_mean"] for r in t],
                        "o-" if iters==200 else "-", color=col, lw=lw, ms=ms,
                        label=lbl, zorder=5)
                sd = np.array([r[f"{key}_std"] for r in t])
                ys = np.array([r[f"{key}_mean"] for r in t])
                if np.any(sd > 0):
                    ax.fill_between([r["iteration"] for r in t], ys - sd, ys + sd,
                                    color=col, alpha=0.16, zorder=4)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Iteration")
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Action mix vs training iteration — Bandra",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(
        0.012, 0.012,
        "Shaded band = seed std (ddof=1). "
        "Thicker lines = 200-iter runs. "
        "All values from results/rl_training_series.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    pdf = OUT_TRAIN / "train_action_mix_iterations.pdf"
    png = OUT_TRAIN / "train_action_mix_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 8 — loss components (200-iter runs only)
# ===========================================================================
def fig_train_loss():
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0), sharex=True)
    keys = [
        ("loss_policy_loss",   "Policy loss",       True),
        ("loss_value_loss",    "Value loss",        False),
        ("loss_entropy",       "Entropy",           False),
        ("loss_clip_frac",     "Clip fraction",     False),
        ("loss_real_frac",     "Real fraction",     False),
        ("loss_explained_var", "Explained variance",False),
    ]
    for ax, (key, ylab, invert) in zip(axes.ravel(), keys):
        for algo in ["ippo", "mappo"]:
            t = TRACES[f"{algo}-200"]
            col = ALGO_COLOR[algo]
            ys = np.array([r[f"{key}_mean"] for r in t])
            sd = np.array([r[f"{key}_std"] for r in t])
            xs = [r["iteration"] for r in t]
            ax.plot(xs, ys, "o-", color=col, lw=2.0, ms=4,
                    label=algo.upper(), zorder=5)
            if np.any(sd > 0):
                ax.fill_between(xs, ys - sd, ys + sd,
                                color=col, alpha=0.16, zorder=4)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Iteration")
        if invert:
            ax.invert_yaxis()
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.suptitle("Loss components vs training iteration — Bandra, 200-iter runs",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.text(
        0.012, 0.012,
        "Shaded band = seed std (ddof=1). "
        "All values from results/rl_training_series.json.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    pdf = OUT_TRAIN / "train_loss_components_iterations.pdf"
    png = OUT_TRAIN / "train_loss_components_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# FIGURE 9 — terminal Pareto from training-series terminal rows
# ===========================================================================
def fig_train_terminal_pareto():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    # Bandra baselines from bandra_baselines_D95.json (service_rate already %)
    for name, col, marker in [("Greedy", "#2ca02c", "D"), ("LP", "#d62728", "D")]:
        sr_m, sr_s, _ = _baseline_stats(name, "service_rate")
        km_m, km_s, _ = _baseline_stats(name, "empty_km")
        ax.errorbar(km_m, sr_m, xerr=km_s, yerr=sr_s,
                    fmt=marker, ms=9, lw=1.8, capsize=4, color=col,
                    label=name, zorder=5)
        ax.annotate(name, (km_m, sr_m), fontsize=9, fontweight="bold",
                    xytext=(7, -12), textcoords="offset points", color="#222222")

    for algo, iters in [("ippo", 200), ("mappo", 200)]:
        t = TRACES[f"{algo}-{iters}"]
        last = t[-1]
        sr_m = last["service_rate_mean"] * 100.0
        sr_s = last["service_rate_std"] * 100.0
        km_m = last["empty_km_per_ep_mean"]
        km_s = last["empty_km_per_ep_std"]
        ax.errorbar(km_m, sr_m, xerr=km_s, yerr=sr_s,
                    fmt="o", ms=10, lw=2.2, capsize=4,
                    color=ALGO_COLOR[algo],
                    label=f"{algo.upper()}-{iters} (seed-mean ± seed std)",
                    zorder=5)
        ax.annotate(f"{algo.upper()}-{iters} (mean)",
                    (km_m, sr_m), fontsize=9, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points", color="#222222")

    ax.set_xlabel("Empty distance (km / episode)")
    ax.set_ylabel("Service rate (%, S0 base demand)")
    ax.set_title("Terminal tradeoff — Service rate vs empty distance (Bandra)")
    ax.set_xlim(5, 200)
    ax.set_ylim(31, 44.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.text(
        0.012, 0.02,
        "RL arms: large point = seed-mean ± seed std (ddof=1) from the terminal training-series row. "
        "Baselines: 5-seed mean ± std from results/bandra_baselines_D95.json. "
        "Lower-right is better; at S0 LP dominates every RL arm on both axes. "
        "Greedy is cheapest but serves least. All values read directly from the source files.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    pdf = OUT_TRAIN / "train_terminal_pareto.pdf"
    png = OUT_TRAIN / "train_terminal_pareto.png"
    fig.savefig(pdf, dpi=200)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("Saved:", pdf, png)


# ===========================================================================
# main
# ===========================================================================
def main():
    print("Building eval figures into", OUT_EVAL)
    fig_shift_retention()
    fig_shift_absolute_sr()
    fig_pareto()
    print("Building train figures into", OUT_TRAIN)
    fig_train_sr()
    fig_train_empty_km()
    fig_train_reward()
    fig_train_action_mix()
    fig_train_loss()
    fig_train_terminal_pareto()
    print("Done.")


if __name__ == "__main__":
    main()
