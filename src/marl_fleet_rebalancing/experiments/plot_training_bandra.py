"""Generate the primary training-trace figures for the Bandra IPPO/MAPPO suite.

Source of truth
---------------
results/rl_training_series.json  — per-(algo,horizon) per-iteration traces,
    seed-mean and ddof=1 std across 5 seeds, for every logged metric.
results/shift_eval/summary.json  — per-shift SR and empty_km (S0 base + shifts).
results/bandra_fleet_D93/sweep_results.json — fleet-size sweep (20..35 drivers).
results/bandra_baselines_D95.json — Greedy / LP baseline per-episode KPIs.

Output directory
----------------
results/figures_train/  (PDF + PNG each)

Every figure is self-contained: title, axis labels, legend, caption.
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
OUT = PROJECT / "results" / "figures_train"
OUT.mkdir(parents=True, exist_ok=True)

def _load_json(path):
    p = PROJECT / path
    return json.loads(p.read_text())

SERIES = _load_json("results/rl_training_series.json")
SUMMARY = _load_json("results/shift_eval/summary.json")
FLEET_SWEEP = _load_json("results/bandra_fleet_D93/sweep_results.json")
BASELINES = _load_json("results/bandra_baselines_D95.json")

# ---------------------------------------------------------------------------
# palette + labels
# ---------------------------------------------------------------------------
ALGO_COLOR = {"ippo": "#1f77b4", "mappo": "#ff7f0e"}
ALGO_LABEL = {"ippo": "IPPO", "mappo": "MAPPO"}
HORIZON_COLOR = {40: "#2ca02c", 80: "#d62728", 200: "#9467bd"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def trace(algo: str, iters: int):
    return SERIES["traces"][f"{algo}-{iters}"]


def _ci(x, s):
    return x, s


def plot_mean_std(ax, recs, key, color, label, fmt="o-", lw=1.8, ms=4,
                  xkey="iteration", y_transform=lambda v: v, fill=True):
    xs = [rec[xkey] for rec in recs]
    ys = [y_transform(rec[f"{key}_mean"]) for rec in recs]
    stds = [rec[f"{key}_std"] for rec in recs]
    ys = np.array([y if y == y else np.nan for y in ys])  # NaN-safe
    stds = np.array([s if s == s else 0.0 for s in stds])
    line, = ax.plot(xs, ys, fmt, color=color, lw=lw, ms=ms, label=label, zorder=5)
    if fill and np.any(stds > 0):
        ax.fill_between(xs, ys - stds, ys + stds, color=color, alpha=0.16, zorder=4)
    return line


# ===========================================================================
# FIGURE 1 — service rate over iterations, all horizons, both algos
# ===========================================================================
def fig_service_rate_iterations():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            col = ALGO_COLOR[algo]
            lbl = f"{algo.upper()}-{iters}"
            plot_mean_std(ax, t, "service_rate", col, lbl,
                          fmt="o-" if iters == 200 else "-",
                          lw=2.4 if iters == 200 else 1.3,
                          ms=5.0 if iters == 200 else 3.0)
    # val_service_rate for each horizon where available
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            vrecs = [r for r in t if r["val_service_rate_mean"] == r["val_service_rate_mean"]]
            if not vrecs:
                continue
            col = ALGO_COLOR[algo]
            ax.plot([r["iteration"] for r in vrecs],
                    [r["val_service_rate_mean"] for r in vrecs],
                    "s--", color=col, ms=5, lw=1.4,
                    label=f"{algo.upper()}-{iters} val(SR)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Service rate (mean over 5 seeds)")
    ax.set_title("Service rate vs training iteration — Bandra")
    ax.set_ylim(0.25, 0.42)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, ncol=2)
    ax.text(0.012, 0.02,
            "Solid = on-policy train SR (eval every iteration). "
            "Dashed square = held-out val SR (eval every 10 iters). "
            "Shaded band = seed std (ddof=1) across 5 seeds.",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout()
    pdf = OUT / "train_service_rate_iterations.pdf"
    png = OUT / "train_service_rate_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 2 — empty km over iterations (cost dimension)
# ===========================================================================
def fig_empty_km_iterations():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            col = ALGO_COLOR[algo]
            plot_mean_std(ax, t, "empty_km_per_ep", col,
                          f"{algo.upper()}-{iters}" if iters == 200 else "",
                          fmt="o-" if iters == 200 else "-",
                          lw=2.2 if iters == 200 else 1.3,
                          ms=4.5 if iters == 200 else 3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Empty km per episode (mean over 5 seeds)")
    ax.set_title("Empty distance vs training iteration — Bandra")
    ax.set_ylim(140, 235)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    ax.text(0.012, 0.02,
            "Shaded band = seed std (ddof=1). Lower is better (less deadhead).",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout()
    pdf = OUT / "train_empty_km_iterations.pdf"
    png = OUT / "train_empty_km_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 3 — wait time over iterations
# ===========================================================================
def fig_wait_iterations():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            col = ALGO_COLOR[algo]
            plot_mean_std(ax, t, "avg_wait_min", col,
                          f"{algo.upper()}-{iters}" if iters == 200 else "",
                          fmt="o-" if iters == 200 else "-",
                          lw=2.2 if iters == 200 else 1.3,
                          ms=4.5 if iters == 200 else 3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg passenger wait (min, mean over 5 seeds)")
    ax.set_title("Average wait vs training iteration — Bandra")
    ax.set_ylim(1.8, 3.8)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    ax.text(0.012, 0.02,
            "Shaded band = seed std (ddof=1). Lower is better.",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout()
    pdf = OUT / "train_avg_wait_iterations.pdf"
    png = OUT / "train_avg_wait_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 4 — reward per episode over iterations
# ===========================================================================
def fig_reward_iterations():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            col = ALGO_COLOR[algo]
            plot_mean_std(ax, t, "reward_per_ep", col,
                          f"{algo.upper()}-{iters}" if iters == 200 else "",
                          fmt="o-" if iters == 200 else "-",
                          lw=2.2 if iters == 200 else 1.3,
                          ms=4.5 if iters == 200 else 3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward per episode (mean over 5 seeds)")
    ax.set_title("Reward per episode vs training iteration — Bandra")
    ax.set_ylim(280, 470)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)
    ax.text(0.012, 0.02,
            "Shaded band = seed std (ddof=1). Reward = served - beta*relocated*km - cancelled penalty.",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout()
    pdf = OUT / "train_reward_iterations.pdf"
    png = OUT / "train_reward_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 5 — action mix over iterations (pct_dispatch / pct_relocate / pct_idle)
# ===========================================================================
def fig_action_mix_iterations():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)
    keys = [("pct_dispatch", "Dispatch (%)"), ("pct_relocate", "Relocate (%)"),
            ("pct_idle", "Idle (%)")]
    for ax, (key, ylab) in zip(axes, keys):
        for algo in ["ippo", "mappo"]:
            for iters in [40, 80, 200]:
                t = trace(algo, iters)
                col = ALGO_COLOR[algo]
                plot_mean_std(ax, t, key, col,
                              f"{algo.upper()}-{iters}" if iters == 200 else "",
                              fmt="o-" if iters == 200 else "-",
                              lw=2.2 if iters == 200 else 1.3,
                              ms=4.5 if iters == 200 else 3.0)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Iteration")
        if key == "pct_idle":
            ax.set_ylim(86, 90.5)
        elif key == "pct_relocate":
            ax.set_ylim(3.5, 7.5)
        else:
            ax.set_ylim(4.5, 8.0)
        ax.text(0.012, 0.02,
                "Shaded band = seed std (ddof=1).",
                transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    for ax, (_, title) in zip(axes, keys):
        ax.set_title(title)
    fig.suptitle("Action mix vs training iteration — Bandra", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    pdf = OUT / "train_action_mix_iterations.pdf"
    png = OUT / "train_action_mix_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 6 — learning curves: loss components over iterations (200-horizon)
# ===========================================================================
def fig_loss_iterations():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True)
    keys = [
        ("loss_policy_loss", "Policy loss", True),
        ("loss_value_loss", "Value loss", False),
        ("loss_entropy", "Entropy", False),
        ("loss_clip_frac", "Clip frac", False),
        ("loss_real_frac", "Real frac", False),
        ("loss_explained_var", "Explained var", False),
    ]
    for ax, (key, ylab, invert) in zip(axes.ravel(), keys):
        for algo in ["ippo", "mappo"]:
            t = trace(algo, 200)
            col = ALGO_COLOR[algo]
            def yt(rec):
                v = rec[f"{key}_mean"]
                return v if v == v else np.nan
            xs = [r["iteration"] for r in t]
            ys = [yt(r) for r in t]
            stds = [r[f"{key}_std"] for r in t]
            ys = np.array([y if y == y else np.nan for y in ys])
            stds = np.array([s if s == s else 0.0 for s in stds])
            ax.plot(xs, ys, "o-", color=col, lw=2.0, ms=4, label=algo.upper(), zorder=5)
            if np.any(stds > 0):
                ax.fill_between(xs, ys - stds, ys + stds, color=col, alpha=0.16, zorder=4)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Iteration")
        if invert:
            ax.invert_yaxis()
        ax.text(0.012, 0.02,
                "Shaded band = seed std (ddof=1).",
                transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
    fig.suptitle("Loss components vs training iteration — Bandra, 200-iter runs", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    pdf = OUT / "train_loss_components_iterations.pdf"
    png = OUT / "train_loss_components_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 7 — validation service rate across iterations (where available)
# ===========================================================================
def fig_val_service_rate():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    plotted = []
    for algo in ["ippo", "mappo"]:
        for iters in [40, 80, 200]:
            t = trace(algo, iters)
            vrecs = [r for r in t if r["val_service_rate_mean"] == r["val_service_rate_mean"]]
            if not vrecs:
                continue
            col = ALGO_COLOR[algo]
            ls = "--" if iters == 80 else "-"
            lw = 2.0 if iters == 200 else 1.4
            ms = 5.5 if iters == 200 else 4.0
            marker = "s" if iters == 80 else "o"
            h = plot_mean_std(
                ax, vrecs, "val_service_rate", col, None,
                fmt=f"{marker}{ls}", lw=lw, ms=ms,
            )
            if h is not None:
                plotted.append((h, f"{algo.upper()}-{iters}"))
    ax.legend(
        [h for h, _ in plotted],
        [l for _, l in plotted],
        loc="upper left", fontsize=9, framealpha=0.95, ncol=2,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation service rate (held-out)")
    ax.set_title("Validation service rate vs iteration — Bandra")
    ax.set_ylim(0.30, 0.38)
    fig.text(
        0.012, 0.012,
        "valSR only logged at eval-interval iters (10). "
        "Shaded band = seed std (ddof=1) across 5 seeds. "
        "Each line = one algorithm x horizon; dashed = 80-iter runs, solid = 40/200-iter runs.",
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    pdf = OUT / "train_val_service_rate_iterations.pdf"
    png = OUT / "train_val_service_rate_iterations.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 8 — terminal comparison: SR vs empty km (Pareto-style, train)
# ===========================================================================
def fig_terminal_pareto_train():
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    # baselines
    def _bl(name):
        rows = BASELINES[name]
        sr = [r["service_rate"] for r in rows]
        km = [r["empty_km"] for r in rows]
        return (float(np.mean(sr)), float(np.std(sr, ddof=1)),
                float(np.mean(km)), float(np.std(km, ddof=1)))
    bl = {"Greedy": _bl("Greedy"), "LP": _bl("LP")}
    for name, (sr_m, sr_s, km_m, km_s) in bl.items():
        ax.errorbar(km_m, sr_m, xerr=km_s, yerr=sr_s, fmt="D", ms=8, lw=1.8,
                    capsize=4, color={"Greedy": "#2ca02c", "LP": "#d62728"}[name],
                    label=name, zorder=5)
        ax.annotate(name, (km_m, sr_m), fontsize=9, fontweight="bold",
                    xytext=(6, -10), textcoords="offset points", color="#222222")
    # RL arms: each seed as a faint point + mean marker + error bar
    for algo, iters in [("ippo", 200), ("mappo", 200)]:
        recs = trace(algo, iters)[-1]
        sr_mean = recs["service_rate_mean"]
        km_mean = recs["empty_km_per_ep_mean"]
        sr_std = recs["service_rate_std"]
        km_std = recs["empty_km_per_ep_std"]
        # seed-level scatter from the terminal rows
        seeds = []
        for s in range(5):
            key = f"{algo}-bandra-{iters}-s{s}"
            rows = [json.loads(l) for l in (Path(PROJECT) / "runs" / key / "metrics.jsonl").read_text().strip().split('\n') if l.strip()]
            seeds.append(rows[-1])
        for r in seeds:
            ax.scatter(r["empty_km_per_ep"], r["service_rate"],
                       color=ALGO_COLOR[algo], alpha=0.35, s=22, zorder=3)
        ax.errorbar(km_mean, sr_mean, xerr=km_std, yerr=sr_std, fmt="o", ms=10,
                    lw=2.2, capsize=4, color=ALGO_COLOR[algo],
                    label=f"{algo.upper()}-{iters} (mean ± seed std)", zorder=5)
        ax.annotate(f"{algo.upper()}-{iters} (mean)", (km_mean, sr_mean),
                    fontsize=9, fontweight="bold",            xytext=(8, 8), textcoords="offset points", color="#222222")
    ax.set_xlabel("Empty distance (km / episode)")
    ax.set_ylabel("Service rate")
    ax.set_title("Terminal tradeoff — Service rate vs empty distance (Bandra)")
    ax.set_xlim(132, 236)
    ax.set_ylim(0.298, 0.376)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", fontsize=9, framealpha=0.95)
    bbox = dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#cccccc", lw=0.6)
    ax.text(
        0.012, 0.02,
        "RL arms: each small point = one seed's terminal row; large point = seed-mean ± seed std (ddof=1). "
        "Baselines: mean of 12 episodes ± episode std. "
        "Lower-left is better (high SR, low empty km). "
        "Greedy / LP are single-seed baseline eval; RL arms are the 200-iter terminal policies.",
        transform=ax.transAxes, fontsize=8, color="#444444", va="bottom",
        bbox=bbox, wrap=True,
    )
    fig.tight_layout()
    # "_legacy": plot_train_charts.fig_train_terminal_pareto owns
    # train_terminal_pareto.* and is the version the report uses. This one draws
    # NYC-free Bandra baselines but with the mean-only view; keep it separate so
    # running this module cannot overwrite the active chart.
    pdf = OUT / "train_terminal_pareto_legacy.pdf"
    png = OUT / "train_terminal_pareto_legacy.png"
    fig.savefig(pdf, dpi=200)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("Saved:", pdf)
    # ---- higher-res redo of the scatter inset so dots are crisp ----
    fig2 = plt.figure(figsize=(6.2, 4.6))
    ax2 = fig2.add_subplot(111)
    for algo, iters in [("ippo", 200), ("mappo", 200)]:
        seeds = []
        for s in range(5):
            key = f"{algo}-bandra-{iters}-s{s}"
            rows = [json.loads(l) for l in (Path(PROJECT) / "runs" / key / "metrics.jsonl").read_text().strip().split('\n') if l.strip()]
            seeds.append(rows[-1])
        for r in seeds:
            ax2.scatter(r["empty_km_per_ep"], r["service_rate"],
                        color=ALGO_COLOR[algo], alpha=0.45, s=28, zorder=3,
                        edgecolor="white", linewidth=0.4)
    from matplotlib.lines import Line2D
    ax2.legend(
        handles=[Line2D([0], [0], marker="o", ls="", color=ALGO_COLOR[a],
                        markersize=7, label=f"{a.upper()}-200") for a in ("ippo", "mappo")],
        loc="lower left", fontsize=9, framealpha=0.95, title="policy (5 seeds each)",
    )
    ax2.set_xlabel("Empty distance (km / episode)")
    ax2.set_ylabel("Service rate")
    ax2.set_title("Terminal tradeoff — seed-level detail (Bandra, 200 iters)")
    ax2.set_xlim(132, 236)
    ax2.set_ylim(0.298, 0.376)
    ax2.grid(True, which="major", linestyle="--", linewidth=0.6, color="#E5E5E5")
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    pdf2 = OUT / "train_terminal_pareto_seed_detail.pdf"
    png2 = OUT / "train_terminal_pareto_seed_detail.png"
    fig2.savefig(pdf2, dpi=200)
    fig2.savefig(png2, dpi=200)
    plt.close(fig2)
    print("Saved:", pdf2)


# ===========================================================================
# FIGURE 9 — fleet-size sweep (from existing results/bandra_fleet_D93)
# ===========================================================================
def _baseline_point(name):
    rows = BASELINES[name]
    sr = [r["service_rate"] for r in rows]
    km = [r["empty_km"] for r in rows]
    return (float(np.mean(sr)), float(np.std(sr, ddof=1)),
            float(np.mean(km)), float(np.std(km, ddof=1)))

def fig_fleet_sweep():
    # group by (policy_prefix, drivers)
    from collections import defaultdict
    by_drivers = defaultdict(lambda: defaultdict(list))
    for e in FLEET_SWEEP:
        algo = e.get("policy", "")
        prefix = algo.split("-")[0].upper()
        d = int(e["drivers"])
        by_drivers[d][prefix].append(e)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {"IPPO": "#1f77b4", "MAPPO": "#ff7f0e"}
    for prefix in ["IPPO", "MAPPO"]:
        xs, ys, xerrs, yerrs = [], [], [], []
        for d in sorted(by_drivers):
            entries = by_drivers[d].get(prefix, [])
            if not entries:
                continue
            srs = [float(e["service_rate"]) for e in entries]
            kms = [float(e["empty_km"]) for e in entries]
            xs.append(np.mean(kms))
            ys.append(np.mean(srs))
            xerrs.append(np.std(kms, ddof=1) if len(kms) > 1 else 0.0)
            yerrs.append(np.std(srs, ddof=1) if len(srs) > 1 else 0.0)
        ax.plot(xs, ys, "o-", color=colors[prefix], lw=2.0, ms=5,
                label=f"{prefix} fleet sweep (20–35 drivers)")
        ax.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt="none",
                    ecolor=colors[prefix], elinewidth=1.2, capsize=3, alpha=0.8)
    greedy_bl = _baseline_point("Greedy")
    lp_bl = _baseline_point("LP")
    for name, (sr_m, sr_s, km_m, km_s), col in [
        ("Greedy", greedy_bl, "#2ca02c"), ("LP", lp_bl, "#d62728")
    ]:
        ax.errorbar(km_m, sr_m, xerr=km_s, yerr=sr_s, fmt="s", ms=8, lw=1.6,
                    capsize=4, color=col, label=name, zorder=5)
    ax.set_xlabel("Empty distance (km / episode, mean)")
    ax.set_ylabel("Service rate (%)")
    ax.set_title("Fleet size sweep — Bandra (IPPO-200-s0, MAPPO-200-s0)")
    ax.set_xlim(50, 270)
    ax.set_ylim(20, 55)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.text(0.012, 0.02,
            "Sweep uses single seed per algo (20–35 drivers, 10 episodes each). "
            "Baselines from 12-episode eval. Service rate shown as percent.",
            transform=ax.transAxes, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout()
    pdf = OUT / "train_fleet_sweep.pdf"
    png = OUT / "train_fleet_sweep.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 10 — distribution shift retention (re-plot from existing summary)
#           but add IPPO/MAPPO 40/80/200 as separate curves so the shift
#           behaviour across horizons is visible.
# ===========================================================================
def _shift_mean_std(keys, shift, metric):
    means = []
    for k in keys:
        v = SUMMARY.get(k, {}).get(shift, {}).get(metric, {})
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


def _rl_keys(iters):
    out = []
    for algo in ("ippo", "mappo"):
        for s in range(5):
            cand = f"{algo}-bandra-{iters}-s{s}"
            if cand in SUMMARY:
                out.append(cand)
    return out


# ---- distribution-shift design (meaningful categories) ----
# S0  : base Bandra demand (seed-window as provided by the train/eval data)
# S1  : zone-label permute — demand magnitudes unchanged, zone identities shuffled
# S2a : demand scale x0.5  — half the base load
# S2b : demand scale x0.75 — 3/4 base load
# S2d : demand scale x1.5  — 1.5x base load
# S2e : demand scale x2.0  — double base load
# S3  : temporal shift — demand time-of-day profile shifted (peak moved)
# S4  : cold-start zone — one zone initialized empty / cold for the episode
# S5  : burst — a short high-demand burst injected mid-episode
# S7  : slow drift — demand gradually drifts away from the base pattern over the episode
#
# Order keeps the scale family (S2*) contiguous and brackets the structural
# shifts (S1, S3, S4, S5, S7) so a reader can compare "same magnitude, different
# structure" vs "same structure, different magnitude".
SHIFTS = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S7"]
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
    "S7":  "S7 drift",
}


def fig_shift_retention_by_horizon():
    """Retention = shifted SR / S0 SR, per method, per horizon.

    Normalizing by each method's own S0 removes the base-demand level so the
    chart answers "how much does each shift cost this policy" rather than
    "which policy has higher absolute SR".  The latter is in the companion
    absolute-SR chart.

    Top panel: grouped bars for 40/80/200 iters plus Greedy/LP baselines
    (line+diamond). Bottom panel: the 200-horizon RL arms alone, with their
    real S4 retention visible rather than hidden behind the bar summary.
    """
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.20},
    )
    x = np.arange(len(SHIFTS))
    w = 0.20
    for i, iters in enumerate([40, 80, 200]):
        keys = _rl_keys(iters)
        s0 = _shift_mean_std(keys, "S0", "service_rate")
        if s0 is None or s0[0] == 0.0:
            continue
        s0_mu = s0[0]
        offs = (i - 1) * w
        ys = []
        yerrs = []
        for s in SHIFTS:
            out = _shift_mean_std(keys, s, "service_rate")
            if out is None or s0_mu == 0.0:
                ys.append(np.nan)
                yerrs.append(np.nan)
            else:
                ret = out[0] / s0_mu
                ys.append(ret)
                yerrs.append(out[1] / s0_mu if out[1] else 0.0)
        ax.bar(x + offs, ys, width=w, color=HORIZON_COLOR[iters],
               label=f"{iters} iters", alpha=0.90, zorder=3,
               yerr=yerrs, capsize=2,
               error_kw={"elinewidth": 1.0, "ecolor": "black"})
    for name, keys, col in [("Greedy", ["Greedy-Baseline"], "#2ca02c"),
                             ("LP", ["LP-Rebalancer"], "#d62728")]:
        s0 = _shift_mean_std(keys, "S0", "service_rate")
        if s0 is None or s0[0] == 0.0:
            continue
        s0_mu = s0[0]
        ys = []
        for s in SHIFTS:
            out = _shift_mean_std(keys, s, "service_rate")
            if out is None or s0_mu == 0.0:
                ys.append(np.nan)
            else:
                ys.append(out[0] / s0_mu)
        ax.plot(x, ys, "D-", color=col, lw=1.6, ms=5, label=name, zorder=5)
    ax.axhline(1.0, color="grey", lw=1.0, ls=":", zorder=2,
               label="_nolegend_")
    ax.set_ylabel("Retention (shifted SR / S0 SR)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="#E5E5E5")
    ax.set_axisbelow(True)
    for algo, col in [("ippo", "#1f6feb"), ("mappo", "#ff7f0e")]:
        keys = [f"{algo}-bandra-200-s{s}" for s in range(5)
                if f"{algo}-bandra-200-s{s}" in SUMMARY]
        if not keys:
            continue
        s0 = _shift_mean_std(keys, "S0", "service_rate")
        if s0 is None or s0[0] == 0.0:
            continue
        s0_mu = s0[0]
        ys = []
        yerrs = []
        for s in SHIFTS:
            out = _shift_mean_std(keys, s, "service_rate")
            if out is None or s0_mu == 0.0:
                ys.append(np.nan)
                yerrs.append(np.nan)
            else:
                ys.append(out[0] / s0_mu)
                yerrs.append(out[1] / s0_mu if out[1] else 0.0)
        ax2.errorbar(
            x, ys, yerr=yerrs, fmt="o--", color=col, lw=1.2, ms=4.0,
            capsize=2, elinewidth=1.0, alpha=0.9,
            label=f"{algo.upper()}-200", zorder=4,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHIFT_LABEL[s] for s in SHIFTS],
                         fontsize=8, rotation=20, ha="right")
    ax2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_yticklabels([f"{v:.1f}" for v in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
                         fontsize=8)
    ax2.set_ylim(0.40, 1.05)
    ax2.set_ylabel("Retention (200-iter RL)")
    ax2.axhline(1.0, color="grey", lw=0.7, ls=":", zorder=1)
    ax2.grid(True, which="major", linestyle="--", linewidth=0.6, color="#E5E5E5")
    ax2.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if l != "_nolegend_"]
    labels = [l for l in labels if l != "_nolegend_"]
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right",
              fontsize=8, framealpha=0.95, ncol=2)
    cap = (
        "Bars: RL arms, mean over 5 seeds/algo, normalized by that algo\'s own S0; "
        "error bar = propagated std (ddof=1). "
        "Diamonds: Greedy / LP baselines (1 seed each, normalized by own S0). "
        "Bottom panel: the 200-iter RL arms alone, so the S4 cold-zone drop is visible "
        "rather than hidden behind the 40/80-iter bars. "
        "Retention < 1 means the shift reduces SR vs base demand."
    )
    fig.text(
        0.012, 0.012, cap,
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    pdf = OUT / "shift_retention_by_horizon.pdf"
    png = OUT / "shift_retention_by_horizon.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# FIGURE 11 — service rate by shift, raw (not normalized), by horizon
# ===========================================================================
def fig_shift_service_rate_by_horizon():
    """Absolute service rate (not normalized) per shift, per horizon.

    Complements the retention chart: retention answers "how much does each shift
    cost", this chart answers "what is the actual service level under each shift".
    The S0 base demand level is visible at the left of every horizon group.

    RL bars are plotted at the policy scale (~0.20-0.40). Greedy/LP sit well
    above that (0.22-0.80) and are given their own top axis so the lines are
    not clipped by the RL ylim.
    """
    fig, (ax, ax_top) = plt.subplots(
        2, 1, figsize=(9.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.1], "hspace": 0.30},
    )
    x = np.arange(len(SHIFTS))
    w = 0.20
    for i, iters in enumerate([40, 80, 200]):
        keys = _rl_keys(iters)
        ys = []
        yerrs = []
        for s in SHIFTS:
            out = _shift_mean_std(keys, s, "service_rate")
            if out is None:
                ys.append(np.nan)
                yerrs.append(np.nan)
            else:
                ys.append(out[0])
                yerrs.append(out[1])
        ax.bar(x + (i - 1) * w, ys, width=w, color=HORIZON_COLOR[iters],
               label=f"{iters} iters", alpha=0.88, zorder=3,
               yerr=yerrs, capsize=2, error_kw={"elinewidth": 1.0, "ecolor": "black"})
    for name, keys, col in [("Greedy", ["Greedy-Baseline"], "#2ca02c"),
                             ("LP", ["LP-Rebalancer"], "#d62728")]:
        ys = []
        yerrs = []
        for s in SHIFTS:
            out = _shift_mean_std(keys, s, "service_rate")
            if out is None:
                ys.append(np.nan); yerrs.append(np.nan)
            else:
                ys.append(out[0]); yerrs.append(out[1])
        ax_top.plot(x, ys, "D-", color=col, lw=1.6, ms=5, label=name, zorder=5)
    ax.set_ylabel("RL policies (mean SR)")
    ax_top.set_ylabel("NYC16 Greedy/LP (SR)")
    ax.set_title("Service rate by shift category — IPPO/MAPPO by horizon")
    ax.set_ylim(0.0, 0.45)
    ax_top.set_ylim(0.20, 0.85)
    ax_top.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    ax_top.grid(True, which="major", linestyle="--", linewidth=0.6, color="#E5E5E5")
    ax_top.set_axisbelow(True)
    ax_top.axhline(0.5, color="grey", lw=0.8, ls=":", zorder=2)
    ax_top.set_xticks(x)
    # Horizontal labels: rotated ones ran into the caption regardless of rect.
    ax_top.set_xticklabels([SHIFT_LABEL[s] for s in SHIFTS],
                           fontsize=7.5, rotation=0, ha="center")
    ax.tick_params(labelbottom=False)
    cap = (
        "RL bars: Bandra, mean over 5 seeds per algo, absolute SR (not normalized), seed std (ddof=1). "
        "Top panel: Greedy / LP on their own scale so the lines are not clipped; "
        "those are NYC16 runs (shift_eval.py builds baselines with NYCConfig), "
        "so do not compare them against the Bandra bars."
    )
    fig.text(
        0.012, 0.012, cap,
        fontsize=8, color="#444444", va="bottom", wrap=True,
    )
    fig.subplots_adjust(bottom=0.17, top=0.90, left=0.10, right=0.98)
    pdf = OUT / "shift_service_rate_by_horizon.pdf"
    png = OUT / "shift_service_rate_by_horizon.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print("Saved:", pdf)


# ===========================================================================
# main
# ===========================================================================
def main():
    print("Building training figures into", OUT)
    fig_service_rate_iterations()
    fig_empty_km_iterations()
    fig_wait_iterations()
    fig_reward_iterations()
    fig_action_mix_iterations()
    fig_loss_iterations()
    fig_val_service_rate()
    fig_terminal_pareto_train()
    fig_fleet_sweep()
    # fig_shift_retention_by_horizon retired: it clipped Greedy/LP above 1.05 and
    # mixed NYC16 baselines into a Bandra axis. Superseded by
    # results/figures/retention_by_shift.png (plot_train_charts.fig_shift_retention).
    fig_shift_service_rate_by_horizon()
    print("Done.")


if __name__ == "__main__":
    main()
