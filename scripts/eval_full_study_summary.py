"""Summaries for results/full_study_v2 (imported by scripts/eval_full_study.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MAIN = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6", "S7"]
ABL = ["S0", "S1", "S3", "S4", "S6"]
BC = ABL + ["S7"]
S1_SWEEP = ["S1@p0.0", "S1@p0.25", "S1@p0.5", "S1@p0.75", "S1@p1.0"]
S4_SWEEP = ["S4@z1@10", "S4@z1@20", "S4@z1@40", "S4@z1@60",
            "S4@z1@80", "S4@z5@60"]
ARM_ORDER = ["Greedy", "LP",
             "IPPO-it200", "MAPPO-it200",
             "IPPO-best", "MAPPO-best",
             "IPPO-it200-sampled", "MAPPO-it200-sampled",
             "BC-LP-rw5", "BC-LP-rw10"]
FAMILIES = {
    "Spatial (S1,S4,S7)": ["S1", "S4", "S7"],
    "Volume/Burst (S2a,S2b,S2d,S2e,S5)": ["S2a", "S2b", "S2d", "S2e", "S5"],
    "Temporal splits (S3,S6)": ["S3", "S6"],
}
COMPARISONS = [("MAPPO-it200", "IPPO-it200"), ("MAPPO-it200", "LP"),
               ("IPPO-it200", "LP")]
METRICS = [("service_rate", "SR", 100.0), ("empty_km", "Empty km", 1.0),
           ("avg_wait_min", "Wait min", 1.0), ("reloc_count", "Relocs", 1.0),
           ("exact_team_reward", "Exact reward", 1.0),
           ("cancel_rate", "Cancel %", 100.0),
           ("mean_km_per_reloc", "km/reloc", 1.0)]


def arm_order(arms: list[str]) -> list[str]:
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in known)


def load(out: Path) -> pd.DataFrame:
    df = pd.read_csv(out / "episodes.csv")
    df["scenario"] = df["scenario"].astype(str)
    for c in df.columns:
        if c.startswith("orig_z"):
            df[c] = df[c].astype(float)
    return df


def _agg_one(g: pd.DataFrame, col: str):
    if g["model_seed"].max() >= 0:
        per = g.groupby("model_seed")[col].mean()
        return float(per.mean()), float(per.std(ddof=1)), int(len(per))
    return float(g[col].mean()), float(g[col].std(ddof=1)), int(len(g))


def aggregates(df: pd.DataFrame) -> dict:
    out = {}
    for (arm, sc), g in df.groupby(["arm", "scenario"]):
        out.setdefault(arm, {})[sc] = {
            m: _agg_one(g, m) for m, _, _ in METRICS
        } | {
            "reloc_leg_km": _agg_one(g, "reloc_leg_km"),
            "pickup_km": _agg_one(g, "pickup_km"),
            "decision_time_sec": _agg_one(g, "decision_time_sec"),
            "env_step_time_sec": _agg_one(g, "env_step_time_sec"),
            "pct_dispatch": _agg_one(g, "pct_dispatch"),
            "pct_relocate": _agg_one(g, "pct_relocate"),
            "pct_idle": _agg_one(g, "pct_idle"),
            "eq1_abs_gap": float(np.max(np.abs(g["exact_team_reward"] - g["eq1_reward"]))),
            "km_attrib_gap": float(np.max(np.abs(
                g["reloc_leg_km"] + g["pickup_km"] - g["empty_km"]))),
        }
    return out


def per_window(df: pd.DataFrame) -> pd.DataFrame:
    """One value per (arm, scenario, seed), averaged over model seeds for RL."""
    return df.groupby(["arm", "scenario", "seed"])[
        [m for m, _, _ in METRICS]
    ].mean().reset_index()


def paired_tests(df: pd.DataFrame, pw: pd.DataFrame, agg: dict) -> list[dict]:
    rows = []
    for sc in MAIN:
        for a, b in COMPARISONS:
            if sc not in set(pw[pw.arm == a].scenario) or sc not in set(pw[pw.arm == b].scenario):
                continue
            wa = pw[(pw.arm == a) & (pw.scenario == sc)].sort_values("seed")
            wb = pw[(pw.arm == b) & (pw.scenario == sc)].sort_values("seed")
            if len(wa) != len(wb) or len(wa) < 3:
                continue
            for m, label, scale in METRICS[:5]:
                xa = wa[m].to_numpy(float)
                xb = wb[m].to_numpy(float)
                d = (xa - xb) * scale
                t_p = float(stats.ttest_rel(xa, xb).pvalue)
                if np.allclose(d, 0.0):
                    w_p = 1.0
                else:
                    w_p = float(stats.wilcoxon(d).pvalue)
                sa = agg[a][sc][m]
                sb = agg[b][sc][m]
                # independent samples: RL seed-level means (n=5), or the
                # baseline's 12 window values (n=12)
                da = df[(df.arm == a) & (df.scenario == sc)]
                db = df[(df.arm == b) & (df.scenario == sc)]
                if da.model_seed.max() >= 0:
                    ga = da.groupby("model_seed")[m].mean()
                else:
                    ga = da.set_index("seed")[m]
                if db.model_seed.max() >= 0:
                    gb = db.groupby("model_seed")[m].mean()
                else:
                    gb = db.set_index("seed")[m]
                welch = float(stats.ttest_ind(ga, gb, equal_var=False).pvalue)
                rows.append({
                    "scenario": sc, "comparison": f"{a} vs {b}", "metric": label,
                    "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
                    "paired_t_p": t_p, "wilcoxon_p": w_p, "welch_p": welch,
                    "n_pairs": int(len(d)), "n_a": int(len(ga)), "n_b": int(len(gb)),
                    "seed_means_a": [float(x) for x in ga],
                    "seed_means_b": [float(x) for x in gb],
                })
    return rows


def retention(agg: dict) -> dict:
    out = {}
    for arm, scs in agg.items():
        base = scs.get("S0", {}).get("service_rate", (np.nan,))[0]
        if not np.isfinite(base) or base <= 0:
            continue
        per = {sc: scs[sc]["service_rate"][0] / base for sc in scs if sc in MAIN}
        fams = {f: float(np.mean([per[s] for s in ss if s in per]))
                for f, ss in FAMILIES.items()}
        out[arm] = {"per_scenario": per, "families": fams,
                    "s0_service_rate": base}
    return out


def _jsd(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) / np.log(base)
                 + 0.5 * np.sum(q * np.log(q / m)) / np.log(base))


def divergence(df: pd.DataFrame) -> tuple[dict, bool]:
    zcols = [f"orig_z{i}" for i in range(16)]
    # demand depends only on scenario+seed; verify every arm/model row agrees
    key = ["scenario", "seed"]
    ref = df[df.arm == "Greedy"] if "Greedy" in set(df.arm) else df
    counts = ref.groupby(key)[zcols].sum()
    refn = counts.div(counts.sum(axis=1), axis=0)
    arm_invariant = True
    for _arm, g in df.groupby("arm"):
        got = g.groupby(["scenario", "seed", "model_seed"])[zcols].sum()
        gotn = got.div(got.sum(axis=1), axis=0)
        for (sc, seed, _ms), row in gotn.iterrows():
            if (sc, seed) not in refn.index:
                continue
            if not np.allclose(row.to_numpy(float), refn.loc[(sc, seed)].to_numpy(float)):
                arm_invariant = False
    p0 = counts.loc["S0"]
    p0 = p0.div(p0.sum(axis=1), axis=0)
    out = {}
    for sc in counts.index.get_level_values("scenario").unique():
        ps = counts.loc[sc]
        ps = ps.div(ps.sum(axis=1), axis=0)
        tv, js, tv_mean, js_mean = [], [], [], []
        for seed in ps.index:
            a = ps.loc[seed].to_numpy(float)
            b = p0.loc[seed].to_numpy(float)
            tv.append(0.5 * np.abs(a - b).sum())
            js.append(_jsd(a, b))
        am = ps.to_numpy(float).mean(axis=0)
        bm = p0.to_numpy(float).mean(axis=0)
        tv_mean.append(0.5 * np.abs(am - bm).sum())
        js_mean.append(_jsd(am, bm))
        out[sc] = {
            "tv_mean": float(np.mean(tv)), "tv_std": float(np.std(tv, ddof=1)),
            "jsd_bits_mean": float(np.mean(js)), "jsd_bits_std": float(np.std(js, ddof=1)),
            "tv_of_mean_dist": float(tv_mean[0]),
            "jsd_bits_of_mean_dist": float(js_mean[0]),
            "n": len(tv),
        }
    return out, arm_invariant


def fmt(v, s, scale=1.0, dec=2):
    if not np.isfinite(v):
        return "n/a"
    return f"{v * scale:.{dec}f} ± {s * scale:.{dec}f}"


def table(agg, arms, scenarios, metric, scale, dec=2, title=""):
    lines = [f"### {title}", "",
             "| Arm | " + " | ".join(scenarios) + " |",
             "|:---|" + "|".join([":--:"] * len(scenarios)) + "|"]
    for arm in arms:
        cells = []
        for sc in scenarios:
            a = agg.get(arm, {}).get(sc)
            cells.append(fmt(*a[metric][:2], scale, dec) if a else "n/a")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def alias_endpoints(agg: dict, div: dict) -> None:
    """Sweep endpoints equal the main S1/S4 scenarios; reuse those rows."""
    for arm, scs in agg.items():
        if "S1" in scs and "S1@p1.0" not in scs:
            scs["S1@p1.0"] = scs["S1"]
        if "S4" in scs and "S4@z1@60" not in scs:
            scs["S4@z1@60"] = scs["S4"]
    if "S1" in div and "S1@p1.0" not in div:
        div["S1@p1.0"] = div["S1"]
    if "S4" in div and "S4@z1@60" not in div:
        div["S4@z1@60"] = div["S4"]


def summarize(out: Path) -> None:
    df = load(out)
    agg = aggregates(df)
    pw = per_window(df)
    tests = paired_tests(df, pw, agg)
    ret = retention(agg)
    div, inv = divergence(df)
    alias_endpoints(agg, div)
    arms_present = arm_order(list(df.arm.unique()))
    main_arms = [a for a in arms_present
                 if a in agg and all(sc in agg[a] for sc in MAIN)]
    abl_arms = [a for a in arms_present if a not in main_arms and "S0" in agg[a]]
    sweep_arms = ["Greedy", "LP", "IPPO-it200", "MAPPO-it200"]
    sweep_arms = [a for a in sweep_arms if a in agg]

    meta = {
        "episodes": int(df.seed.nunique()), "base_seed": int(df.seed.min()),
        "seeds": sorted(int(s) for s in df.seed.unique()),
        "models_per_arm": {a: sorted(int(s) for s in df[df.arm == a].model_seed.unique())
                           for a in arms_present},
        "windows_by_scenario": {
            sc: sorted(int(w) for w in df[df.scenario == sc].window.unique())
            for sc in df.scenario.unique()},
        "unique_windows_by_scenario": {
            sc: int(df[df.scenario == sc].window.nunique())
            for sc in df.scenario.unique()},
        "demand_arm_invariant": inv,
        "eq1_max_abs_gap": max(v["eq1_abs_gap"] for a in agg.values() for v in a.values()),
        "km_attrib_max_abs_gap": max(v["km_attrib_gap"] for a in agg.values() for v in a.values()),
    }

    md = ["# Full paired 12-window study (v2)", ""]
    md += [
        f"- Episodes: {meta['episodes']} paired windows, seeds "
        f"{meta['seeds'][0]}..{meta['seeds'][-1]}.",
        "- Baselines (Greedy, LP) are deterministic: the 12 numbers are windows.",
        "- RL arms: 5 model seeds x 12 windows; mean ± sample std (`ddof=1`) across the 5 seed means.",
        "- Every policy is stepped sequentially zone-by-zone through the AEC loop "
        "(same loop as `scripts/diag_objective_gap.py`).",
        "- LP = tuned defaults: lookahead_minutes=5.0, deficit_penalty_scale=1.0.",
        "- `exact_team_reward` is the call-through sum of `env._compute_zone_rewards` "
        "under the Eq. (1) reward config (`relocator_share=0`, `targeting_bonus=0`).",
        f"- Eq. (1) check: max |exact − gamma*served + beta*km + alpha*wait + cancel| "
        f"= {meta['eq1_max_abs_gap']:.2e} across all rows.",
        f"- Empty-km attribution check: max |reloc_leg_km + pickup_km − empty_km| "
        f"= {meta['km_attrib_max_abs_gap']:.2e} km.",
        f"- Demand distribution is arm-invariant (checked): {inv}.",
        "- Checkpoints: `results/repro_nyc16_v1.0.0/checkpoints/{ippo,mappo}/"
        "seed{0..4}/checkpoint_it0200.pt` (main, `deterministic=True`), "
        "`checkpoint_best.pt` (validation-best ablation), the same it200 "
        "checkpoints with `deterministic=False` (sampled ablation), and "
        "`runs/nyc16-bc-lp-rw{5,10}/checkpoint.pt` (BC-LP control).",
        "- Sampled arms seed `torch`/`numpy` with the episode seed (1000..1011) "
        "so the stochastic evaluation is reproducible.",
        "",
        "Windows per scenario (the generator's window index, seeds "
        f"{meta['seeds'][0]}..{meta['seeds'][-1]}). S3 walks the 12 val windows "
        "and S6 the 12 test windows exactly once each; train-mode scenarios draw "
        "a window per seed with replacement, so their `unique` column can be "
        "below 12 while every arm sees the same 12 paired draws.",
        "",
        "| Scenario | window indices | unique |", "|:---|:---|:--:|",
    ]
    main_present = [sc for sc in MAIN if sc in df.scenario.unique()]
    for sc in main_present:
        md.append(f"| {sc} | {meta['windows_by_scenario'][sc]} | "
                  f"{meta['unique_windows_by_scenario'][sc]} |")
    md.append("")

    md.append(table(agg, main_arms, main_present, "service_rate", 100.0, 1,
                    "Main grid — service rate % (sd)") )
    md.append(table(agg, main_arms, main_present, "empty_km", 1.0, 1,
                    "Main grid — empty km"))
    md.append(table(agg, main_arms, main_present, "avg_wait_min", 1.0, 2,
                    "Main grid — average wait (min)"))
    md.append(table(agg, main_arms, main_present, "reloc_count", 1.0, 1,
                    "Main grid — executed relocations"))
    md.append(table(agg, main_arms, main_present, "exact_team_reward", 1.0, 1,
                    "Main grid — exact team reward (Eq. 1)"))
    md.append(table(agg, main_arms, main_present, "cancel_rate", 100.0, 1,
                    "Main grid — cancel rate %"))
    md.append(table(agg, main_arms, main_present, "mean_km_per_reloc", 1.0, 2,
                    "Main grid — empty km / max(reloc_count, 1) "
                    "(Greedy executes 0 relocations, so its entry equals its empty km)"))
    md.append(table(agg, main_arms, main_present, "reloc_leg_km", 1.0, 1,
                    "Main grid — relocation-leg empty km"))
    md.append(table(agg, main_arms, main_present, "pickup_km", 1.0, 1,
                    "Main grid — pickup-leg empty km"))

    md.append("## Mechanistic ablations\n")
    abl_present = [sc for sc in ABL if sc in df.scenario.unique()]
    md.append("Ablation scenarios: " + ", ".join(abl_present) + ".\n")
    md.append(table(agg, abl_arms, abl_present, "service_rate", 100.0, 1,
                    "Ablations — service rate %"))
    md.append(table(agg, abl_arms, abl_present, "empty_km", 1.0, 1,
                    "Ablations — empty km"))
    md.append(table(agg, abl_arms, abl_present, "exact_team_reward", 1.0, 1,
                    "Ablations — exact team reward"))
    md.append(table(agg, abl_arms, abl_present, "reloc_count", 1.0, 1,
                    "Ablations — executed relocations"))
    bc_arms = [a for a in abl_arms if a.startswith("BC-LP")]
    bc_present = [sc for sc in BC if sc in df.scenario.unique()]
    if bc_arms and bc_present:
        md.append(table(agg, bc_arms, bc_present, "service_rate", 100.0, 1,
                        "BC-LP representation control — service rate %"))
        md.append(table(agg, bc_arms, bc_present, "empty_km", 1.0, 1,
                        "BC-LP representation control — empty km"))
        md.append(table(agg, bc_arms, bc_present, "exact_team_reward", 1.0, 1,
                        "BC-LP representation control — exact team reward"))
        md.append(table(agg, bc_arms, bc_present, "reloc_count", 1.0, 1,
                        "BC-LP representation control — executed relocations"))

    md.append("## Parametric spatial shifts\n")
    md.append("The sweep endpoints equal the main-grid scenarios: p=1.0 reuses S1 "
              "and 60% on zone 1 reuses S4 (same parameters, same seeds).\n")
    md.append(table(agg, sweep_arms, S1_SWEEP, "service_rate", 100.0, 1,
                    "S1 permutation strength p — service rate %"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "service_rate", 100.0, 1,
                    "S4 concentration on zone 1 (zone 79) and cold zone 5 (zone 162) — service rate %"))
    md.append(table(agg, sweep_arms, S1_SWEEP, "empty_km", 1.0, 1,
                    "S1 sweep — empty km"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "empty_km", 1.0, 1,
                    "S4 sweep — empty km"))
    md.append(table(agg, sweep_arms, S1_SWEEP, "exact_team_reward", 1.0, 1,
                    "S1 sweep — exact team reward"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "exact_team_reward", 1.0, 1,
                    "S4 sweep — exact team reward"))

    md.append("## Paired per-window statistical tests\n")
    md.append(
        "`paired t` / `Wilcoxon` are on the 12 window-paired means (RL arm = mean over "
        "its 5 seeds for that window). `Welch` uses the independent seed-level means "
        "(n shown); for RL-vs-LP, the LP side is its 12 window values. "
        "Differences are arm A − arm B in the metric's own units.\n")
    md.append("| Scenario | Comparison | Metric | Diff (mean ± sd) | paired t p | Wilcoxon p | Welch p | n pairs | n A / n B |")
    md.append("|:---|:---|:---|---:|---:|---:|---:|---:|---:|")
    for r in tests:
        md.append(
            f"| {r['scenario']} | {r['comparison']} | {r['metric']} | "
            f"{r['diff_mean']:+.3f} ± {r['diff_std']:.3f} | "
            f"{r['paired_t_p']:.2e} | {r['wilcoxon_p']:.2e} | "
            f"{r['welch_p']:.2e} | {r['n_pairs']} | {r['n_a']} / {r['n_b']} |")
    md.append("")

    md.append("## Shift-family retention (service rate relative to S0)\n")
    md.append(
        "Retention is per-scenario SR / S0 SR. Family means are shown separately; "
        "no pooled average across families is reported. S2a (magnitude x0.5) is "
        ">100% for every arm because half the demand is easier to serve, so it is "
        "never mixed into a single retention number. S1 and S4 retention reuse "
        "their main-grid rows (identical parameters to the sweep endpoints).\n")
    for fam, sss in FAMILIES.items():
        md.append(f"### {fam}\n")
        md.append("| Arm | " + " | ".join(sss) + " | Family mean |")
        md.append("|:---|" + "|".join([":--:"] * len(sss)) + "|:--:|")
        for arm in main_arms:
            if arm not in ret:
                continue
            per = ret[arm]["per_scenario"]
            cells = [f"{100 * per[s]:.1f}%" if s in per else "n/a" for s in sss]
            md.append(f"| {arm} | " + " | ".join(cells)
                      + f" | {100 * ret[arm]['families'][fam]:.1f}% |")
        md.append("")
    md.append("Per-scenario retention (all main-grid arms):\n")
    md.append("| Arm | " + " | ".join(f"{s} vs S0" for s in MAIN[1:]) + " |")
    md.append("|:---|" + "|".join([":--:"] * (len(MAIN) - 1)) + "|")
    for arm in main_arms:
        if arm not in ret:
            continue
        cells = []
        for s in MAIN[1:]:
            v = ret[arm]["per_scenario"].get(s)
            cells.append(f"{100 * v:.1f}%" if v is not None and np.isfinite(v) else "n/a")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Spatial divergence of demand vs S0\n")
    md.append(
        "16-bin empirical origin-zone distributions (nearest-centroid assignment, "
        "same rule as the reward code), per episode. TV = 1/2 sum |p-q|, JSD in bits. "
        "Mean ± sd across the 12 paired windows. Demand is identical for every arm "
        f"(verified: {inv}), so these numbers apply to all arms.\n")
    md.append("The S1 sweep at p=0.0 must reproduce S0 (TV = 0); it does.\n")
    md.append("| Scenario | TV mean ± sd | JSD bits mean ± sd | TV of window-mean dist | JSD of window-mean dist |")
    md.append("|:---|---:|---:|---:|---:|")
    order = MAIN + [s for s in S1_SWEEP + S4_SWEEP if s in div]
    for sc in order:
        if sc not in div:
            continue
        d = div[sc]
        md.append(f"| {sc} | {d['tv_mean']:.3f} ± {d['tv_std']:.3f} | "
                  f"{d['jsd_bits_mean']:.3f} ± {d['jsd_bits_std']:.3f} | "
                  f"{d['tv_of_mean_dist']:.3f} | {d['jsd_bits_of_mean_dist']:.3f} |")
    md.append("")

    md.append("## Decision vs environment-step time\n")
    md.append("| Scenario | Arm | decision s/ep | env step s/ep |")
    md.append("|:---|:---|---:|---:|")
    for sc in main_present:
        for arm in main_arms:
            a = agg.get(arm, {}).get(sc)
            if not a:
                continue
            md.append(f"| {sc} | {arm} | {a['decision_time_sec'][0]:.2f} | "
                      f"{a['env_step_time_sec'][0]:.2f} |")
    md.append("")

    (out / "summary.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps({
        "meta": meta, "aggregates": agg, "tests": tests,
        "retention": ret, "divergence": div,
        "families": FAMILIES,
    }, indent=2))
    print(f"[study] wrote {out / 'summary.json'}")
    print(f"[study] wrote {out / 'summary.md'}")
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/full_study_v2")
    a = ap.parse_args()
    summarize(Path(a.out_dir))
