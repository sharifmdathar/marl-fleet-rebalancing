"""Summary for results/plan_b_v1 (D100 study).

Compares the fresh Plan B arms against each other, against LP re-run on the
same paired windows, and against the existing MLP-MAPPO arm read from
``results/full_study_v2/episodes.csv`` (same 12 windows, seeds 1000..1011).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eval_full_study_summary import (
    ABL, FAMILIES, MAIN, METRICS, S1_SWEEP, S4_SWEEP,
    aggregates, alias_endpoints, divergence, fmt, load, per_window, retention,
    table,
)

ROOT = Path(__file__).resolve().parents[1]
OLD_CSV = ROOT / "results" / "full_study_v2" / "episodes.csv"

ARM_ORDER = ["LP", "MLP-MAPPO", "Attention-MAPPO", "DR-MLP-MAPPO",
             "DR-Attention-MAPPO"]
VARIANT_ARMS = [
    "Attention-MAPPO", "Attention-MAPPO-best", "Attention-MAPPO-it200-sampled",
    "DR-MLP-MAPPO", "DR-MLP-MAPPO-best", "DR-MLP-MAPPO-it200-sampled",
    "DR-Attention-MAPPO", "DR-Attention-MAPPO-best",
    "DR-Attention-MAPPO-it200-sampled",
]
COMPARISONS = [
    ("Attention-MAPPO", "MLP-MAPPO"),
    ("DR-MLP-MAPPO", "MLP-MAPPO"),
    ("DR-Attention-MAPPO", "MLP-MAPPO"),
    ("DR-Attention-MAPPO", "Attention-MAPPO"),
    ("DR-Attention-MAPPO", "DR-MLP-MAPPO"),
    ("Attention-MAPPO", "LP"),
    ("DR-Attention-MAPPO", "LP"),
    ("MLP-MAPPO", "LP"),
]
TEST_METRICS = [("service_rate", "SR", 100.0), ("empty_km", "Empty km", 1.0),
                ("avg_wait_min", "Wait min", 1.0), ("reloc_count", "Relocs", 1.0),
                ("exact_team_reward", "Exact reward", 1.0)]


def combined(out: Path) -> pd.DataFrame:
    """Plan B rows + the existing MLP-MAPPO control rows."""
    df = load(out)
    old = pd.read_csv(OLD_CSV)
    mlp = old[old["arm"] == "MAPPO-it200"].copy()
    if mlp.empty:
        raise RuntimeError(f"no MAPPO-it200 rows in {OLD_CSV}")
    mlp["arm"] = "MLP-MAPPO"
    cols = [c for c in df.columns if c in mlp.columns]
    df = pd.concat([df[cols], mlp[cols]], ignore_index=True)
    for c in df.columns:
        if c.startswith("orig_z"):
            df[c] = df[c].astype(float)
    return df


def ordered(arms) -> list[str]:
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in known)


def paired_tests(df: pd.DataFrame, pw: pd.DataFrame) -> list[dict]:
    rows = []
    for sc in MAIN:
        for a, b in COMPARISONS:
            wa = pw[(pw.arm == a) & (pw.scenario == sc)].sort_values("seed")
            wb = pw[(pw.arm == b) & (pw.scenario == sc)].sort_values("seed")
            if len(wa) != len(wb) or len(wa) < 3:
                continue
            for m, label, scale in TEST_METRICS:
                xa = wa[m].to_numpy(float)
                xb = wb[m].to_numpy(float)
                d = (xa - xb) * scale
                if float(np.ptp(d)) < 1e-12:
                    t_p = w_p = 1.0
                else:
                    t_p = float(stats.ttest_rel(xa, xb).pvalue)
                    w_p = float(stats.wilcoxon(d).pvalue)
                da = df[(df.arm == a) & (df.scenario == sc)]
                db = df[(df.arm == b) & (df.scenario == sc)]
                ga = (da.groupby("model_seed")[m].mean()
                      if da.model_seed.max() >= 0 else da.set_index("seed")[m])
                gb = (db.groupby("model_seed")[m].mean()
                      if db.model_seed.max() >= 0 else db.set_index("seed")[m])
                if float(ga.std(ddof=1)) == 0.0 and float(gb.std(ddof=1)) == 0.0:
                    welch = 1.0
                else:
                    welch = float(stats.ttest_ind(ga, gb, equal_var=False).pvalue)
                rows.append({
                    "scenario": sc, "comparison": f"{a} vs {b}",
                    "metric": label, "diff_mean": float(d.mean()),
                    "diff_std": float(d.std(ddof=1)),
                    "paired_t_p": t_p, "wilcoxon_p": w_p, "welch_p": welch,
                    "n_pairs": int(len(d)), "n_a": int(len(ga)), "n_b": int(len(gb)),
                })
    return rows


def family_tests(pw: pd.DataFrame) -> list[dict]:
    rows = []
    for fam, scenarios in FAMILIES.items():
        present = [s for s in scenarios if s in set(pw.scenario)]
        if not present:
            continue
        for a, b in COMPARISONS:
            ga = (pw[(pw.arm == a) & (pw.scenario.isin(present))]
                  .groupby("seed")["service_rate"].mean())
            gb = (pw[(pw.arm == b) & (pw.scenario.isin(present))]
                  .groupby("seed")["service_rate"].mean())
            common = ga.index.intersection(gb.index)
            if len(common) < 3:
                continue
            d = (ga.loc[common] - gb.loc[common]).to_numpy(float) * 100.0
            p = (1.0 if float(np.ptp(d)) < 1e-12 else
                 float(stats.ttest_rel(ga.loc[common], gb.loc[common]).pvalue))
            rows.append({
                "family": fam, "comparison": f"{a} vs {b}",
                "diff_pp_mean": float(d.mean()), "diff_pp_std": float(d.std(ddof=1)),
                "paired_t_p": p,
                "n_pairs": int(len(d)),
            })
    return rows


def main_tables(md: list[str], agg: dict, arms: list[str], scenarios: list[str]) -> None:
    for metric, scale, dec, title in (
        ("service_rate", 100.0, 1, "Main grid — service rate % (sd)"),
        ("empty_km", 1.0, 1, "Main grid — empty km"),
        ("avg_wait_min", 1.0, 2, "Main grid — average wait (min)"),
        ("exact_team_reward", 1.0, 1, "Main grid — exact team reward (Eq. 1)"),
        ("reloc_count", 1.0, 1, "Main grid — executed relocations"),
        ("cancel_rate", 100.0, 1, "Main grid — cancel rate %"),
    ):
        md.append(table(agg, arms, scenarios, metric, scale, dec, title))


def summarize(out: Path) -> None:
    df = combined(out)
    agg = aggregates(df)
    pw = per_window(df)
    tests = paired_tests(df, pw)
    fam_tests = family_tests(pw)
    ret = retention(agg)
    div, invariant = divergence(df)
    alias_endpoints(agg, div)

    arms = ordered(list(df.arm.unique()))
    main_arms = [a for a in arms if all(sc in agg.get(a, {}) for sc in MAIN)]
    variant_arms = [a for a in VARIANT_ARMS if a in agg]
    sweep_arms = [a for a in ARM_ORDER if a in agg]

    eq1 = max(v["eq1_abs_gap"] for a in agg.values() for v in a.values())
    kmgap = max(v["km_attrib_gap"] for a in agg.values() for v in a.values())
    windows = {sc: sorted(int(w) for w in df[df.scenario == sc].window.unique())
               for sc in df.scenario.unique()}

    meta = {
        "episodes": int(df.seed.nunique()), "base_seed": int(df.seed.min()),
        "seeds": sorted(int(s) for s in df.seed.unique()),
        "models_per_arm": {a: sorted(int(s) for s in df[df.arm == a].model_seed.unique())
                           for a in arms},
        "windows_by_scenario": windows,
        "unique_windows_by_scenario": {sc: len(v) for sc, v in windows.items()},
        "demand_arm_invariant": bool(invariant),
        "eq1_max_abs_gap": float(eq1),
        "km_attrib_max_abs_gap": float(kmgap),
        "mlp_source": str(OLD_CSV.relative_to(ROOT)),
        "plan_b_episodes_source": str(out / "episodes.csv"),
    }

    md = ["# Plan B (D100): spatial equivariance + domain randomization", ""]
    md += [
        "- Protocol identical to `results/full_study_v2`: 12 paired windows, "
        "seeds 1000..1011, sequential zone-by-zone AEC stepping, deterministic "
        "argmax unless marked sampled, Eq. (1) reward config "
        "(`relocator_share=0`, `targeting_bonus=0`).",
        "- RL arms: 5 model seeds × 12 windows; mean ± sample std (`ddof=1`) "
        "across the 5 seed means.",
        "- `MLP-MAPPO` is the existing control read from "
        "`results/full_study_v2/episodes.csv` (arm `MAPPO-it200`), not re-run.",
        "- `Attention-MAPPO` = D100 equivariant attention trunk, no domain "
        "randomization; `DR-*` arms train with `--domain-randomize`.",
        "- Evaluation is always clean: `domain_randomize=False`, forced in "
        "`load_trained_policy` and in `_make_validation_env`.",
        f"- Eq. (1) check: max |exact − (gamma*served − beta*km − alpha*wait "
        f"− cancel)| = {eq1:.2e}.",
        f"- Empty-km attribution check: max |reloc_leg_km + pickup_km − empty_km| "
        f"= {kmgap:.2e} km.",
        f"- Demand distribution arm-invariant (checked): {invariant}.",
        "",
        "| Scenario | window indices | unique |", "|:---|:---|:--:|",
    ]
    for sc in MAIN + [s for s in S1_SWEEP + S4_SWEEP if s in windows]:
        if sc in windows:
            md.append(f"| {sc} | {windows[sc]} | {len(windows[sc])} |")
    md.append("")

    main_tables(md, agg, main_arms, MAIN)

    md.append("## Paired deltas vs MLP-MAPPO control (service rate, pp)\n")
    md.append("| Arm | " + " | ".join(MAIN) + " |")
    md.append("|:---|" + "|".join([":--:"] * len(MAIN)) + "|")
    for arm in ["Attention-MAPPO", "DR-MLP-MAPPO", "DR-Attention-MAPPO"]:
        cells = []
        for sc in MAIN:
            a = agg.get(arm, {}).get(sc)
            b = agg.get("MLP-MAPPO", {}).get(sc)
            cells.append(f"{100 * (a['service_rate'][0] - b['service_rate'][0]):+.1f}"
                         if a and b else "n/a")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Mechanistic ablations\n")
    abl_present = [s for s in ABL if s in df.scenario.unique()]
    md.append(table(agg, variant_arms, abl_present, "service_rate", 100.0, 1,
                    "it200 / validation-best / sampled — service rate %"))
    md.append(table(agg, variant_arms, abl_present, "empty_km", 1.0, 1,
                    "it200 / validation-best / sampled — empty km"))
    md.append(table(agg, variant_arms, abl_present, "exact_team_reward", 1.0, 1,
                    "it200 / validation-best / sampled — exact team reward"))
    md.append("")

    md.append("## Parametric spatial shifts\n")
    md.append(table(agg, sweep_arms, S1_SWEEP, "service_rate", 100.0, 1,
                    "S1 permutation strength p — service rate %"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "service_rate", 100.0, 1,
                    "S4 concentration on zone 1 and cold zone 5 — service rate %"))
    md.append(table(agg, sweep_arms, S1_SWEEP, "empty_km", 1.0, 1,
                    "S1 sweep — empty km"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "empty_km", 1.0, 1,
                    "S4 sweep — empty km"))
    md.append(table(agg, sweep_arms, S1_SWEEP, "exact_team_reward", 1.0, 1,
                    "S1 sweep — exact team reward"))
    md.append(table(agg, sweep_arms, S4_SWEEP, "exact_team_reward", 1.0, 1,
                    "S4 sweep — exact team reward"))
    md.append("")

    md.append("## Shift-family retention (SR relative to S0)\n")
    md.append("Retention is per-scenario SR / S0 SR; family mean is shown "
              "separately per family, never pooled across families.\n")
    for fam, scenarios in FAMILIES.items():
        md.append(f"### {fam}\n")
        md.append("| Arm | " + " | ".join(scenarios) + " | Family mean |")
        md.append("|:---|" + "|".join([":--:"] * len(scenarios)) + "|:--:|")
        for arm in main_arms:
            if arm not in ret:
                continue
            per = ret[arm]["per_scenario"]
            cells = [f"{100 * per[s]:.1f}%" if s in per else "n/a" for s in scenarios]
            md.append(f"| {arm} | " + " | ".join(cells)
                      + f" | {100 * ret[arm]['families'][fam]:.1f}% |")
        md.append("")

    md.append("## Paired statistical tests (12 window-paired means)\n")
    md.append("`paired t` / `Wilcoxon` pair the 12 windows after averaging an RL "
              "arm over its 5 model seeds. `Welch` uses the independent "
              "seed-level means. Difference = arm A − arm B.\n")
    focus = {"S0", "S1", "S4", "S6", "S7"}
    for metric in ("SR", "Empty km", "Exact reward"):
        md.append(f"### {metric}\n")
        md.append("| Scenario | Comparison | Diff (mean ± sd) | paired t p | Wilcoxon p |")
        md.append("|:---|:---|---:|---:|---:|")
        for r in tests:
            if r["metric"] != metric or r["scenario"] not in focus:
                continue
            md.append(
                f"| {r['scenario']} | {r['comparison']} | "
                f"{r['diff_mean']:+.3f} ± {r['diff_std']:.3f} | "
                f"{r['paired_t_p']:.2e} | {r['wilcoxon_p']:.2e} |")
        md.append("")

    md.append("## Family-level paired SR deltas\n")
    md.append("| Family | Comparison | Δ SR (pp) | paired t p | n |")
    md.append("|:---|:---|---:|---:|---:|")
    for r in fam_tests:
        md.append(f"| {r['family']} | {r['comparison']} | "
                  f"{r['diff_pp_mean']:+.2f} ± {r['diff_pp_std']:.2f} | "
                  f"{r['paired_t_p']:.2e} | {r['n_pairs']} |")
    md.append("")

    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "meta": meta, "aggregates": agg, "tests": tests,
        "family_tests": fam_tests, "retention": ret,
        "divergence": div, "families": FAMILIES,
    }, indent=2))
    print(f"[plan-b] wrote {out / 'summary.json'}")
    print(f"[plan-b] wrote {out / 'summary.md'}")
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/plan_b_v1")
    args = ap.parse_args()
    summarize(Path(args.out_dir))
