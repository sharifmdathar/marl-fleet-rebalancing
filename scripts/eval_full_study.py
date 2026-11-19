"""Full paired 12-window evaluation study (main grid, ablations, spatial sweeps).

Stepping, imports and env construction follow `scripts/diag_objective_gap.py`
(sequential zone-by-zone AEC loop, per-arm action space from the manifest,
Eq. (1) reward config) and `experiments/shift_eval.py` (`SHIFTS`,
`make_shifted_env`, action-type counting). Nothing in `src/` or any checkpoint
is modified.

    uv run python scripts/eval_full_study.py --workers 4
    uv run python scripts/eval_full_study.py --summarize-only
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(os.environ.get("EVAL_FULL_STUDY_OUT",
                         str(ROOT / "results" / "full_study_v2")))
PARTIAL = OUT / "_partial"
CKPT = ROOT / "results" / "repro_nyc16_v1.0.0" / "checkpoints"
BC_CKPTS = {"BC-LP-rw5": ROOT / "runs/nyc16-bc-lp-rw5/checkpoint.pt",
            "BC-LP-rw10": ROOT / "runs/nyc16-bc-lp-rw10/checkpoint.pt"}

EPISODES = int(os.environ.get("EVAL_FULL_STUDY_EPISODES", "12"))
BASE_SEED = 1000
RL_SEEDS = range(5)
NZ = 16
MAIN_SCENARIOS = ["S0", "S1", "S2a", "S2b", "S2d", "S2e", "S3", "S4", "S5", "S6", "S7"]
ABL_SCENARIOS = ["S0", "S1", "S3", "S4", "S6"]
BC_SCENARIOS = ABL_SCENARIOS + ["S7"]
S1_P = [0.0, 0.25, 0.5, 0.75, 1.0]
S4_PCT = [10.0, 20.0, 40.0, 60.0, 80.0]
COLD_ZONE = 5
S4_ZONE = 1
WINDOW = 130


# --------------------------------------------------------------------- tasks
def scenario_specs() -> dict[str, tuple]:
    """label -> (shift_label, window_limit, target_zone, pct, shift_param)."""
    from marl_fleet_rebalancing.experiments.shift_eval import SHIFTS
    specs = {s[0]: (s[0], s[1], s[2], s[3], s[4]) for s in SHIFTS}
    for p in S1_P:
        specs[f"S1@p{p}"] = ("S1", WINDOW, None, p, "permute")
    for pct in S4_PCT:
        specs[f"S4@z{S4_ZONE}@{pct:g}"] = ("S4", WINDOW, S4_ZONE, pct, "concentrate")
    specs[f"S4@z{COLD_ZONE}@60"] = ("S4", WINDOW, COLD_ZONE, 60.0, "concentrate")
    return specs


def arm_spec(name, kind, ckpt=None, deterministic=True, model_seed=-1):
    return {"arm": name, "kind": kind, "ckpt": str(ckpt) if ckpt else None,
            "deterministic": deterministic, "model_seed": model_seed}


def rl_arms(label_prefix, algo, ckpt_name, deterministic):
    return [arm_spec(f"{label_prefix}", algo, CKPT / algo / f"seed{s}" / ckpt_name,
                     deterministic, s) for s in RL_SEEDS]


def build_tasks() -> list[dict]:
    base = [arm_spec("Greedy", "greedy"), arm_spec("LP", "lp")]
    it200 = (rl_arms("IPPO-it200", "ippo", "checkpoint_it0200.pt", True)
             + rl_arms("MAPPO-it200", "mappo", "checkpoint_it0200.pt", True))
    tasks: list[dict] = []
    for a in base + it200:
        for sc in MAIN_SCENARIOS:
            tasks.append({"study": "main", "arm": a, "scenario": sc})
    abl = (rl_arms("IPPO-best", "ippo", "checkpoint_best.pt", True)
           + rl_arms("MAPPO-best", "mappo", "checkpoint_best.pt", True)
           + rl_arms("IPPO-it200-sampled", "ippo", "checkpoint_it0200.pt", False)
           + rl_arms("MAPPO-it200-sampled", "mappo", "checkpoint_it0200.pt", False))
    for a in abl:
        for sc in ABL_SCENARIOS:
            tasks.append({"study": "ablation", "arm": a, "scenario": sc})
    for name, path in BC_CKPTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/diag_bc_lp.py first")
        for sc in BC_SCENARIOS:
            tasks.append({"study": "ablation",
                          "arm": arm_spec(name, "bc", path, True, -1), "scenario": sc})
    # Sweep points identical to a main scenario (S1 p=1.0, S4 z1 @60) reuse the
    # main rows instead of re-running the same seeds.
    skip = {"S1@p1.0", "S4@z1@60"}
    sweep_labels = [k for k in scenario_specs() if "@" in k and k not in skip]
    for a in base + it200:
        for sc in sweep_labels:
            tasks.append({"study": "sweep", "arm": a, "scenario": sc})
    return tasks


def task_key(t: dict) -> str:
    a = t["arm"]
    return (f'{t["study"]}__{a["arm"]}__s{a["model_seed"]}__{t["scenario"]}'
            f'__e{EPISODES}').replace("@", "_")


# ------------------------------------------------------------------- worker
_POLICIES: dict = {}


def _init_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _get_policy(arm: dict):
    from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
    from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
    key = (arm["kind"], arm["ckpt"])
    if key in _POLICIES:
        return _POLICIES[key]
    if arm["kind"] == "greedy":
        out = (GreedyZoneDispatcher(), {})
    elif arm["kind"] == "lp":
        # defaults are the tuned values (lookahead 5.0, deficit_penalty_scale 1.0)
        pol = ZoneLPRebalancer()
        assert pol.config.lookahead_minutes == 5.0 and pol.config.deficit_penalty_scale == 1.0
        out = (pol, {})
    else:
        learner, manifest, _ = load_trained_policy(Path(arm["ckpt"]))
        out = (learner, manifest.get("env_config", {}))
    _POLICIES[key] = out
    return out


def _arm_cfg(arm: dict, env_cfg: dict):
    from marl_fleet_rebalancing.config import NYCConfig
    cfg = NYCConfig()
    cfg.relocator_share = 0.0     # Eq. (1) objective, as diag_objective_gap defaults
    cfg.targeting_bonus = 0.0
    if arm["kind"] in ("ippo", "mappo", "bc"):
        cfg.relocation_nearest_k = int(env_cfg.get("relocation_nearest_k", 0))
        cfg.obs_inbound = bool(env_cfg.get("obs_inbound", False))
    return cfg


def _window_index(env) -> int:
    """Window index of the demand generator, unwrapping the shift wrappers."""
    gen = getattr(env.env, "order_generator", None) if hasattr(env, "env") else None
    while gen is not None:
        w = getattr(gen, "last_window_index", None)
        if w is not None:
            return int(w)
        gen = getattr(gen, "base_generator", None)
    return -1


def run_episode(policy, is_rl, deterministic, env, seed: int, model_seed: int) -> dict:
    import torch
    from marl_fleet_rebalancing.networks.encoder import obs_to_vector

    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed=seed)
    if is_rl and not deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cfg = env._cfg
    cents = np.asarray(env.env.relocation_points, dtype=float)
    mpend = env._max_pending
    acc = {"reward": 0.0, "wait_pen": 0.0, "reloc_km": 0.0, "pickup_km": 0.0}
    step_ctx: dict = {}

    # Call-through recorders. They read state and never mutate it (AGENTS.md).
    inner_rewards = env._compute_zone_rewards
    inner_step = env.env.step

    def ride_step(actions, *a, **k):
        old = env._last_obs
        step_ctx["status"] = {d: old[d]["self"]["status"] for d in range(env._num_drivers)}
        step_ctx["actions"] = actions
        return inner_step(actions, *a, **k)

    def rewards(events, *a, **k):
        # waiting-time penalty exactly as in step 3 of the env (read-only)
        w = sum(float(o.get("waiting_time", 0.0)) for o in (env._current_pending or []))
        acc["wait_pen"] += cfg.alpha * w / max(cfg.order_timeout, 1e-9)
        for did, ev in events.items():
            if not ev.get("is_empty_move", False):
                continue
            km = float(ev.get("distance_moved", 0.0)) / 1000.0
            act = (step_ctx.get("actions") or {}).get(did) or {}
            if "orders" in act:
                acc["pickup_km"] += km
            elif "relocate" in act or step_ctx["status"].get(did) == "relocating":
                acc["reloc_km"] += km
            else:
                acc["pickup_km"] += km
        out = inner_rewards(events, *a, **k)
        acc["reward"] += float(np.sum(out))
        return out

    env.env.step = ride_step
    env._compute_zone_rewards = rewards

    counts = {"idle": 0, "dispatch": 0, "reloc": 0}
    t_decide = t_step = 0.0
    t_wall0 = time.perf_counter()
    while env.agents:
        if env._terminated:
            for _ in range(len(env.agents)):
                env.step(None)
            break
        ag = env.agent_selection
        t0 = time.perf_counter()
        m = env.action_mask(ag)
        obs = env.observe(ag)
        if is_rl:
            with torch.no_grad():
                a, _, _ = policy.act(obs_to_vector(obs)[None, :], m[None, :],
                                     deterministic=deterministic)
            a = int(a[0])
        else:
            a = int(policy.select_action(ag, env, obs))
        t1 = time.perf_counter()
        if not m[a]:
            raise RuntimeError(f"{ag} chose masked action {a}")
        if a == 0:
            counts["idle"] += 1
        elif a <= mpend:
            counts["dispatch"] += 1
        else:
            counts["reloc"] += 1
        env.step(a)
        t2 = time.perf_counter()
        t_decide += t1 - t0
        t_step += t2 - t1
    wall = time.perf_counter() - t_wall0

    k = env.kpi_summary()
    total = max(float(k["total"]), 1.0)
    executed = int(env._reloc_out["executed"])
    nact = max(sum(counts.values()), 1)
    # origin-zone demand: nearest centroid, same rule as the env's reward code
    org = np.array([o.origin for o in env.env.orders.values()], dtype=float)
    zi = np.argmin(((org[:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)
    oc = np.bincount(zi, minlength=NZ)
    eq1 = (cfg.gamma * k["served"] - cfg.beta * k["empty_km"] / max(cfg.empty_distance_norm_km, 1e-9)
           - acc["wait_pen"] - cfg.cancel_penalty * k["cancelled"])
    row = {
        "seed": seed,
        "window": _window_index(env),
        "served": int(k["served"]), "cancelled": int(k["cancelled"]), "total": int(k["total"]),
        "service_rate": k["served"] / total, "cancel_rate": k["cancelled"] / total,
        "empty_km": float(k["empty_km"]), "avg_wait_min": float(k["avg_wait"]),
        "reloc_count": executed, "reloc_attempted": counts["reloc"],
        "mean_km_per_reloc": float(k["empty_km"]) / max(executed, 1),
        "reloc_leg_km": acc["reloc_km"], "pickup_km": acc["pickup_km"],
        "pct_dispatch": 100.0 * counts["dispatch"] / nact,
        "pct_relocate": 100.0 * counts["reloc"] / nact,
        "pct_idle": 100.0 * counts["idle"] / nact,
        "exact_team_reward": acc["reward"], "accumulated_wait_penalty": acc["wait_pen"],
        "eq1_reward": eq1,
        "wall_time_sec": wall, "decision_time_sec": t_decide, "env_step_time_sec": t_step,
    }
    for z in range(NZ):
        row[f"orig_z{z}"] = int(oc[z])
    return row


def run_task(task: dict) -> str:
    from marl_fleet_rebalancing.experiments.shift_eval import make_shifted_env
    path = PARTIAL / (task_key(task) + ".json")
    if path.exists():
        return str(path)
    arm = task["arm"]
    policy, env_cfg = _get_policy(arm)
    cfg = _arm_cfg(arm, env_cfg)
    shift_label, limit, tz, pct, param = scenario_specs()[task["scenario"]]
    is_rl = arm["kind"] in ("ippo", "mappo", "bc")
    rows = []
    for i in range(EPISODES):
        seed = BASE_SEED + i
        env = make_shifted_env(shift_label, limit, tz, pct, shift_param=param,
                               seed=seed, base_cfg=copy.deepcopy(cfg))
        r = run_episode(policy, is_rl, arm["deterministic"], env, seed, arm["model_seed"])
        r.update({"study": task["study"], "arm": arm["arm"], "model_seed": arm["model_seed"],
                  "scenario": task["scenario"], "deterministic": arm["deterministic"]})
        rows.append(r)
        del env
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.rename(path)
    return str(path)


# ------------------------------------------------------------------- driver
def run_all(workers: int, limit: int = 0, out: Path = OUT, only: str = "") -> None:
    PARTIAL.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    if only:
        tasks = [t for t in tasks if only in t["arm"]["arm"]]
    todo = [t for t in tasks if not (PARTIAL / (task_key(t) + ".json")).exists()]
    if limit:
        tasks = todo[:limit]
        todo = tasks
    print(f"[study] {len(tasks)} tasks, {len(todo)} to run, workers={workers}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_task, todo), 1):
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[study] {i}/{len(todo)} tasks, {el:.0f}s, eta {el / i * (len(todo) - i):.0f}s",
                      flush=True)
    import pandas as pd
    rows = []
    for t in tasks:
        rows.extend(json.loads((PARTIAL / (task_key(t) + ".json")).read_text()))
    df = pd.DataFrame(rows)
    lead = ["study", "arm", "model_seed", "deterministic", "scenario", "seed", "window"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(OUT / "episodes.csv", index=False)
    print(f"[study] wrote {OUT / 'episodes.csv'} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--limit-tasks", type=int, default=0, help="debug: run only N tasks")
    ap.add_argument("--only", default="", help="debug: only arms whose name contains this")
    args = ap.parse_args()
    EPISODES = args.episodes
    OUT = Path(args.out_dir)
    PARTIAL = OUT / "_partial"
    os.environ["EVAL_FULL_STUDY_OUT"] = str(OUT)
    os.environ["EVAL_FULL_STUDY_EPISODES"] = str(EPISODES)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        run_all(args.workers, args.limit_tasks, OUT, args.only)
    from eval_full_study_summary import summarize  # noqa: E402
    summarize(OUT)
