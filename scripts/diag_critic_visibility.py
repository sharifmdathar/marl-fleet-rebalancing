"""Can MAPPO's centralized critic SEE the coordination failure it is meant to fix?

Read-only diagnostic. No training, no mutation of any quantity that is scored:
every number below comes from rolling a stored checkpoint forward on the env's
own train windows, or from comparing two matched rollouts that differ only in
which relocation destination one zone picked.

Sections
  A  enumerate every field concatenated into ``state()`` and into the critic's
     actual input (``[state | own obs]``, train.py:342) and mark which are
     driver-status aware.
  B  H2 counterfactual: two rollouts from the same seed, identical except for
     ONE zone's relocation destination. If the critic state at the following
     turns is identical, the destination is invisible to the critic while the
     car is in transit. Positive control included (the probe must be able to
     detect a change that IS visible).
  C  H1/H3: what the critic actually outputs. Variance of V decomposed into a
     shared turn-level part and an agent-specific part, for MAPPO's
     centralized critic and IPPO's decentralized critic on the same windows.
  D  H4: what the empty distance is actually made of. Split empty km into
     relocation-leg km vs other empty km, plus the churn decomposition, for
     both arms on identical (paired) windows.

    uv run python scripts/diag_critic_visibility.py --ippo runs/d42-ippo-s0/checkpoint_best.pt --mappo runs/d42-mappo-s0/checkpoint_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

M_PER_KM = 1000.0


def build_env(manifest_path: str, split: str = "train") -> RideGymAEC:
    man = json.loads(Path(manifest_path).read_text())
    cfg = NYCConfig(**{
        k: v for k, v in man["env_config"].items() if hasattr(NYCConfig, k)
    })
    cfg.nyc_split = split
    return RideGymAEC(config=cfg, seed=man.get("seed", 0))


# ---------------------------------------------------------------------- section A

def section_a(env: RideGymAEC) -> None:
    print("\n=== A. fields in the critic's input ===")
    env.reset(seed=1000)
    # advance a few turns so caches are populated the way training sees them
    for _ in range(8):
        while env.agents and not env._terminated:
            env.step(0)
    st = env.state()
    from marl_fleet_rebalancing.networks.encoder import obs_dim
    o = env.observation_space(env.possible_agents[0])
    odim = obs_dim(o)
    print("state() dims                 %d" % len(st))
    print("obs dims                     %d" % odim)
    print("critic input (train.py:342)  %d  = state + own obs" % (len(st) + odim))

    fields = [
        ("_zone_fleet_state[:,0] fleet total/num_drivers", env._zone_fleet_state[:, 0].size,
         "region membership only"),
        ("_zone_fleet_state[:,1] idle/num_drivers", env._zone_fleet_state[:, 1].size,
         "idle vs NOT idle; relocating is folded into 'not idle'"),
        ("_zone_fleet_state[:,2] pending demand", env._zone_fleet_state[:, 2].size, ""),
        ("_zone_fleet_state[:,3] deficit", env._zone_fleet_state[:, 3].size, ""),
        ("_pending_features", np.size(env._pending_features),
         "one zone's orders (which zone: see below)"),
        ("_pending_mask", np.size(env._pending_mask), ""),
        ("_demand_hist_arr", env._demand_hist_arr.size, ""),
        ("_legal_target_mask_flat()", env._num_zones, "static adjacency"),
        ("time_frac", 1, ""),
    ]
    for name, size, note in fields:
        print("  %-46s %4d  %s" % (name, size, note))

    # which zone's pending orders does state() carry?
    same_as_any = any(
        np.array_equal(env._pending_features, env._zone_pending_features[z])
        for z in range(env._num_zones)
    )
    print("\n  state()'s _pending_features equals ANY single zone's order "
          "block: %s" % same_as_any)
    print("  it is the GLOBALLY most-urgent-orders list "
          "(pettingzoo_env.py:550-553), carrying no zone label.")

    # the one quantity that IS the redundancy signal, and where it goes
    print("\n  _reloc_inbound_this_turn (cars inbound THIS turn): %s"
          % "NOT in state(), NOT in observe() -- consumed only by "
             "_legal_relocate_targets() -> action_mask()")
    print("  _driver_relocation_target (per-car destination):    %d entries, "
          "NOT in state(), NOT in observe()" % len(env._driver_relocation_target))
    print("  _zone_dispatchable_counts (idle OR relocating):    NOT in state(); "
          "read only by action_mask() at pettingzoo_env.py:852")


# ---------------------------------------------------------------------- section B

STATE_LAYOUT = [
    ("fleet_col0_total", 0, 16), ("fleet_col1_idle", 16, 16),
    ("fleet_col2_pending", 32, 16), ("fleet_col3_deficit", 48, 16),
    ("global_urgent_orders", 64, 96), ("global_urgent_mask", 160, 16),
    ("demand_hist", 176, 64), ("static_adjacency", 240, 16),
    ("time_frac", 256, 1),
]


def describe_dims(idx: list[int]) -> str:
    out = []
    for name, lo, hi in STATE_LAYOUT:
        k = sum(1 for i in idx if lo <= i < hi)
        if k:
            out.append("%s x%d" % (name, k))
    return ", ".join(out) if out else "(none)"


def _forced_target_rollout(env: RideGymAEC, force_zone: int | None,
                           force_target: int | None, n_turns: int
                           ) -> tuple[list[np.ndarray], dict]:
    """Roll a FRESH episode; all agents idle except one forced choice.

    If ``force_target`` is None the forced agent relocates to whichever legal
    target it has first (used to vary the SOURCE zone while holding the
    destination fixed). The legality of the forced action is checked at the
    moment it is taken and reported -- a silently-downgraded force would make
    this probe report "no difference" for the wrong reason.

    Returns (per-turn state() snapshots, forced-action record)."""
    env.reset(seed=1000)
    states: list[np.ndarray] = []
    rec = {"zone": force_zone, "target": None, "taken": False,
           "legal": False, "turn_forced": None, "n_legal_targets": 0}
    last_time = env.env.time
    turn = -1
    while env.agents and not env._terminated and len(states) < n_turns:
        a = env.agent_selection
        if a is None:
            break
        # _agent_order is re-permuted every turn (pettingzoo_env.py:1201), so a
        # turn boundary is detected from the sim clock, not from a fixed agent.
        if env.env.time != last_time:
            last_time = env.env.time
            turn += 1
            states.append(env.state().copy())
        z = env._zone_int_id(a)
        mask = env.action_mask(a)
        act = int(np.flatnonzero(mask)[0])          # idle
        legal = [j for j in range(env._num_zones)
                 if mask[1 + env._max_pending + j]]
        if (not rec["taken"]) and z == force_zone and turn == 1 and legal:
            rec["n_legal_targets"] = len(legal)
            tgt = legal[0] if force_target is None else force_target
            if tgt in legal:
                act = 1 + env._max_pending + tgt
                rec["target"] = tgt
                rec["taken"] = True
                rec["legal"] = True
                rec["turn_forced"] = turn
        env.step(act)
    return states, rec


def _report_pair(name: str, sA: list[np.ndarray], sB: list[np.ndarray]) -> None:
    n = min(len(sA), len(sB))
    print("  %s  (turns recorded %d/%d)" % (name, len(sA), len(sB)))
    for t in range(n):
        d = np.abs(sA[t] - sB[t])
        idx = [int(i) for i in np.flatnonzero(d > 1e-6)]
        print("    turn %2d: max|delta| = %.6f   dims differing = %3d/%d   [%s]"
              % (t, float(d.max()) if d.size else 0.0, len(idx), len(sA[t]),
                 describe_dims(idx)))


def section_b(env: RideGymAEC) -> None:
    print("\n=== B. H2 counterfactual: is a relocation destination visible "
          "in state()? ===")
    print("  Protocol: fresh episode from seed 1000, ALL agents idle except one")
    print("  forced relocate on turn 1. The two variants differ in exactly one")
    print("  number. Legality of the forced action is verified, not assumed.")

    # Which zone has >= 2 legal targets at the moment of forcing? Probe with a
    # throwaway rollout, then reuse the answer (same seed => same trajectory).
    probe_states, probe_rec = _forced_target_rollout(env, None, None, 2)
    zone = probe_rec["zone"]
    print("\n  probe for a legal source zone ...")
    # find the first zone (in selection order) that can relocate at turn 1
    env.reset(seed=1000)
    last_time = env.env.time
    turn = -1
    cand = None
    cand_targets: list[int] = []
    while env.agents and not env._terminated and turn < 1:
        a = env.agent_selection
        if a is None:
            break
        if env.env.time != last_time:
            last_time = env.env.time
            turn += 1
            continue
        z = env._zone_int_id(a)
        m = env.action_mask(a)
        legal = [j for j in range(env._num_zones) if m[1 + env._max_pending + j]]
        if len(legal) >= 2 and cand is None:
            cand, cand_targets = z, legal
        env.step(int(np.flatnonzero(m)[0]))
    if cand is None:
        print("  no zone with >=2 legal targets at turn 1; skipping")
        return
    zone = cand
    print("  source zone %d, %d legal targets at turn 1: %s"
          % (zone, len(cand_targets), cand_targets[:8]))

    print("\n  TEST: zone %d -> target %d  vs  zone %d -> target %d"
          % (zone, cand_targets[0], zone, cand_targets[1]))
    sA, rA = _forced_target_rollout(env, zone, cand_targets[0], 14)
    sB, rB = _forced_target_rollout(env, zone, cand_targets[1], 14)
    for tag, r in (("A", rA), ("B", rB)):
        print("    variant %s forced=%s target=%s legal=%s n_legal=%d"
              % (tag, r["taken"], r["target"], r["legal"], r["n_legal_targets"]))
    if not (rA["taken"] and rB["taken"]):
        print("    ABORT: a forced action did not execute")
        return
    _report_pair("state() diff", sA, sB)

    print("\n  POSITIVE CONTROL (vary the SOURCE zone, hold destination fixed):")
    print("    the probe must detect a change that IS in state().")
    sC, rC = _forced_target_rollout(env, zone, cand_targets[0], 14)
    sD, rD = _forced_target_rollout(env, zone + 1, cand_targets[0], 14)
    for tag, r in (("C", rC), ("D", rD)):
        print("    variant %s forced=%s zone=%s target=%s legal=%s n_legal=%d"
              % (tag, r["taken"], r["zone"], r["target"], r["legal"],
                 r["n_legal_targets"]))
    if not (rC["taken"] and rD["taken"]):
        print("    control partially executed; "
              "showing diff anyway (may be degenerate)")
    _report_pair("state() diff", sC, sD)


# ---------------------------------------------------------------------- section C

def rollout_values(env: RideGymAEC, ckpt: str, episodes: int, base_seed: int,
                   deterministic: bool = False):
    """Roll a checkpoint forward and collect every agent's critic output,
    grouped by TURN.

    A turn boundary is detected from the sim clock (env.env.time), because
    _agent_order is re-permuted every turn and env.agents stays non-empty for
    the whole episode -- keying on either would silently produce one record per
    EPISODE instead of per turn, which turns a within-turn variance into a
    within-episode one.
    """
    learner, _man, _ = load_trained_policy(Path(ckpt))
    is_mappo = learner.__class__.__name__ == "MAPPO"
    out = []
    for ep in range(episodes):
        seed = base_seed + ep
        env.reset(seed=seed)
        rec: list[np.ndarray] = []
        cur: list[float] = []
        last_time = env.env.time
        while env.agents and not env._terminated:
            if env.env.time != last_time:
                last_time = env.env.time
                if cur:
                    rec.append(np.array(cur, dtype=np.float64))
                cur = []
            g = env.state()
            a = env.agent_selection
            if a is None:
                break
            slot = env._zone_int_id(a)
            obs_vec = obs_to_vector(env.observe(a))
            mask = env.action_mask(a)
            if is_mappo:
                sv = np.concatenate([g, np.asarray(obs_vec, dtype=np.float32)])
                v = float(learner.value(sv[None, :], np.asarray([slot]))[0])
            else:
                with torch.no_grad():
                    _l, vt = learner.policy.forward(
                        torch.as_tensor(obs_vec[None, :]).to(learner.device))
                v = float(vt[0])
            cur.append(v)
            act, _lp, _v = learner.act(obs_vec[None, :], mask[None, :],
                                       deterministic=deterministic)
            env.step(int(act[0]))
        if cur:
            rec.append(np.array(cur, dtype=np.float64))
        out.append(rec)
    return out, is_mappo


def decompose(records) -> dict:
    """Split critic-output variance into a shared turn-level part and an
    agent-specific residual part. For each turn t with agent values v_t (one
    per zone): total = var over all (t,a); between = var of turn means;
    within = mean over turns of var within the turn."""
    allv = np.concatenate([v for rec in records for v in rec])
    means = np.array([v.mean() for rec in records for v in rec])
    withins = np.array([v.var() for rec in records for v in rec])
    total = float(allv.var())
    between = float(means.var())
    within = float(withins.mean())
    n_turns = len(means)
    n_ag = allv.size / max(n_turns, 1)
    return {
        "turns": n_turns,
        "agents_per_turn": n_ag,
        "mean_V": float(allv.mean()),
        "sd_V": float(allv.std()),
        "total_var": total,
        "between_turn_var": between,
        "within_turn_var": within,
        "within_share": within / max(total, 1e-12),
        "between_share": between / max(total, 1e-12),
        "mean_within_sd": float(np.sqrt(max(within, 0.0))),
        "sd_of_turn_means": float(np.sqrt(max(between, 0.0))),
    }


def section_c(env: RideGymAEC, ippo_ckpt: str, mappo_ckpt: str,
              episodes: int, base_seed: int) -> None:
    print("\n=== C. H3: is the critic's output agent-specific or turn-shared? ===")
    print("  (%d episodes, base seed %d, SAMPLED actions = training-time "
          "distribution)" % (episodes, base_seed))
    for label, ck in (("IPPO (decentralized critic)", ippo_ckpt),
                      ("MAPPO (centralized critic)", mappo_ckpt)):
        recs, is_mappo = rollout_values(env, ck, episodes, base_seed,
                                        deterministic=False)
        d = decompose(recs)
        print("\n  %s  [detected MAPPO=%s]" % (label, is_mappo))
        print("    turns=%d  agents/turn=%.0f  mean V=%.4f  sd V=%.4f"
              % (d["turns"], d["agents_per_turn"], d["mean_V"], d["sd_V"]))
        print("    shared (between-turn) var = %.6f   sd=%.4f   share=%.1f%%"
              % (d["between_turn_var"], d["sd_of_turn_means"],
                 100 * d["between_share"]))
        print("    agent-specific (within) var = %.6f   sd=%.4f   share=%.1f%%"
              % (d["within_turn_var"], d["mean_within_sd"],
                 100 * d["within_share"]))
        print("    within/between sd ratio    = %.3f"
              % (d["mean_within_sd"] / max(d["sd_of_turn_means"], 1e-12)))


# ---------------------------------------------------------------------- section D

class SplitEmptyProbe(RideGymAEC):
    """Records how much empty distance was driven by a car that is in a
    relocation chain vs everything else. Scores through the unmodified parent
    first, then reads the same events dict -- nothing the parent reads is
    consumed or altered."""

    def _compute_zone_rewards(self, events, dispatched_this_turn):
        reloc_m = 0.0
        other_m = 0.0
        for did, ev in events.items():
            if not ev.get("is_empty_move", False):
                continue
            m = float(ev.get("distance_moved", 0.0))
            if did in self._driver_relocation_target:
                reloc_m += m
            else:
                other_m += m
        rewards = super()._compute_zone_rewards(events, dispatched_this_turn)
        self.split_reloc_m += reloc_m
        self.split_other_m += other_m
        return rewards


def rollout_flow(env: SplitEmptyProbe, ckpt: str, episodes: int, base_seed: int,
                 deterministic: bool = True) -> list[dict]:
    learner, _man, _ = load_trained_policy(Path(ckpt))
    out = []
    n = env._num_zones
    mp = env._max_pending
    for ep in range(episodes):
        seed = base_seed + ep
        env.reset(seed=seed)
        env.split_reloc_m = 0.0
        env.split_other_m = 0.0
        inflow, outflow = Counter(), Counter()
        gross = wasted = 0
        reloc_leg_m = 0.0
        while env.agents and not env._terminated:
            a = env.agent_selection
            if a is None:
                break
            z = env._zone_int_id(a)
            obs_vec = obs_to_vector(env.observe(a))
            mask = env.action_mask(a)
            act, _lp, _v = learner.act(obs_vec[None, :], mask[None, :],
                                       deterministic=deterministic)
            act_val = int(act[0])
            if not mask[act_val]:
                act_val = int(np.flatnonzero(mask)[0])
            if act_val > mp:
                t = act_val - mp - 1
                outflow[z] += 1
                inflow[t] += 1
                gross += 1
                pend = float(env._pending_per_zone[t] * env._num_drivers)
                if round(pend) < 1:
                    wasted += 1
            env.step(act_val)
        churn = sum(min(inflow[i], outflow[i]) for i in range(n))
        k = env.kpi_summary()
        tot = max(int(k["served"]) + int(k["cancelled"]), 1)
        out.append({
            "seed": seed,
            "gross": gross,
            "churn": churn,
            "net": gross - churn,
            "wasted": wasted,
            "empty_km": float(k["empty_km"]),
            "reloc_km": env.split_reloc_m / M_PER_KM,
            "other_km": env.split_other_m / M_PER_KM,
            "sr": 100.0 * int(k["served"]) / tot,
        })
    return out


def section_d(ippo_ckpt: str, mappo_ckpt: str, episodes: int, base_seed: int,
              deterministic: bool) -> None:
    print("\n=== D. H4: what is the empty distance made of? ===")
    print("  (%d PAIRED windows, base seed %d, %s actions)"
          % (episodes, base_seed,
             "DETERMINISTIC" if deterministic else "SAMPLED"))
    res = {}
    for label, ck in (("IPPO", ippo_ckpt), ("MAPPO", mappo_ckpt)):
        env = SplitEmptyProbe.__new__(SplitEmptyProbe)
        man = json.loads((Path(ck).parent / "config.json").read_text())
        cfg = NYCConfig(**{k: v for k, v in man["env_config"].items()
                           if hasattr(NYCConfig, k)})
        cfg.nyc_split = "train"
        RideGymAEC.__init__(env, config=cfg, seed=man.get("seed", 0))
        env.split_reloc_m = 0.0
        env.split_other_m = 0.0
        res[label] = rollout_flow(env, ck, episodes, base_seed,
                                  deterministic=deterministic)

    print("\n  %-8s %8s %8s %8s %8s %9s %9s %9s"
          % ("arm", "SR%", "gross", "churn", "wasted%", "empty km",
             "reloc km", "other km"))
    for label, rows in res.items():
        g = lambda k_: np.mean([r[k_] for r in rows])
        print("  %-8s %8.2f %8.1f %8.1f %8.1f %9.2f %9.2f %9.2f"
              % (label, g("sr"), g("gross"), g("churn"),
                 100.0 * g("wasted") / max(g("gross"), 1e-9),
                 g("empty_km"), g("reloc_km"), g("other_km")))

    print("\n  churn share of gross legs:")
    for label, rows in res.items():
        g = lambda k_: np.mean([r[k_] for r in rows])
        print("    %-8s %5.1f%%  (km per gross leg = %.3f, "
              "km per net-changing leg = %.3f)"
              % (label,
                 100.0 * g("churn") / max(g("gross"), 1e-9),
                 g("empty_km") / max(g("gross"), 1e-9),
                 g("empty_km") / max(g("net"), 1e-9)))

    print("\n  PAIRED per-window deltas (MAPPO - IPPO), mean +/- sd:")
    A = res["IPPO"]
    B = res["MAPPO"]
    for key, unit in (("empty_km", "km"), ("reloc_km", "km"), ("other_km", "km"),
                      ("gross", "legs"), ("sr", "pp")):
        d = np.array([b[key] - a[key] for a, b in zip(A, B)])
        print("    d%-9s %+8.3f +/- %-7.3f %s"
              % (key, d.mean(), d.std(), unit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default="runs/d42-ippo-s0/checkpoint_best.pt")
    ap.add_argument("--mappo", default="runs/d42-mappo-s0/checkpoint_best.pt")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--sections", default="ABCD")
    ap.add_argument("--deterministic", action="store_true",
                    help="section D: argmax actions (default: sampled, which is "
                         "what the training-metric table reports)")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    man = json.loads((Path(args.mappo).parent / "config.json").read_text())
    env = build_env(
        str(Path(args.mappo).parent / "config.json"), split="train")

    if "A" in args.sections:
        section_a(env)
    if "B" in args.sections:
        section_b(env)
    if "C" in args.sections:
        section_c(env, args.ippo, args.mappo, args.episodes, args.base_seed)
    if "D" in args.sections:
        section_d(args.ippo, args.mappo, args.episodes, args.base_seed,
                  args.deterministic)


if __name__ == "__main__":
    main()