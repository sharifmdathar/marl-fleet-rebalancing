"""Is the relocation failure an observability failure? An oracle-forecast probe.

Every relocation-capable candidate fails the two-zone competence test: on
trained d42/d43 checkpoints, the `greedy` floor and `lp`, ALL end phase 2 with
LESS fleet share at the new hotspot B than they started phase 1 with. RL
relocation precision sits at 13.5% against a 12.5% chance rate over 8 legal
targets -- i.e. RL targets relocation destinations at chance.

Two explanations are consistent with that, and they call for opposite fixes:

  (a) OBSERVABILITY. The actor's only temporal demand channel is `demand_hist`,
      shape (16, demand_history_len=4), appended once per turn. With dt = 0.5
      that is 4 * 0.5 = 2.0 minutes of memory in a 60-minute episode (3.3%),
      against a probe that switches demand at minute 25. The LP rebalancer has
      the same problem on the same axis: `lookahead_minutes = 5.0`. Neither
      family can see a shift that has not happened yet, so both correctly
      conclude "stay put" -- which is exactly what losing fleet share at B looks
      like.

  (b) OBJECTIVE / CREDIT ASSIGNMENT. Both families can see demand arrive at B
      but neither is rewarded enough for pre-positioning, so they still stay put.

This script separates them WITHOUT retraining and WITHOUT changing the
environment. It injects a *perfect demand forecast* straight into the existing
checkpoint's observation vector, so no obs-dim change and no env version bump
is required: the checkpoint loads, sees a forecast it was never trained to
rely on, and must either act on it or not.

    uv run python scripts/diag_oracle_forecast.py --seeds 0 --episodes 1 --dry-run
    uv run python scripts/diag_oracle_forecast.py

Injection rule. Only the `demand_hist` slice is touched, and within it all four
history columns are overwritten with the demand that WILL occur, evaluated at
k = (10, 20, 30, 40) turns ahead (5 / 10 / 15 / 20 minutes ahead at dt = 0.5).
The value comes from the analytic hub schedule the probe itself builds --
hub(t) = localhub_stage1_zone while t < localhub_switch_min else
localhub_stage2_zone -- broadcast as 1.0 at the hub zone and 0.0 elsewhere, then
divided by the env's live `_demand_hist_scale` and clipped to [0, 1] so it
occupies exactly the normalized range the encoder was trained on. Every other
slice is left bit-identical, which `--dry-run` asserts slice by slice.

Because `scale` is a running max of a fleet-normalized pending count it is
normally well below 1.0, so the oracle channel saturates at the maximum
normalized demand the encoder has ever seen. The forecast therefore lives in the
SPATIAL channel -- which zone is about to be hot -- not in a graded intensity.
That is the axis under test.

Scenario construction, the `verify_shift` gate and the phase accounting are
reused verbatim from scripts/competence_two_zone.py: if demand did not actually
concentrate at A then B, the comparison is meaningless and this script refuses
to report.

Relocation precision -- the primary read -- is the share of relocation legs
whose destination zone had at least one unmet order at the moment the move was
commanded. Chance rate is 1/8 = 12.5% over the 8 legal relocation targets.
Sampling the decision (`deterministic=False`) matches the competence probe.

Written to results/oracle_forecast/oracle_forecast.json.
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

from marl_fleet_rebalancing.baselines import ZoneLPRebalancer
from marl_fleet_rebalancing.baselines.lp_baseline import LPBaselineConfig
from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

from ride_gym.enums import OrderStatus

# Scenario construction, shift gate and pick_far_pair are the verified
# competence-test machinery; reuse, do not reimplement.
from competence_two_zone import pick_far_pair, verify_shift  # noqa: E402

# Horizon ladder in TURNS. With dt = 0.5 these are 5 / 10 / 15 / 20 minutes,
# matching `demand_history_len = 4` so each history column gets its own rung.
K_TURNS = (10, 20, 30, 40)

# Deepest rung in minutes. The LP oracle is given the same foresight as the
# deepest RL channel so the two families are compared on equal information.
ORACLE_HORIZON_MIN = 20.0


def slice_table(num_zones: int, max_pending: int, hist_len: int) -> dict[str, tuple[int, int]]:
    """Flat-vector slice bounds, derived from `obs_to_vector`'s key order.

    Derived rather than hard-coded because the offsets depend only on
    num_zones, max_pending and demand_history_len -- all of which live in the
    run's config.json, not on any magic constant.
    """
    dims = [
        ("zone_location", 2),
        ("zone_state", 4),
        ("all_zone_state", 4 * num_zones),
        ("pending_orders", 6 * max_pending),
        ("pending_mask", max_pending),
        ("pending_passengers", max_pending),
        ("neighbor_mask", num_zones),
        ("demand_hist", num_zones * hist_len),
        ("time_frac", 1),
    ]
    out: dict[str, tuple[int, int]] = {}
    at = 0
    for name, d in dims:
        out[name] = (at, at + d)
        at += d
    out["_total"] = (0, at)
    return out


def hub_at(t_min: float, stage1: int, stage2: int, switch_min: float) -> int:
    """The analytic demand schedule the probe itself builds."""
    return stage1 if t_min < switch_min else stage2


def oracle_demand_block(
    env,
    stage1: int,
    stage2: int,
    switch_min: float,
    k_turns: tuple[int, ...] = K_TURNS,
) -> np.ndarray:
    """Perfect (num_zones, len(k_turns)) demand forecast, normalized as the encoder saw it.

    Column j is the demand `k_turns[j]` turns from now. Values are 1.0 at the
    then-current hub and 0.0 elsewhere, divided by the env's live
    `_demand_hist_scale` and clipped to [0, 1].
    """
    n = env._num_zones
    now = float(env.env.time)
    dt = float(getattr(env._cfg, "dt", 0.5))
    block = np.zeros((n, len(k_turns)), dtype=np.float32)
    for j, k in enumerate(k_turns):
        block[hub_at(now + k * dt, stage1, stage2, switch_min), j] = 1.0
    scale = float(getattr(env, "_demand_hist_scale", 0.0))
    if scale <= 1e-9:
        # No history yet this episode, so nothing has established a scale.
        # 1.0 is the encoder's ceiling; use it rather than dividing by ~0.
        scale = 1.0
    return np.clip(block / scale, 0.0, 1.0).astype(np.float32)


def inject_demand_hist(vec: np.ndarray, block: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    """Return a copy of `vec` with ONLY the demand_hist slice replaced."""
    lo, hi = bounds
    out = vec.copy()
    out[lo:hi] = np.clip(block.ravel(), 0.0, 1.0)
    return out


class OracleLP(ZoneLPRebalancer):
    """LP rebalancer whose deficit estimate sees the hub schedule directly.

    Two changes from the shipped baseline, both about INFORMATION only:

      * `_update_demand_rates` is replaced. Instead of an EMA of recently
        observed pending orders, the rate vector is zero everywhere except the
        zone that will be hot at the end of the planning horizon. Magnitude is
        calibrated to the peak rate the policy has itself observed so far in
        this episode -- so the forecast is oracle in *where and when* only, and
        the magnitude distribution stays identical to the blind arm.
      * `lookahead_minutes` is raised to the deepest RL forecast rung
        (20 min). Without this the LP sees the shift only after it happens.
        Raising the horizon is safe here in a way it is not for the shipped
        baseline: the oracle rate vector is zero off-hub, so inflating the
        horizon scales that single zone's deficit and cannot destroy the
        spatial selectivity that a long horizon wrecks in the real estimator.

    The confounds are stated rather than hidden: this arm differs from `lp-blind`
    in direction AND horizon. The decision rule is about whether oracle-informed
    relocation rises above chance at all, not about attributing the two effects
    separately.
    """

    def __init__(self, stage1: int, stage2: int, switch_min: float) -> None:
        cfg = LPBaselineConfig(lookahead_minutes=ORACLE_HORIZON_MIN)
        super().__init__(cfg)
        self._stage1 = stage1
        self._stage2 = stage2
        self._switch_min = switch_min
        self._peak_rate = float(cfg.demand_init_rate)

    def _update_demand_rates(self, env: RideGymAEC) -> None:
        if self._n_zones is None or self._demand_rates is None:
            return
        # Track the magnitude from the REAL observed rate, before discarding it.
        super()._update_demand_rates(env)
        self._peak_rate = max(self._peak_rate, float(np.max(self._demand_rates)))
        ahead = float(env.env.time) + float(self.config.lookahead_minutes)
        z = hub_at(ahead, self._stage1, self._stage2, self._switch_min)
        rates = np.zeros(self._n_zones, dtype=float)
        if 0 <= z < self._n_zones:
            rates[z] = self._peak_rate
        self._demand_rates = rates


class RandomRelocator:
    """Control arm: dispatch if possible, else relocate to a uniformly random
    legal target. Measures relocation precision at CHANCE under the probe's own
    legal-target distribution.

    This arm exists because the chance rate had been assumed to be 1/8 = 12.5%
    from a fixed 8-target neighbourhood. With `neighbor_relocation_only` and the
    live inbound-relocation claim, the legal set is far smaller and varies per
    turn, so assuming 12.5% would have made a policy's precision look like
    signal when it is indistinguishable from random -- or worse.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, agent, env, obs=None) -> int:
        m = env.action_mask(agent)
        mp = env._max_pending
        d = np.flatnonzero(m[1:1 + mp])
        if len(d):
            return int(d[0]) + 1
        r = np.flatnonzero(m[1 + mp:])
        if len(r):
            return int(r[self._rng.integers(len(r))]) + 1 + mp
        return 0


def choose(env, policy, oracle: dict | None):
    """Pick this agent's action. Returns (action, destination_zone_or_None)."""
    ag = env.agent_selection
    m = env.action_mask(ag)
    mp = env._max_pending
    if hasattr(policy, "act"):
        vec = obs_to_vector(env.observe(ag))
        if oracle is not None:
            vec = inject_demand_hist(vec, oracle["block"], oracle["bounds"]["demand_hist"])
        with torch.no_grad():
            act, _, _ = policy.act(vec[None, :], m[None, :], deterministic=False)
        a = int(act[0])
    else:
        a = int(policy.select_action(ag, env, env.observe(ag)))
    if not m[a]:
        a = int(np.flatnonzero(m)[0])
    dest = a - mp - 1 if a > mp else None
    return a, (int(dest) if dest is not None else None)


def unmet_orders_at(env, zone: int) -> bool:
    """True if `zone` had at least one unassigned order at this instant.

    `_pending_per_zone` is the pending count normalized by fleet size (D22), so
    multiply back out before thresholding.
    """
    n = max(int(getattr(env, "_num_drivers", 1) or 1), 1)
    return float(env._pending_per_zone[zone]) * n >= 0.5


def run_arm(cfg, policy, episodes, seeds, switch_min, a, b, oracle_on, bounds):
    """Run one arm; return per-seed phase counters plus relocation precision."""
    env = RideGymAEC(cfg)
    per_seed = []
    for sd in seeds:
        if hasattr(policy, "reset") and not hasattr(policy, "act"):
            policy.reset(seed=sd)
        env.reset(seed=sd)
        st: dict[int, Counter] = {1: Counter(), 2: Counter()}
        hit = miss = 0
        chance = 0.0
        legal_sum = legal_n = 0
        while env.agents:
            ag = env.agent_selection
            phase = 1 if env.env.time < switch_min else 2
            oracle = None
            if oracle_on:
                oracle = {
                    "block": oracle_demand_block(
                        env, cfg.localhub_stage1_zone,
                        cfg.localhub_stage2_zone, switch_min,
                    ),
                    "bounds": bounds,
                }
            a_val, dest = choose(env, policy, oracle)
            # The chance rate for this move is 1 / (#legal targets right now),
            # not the 1/8 implied by a fixed neighbourhood size. Record it so
            # precision can be read against the real rate.
            n_legal = int(env.action_mask(ag)[1 + env._max_pending:].sum())
            legal_sum += n_legal
            legal_n += 1

            if dest is not None:
                # Read demand state BEFORE the env's own bookkeeping consumes it.
                st[phase]["reloc"] += 1
                chance += 1.0 / max(n_legal, 1)
                if unmet_orders_at(env, dest):
                    hit += 1
                else:
                    miss += 1
            elif a_val > 0:
                st[phase]["disp"] += 1
            env.step(a_val)

            st[phase]["fleet_at_B"] += int(env._zone_total_counts[b])
            st[phase]["fleet_total"] += int(env._zone_total_counts.sum())


        for o in env.env.orders.values():
            pt = float(o.pickup_time) if o.pickup_time is not None else 0.0
            st[1 if pt < switch_min else 2]["served"] += int(o.status == OrderStatus.COMPLETED)
        for o in env.env.orders.values():
            if o.status == OrderStatus.CANCELLED:
                ct = float(o.cancel_time) if o.cancel_time is not None else 0.0
                st[1 if ct < switch_min else 2]["cancelled"] += 1

        out = {"seed": sd, "reloc_prec": hit / max(hit + miss, 1),
               "reloc_legs": hit + miss,
               # Expected precision if targets were picked uniformly at random
               # among the legal set actually available on each move.
               "chance_prec": chance / max(hit + miss, 1),
               "mean_legal_targets": legal_sum / max(legal_n, 1)}
        for ph in (1, 2):
            s = st[ph]
            out[f"sr_p{ph}"] = 100.0 * s["served"] / max(s["served"] + s["cancelled"], 1)
            out[f"served_p{ph}"] = s["served"]
            out[f"cancelled_p{ph}"] = s["cancelled"]
            out[f"fleetB_p{ph}"] = s["fleet_at_B"] / max(s["fleet_total"], 1)
            out[f"reloc_p{ph}"] = s["reloc"]
            out[f"disp_p{ph}"] = s["disp"]
        per_seed.append(out)
    return per_seed


def dry_run(cfg, ckpt: Path | None, bounds) -> None:
    """Prove the injection touches exactly the 64 claimed dims and nothing else."""
    env = RideGymAEC(cfg)
    env.reset(seed=0)
    # Advance a few turns so `_demand_hist_scale` is populated and history is non-trivial.
    for _ in range(12):
        ag = env.agent_selection
        m = env.action_mask(ag)
        mp = env._max_pending
        d = np.flatnonzero(m[1:1 + mp])
        env.step(int(d[0]) + 1 if len(d) else 0)

    lo, hi = bounds["demand_hist"]
    before = obs_to_vector(env.observe(env.agent_selection))
    block = oracle_demand_block(
        env, cfg.localhub_stage1_zone, cfg.localhub_stage2_zone,
        cfg.localhub_switch_min,
    )
    after = inject_demand_hist(before, block, bounds["demand_hist"])

    total = bounds["_total"][1]
    print(f"\nDRY RUN: obs vector dim = {total}")
    if ckpt is not None:
        pol, man, _ = load_trained_policy(ckpt)
        exp = int(man["num_zones"]) if "num_zones" in man else None
        print(f"  checkpoint {ckpt.parent.name} num_zones={exp}")
    assert hi - lo == 64, f"demand_hist slice is {hi - lo} dims, expected 64"
    print(f"  demand_hist slice [{lo}:{hi}] = {hi - lo} dims (as claimed)")

    outside = np.concatenate([np.arange(0, lo), np.arange(hi, total)])
    same = np.array_equal(before[outside], after[outside])
    print(f"  indices outside [{lo}:{hi}] bit-identical: {same}")
    assert same, "injection leaked outside the demand_hist slice"

    changed = int(np.count_nonzero(before[lo:hi] != after[lo:hi]))
    print(f"  demand_hist dims changed: {changed}/64")
    raw = before[lo:hi].reshape(-1, len(K_TURNS))
    new = after[lo:hi].reshape(-1, len(K_TURNS))
    nz_raw = np.flatnonzero(raw.max(axis=1) > 0)
    nz_new = np.flatnonzero(new.max(axis=1) > 0)
    print(f"  zones with history signal  real={nz_raw.tolist()}  oracle={nz_new.tolist()}")
    print(f"  injected values at hub zones: {sorted(set(np.round(new[new > 0], 4).tolist()))}")
    if not same:
        raise AssertionError("dry run failed")
    print("  -> injection is confined to demand_hist. DRY RUN OK.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--switch-min", type=float, default=25.0)
    ap.add_argument("--pct", type=float, default=90.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ippo", default="runs/d43-ippo-s0/checkpoint_best.pt")
    ap.add_argument("--mappo", default="runs/d43-mappo-s0/checkpoint_best.pt")
    ap.add_argument("--out", default="results/oracle_forecast/oracle_forecast.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = NYCConfig()
    # localhub: EVERY order originates at stage1_zone before the switch and at
    # stage2_zone after it, with nothing elsewhere. `drift` cannot do this.
    cfg.demand_shift_mode = "localhub"
    cfg.drift_source_pct = args.pct
    cfg.drift_ramp_start_min = args.switch_min
    cfg.drift_ramp_end_min = args.switch_min + 0.5
    env0 = RideGymAEC(cfg)
    a, b = pick_far_pair(env0)
    cfg.drift_source_zone = a
    cfg.drift_dest_zone = b
    cfg.localhub_stage1_zone = a
    cfg.localhub_stage2_zone = b
    cfg.localhub_switch_min = args.switch_min
    cfg.nyc_split = args.split

    bounds = slice_table(env0._num_zones, env0._max_pending, env0._hist_len)
    print("oracle-forecast relocation probe")
    print("  phase 1 (t < %.0f min): demand at zone %d" % (args.switch_min, a))
    print("  phase 2 (t > %.0f min): demand moves to zone %d" % (args.switch_min, b))
    print("  forecast ladder (turns): %s  = %.0f/%.0f/%.0f/%.0f min ahead"
          % (list(K_TURNS), *[k * float(getattr(cfg, "dt", 0.5)) for k in K_TURNS]))
    print("  obs demand_hist slice: [%d:%d] of %d dims"
          % (bounds["demand_hist"][0], bounds["demand_hist"][1], bounds["_total"][1]))

    ckpts = {"ippo": Path(args.ippo), "mappo": Path(args.mappo)}
    for k, p in ckpts.items():
        if p.exists():
            ckpts[k] = p
        else:
            print(f"  !! {k} checkpoint missing at {p}; arm skipped")
            ckpts[k] = None

    if args.dry_run:
        dry_run(cfg, ckpts["ippo"], bounds)
        return

    ok, share_a, share_b = verify_shift(cfg, a, b, args.switch_min, args.base_seed)
    print("\n  SHIFT CHECK: early share at A=%.2f, late share at B=%.2f" % (share_a, share_b))
    if not ok:
        print("  *** DEMAND DID NOT CONCENTRATE AT THE HUBS. Numbers below would")
        print("  *** be meaningless -- not a result about any policy.")
        raise SystemExit("refusing to report: shift check failed")
    print("  SHIFT CHECK: early share at A > 0.5, late share at B > 0.5 -- comparison is meaningful.")

    arms = [("random", lambda: RandomRelocator(), False),
            ("lp-blind", lambda: ZoneLPRebalancer(), False),
            ("lp-oracle", lambda: OracleLP(a, b, args.switch_min), False)]
    for key in ("ippo", "mappo"):
        if ckpts[key] is not None:
            arms.append((f"{key}-blind", lambda c=ckpts[key]: load_trained_policy(c)[0], False))
            arms.append((f"{key}-oracle", lambda c=ckpts[key]: load_trained_policy(c)[0], True))

    print("\n%-16s %8s %8s %14s %10s %10s"
          % ("arm", "P1 SR%", "P2 SR%", "fleet@B P1->P2", "prec", "chance"))
    print("-" * 74)
    results = {}
    for name, make, oracle_on in arms:
        ep_seeds = [args.base_seed + s * args.episodes + i
                    for s in args.seeds for i in range(args.episodes)]
        per = run_arm(cfg, make(), len(ep_seeds), ep_seeds, args.switch_min, a, b,
                      oracle_on, bounds)
        agg = {
            k: float(np.mean([p[k] for p in per]))
            for k in ("sr_p1", "sr_p2", "fleetB_p1", "fleetB_p2",
                      "reloc_prec", "chance_prec", "mean_legal_targets")
        }
        agg["reloc_legs"] = int(sum(p["reloc_legs"] for p in per))
        results[name] = {"aggregate": agg, "per_episode": per}
        print("%-16s %7.1f%% %7.1f%% %13.0f%%->%2.0f%% %9.1f%% %9.1f%%"
              % (name, agg["sr_p1"], agg["sr_p2"],
                 100 * agg["fleetB_p1"], 100 * agg["fleetB_p2"],
                 100 * agg["reloc_prec"], 100 * agg["chance_prec"]))

    print("\n  `chance` is 1/#legal relocation targets actually available on each")
    print("  move, not the 1/8 a fixed 8-target neighbourhood would imply.")
    print("  decision rule, stated in advance:")
    print("    observability  <- oracle precision rises well above chance AND P2 SR rises")
    print("    objective      <- oracle precision stays at chance")
    print("    mixed          <- precision rises but P2 SR does not")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "oracle_forecast",
        "zones": {"A": a, "B": b, "switch_min": args.switch_min},
        "k_turns": list(K_TURNS),
        "oracle_horizon_min": ORACLE_HORIZON_MIN,
        "episode_seeds": ep_seeds,
        "arms": results,
    }, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()