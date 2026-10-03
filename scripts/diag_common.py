"""Shared, self-verifying helpers for the scripts/ diagnostics.

Why this exists
---------------
Every script under scripts/ that picks an action by hand used to re-derive the
mask layout, and the flatnonzero-on-a-slice bug (indices come back relative to
the slice, so `int(rel[0])` dispatched action 0 = NOOP instead of a relocation)
turned several probes into no-op probes that silently reported the wrong thing.

Action layout in env/pettingzoo_env.py:
    action_mask(a)[0]                    -> NOOP, always legal
    action_mask(a)[1 + i]                -> dispatch candidate order i
    action_mask(a)[1 + max_pending + j]  -> relocate one idle driver to zone j

Two rules are enforced here rather than repeated per script:

1. Every chosen action is asserted legal in the SAME mask that produced it.
2. Every policy declares the semantics it INTENDED ("dispatch" / "relocate" /
   "noop") independently of the index it computes, and the declaration is
   checked against ``env._decode_action``. A policy that computes the wrong
   index therefore fails loudly instead of quietly measuring a no-op.

Observation is strictly READ-ONLY. ``ObservedEnv`` subclasses RideGymAEC and
overrides ``_compute_zone_rewards`` only to *read* ``self._driver_relocator`` /
``self._order_dispatcher`` / ``self.env.orders`` immediately before delegating to
``super()``. It never consumes ``_seen_completed_ids`` -- it keeps its OWN set of
completed ids, which is the mistake that made an earlier probe report "0 credits"
twice (see AGENTS.md). Per-turn arrival/dispatch history is rebuilt by scanning
driver status in ``env._last_obs`` after each flush.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC

NOOP = 0


def legal_dispatch(mask: np.ndarray, max_pending: int) -> np.ndarray:
    """Slice-relative indices of legal DISPATCH actions."""
    return np.flatnonzero(mask[1 : 1 + max_pending])


def legal_relocate(mask: np.ndarray, max_pending: int) -> np.ndarray:
    """Slice-relative indices of legal RELOCATE targets."""
    return np.flatnonzero(mask[1 + max_pending :])


def relocate_action(max_pending: int, target_zone: int) -> int:
    return 1 + max_pending + int(target_zone)


def dispatch_action(candidate_index: int) -> int:
    return 1 + int(candidate_index)


def intended_kind(max_pending: int, action: int) -> str:
    a = int(action)
    if a == 0:
        return "noop"
    if a <= max_pending:
        return "dispatch"
    return "relocate"


# --- policies. Each returns (action, intended_kind). ------------------------
def _fallback(mask, max_pending):
    """Noop is only a legal fallback once dispatch AND relocate are both out."""
    if len(legal_dispatch(mask, max_pending)) or len(legal_relocate(mask, max_pending)):
        raise AssertionError("policy wanted noop but a real action was legal")
    return NOOP, "noop"


def policy_dispatch_first(env, agent, rng):
    mask = env.action_mask(agent)
    d = legal_dispatch(mask, env._max_pending)
    if len(d):
        return dispatch_action(int(d[0])), "dispatch"
    r = legal_relocate(mask, env._max_pending)
    if len(r):
        return relocate_action(env._max_pending, int(r[0])), "relocate"
    return _fallback(mask, env._max_pending)


def policy_relocate_first(env, agent, rng, target_rule: str = "first"):
    mask = env.action_mask(agent)
    r = legal_relocate(mask, env._max_pending)
    if len(r):
        if target_rule == "first":
            tgt = int(r[0])
        elif target_rule == "rand":
            tgt = int(r[int(rng.integers(len(r)))])
        elif target_rule == "best":
            pend = env._pending_per_zone * max(env._num_drivers, 1)
            tgt = int(max(r, key=lambda j: (round(float(pend[j])), -int(j))))
        else:
            raise ValueError(f"unknown target_rule {target_rule!r}")
        return relocate_action(env._max_pending, tgt), "relocate"
    d = legal_dispatch(mask, env._max_pending)
    if len(d):
        return dispatch_action(int(d[0])), "dispatch"
    return _fallback(mask, env._max_pending)


def policy_random_legal(env, agent, rng):
    """Uniform over ALL legal actions, noop included."""
    mask = env.action_mask(agent)
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise AssertionError("action_mask had no legal action at all")
    a = int(legal[int(rng.integers(len(legal)))])
    return a, intended_kind(env._max_pending, a)


def policy_dispatch_only(env, agent, rng):
    """Dispatch when legal, otherwise NOOP. Never relocates.

    This is the causal control for "does relocation produce service": paired
    against an arm that also relocates, on identical seeds, any difference in
    completions is attributable to the relocation actions alone.
    """
    mask = env.action_mask(agent)
    d = legal_dispatch(mask, env._max_pending)
    if len(d):
        return dispatch_action(int(d[0])), "dispatch"
    return NOOP, "noop"


def policy_relocate_only(env, agent, rng):
    """The BROKEN probe, kept as a labelled reference.

    Relocates when it can and otherwise NOOPs, so it is structurally incapable
    of dispatching and must report ~0 service at every timeout. It is included
    only to show that the earlier "0 orders served" number is a property of this
    policy, not of relocation.
    """
    mask = env.action_mask(agent)
    r = legal_relocate(mask, env._max_pending)
    if len(r):
        return relocate_action(env._max_pending, int(r[0])), "relocate"
    return NOOP, "noop"


def policy_noop(env, agent, rng):
    """Unconditional do-nothing reference floor. NOOP is always legal."""
    return NOOP, "noop"


def make_policies() -> dict[str, Callable]:
    return {
        "dispatch-first": policy_dispatch_first,
        "dispatch-only": policy_dispatch_only,
        "relocate-first-first": lambda e, a, r: policy_relocate_first(e, a, r, "first"),
        "relocate-first-best": lambda e, a, r: policy_relocate_first(e, a, r, "best"),
        "relocate-first-rand": lambda e, a, r: policy_relocate_first(e, a, r, "rand"),
        "random-legal": policy_random_legal,
        "relocate-only(BROKEN)": policy_relocate_only,
        "noop": policy_noop,
    }


# --- self-verification -----------------------------------------------------
def verify_action(env, agent: str, action: int, intended: str) -> int:
    """Assert the action is legal AND decodes to the declared intent."""
    mask = env.action_mask(agent)
    zi = env._zone_int_id(agent)
    action = int(action)
    if not (0 <= action < len(mask)) or not bool(mask[action]):
        raise AssertionError(
            f"{agent}: chose ILLEGAL action {action}; legal = {np.flatnonzero(mask).tolist()}"
        )
    decoded = env._decode_action(action, zi)
    if intended == "dispatch":
        ok = isinstance(decoded, dict) and "orders" in decoded
    elif intended == "relocate":
        ok = isinstance(decoded, dict) and "relocate" in decoded
    elif intended == "noop":
        ok = action == 0 and decoded is None
    elif intended == "any":
        ok = isinstance(decoded, dict) and ("orders" in decoded or "relocate" in decoded)
    else:
        raise ValueError(intended)
    if not ok:
        raise AssertionError(
            f"{agent}: policy declared '{intended}' but action {action} decoded to "
            f"{decoded!r} (slice-offset bug or mask/policy disagreement)"
        )
    return action


# --- read-only observation -------------------------------------------------
class ObservedEnv(RideGymAEC):
    """RideGymAEC + a read-only record of relocation -> completion events.

    Nothing here writes to env state. The only override reads, then calls
    super(), and it keeps its OWN ``_probe_seen_completed`` set so the env's
    ``_seen_completed_ids`` is never consumed by the probe.
    """

    def reset_observer(self) -> None:
        self._probe_seen_completed: set[int] = set()
        self._probe_prev_status: dict[int, str] = {}
        self.reloc_starts: dict[int, list[tuple[float, int, int]]] = {}
        self.reloc_arrivals: dict[int, list[tuple[float, int]]] = {}
        self.assignments: dict[int, tuple[float, int, int]] = {}
        self.completions: list[dict] = []
        self._last_sample_time: float | None = None
        self._streak: Counter = Counter()
        self._best_streak: Counter = Counter()
        self.status_samples: Counter = Counter()
        self.per_driver_reloc_turns: Counter = Counter()
        self.per_driver_turns: Counter = Counter()
        self.reloc_cmds_this_ep = 0
        self._prev_reloc_targets: set[int] = set()

    def _compute_zone_rewards(self, events, dispatched_this_turn=None):
        from ride_gym.enums import OrderStatus

        done = {
            oid for oid, o in self.env.orders.items()
            if o.status == OrderStatus.COMPLETED
        }
        new = done - self._probe_seen_completed
        self._probe_seen_completed = done
        now = float(self.env.time)
        for oid in sorted(new):
            order = self.env.orders[oid]
            did = getattr(order, "assigned_driver", None)
            chain = list(self._driver_relocator.get(did, [])) if did is not None else []
            dispatcher = self._order_dispatcher.get(oid)
            asg = self.assignments.get(oid)
            asg_t = asg[0] if asg else None
            arrive = self.reloc_arrivals.get(did, []) if did is not None else []
            starts = self.reloc_starts.get(did, []) if did is not None else []
            # Most recent relocation arrival into the dispatching zone that
            # happened AFTER this rider asked and BEFORE the car was sent.
            fresh = None
            if did is not None and dispatcher is not None and asg_t is not None:
                cands = [
                    (t, z) for (t, z) in arrive
                    if z == dispatcher and float(order.request_time) <= t <= asg_t
                ]
                if cands:
                    fresh = max(cands)
            prior_starts = [s for s in starts if asg_t is not None and s[0] <= asg_t]
            self.completions.append({
                "oid": int(oid),
                "did": None if did is None else int(did),
                "dispatcher": None if dispatcher is None else int(dispatcher),
                "req": float(order.request_time),
                "assign_t": asg_t,
                "chain": chain,
                # env's own definition: the env pays relocator_share on this
                "conv_env": bool(chain),
                # came from a different zone than the one that dispatched it
                "conv_final_hop": bool(chain) and dispatcher is not None
                and int(chain[-1]) != int(dispatcher),
                # arrived in the dispatching zone AFTER this rider asked
                "conv_fresh": fresh is not None,
                "fresh_delay": (None if fresh is None
                                else fresh[0] - float(order.request_time)),
                "relocs_since_service": len(prior_starts),
            })
        return super()._compute_zone_rewards(events, dispatched_this_turn)

    # -- per-turn observation, called by the runner after each env.step --------
    def observe_turn(self) -> None:
        now = float(self.env.time)
        if self._last_sample_time is not None and now == self._last_sample_time:
            return
        self._last_sample_time = now
        for did in range(self._num_drivers):
            info = self._last_obs[did]["self"]
            st = info["status"]
            zi = int(max(0, min(int(info.get("current_region", 0)), self._num_zones - 1)))
            self.status_samples[st] += 1
            self.per_driver_turns[did] += 1
            prev = self._probe_prev_status.get(did)
            if st == "relocating":
                self.per_driver_reloc_turns[did] += 1
                if prev != "relocating":
                    self.reloc_starts.setdefault(did, []).append(
                        (now, zi, int(self._driver_relocation_target.get(did, -1)))
                    )
            elif prev == "relocating":
                # finished the leg; current_region is now the destination
                self.reloc_arrivals.setdefault(did, []).append((now, zi))
            self._probe_prev_status[did] = st
            if st == "relocating":
                self._streak[did] += 1
                self._best_streak[did] = max(self._best_streak[did], self._streak[did])
            else:
                self._streak[did] = 0
        # Relocation commands actually executed: relocate actions consume only
        # IDLE drivers (pettingzoo_env.py:1063), so every command puts a driver
        # into _driver_relocation_target that was not there before the flush.
        cur = set(self._driver_relocation_target)
        self.reloc_cmds_this_ep += len(cur - self._prev_reloc_targets)
        self._prev_reloc_targets = cur
        # Newly assigned orders (assigned_driver is stamped at dispatch time).
        for oid, o in self.env.orders.items():
            if oid in self.assignments:
                continue
            if getattr(o, "assigned_driver", None) is not None:
                self.assignments[int(oid)] = (
                    now, int(o.assigned_driver), int(self._order_dispatcher.get(int(oid), -1))
                )

    # -- summaries ------------------------------------------------------------
    def longest_reloc_streak(self) -> int:
        return max(self._best_streak.values()) if self._best_streak else 0

    def max_driver_reloc_frac(self) -> float:
        return max(
            (self.per_driver_reloc_turns[d] / max(self.per_driver_turns[d], 1))
            for d in range(self._num_drivers)
        )

    def reloc_driver_time(self) -> float:
        tot = sum(self.status_samples.values())
        return self.status_samples.get("relocating", 0) / max(tot, 1)


# --- episode runner --------------------------------------------------------
def run_episode(env: ObservedEnv, policy: Callable, seed: int, rng: np.random.Generator) -> dict:
    env.reset(seed=seed)
    env.reset_observer()
    kinds: Counter = Counter()
    while env.agents:
        agent = env.agent_selection
        action, intended = policy(env, agent, rng)
        action = verify_action(env, agent, action, intended)
        kinds[intended] += 1
        env.step(action)
        env.observe_turn()
    kpi = env.kpi_summary()
    served, cancelled = float(kpi["served"]), float(kpi["cancelled"])
    comp = env.completions
    delays = [c["fresh_delay"] for c in comp if c["conv_fresh"]]
    return {
        "seed": seed,
        "served": served,
        "cancelled": cancelled,
        "sr": 100.0 * served / max(served + cancelled, 1.0),
        "empty_km": float(kpi["empty_km"]),
        "avg_wait": float(kpi["avg_wait"]),
        "actions": sum(kinds.values()),
        "n_dispatch": kinds["dispatch"],
        "n_relocate": kinds["relocate"],
        "n_noop": kinds["noop"],
        "reloc_cmds": env.reloc_cmds_this_ep,
        "completions": len(comp),
        "conv_env": sum(c["conv_env"] for c in comp),
        "conv_final_hop": sum(c["conv_final_hop"] for c in comp),
        "conv_fresh": sum(c["conv_fresh"] for c in comp),
        "fresh_delay_mean": float(np.mean(delays)) if delays else float("nan"),
        "fresh_delay_max": float(np.max(delays)) if delays else float("nan"),
        "reloc_arrivals": sum(len(v) for v in env.reloc_arrivals.values()),
        "reloc_starts": sum(len(v) for v in env.reloc_starts.values()),
        "dispatches": len(env.assignments),
        "reloc_driver_time": env.reloc_driver_time(),
        "longest_reloc_streak": env.longest_reloc_streak(),
        "max_driver_reloc_frac": env.max_driver_reloc_frac(),
    }


def paired(a: list[float], b: list[float], label: str = "") -> dict:
    """Paired per-seed difference a-b (docs/eval-noise.md): never two means."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return {
        "label": label,
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
        "n": int(len(d)),
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
    }


def fmt_paired(p: dict, unit: str = "") -> str:
    return (f"{p['mean']:+.2f}{unit} +/- {p['std']:.2f} sd "
            f"(sem {p['sem']:.2f}, n={p['n']}, {p['wins']}W/{p['losses']}L/{p['ties']}T)")