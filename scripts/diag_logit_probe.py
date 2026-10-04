"""What does the policy's decision layer actually see and output?

`diag_obs_audit.py` proved the demand channel arrives correctly indexed. This
looks at the other end: take a real observation, hold everything fixed, vary
ONLY the demand column, and watch the relocation logits.

That converts the question from "is the signal present" to "does the network's
output depend on it", with no confound from RL sampling, no randomness in the
action choice, and no dependence on where the car ends up.

Reports, for a real zone observation:
  * the top-ranked relocation action by logit
  * how the relocation logit spread moves when demand is moved between zones
  * whether the argmax target changes at all
  * the same under a demand pattern that is wildly exaggerated, to see whether
    the logits are simply insensitive or genuinely flat

A network whose argmax target never moves when the demand pattern is inverted is
not weakly reading demand -- it is ignoring it.

    uv run python scripts/diag_logit_probe.py --ippo <ckpt> --mappo <ckpt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.experiments.evaluate import load_trained_policy
from marl_fleet_rebalancing.networks.encoder import obs_to_vector

A0 = 6          # all_zone_state offset in the flat vector (from diag_obs_audit)
NZ = 16
COL_PEND = 2
COL_DEFICIT = 3


def relocate_logits(policy, v: np.ndarray, mp: int) -> np.ndarray:
    """Logits over the num_zones relocation actions for one observation."""
    with torch.no_grad():
        if hasattr(policy, "act"):
            logits, _, _ = policy.policy.evaluate_actions(
                torch.as_tensor(v)[None, :], None)[:3] \
                if hasattr(policy, "policy") else (None, None, None)
    return logits


def logits_via_act(policy, v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Action logits, obtained by asking the learner for its distribution."""
    with torch.no_grad():
        if hasattr(policy, "act"):
            out = policy.act(torch.as_tensor(v)[None, :],
                             torch.as_tensor(m)[None, :], deterministic=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ippo", default=None)
    ap.add_argument("--mappo", default=None)
    ap.add_argument("--base-seed", type=int, default=8000)
    args = ap.parse_args()

    cfg = NYCConfig()
    cfg.nyc_split = "val"
    env = RideGymAEC(cfg)
    env.reset(seed=args.base_seed)
    mp = env._max_pending

    # Take a real observation from a zone that has a car available, so the
    # relocation actions are actually legal and their logits meaningful.
    while env.agents:
        found = None
        for _ in range(env._num_zones):
            ag = env.agent_selection
            zi = env._zone_int_id(ag)
            if env._zone_idle_counts[zi] > 0 and env._legal_relocate_targets(zi):
                found = (ag, zi)
                break
            m = env.action_mask(ag)
            d = np.flatnonzero(m[1: 1 + mp])
            act = int(d[0]) + 1 if d.size else 0
            if not m[act]:
                act = int(np.flatnonzero(m)[0])
            env.step(act)
        if found:
            ag, zi = found
            break
        if not env.agents:
            print("no zone with an idle car found")
            return

    obs = env.observe(ag)
    v0 = obs_to_vector(obs).copy()
    m = env.action_mask(ag).copy()
    legal = env._legal_relocate_targets(zi)
    print(f"zone {zi}, {len(legal)} legal relocation targets")
    print(f"relocation actions are indices [{1+mp} : {env._action_dim})")

    def set_demand(v: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        w = v.copy()
        for z in range(NZ):
            w[A0 + z * 4 + COL_PEND] = float(pattern[z])
        return w

    scenarios = {
        "all demand in zone 3": np.eye(NZ)[3],
        "all demand in zone 11": np.eye(NZ)[11],
        "demand spread evenly": np.ones(NZ) / NZ,
        "no demand anywhere": np.zeros(NZ),
    }

    for label, ck in (("IPPO", args.ippo), ("MAPPO", args.mappo)):
        if not ck:
            continue
        learner, _mm, _ = load_trained_policy(Path(ck))
        print("\n" + "=" * 72)
        print(f"{label}: relocation logits under controlled demand")
        print("=" * 72)
        picks = {}
        for name, pat in scenarios.items():
            v = set_demand(v0, pat)
            out = learner.act(torch.as_tensor(v)[None, :],
                              torch.as_tensor(m)[None, :], deterministic=True)
            a = int(out[0][0]) if isinstance(out, tuple) else int(out)
            tgt = a - 1 - mp if a > mp else None
            picks[name] = tgt
            print(f"  {name:<24} -> action {a:3d}  target zone "
                  f"{tgt if tgt is not None else '(not a relocation)'}")
        uniq = set(picks.values())
        print()
        if len(uniq) == 1:
            print(f"  The argmax action is IDENTICAL under every demand pattern.")
            print(f"  The network's choice does not depend on where demand is.")
        else:
            print(f"  The argmax action changed across demand patterns: {uniq}")


if __name__ == "__main__":
    main()
