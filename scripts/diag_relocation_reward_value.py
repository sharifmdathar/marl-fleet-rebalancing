"""Does the reward actually PAY for relocation? Tests MAPPO's mechanism directly.

Claim under test: MAPPO's better critic (explained_var 0.28-0.38 vs IPPO's
0.07-0.14) correctly resolves that relocation is NOT profitable under the
current reward, so seed 0 stopped relocating (5.3% of actions vs IPPO's 9.5%).

If true, the net reward of the relocation decisions themselves must be negative.
Relocation costs beta*km = 0.149 immediately but pays relocator_share*gamma =
0.45 only on completion, several turns later, discounted, and split across the
relay chain (0.45/len(chain)).

This measures that net value empirically by differencing two policies on
identical windows:

    dispatch-only    never relocates            (SR 46.7%, 69 km, from base S0)
    dispatch-first   relocates when it cannot dispatch

    net_value_of_relocation = reward(dispatch-first) - reward(dispatch-only)

per episode, on the env's own cumulative reward. Negative => the reward punishes
the decision it is meant to reward, and a low-variance critic would correctly
learn to stop relocating. That is the whole mechanism, measured.

    uv run python scripts/diag_relocation_reward_value.py --episodes 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC


def run_arm(arm: str, cfg: NYCConfig, episodes: int, base_seed: int):
    env = RideGymAEC(cfg)
    mp = env._max_pending
    rows = []
    for ep in range(episodes):
        env.reset(seed=base_seed + ep)
        # Accumulate the reward the env actually hands the agents.
        total_r = 0.0
        n_reloc = 0
        n_disp = 0
        while env.agents:
            a = env.agent_selection
            m = env.action_mask(a)
            d = np.flatnonzero(m[1 : 1 + mp])
            r = np.flatnonzero(m[1 + mp :])
            if arm == "dispatch-only":
                # TRUE dispatch-only: NOOP when nothing can be dispatched.
                # An earlier version fell through to the relocate branch here,
                # which made both arms byte-identical (measured reward delta
                # exactly 0.00, 0 differing relocations) and measured nothing.
                if len(d):
                    act = int(d[0]) + 1
                else:
                    act = 0
            else:  # dispatch-first
                if len(d):
                    act = int(d[0]) + 1
                elif len(r):
                    act = int(r[0]) + 1 + mp
                else:
                    act = 0
            if act > mp:
                n_reloc += 1
            elif act > 0:
                n_disp += 1
            env.step(act)
            total_r += float(sum(env.rewards.values()))
        k = env.kpi_summary()
        rows.append({
            "reward": total_r,
            "sr": 100.0 * k["served"] / max(k["served"] + k["cancelled"], 1),
            "empty": float(k["empty_km"]),
            "reloc": n_reloc,
            "disp": n_disp,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--relocator-share", type=float, default=None)
    args = ap.parse_args()

    cfg = NYCConfig()
    if args.relocator_share is not None:
        cfg.relocator_share = args.relocator_share
    print("config: beta=%.2f relocator_share=%.2f gamma=%.2f timeout=%.1f"
          % (cfg.beta, cfg.relocator_share, cfg.gamma, cfg.order_timeout))

    no = run_arm("dispatch-only", cfg, args.episodes, args.base_seed)
    yes = run_arm("dispatch-first", cfg, args.episodes, args.base_seed)

    def m(rs, k):
        return float(np.mean([r[k] for r in rs]))

    print("\n%-16s %10s %8s %10s %8s" % ("arm", "reward/ep", "SR%", "empty km", "reloc"))
    for name, rs in (("dispatch-only", no), ("dispatch-first", yes)):
        print("%-16s %10.2f %7.1f%% %10.1f %8d"
              % (name, m(rs, "reward"), m(rs, "sr"), m(rs, "empty"),
                 int(m(rs, "reloc"))))

    dr = [y["reward"] - x["reward"] for x, y in zip(no, yes)]
    de = [y["empty"] - x["empty"] for x, y in zip(no, yes)]
    ds = [y["sr"] - x["sr"] for x, y in zip(no, yes)]
    n_leg = max(m(yes, "reloc"), 1e-9)

    print("\npaired difference (dispatch-first MINUS dispatch-only), same seeds:")
    print("  reward/ep   %+.2f  (sd %.2f, %d/%d favour relocation)"
          % (np.mean(dr), np.std(dr), sum(1 for x in dr if x > 0), len(dr)))
    print("  service rate %+.2f pp" % np.mean(ds))
    print("  empty km     %+.1f" % np.mean(de))
    print("  extra legs   %.1f" % (m(yes, "reloc") - m(no, "reloc")))
    print("\n  NET REWARD PER EXTRA RELOCATION LEG: %+.4f"
          % (np.mean(dr) / max(m(yes, "reloc") - m(no, "reloc"), 1e-9)))
    print("\n  Negative => the reward PUNISHES relocation. A low-variance critic")
    print("  would correctly learn to stop relocating; a noisy one would not.")
    print("  That is the proposed MAPPO mechanism, tested.")


if __name__ == "__main__":
    main()