"""Evaluation and visual playback tool for trained IPPO / MAPPO policies.

Usage:
    # 1. Evaluate a trained checkpoint over 5 evaluation seeds:
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --checkpoint runs/ippo-S0-base-bandra-seed0-.../checkpoint.pt \
        --episodes 5

    # 2. Render an MP4 animation of the policy in action:
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --checkpoint runs/ippo-S0-base-bandra-seed0-.../checkpoint.pt \
        --output policy_rollout.mp4

    # 3. Open the interactive GUI viewer:
    uv run python -m marl_fleet_rebalancing.experiments.evaluate \
        --checkpoint runs/ippo-S0-base-bandra-seed0-.../checkpoint.pt -i
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_fleet_rebalancing.agents.ippo import IPPO, IPPOConfig
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.baselines import GreedyZoneDispatcher, ZoneLPRebalancer
from marl_fleet_rebalancing.config import BandraConfig, EnvConfig, NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_dim, obs_to_vector
from marl_fleet_rebalancing.visualization import interactive_view, render_animation
from ride_gym.visualize import TrajectoryRecorder


def load_trained_policy(
    checkpoint_path: Path,
    target_zone: int | None = None,
    target_zone_demand_pct: float | None = None,
    num_drivers: int | None = None,
    max_dispatch_radius_km: float | None = None,
) -> tuple[IPPO | MAPPO, dict, RideGymAEC]:
    """Load policy weights, run manifest, and an aligned evaluation env."""
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = run_dir / "config.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Neither manifest.json nor config.json found at {run_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    algo = run_manifest.get("algo", "ippo").lower()
    scenario_id = run_manifest.get("scenario_id", "")
    is_nyc = "nyc" in scenario_id.lower() or "nyc" in str(run_dir).lower()

    # Reconstruct EnvConfig
    env_cfg_dict = run_manifest.get("env_config", {})
    if is_nyc:
        env_cfg = NYCConfig(**{k: v for k, v in env_cfg_dict.items() if hasattr(NYCConfig, k)})
    else:
        env_cfg = BandraConfig(**{k: v for k, v in env_cfg_dict.items() if hasattr(BandraConfig, k)})

    if target_zone is not None:
        env_cfg.target_zone = target_zone
    if target_zone_demand_pct is not None:
        pct = target_zone_demand_pct
        if pct > 1.0:
            pct = pct / 100.0
        env_cfg.target_zone_demand_pct = pct
    if num_drivers is not None:
        env_cfg.num_drivers = num_drivers
    if max_dispatch_radius_km is not None:
        env_cfg.max_dispatch_radius_km = max_dispatch_radius_km

    # Build environment
    eval_env = RideGymAEC(config=env_cfg, seed=run_manifest.get("seed", 42))
    eval_env.reset(seed=run_manifest.get("seed", 42))

    any_agent = eval_env.possible_agents[0]
    observation_dim = obs_dim(eval_env.observation_space(any_agent))
    action_dim = eval_env._action_dim
    use_state = algo == "mappo"

    # Pre-inspect checkpoint weights to accurately size actor and critic (backward compatibility)
    weights = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    ckpt_arch = run_manifest.get("algo_config", {}).get("arch", "mlp")

    # trunk.0 is the flat-MLP's first layer (input=observation_dim); the
    # attention encoder's trunk.0 takes the pooled context dim instead, so
    # this backward-compat resize only applies to the mlp arch (D33).
    if ckpt_arch == "mlp" and "actor" in weights and "trunk.0.weight" in weights["actor"]:
        ckpt_obs_dim = weights["actor"]["trunk.0.weight"].shape[1]
        if ckpt_obs_dim != observation_dim:
            observation_dim = ckpt_obs_dim

    state_dim = None
    if use_state:
        if "critic" in weights and "trunk.0.weight" in weights["critic"]:
            state_dim = weights["critic"]["trunk.0.weight"].shape[1]
        else:
            sample_obs = eval_env.observe(any_agent)
            zone_loc_dim = len(sample_obs["zone_location"].ravel())
            state_dim = len(eval_env.state()) + zone_loc_dim

    # Reconstruct Learner
    if use_state:
        algo_cfg = MAPPOConfig(**{k: v for k, v in run_manifest.get("algo_config", {}).items() if hasattr(MAPPOConfig, k)})
        learner: IPPO | MAPPO = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=algo_cfg,
        )
    else:
        algo_cfg = IPPOConfig(**{k: v for k, v in run_manifest.get("algo_config", {}).items() if hasattr(IPPOConfig, k)})
        learner = IPPO(
            observation_dim=observation_dim,
            action_dim=action_dim,
            config=algo_cfg,
        )

    learner.load_state_dict(weights)
    return learner, run_manifest, eval_env


def evaluate_policy(
    policy: Any,
    env: RideGymAEC,
    policy_type: str = "checkpoint",
    num_episodes: int = 5,
    base_seed: int = 1000,
    deterministic: bool = True,
    record_first_episode: bool = False,
) -> tuple[list[dict[str, float]], TrajectoryRecorder | None]:
    """Run evaluation episodes over fresh seeds and record performance metrics."""
    recorder = TrajectoryRecorder() if record_first_episode else None

    episode_results: list[dict[str, float]] = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset(seed=seed)
        should_record = recorder is not None and ep == 0

        acts_idle = 0
        acts_dispatch = 0
        acts_relocate = 0
        total_reward = 0.0
        last_time = env.env.time

        if should_record:
            if not hasattr(recorder, "relocation_points"):
                recorder.relocation_points = list(getattr(env.env, "relocation_points", []))
            if not hasattr(recorder, "target_zone"):
                recorder.target_zone = getattr(env._cfg, "target_zone", None)
                recorder.target_zone_pct = getattr(env._cfg, "target_zone_demand_pct", None)
            recorder.snapshot(env.env)

        while env.agents and not env._terminated:
            agent = env.agent_selection
            _obs, reward, term, trunc, _info = env.last()
            total_reward += float(reward)

            if term or trunc:
                env.step(None)
                continue

            obs = env.observe(agent)
            mask = env.action_mask(agent)

            if policy_type == "checkpoint":
                learner = policy
                # Backward compatibility: if checkpoint was trained with legacy (Z, 2) fleet state
                if hasattr(learner, "actor") and learner.actor.trunk[0].in_features < obs_dim(env.observation_space(agent)):
                    obs_in = dict(obs)
                    obs_in["all_zone_state"] = obs["all_zone_state"][:, :2]
                    obs_vec = obs_to_vector(obs_in)
                else:
                    obs_vec = obs_to_vector(obs)

                action, _logprob, _val = learner.act(
                    obs_vec[None, :], mask[None, :], deterministic=deterministic,
                )
                act_val = int(action[0])
            elif policy_type in ("greedy", "lp"):
                act_val = policy.select_action(agent, env, obs)
            elif policy_type == "random":
                valid_acts = np.where(mask)[0]
                act_val = int(np.random.choice(valid_acts)) if len(valid_acts) > 0 else 0
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

            if act_val == 0:
                acts_idle += 1
            elif 1 <= act_val <= env._max_pending:
                acts_dispatch += 1
            else:
                acts_relocate += 1

            env.step(act_val)

            if should_record and env.env.time > last_time:
                recorder.snapshot(env.env)
                last_time = env.env.time

        # Harvest final episode KPIs
        kpi = env.kpi_summary()
        total_acts = max(acts_idle + acts_dispatch + acts_relocate, 1)
        sr = float(kpi["served"]) / max(float(kpi["served"]) + float(kpi["cancelled"]), 1.0)

        res = {
            "episode": float(ep + 1),
            "seed": float(seed),
            "served": float(kpi["served"]),
            "cancelled": float(kpi["cancelled"]),
            "service_rate": sr,
            "avg_wait_min": float(kpi["avg_wait"]),
            "empty_km": float(kpi["empty_km"]),
            "total_reward": total_reward,
            "pct_dispatch": float(100.0 * acts_dispatch / total_acts),
            "pct_relocate": float(100.0 * acts_relocate / total_acts),
            "pct_idle": float(100.0 * acts_idle / total_acts),
        }

        # Driver-level status percentages from recorded frames.
        # Unlike the zone-agent action counts above, these count every
        # (driver, frame) pair across the episode — measuring actual fleet
        # time spent in each state, not zone-level "no driver in zone" idle.
        if recorder is not None and recorder.frames:
            _idles, _relocates = 0, 0
            from ride_gym.enums import DriverStatus
            for _frame in recorder.frames:
                for _d in _frame["drivers"].values():
                    if _d["status"] == DriverStatus.IDLE:
                        _idles += 1
                    elif _d["status"] == DriverStatus.RELOCATING:
                        _relocates += 1
            _n_driver_frames = len(recorder.frames) * len(recorder.frames[0]["drivers"])
            res["pct_idle_drivers"] = (
                100.0 * _idles / _n_driver_frames if _n_driver_frames else 0.0
            )
            res["pct_relocate_drivers"] = (
                100.0 * _relocates / _n_driver_frames if _n_driver_frames else 0.0
            )
        episode_results.append(res)

    return episode_results, recorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MARL checkpoint or baseline policy")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt file (required if --policy is checkpoint)")
    parser.add_argument("--policy", type=str, default=None,
                        choices=["checkpoint", "greedy", "lp", "random"],
                        help="Policy to evaluate: checkpoint, greedy, lp, or random (default: checkpoint if --checkpoint given, else greedy)")
    parser.add_argument("--env", type=str, default="bandra", choices=["bandra", "nyc"],
                        help="Environment preset for baseline policies: bandra or nyc (default: bandra)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--base-seed", type=int, default=1000,
                        help="Base seed for evaluation episodes (default: 1000)")
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="Sample actions stochastically instead of greedy argmax")
    parser.add_argument("-o", "--output", "--out-mp4", dest="output", type=str, default=None,
                        help="Output MP4 path to render animation (e.g. rollout.mp4)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for rendered MP4 animation (default: 10)")
    parser.add_argument("-i", "--interactive", action="store_true", default=False,
                        help="Open interactive viewer for the evaluation episode")
    parser.add_argument("--heatmap", action="store_true", default=False,
                        help="Enable zone outline-based demand heatmap overlay")
    parser.add_argument("--target-zone", type=int, default=None,
                        help="Target zone ID for parameterized demand injection")
    parser.add_argument("--target-zone-demand-pct", type=float, default=None,
                        help="Fraction or percentage of total orders originating from target zone (e.g. 0.08 or 8 for 8%%)")
    parser.add_argument("--num-drivers", type=int, default=None,
                        help="Override number of active drivers in the fleet")
    parser.add_argument("--max-dispatch-radius", type=float, default=None,
                        help="Override maximum pickup distance in km from zone centroid")
    args = parser.parse_args()

    policy_type = args.policy
    if policy_type is None:
        policy_type = "checkpoint" if args.checkpoint is not None else "greedy"

    if policy_type == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint <path> is required when --policy is checkpoint")
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        print(f"Loading policy from {ckpt_path}...")
        policy, manifest, env = load_trained_policy(
            ckpt_path,
            target_zone=args.target_zone,
            target_zone_demand_pct=args.target_zone_demand_pct,
            num_drivers=args.num_drivers,
            max_dispatch_radius_km=args.max_dispatch_radius,
        )
        algo = manifest.get("algo", "unknown").upper()
        scenario = manifest.get("scenario_id", "unknown")
    else:
        # Build baseline policy and environment
        if args.env == "nyc":
            env_cfg = NYCConfig()
        else:
            env_cfg = BandraConfig()

        if args.target_zone is not None:
            env_cfg.target_zone = args.target_zone
        if args.target_zone_demand_pct is not None:
            pct = args.target_zone_demand_pct
            if pct > 1.0:
                pct = pct / 100.0
            env_cfg.target_zone_demand_pct = pct
        if args.num_drivers is not None:
            env_cfg.num_drivers = args.num_drivers
        if args.max_dispatch_radius is not None:
            env_cfg.max_dispatch_radius_km = args.max_dispatch_radius

        env = RideGymAEC(config=env_cfg, seed=args.base_seed)
        algo = policy_type.upper()
        scenario = f"base-{args.env}"

        if policy_type == "greedy":
            policy = GreedyZoneDispatcher()
        elif policy_type == "lp":
            policy = ZoneLPRebalancer()
        elif policy_type == "random":
            policy = None
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    print(f"Config loaded: Algo={algo} | Scenario={scenario} | Zones={env._num_zones} | Drivers={env._num_drivers}")

    record_animation = args.output is not None or args.interactive

    # Interactive inspect mode: run a single episode — the viewer only shows
    # the first one anyway, so running 5 wastes resources.
    effective_episodes = 1 if args.interactive else args.episodes

    deterministic = not args.stochastic

    print(f"\nRunning {effective_episodes} evaluation episode(s) "
          f"(deterministic={deterministic}, base_seed={args.base_seed})..."
          + (" (interactive inspect — single episode)" if args.interactive else ""))
    results, recorder = evaluate_policy(
        policy=policy,
        env=env,
        policy_type=policy_type,
        num_episodes=effective_episodes,
        base_seed=args.base_seed,
        deterministic=deterministic,
        record_first_episode=record_animation,
    )

    # Compute Aggregate Stats
    served_vals = [r["served"] for r in results]
    cancelled_vals = [r["cancelled"] for r in results]
    sr_vals = [r["service_rate"] * 100 for r in results]
    wait_vals = [r["avg_wait_min"] for r in results]
    empty_vals = [r["empty_km"] for r in results]
    reward_vals = [r["total_reward"] for r in results]
    disp_vals = [r["pct_dispatch"] for r in results]
    reloc_vals = [r["pct_relocate"] for r in results]

    # Driver-level status counts come from the TrajectoryRecorder, which only
    # captures episode 0 (when record_first_episode=True). They are only
    # available in single-episode/inspect mode — not meaningful as an
    # aggregate across non-recorded episodes.
    _all_have_drv = all("pct_idle_drivers" in r for r in results)
    idle_drv_vals = [r.get("pct_idle_drivers", 0.0) for r in results] if _all_have_drv else []
    reloc_drv_vals = [r.get("pct_relocate_drivers", 0.0) for r in results] if _all_have_drv else []

    if effective_episodes == 1:
        r = results[0]
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULT ({algo} on {scenario} | seed={int(r['seed'])})")
        print("=" * 60)
        print(f"  Served Orders:      {r['served']:.0f}")
        print(f"  Cancelled Orders:   {r['cancelled']:.0f}")
        print(f"  Service Rate:       {r['service_rate'] * 100:.1f}%")
        print(f"  Avg Wait Time:      {r['avg_wait_min']:.2f} min")
        print(f"  Empty Distance:     {r['empty_km']:.1f} km")
        print(f"  Cumulative Reward:  {r['total_reward']:+.1f}")
        if "pct_idle_drivers" in r:
            print(f"  Idle Drivers:       {r['pct_idle_drivers']:.1f}%")
            print(f"  Relocating Drivers: {r['pct_relocate_drivers']:.1f}%")
            print(f"  Busy Drivers:       {100.0 - r['pct_idle_drivers'] - r['pct_relocate_drivers']:.1f}%")
        print("=" * 60)
    else:
        print("\n" + "=" * 78)
        print(f"EVALUATION SUMMARY ({algo} on {scenario} over {args.episodes} seeds)")
        print("=" * 78)
        print(f"{'Metric':<25} | {'Mean ± Std':<20} | {'Range [Min, Max]'}")
        print("-" * 78)
        print(f"{'Served Orders':<25} | {np.mean(served_vals):6.1f} ± {np.std(served_vals):<10.1f} | [{min(served_vals):.0f}, {max(served_vals):.0f}]")
        print(f"{'Cancelled Orders':<25} | {np.mean(cancelled_vals):6.1f} ± {np.std(cancelled_vals):<10.1f} | [{min(cancelled_vals):.0f}, {max(cancelled_vals):.0f}]")
        print(f"{'Service Rate (%)':<25} | {np.mean(sr_vals):6.1f}% ± {np.std(sr_vals):<9.1f}% | [{min(sr_vals):.1f}%, {max(sr_vals):.1f}%]")
        print(f"{'Avg Waiting Time (min)':<25} | {np.mean(wait_vals):6.2f} ± {np.std(wait_vals):<10.2f} | [{min(wait_vals):.2f}, {max(wait_vals):.2f}]")
        print(f"{'Empty Distance (km)':<25} | {np.mean(empty_vals):6.1f} ± {np.std(empty_vals):<10.1f} | [{min(empty_vals):.1f}, {max(empty_vals):.1f}]")
        print(f"{'Cumulative Reward':<25} | {np.mean(reward_vals):6.1f} ± {np.std(reward_vals):<10.1f} | [{min(reward_vals):.1f}, {max(reward_vals):.1f}]")
        print(f"{'Dispatch Actions (%)':<25} | {np.mean(disp_vals):6.1f}% ± {np.std(disp_vals):<9.1f}% | [{min(disp_vals):.1f}%, {max(disp_vals):.1f}%]")
        print(f"{'Relocate Actions (%)':<25} | {np.mean(reloc_vals):6.1f}% ± {np.std(reloc_vals):<9.1f}% | [{min(reloc_vals):.1f}%, {max(reloc_vals):.1f}%]")
        if idle_drv_vals:
            print(f"{'Idle Drivers (%)':<25} | {np.mean(idle_drv_vals):6.1f}% ± {np.std(idle_drv_vals):<9.1f}% | [{min(idle_drv_vals):.1f}%, {max(idle_drv_vals):.1f}%]")
            print(f"{'Relocating Drivers (%)':<25} | {np.mean(reloc_drv_vals):6.1f}% ± {np.std(reloc_drv_vals):<9.1f}% | [{min(reloc_drv_vals):.1f}%, {max(reloc_drv_vals):.1f}%]")
        print("=" * 78)

    if recorder is not None:
        if args.interactive:
            print("\nOpening interactive viewer...")
            interactive_view(recorder, detailed=True, initial_heatmap=args.heatmap)
        elif args.output:
            print(f"\nRendering {len(recorder.frames)} frames to {args.output} (fps={args.fps}, heatmap={args.heatmap})...")
            render_animation(recorder, out_path=args.output, fps=args.fps, detailed=True, show_heatmap=args.heatmap)
            print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
