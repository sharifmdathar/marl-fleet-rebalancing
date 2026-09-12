"""Training entry point for IPPO/MAPPO on the zone-controller fleet env.

Run (from repo root):
    uv run python -m marl_fleet_rebalancing.experiments.train \
        --env bandra --algo mappo --seed 0 --iterations 20

Every run logs: full config (env + algo), seed(s), demand scenario ID,
and per-iteration KPIs (AGENTS.md experiment rules). Checkpoints are written
to the run dir but never committed (git-ignored).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from marl_fleet_rebalancing.agents.ippo import IPPO, IPPOConfig, RolloutBuffer
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.config import BandraConfig, EnvConfig, NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_dim, obs_to_vector


@dataclass
class TrainConfig:
    """Training-run parameters (everything logged, nothing hardcoded)."""

    env_config: EnvConfig
    algo_config: IPPOConfig | MAPPOConfig
    seed: int = 0
    iterations: int = 20           # PPO updates; each = rollout_episodes episodes
    scenario_id: str = "S0-base"   # demand scenario per docs/demand-shift.md
    out_root: Path = Path("runs")
    run_name: str | None = None
    checkpoint_interval: int = 10
    verbose: bool = False

    def to_dict(self) -> dict:
        return {
            "env_config": dataclasses.asdict(self.env_config),
            "algo_config": self.algo_config.to_dict(),
            "seed": self.seed,
            "iterations": self.iterations,
            "scenario_id": self.scenario_id,
            "checkpoint_interval": self.checkpoint_interval,
        }


def _episode_turns(cfg: EnvConfig) -> int:
    return int(round(cfg.horizon / cfg.dt))


def _make_env(
    name: str,
    seed: int,
    nyc_window_limit: int | None,
    base_env_cfg: EnvConfig | None = None,
) -> tuple[RideGymAEC, EnvConfig, str]:
    if base_env_cfg is not None:
        env_cfg = base_env_cfg
        if nyc_window_limit is not None and hasattr(env_cfg, "nyc_window_limit"):
            env_cfg.nyc_window_limit = nyc_window_limit
        scenario = "S0-base-bandra" if name == "bandra" else "S0-base-nyc-train"
    elif name == "bandra":
        env_cfg = BandraConfig()
        scenario = "S0-base-bandra"
    elif name == "nyc":
        env_cfg = NYCConfig(nyc_window_limit=nyc_window_limit)
        scenario = "S0-base-nyc-train"
    else:
        raise ValueError(f"Unknown env: {name!r} (expected 'bandra' or 'nyc')")
    if env_cfg.target_zone is not None:
        scenario += f"-target-zone-{env_cfg.target_zone}"
    env = RideGymAEC(config=env_cfg, seed=seed)
    return env, env_cfg, scenario


def collect_rollout(
    env: RideGymAEC,
    learner: IPPO | MAPPO,
    buffer: RolloutBuffer,
    total_turns: int,
    use_state: bool = False,
) -> dict:
    """Collect agent transitions for ``total_turns`` env turns (AEC pattern).

    Rewards close at the agent's next selection (PettingZoo delayed-close
    semantics): the reward read for an agent is the one produced by the env
    step that followed its previous action. Rollouts must end exactly at
    episode boundaries (total_turns = k * episode_turns), so no value
    bootstrap is needed. Returns episode KPIs.

    With ``use_state`` (MAPPO), each transition also stores the centralized
    global state and values come from the centralized critic; otherwise
    values are decentralized (IPPO).

    Episode accounting: when the AEC termination dance empties ``env.agents``,
    the ended episode's KPIs are read from the env BEFORE the next reset.
    """
    num_zones = env._num_zones
    counts = np.zeros(num_zones, dtype=int)
    pending: dict[str, tuple[int, int]] = {}
    episodes = 0
    served_total = 0
    cancelled_total = 0
    empty_km_total = 0.0
    wait_times_all: list[float] = []
    acts_idle = 0
    acts_dispatch = 0
    acts_relocate = 0
    episode_ended = False  # an episode finished; KPIs not yet accounted

    while counts.min() < total_turns or env.agents:
        if not env.agents:
            if episode_ended:
                kpi = env.kpi_summary()
                served_total += int(kpi["served"])
                cancelled_total += int(kpi["cancelled"])
                empty_km_total += float(kpi["empty_km"])
                if kpi["avg_wait"] > 0:
                    wait_times_all.append(kpi["avg_wait"])
                episodes += 1
                episode_ended = False
            if counts.min() < total_turns:
                env.reset(seed=None)
            continue
        agent = env.agent_selection
        _obs, reward, term, trunc, _info = env.last()
        slot = int(agent.split("_")[1])
        if agent in pending:
            s_slot, s_turn = pending.pop(agent)
            buffer.close(s_slot, s_turn, reward, done=term or trunc)
        if term or trunc:
            episode_ended = True
            env.step(None)
            continue
        obs = env.observe(agent)
        obs_vec = obs_to_vector(obs)
        mask = env.action_mask(agent)
        action, logprob, decentral_value = learner.act(obs_vec[None, :], mask[None, :])
        act_val = int(action[0])
        if act_val == 0:
            acts_idle += 1
        elif 1 <= act_val <= env._max_pending:
            acts_dispatch += 1
        else:
            acts_relocate += 1

        if use_state:
            zone_loc = obs["zone_location"].ravel()
            state_vec = np.concatenate([env.state(), zone_loc]).astype(np.float32)
            value = float(learner.value(state_vec[None, :])[0])
        else:
            state_vec = None
            value = float(decentral_value[0])
        turn = int(counts[slot])
        buffer.store(
            slot, turn, obs_vec, mask,
            act_val, float(logprob[0]), value,
            state=state_vec,
        )
        counts[slot] += 1
        pending[agent] = (slot, turn)
        env.step(act_val)

    # The rollout ends inside the final episode's termination dance; account
    # its KPIs now (env still holds the finished episode's orders until reset).
    if episode_ended:
        kpi = env.kpi_summary()
        served_total += int(kpi["served"])
        cancelled_total += int(kpi["cancelled"])
        empty_km_total += float(kpi["empty_km"])
        if kpi["avg_wait"] > 0:
            wait_times_all.append(kpi["avg_wait"])
        episodes += 1

    total_acts = max(acts_idle + acts_dispatch + acts_relocate, 1)
    avg_wait = float(np.mean(wait_times_all)) if wait_times_all else 0.0

    return {
        "episodes": episodes,
        "served_total": served_total,
        "cancelled_total": cancelled_total,
        "empty_km_total": empty_km_total,
        "avg_wait": avg_wait,
        "pct_dispatch": float(100.0 * acts_dispatch / total_acts),
        "pct_relocate": float(100.0 * acts_relocate / total_acts),
        "pct_idle": float(100.0 * acts_idle / total_acts),
    }


def _episode_kpis(env: RideGymAEC) -> dict[str, float]:
    """KPIs of the episode that just ended."""
    return env.kpi_summary()


def train(
    train_cfg: TrainConfig,
    env_name: str,
    nyc_window_limit: int | None,
    algo_name: str,
    resume_from: str | None = None,
) -> Path:
    env, env_cfg, scenario_id = _make_env(
        env_name, train_cfg.seed, nyc_window_limit, base_env_cfg=train_cfg.env_config
    )
    train_cfg.env_config = env_cfg
    train_cfg.scenario_id = scenario_id
    env.reset(seed=train_cfg.seed)

    # ---- setup directory structure ----
    if train_cfg.run_name:
        run_id = train_cfg.run_name
    else:
        timestamp = _dt.datetime.now().strftime("%m%d-%H%M%S")
        run_id = f"{algo_name}-{env_name}-{timestamp}"
    run_dir = train_cfg.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- policy & networks ----
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    any_agent = env.possible_agents[0]
    observation_dim = obs_dim(env.observation_space(any_agent))
    action_dim = env._action_dim
    use_state = algo_name == "mappo"
    sample_obs = env.observe(any_agent)
    zone_loc_dim = len(sample_obs["zone_location"].ravel())
    state_dim = (len(env.state()) + zone_loc_dim) if use_state else None
    if use_state:
        assert isinstance(train_cfg.algo_config, MAPPOConfig)
        learner: IPPO | MAPPO = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=train_cfg.algo_config,
        )
    else:
        assert isinstance(train_cfg.algo_config, IPPOConfig)
        learner = IPPO(
            observation_dim=observation_dim,
            action_dim=action_dim,
            config=train_cfg.algo_config,
        )

    # ---- run manifest: config + seed + scenario (AGENTS rules) -----------
    manifest = {
        **train_cfg.to_dict(),
        "algo": algo_name,
        "torch_device": str(learner.device),
        "num_zones": env._num_zones,
        "num_drivers": env._num_drivers,
        "episode_turns": _episode_turns(env_cfg),
    }
    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        learner.load_state_dict(
            torch.load(ckpt_path, map_location=learner.device, weights_only=True)
        )
        manifest["resumed_from"] = str(ckpt_path)
        print(f"[train] resumed weights from {ckpt_path} (fresh optimizer)")
    if state_dim is not None:
        manifest["state_dim"] = state_dim
    with open(run_dir / "config.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    metrics_path = run_dir / "metrics.jsonl"
    total_turns = train_cfg.algo_config.rollout_episodes * _episode_turns(env_cfg)
    buffer = RolloutBuffer(
        num_agents=env._num_zones,
        num_turns=total_turns,
        obs_dim=observation_dim,
        action_dim=action_dim,
        state_dim=state_dim,
    )

    print(f"[train] run={run_id} device={learner.device} zones={env._num_zones} "
          f"obs={observation_dim} actions={action_dim} "
          f"rollout_turns={total_turns}")
    print(f"[train] manifest: {json.dumps({k: manifest[k] for k in ('algo', 'scenario_id', 'seed')})}")

    for it in range(1, train_cfg.iterations + 1):
        kpis = collect_rollout(env, learner, buffer, total_turns, use_state=use_state)
        losses = learner.update(buffer)
        ep = max(kpis["episodes"], 1)
        sr = kpis["served_total"] / max(kpis["served_total"] + kpis["cancelled_total"], 1)
        reward_per_ep = float(buffer.rewards[buffer.filled].sum() / ep)
        record = {
            "iteration": it,
            "episodes": kpis["episodes"],
            "reward_per_ep": reward_per_ep,
            "mean_reward_per_turn": float(buffer.rewards[buffer.filled].mean()),
            "service_rate": sr,
            "served_per_ep": kpis["served_total"] / ep,
            "cancelled_per_ep": kpis["cancelled_total"] / ep,
            "avg_wait_min": kpis["avg_wait"],
            "empty_km_per_ep": kpis["empty_km_total"] / ep,
            "pct_dispatch": kpis["pct_dispatch"],
            "pct_relocate": kpis["pct_relocate"],
            "pct_idle": kpis["pct_idle"],
            **{f"loss_{k}": v for k, v in losses.items()},
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if train_cfg.verbose:
            print(
                f"[iter {it:3d}/{train_cfg.iterations}] eps={ep} | "
                f"served={record['served_per_ep']:5.1f}/ep (sr={sr*100:4.1f}%) | "
                f"wait={kpis['avg_wait']:4.2f}m | "
                f"empty={record['empty_km_per_ep']:5.1f}km | "
                f"acts: disp={kpis['pct_dispatch']:4.1f}% reloc={kpis['pct_relocate']:4.1f}% idle={kpis['pct_idle']:4.1f}% | "
                f"reward={reward_per_ep:+6.1f}/ep "
                f"V-loss={losses['value_loss']:6.2f} "
                f"H={losses['entropy']:.3f} EV={losses['explained_var']:+.2f}"
            )
        else:
            print(
                f"[iter {it:3d}/{train_cfg.iterations}] "
                f"served={record['served_per_ep']:4.0f}/ep (sr={sr*100:4.1f}%) | "
                f"wait={kpis['avg_wait']:4.2f}m | "
                f"empty={record['empty_km_per_ep']:5.1f}km | "
                f"reward={reward_per_ep:+6.1f}/ep"
            )

        if it % train_cfg.checkpoint_interval == 0 or it == train_cfg.iterations:
            torch.save(learner.state_dict(), run_dir / "checkpoint.pt")

    print(f"[train] done -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPPO/MAPPO on the fleet env")
    parser.add_argument("--env", choices=["bandra", "nyc"], default="bandra")
    parser.add_argument("--algo", choices=["ippo", "mappo"], default="ippo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Custom run directory name (default: {algo}-{env}-{timestamp})")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N iterations (default: 10)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print internal RL statistics (entropy, value loss, EV)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight for waiting time penalty (default: 1.0)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Weight for empty distance penalty (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Weight for served orders reward (default: 5.0)")
    parser.add_argument("--team-share", type=float, default=None,
                        help="Weight for shared global reward vs local credit (default: 0.2)")
    parser.add_argument("--cancel-penalty", type=float, default=None,
                        help="Penalty per cancelled order in zone (default: 2.0)")
    parser.add_argument("--nyc-window-limit", type=int, default=None,
                        help="Per-window order cap for NYC (sample-rate lever)")
    parser.add_argument("--rollout-episodes", type=int, default=None,
                        help="Episodes per PPO update (default: algo config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint.pt to warm-start from (fresh optimizer, "
                             "parent recorded in manifest)")
    parser.add_argument("--target-zone", type=int, default=None,
                        help="Target zone ID for parameterized demand injection")
    parser.add_argument("--target-zone-demand-pct", type=float, default=None,
                        help="Fraction or percentage of total orders originating from target zone (e.g. 0.08 or 8 for 8%%)")
    parser.add_argument("--num-drivers", type=int, default=None,
                        help="Number of active drivers in the fleet (default: 30 for Bandra)")
    parser.add_argument("--max-dispatch-radius", type=float, default=None,
                        help="Maximum pickup distance in km from zone centroid (default: 1.2 km)")
    parser.add_argument("--out-root", type=str, default="runs")
    args = parser.parse_args()

    algo_cfg: IPPOConfig | MAPPOConfig = (
        MAPPOConfig() if args.algo == "mappo" else IPPOConfig()
    )
    if args.rollout_episodes is not None:
        algo_cfg.rollout_episodes = args.rollout_episodes

    env_cfg = BandraConfig() if args.env == "bandra" else NYCConfig()
    if args.alpha is not None:
        env_cfg.alpha = args.alpha
    if args.beta is not None:
        env_cfg.beta = args.beta
    if args.gamma is not None:
        env_cfg.gamma = args.gamma
    if args.cancel_penalty is not None:
        env_cfg.cancel_penalty = args.cancel_penalty
    if args.team_share is not None:
        env_cfg.team_share = args.team_share
    if args.num_drivers is not None:
        env_cfg.num_drivers = args.num_drivers
    if args.max_dispatch_radius is not None:
        env_cfg.max_dispatch_radius_km = args.max_dispatch_radius
    if args.nyc_window_limit is not None:
        env_cfg.nyc_window_limit = args.nyc_window_limit
    if args.target_zone is not None:
        env_cfg.target_zone = args.target_zone
    if args.target_zone_demand_pct is not None:
        pct = args.target_zone_demand_pct
        if pct > 1.0:
            pct = pct / 100.0
        env_cfg.target_zone_demand_pct = pct

    train_cfg = TrainConfig(
        env_config=env_cfg,
        algo_config=algo_cfg,
        seed=args.seed,
        iterations=args.iterations,
        out_root=Path(args.out_root),
        run_name=args.name,
        checkpoint_interval=args.checkpoint_interval,
        verbose=args.verbose,
    )
    train(train_cfg, args.env, args.nyc_window_limit, args.algo,
          resume_from=args.resume)


if __name__ == "__main__":
    main()
