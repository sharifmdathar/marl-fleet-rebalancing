"""Training entry point for IPPO/MAPPO on the zone-controller fleet env.

Run (from repo root):
    uv run python -m marl_fleet_rebalancing.experiments.train \
        --env bandra --algo mappo --seed 0 --iterations 20

Every run logs: full config (env + algo), seed(s), git commit hash, demand
scenario ID, and per-iteration KPIs (AGENTS.md experiment rules). Checkpoints
are written to the run dir but never committed (git-ignored).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import subprocess
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

    def to_dict(self) -> dict:
        return {
            "env_config": dataclasses.asdict(self.env_config),
            "algo_config": self.algo_config.to_dict(),
            "seed": self.seed,
            "iterations": self.iterations,
            "scenario_id": self.scenario_id,
        }


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _episode_turns(cfg: EnvConfig) -> int:
    return int(round(cfg.horizon / cfg.dt))


def _make_env(name: str, seed: int, nyc_window_limit: int | None) -> tuple[RideGymAEC, EnvConfig, str]:
    if name == "bandra":
        env_cfg: EnvConfig = BandraConfig()
        scenario = "S0-base-bandra"
    elif name == "nyc":
        env_cfg = NYCConfig(nyc_window_limit=nyc_window_limit)
        scenario = "S0-base-nyc-train"
    else:
        raise ValueError(f"Unknown env: {name!r} (expected 'bandra' or 'nyc')")
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
    episode_ended = False  # an episode finished; KPIs not yet accounted

    while counts.min() < total_turns or env.agents:
        if not env.agents:
            if episode_ended:
                served, cancelled = _episode_kpis(env)
                served_total += served
                cancelled_total += cancelled
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
        state_vec = env.state() if use_state else None
        if use_state:
            value = float(learner.value(state_vec[None, :])[0])
        else:
            value = float(decentral_value[0])
        turn = int(counts[slot])
        buffer.store(
            slot, turn, obs_vec, mask,
            int(action[0]), float(logprob[0]), value,
            state=state_vec,
        )
        counts[slot] += 1
        pending[agent] = (slot, turn)
        env.step(int(action[0]))

    # The rollout ends inside the final episode's termination dance; account
    # its KPIs now (env still holds the finished episode's orders until reset).
    if episode_ended:
        served, cancelled = _episode_kpis(env)
        served_total += served
        cancelled_total += cancelled
        episodes += 1

    return {
        "episodes": episodes,
        "served_total": served_total,
        "cancelled_total": cancelled_total,
    }


def _episode_kpis(env: RideGymAEC) -> tuple[int, int]:
    """KPIs of the episode that just ended (from the env's final order set)."""
    from ride_gym.enums import OrderStatus

    served = sum(
        1 for o in env.env.orders.values() if o.status == OrderStatus.COMPLETED
    )
    cancelled = sum(
        1 for o in env.env.orders.values() if o.status == OrderStatus.CANCELLED
    )
    return served, cancelled


def train(
    train_cfg: TrainConfig,
    env_name: str,
    nyc_window_limit: int | None,
    algo_name: str = "ippo",
) -> Path:
    run_id = (
        f"{algo_name}-{train_cfg.scenario_id}-seed{train_cfg.seed}-"
        f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = train_cfg.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env, env_cfg, scenario = _make_env(env_name, train_cfg.seed, nyc_window_limit)
    train_cfg.scenario_id = scenario
    env.reset(seed=train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    any_agent = env.possible_agents[0]
    observation_dim = obs_dim(env.observation_space(any_agent))
    action_dim = env.action_space(any_agent).n
    use_state = algo_name == "mappo"
    state_dim = len(env.state()) if use_state else None
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

    # ---- run manifest: config + seed + git hash + scenario (AGENTS rules) --
    manifest = {
        **train_cfg.to_dict(),
        "algo": algo_name,
        "git_commit": _git_hash(),
        "torch_device": str(learner.device),
        "num_zones": env._num_zones,
        "num_drivers": env._num_drivers,
        "episode_turns": _episode_turns(env_cfg),
    }
    if state_dim is not None:
        manifest["state_dim"] = state_dim
    with open(run_dir / "config.json", "w") as f:
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
    print(f"[train] manifest: {json.dumps({k: manifest[k] for k in ('algo', 'scenario_id', 'seed', 'git_commit')})}")

    for it in range(1, train_cfg.iterations + 1):
        kpis = collect_rollout(env, learner, buffer, total_turns, use_state=use_state)
        losses = learner.update(buffer)
        ep = max(kpis["episodes"], 1)
        record = {
            "iteration": it,
            "episodes": kpis["episodes"],
            "mean_reward_per_turn": float(buffer.rewards[buffer.filled].mean()),
            "service_rate": kpis["served_total"]
            / max(kpis["served_total"] + kpis["cancelled_total"], 1),
            "served_total": kpis["served_total"],
            "cancelled_total": kpis["cancelled_total"],
            **{f"loss_{k}": v for k, v in losses.items()},
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[iter {it:3d}] eps={kpis['episodes']} "
            f"served={kpis['served_total']:5d} "
            f"cancelled={kpis['cancelled_total']:6d} "
            f"service_rate={record['service_rate']:.3f} "
            f"R/turn={record['mean_reward_per_turn']:+.3f} "
            f"V-loss={losses['value_loss']:.3f} "
            f"H={losses['entropy']:.3f} EV={losses['explained_var']:+.2f}"
        )
        torch.save(learner.state_dict(), run_dir / "checkpoint.pt")

    print(f"[train] done -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPPO/MAPPO on the fleet env")
    parser.add_argument("--env", choices=["bandra", "nyc"], default="bandra")
    parser.add_argument("--algo", choices=["ippo", "mappo"], default="ippo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--nyc-window-limit", type=int, default=None,
                        help="Per-window order cap for NYC (sample-rate lever)")
    parser.add_argument("--rollout-episodes", type=int, default=None,
                        help="Episodes per PPO update (default: algo config)")
    parser.add_argument("--out-root", type=str, default="runs")
    args = parser.parse_args()

    algo_cfg: IPPOConfig | MAPPOConfig = (
        MAPPOConfig() if args.algo == "mappo" else IPPOConfig()
    )
    if args.rollout_episodes is not None:
        algo_cfg.rollout_episodes = args.rollout_episodes
    train_cfg = TrainConfig(
        env_config=BandraConfig() if args.env == "bandra" else NYCConfig(),
        algo_config=algo_cfg,
        seed=args.seed,
        iterations=args.iterations,
        out_root=Path(args.out_root),
    )
    train(train_cfg, args.env, args.nyc_window_limit, args.algo)


if __name__ == "__main__":
    main()
