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
import copy
import dataclasses
import datetime as _dt
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl_fleet_rebalancing.agents.ippo import IPPO, IPPOConfig, RolloutBuffer
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.config import BandraConfig, EnvConfig, NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC
from marl_fleet_rebalancing.networks.encoder import obs_dim, obs_to_vector

EvalLearner = Any  # IPPO | MAPPO | scripted act()-compatible stand-in (tests)


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
    # Evaluate deterministically every N iterations and save checkpoint_best.pt
    # when validation service rate improves (0 = disabled). Selection must run
    # on the frozen, deterministic protocol — training SR is stochastic and
    # rewarded a collapsed policy 23 points above its frozen value.
    # Selection runs in a DEDICATED env on the held-out NYC val split (never
    # the training env: seeded val resets used to rewrite the training RNG
    # stream, see tests/test_training_validation.py).
    eval_interval: int = 0
    eval_episodes: int = 3
    eval_seed: int = 9000  # disjoint from the training seed stream
    eval_split: str = "val"  # held-out split used for selection (never "train")
    eval_window_limit: int | None = 500  # matches the training window budget
    verbose: bool = False
    num_workers: int = 1  # parallel rollout envs (1 = serial, current behavior)

    def to_dict(self) -> dict:
        return {
            "env_config": dataclasses.asdict(self.env_config),
            "algo_config": self.algo_config.to_dict(),
            "seed": self.seed,
            "iterations": self.iterations,
            "scenario_id": self.scenario_id,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "eval_seed": self.eval_seed,
            "eval_split": self.eval_split,
            "eval_window_limit": self.eval_window_limit,
            "num_workers": self.num_workers,
        }


def _episode_turns(cfg: EnvConfig) -> int:
    return int(round(cfg.horizon / cfg.dt))


def evaluate_deterministic(
    env: RideGymAEC,
    learner: EvalLearner,
    episodes: int,
    base_seed: int,
) -> dict:
    """Frozen-policy deterministic evaluation on fresh seeds (no learning).

    Used for best-checkpoint selection: training KPIs come from stochastic
    rollouts on the training stream, which made a mid-training policy look 23
    points better frozen than training (see docs/shift-results.md, IPPO
    collapse). Selection must use the same protocol as the final evaluation:
    deterministic argmax actions on unseen seeds.

    Runs on a DEDICATED validation env (see ``_make_validation_env``), never
    on the training env — resetting the training env with the val seeds
    rewrote its RNG stream and silently changed subsequent training episodes.
    The caller owns both envs; this function never resets anything after the
    val episodes.

    Sequential-decoding envs are evaluated with sequential semantics: zones
    are observed and stepped one at a time so each zone's observation sees
    the shadow-state effects of earlier zones' actions, matching the
    training-time rollout protocol. Batched envs keep the single batched
    forward pass.
    """
    served = cancelled = 0
    empty_km = 0.0
    wait_sum = 0.0
    sequential = bool(getattr(env, "_sequential_mode", False))
    for i in range(episodes):
        env.reset(seed=base_seed + i)
        while env.agents and not env._terminated:
            order = [int(z) for z in env._agent_order]
            if env.terminations.get(env.agent_selection, False):
                for _ in order:
                    if not env.agents:
                        break
                    env.step(None)
                continue

            if sequential:
                for slot in order:
                    agent = f"zone_{slot}"
                    obs_vec = obs_to_vector(env.observe(agent))
                    mask = env.action_mask(agent)
                    act, _, _ = learner.act(
                        obs_vec[None, :], mask[None, :], deterministic=True,
                    )
                    act_val = int(act[0])
                    env.step(act_val)
                    env._apply_action_sequentially(act_val, slot)
                continue

            obs_list, mask_list = [], []
            for slot in order:
                agent = f"zone_{slot}"
                obs_list.append(obs_to_vector(env.observe(agent)))
                mask_list.append(env.action_mask(agent))
            actions, _, _ = learner.act(
                np.stack(obs_list), np.stack(mask_list), deterministic=True,
            )
            for slot, act_v in zip(order, actions):
                env.step(int(act_v))
        kpi = env.kpi_summary()
        served += int(kpi["served"])
        cancelled += int(kpi["cancelled"])
        empty_km += float(kpi["empty_km"])
        wait_sum += float(kpi["avg_wait"])
    n = max(episodes, 1)
    return {
        "episodes": episodes,
        "service_rate": served / max(served + cancelled, 1),
        "served_per_ep": served / n,
        "empty_km_per_ep": empty_km / n,
        "avg_wait_min": wait_sum / n,
    }


def _make_validation_env(
    train_cfg: TrainConfig,
    seed: int,
) -> RideGymAEC:
    """Build the dedicated selection env: same sim, held-out demand split.

    Deep-copies the training env config and overrides ONLY the demand split
    (and window limit if set), so selection sees the held-out ``val`` windows
    under an identical fleet/network/reward. A fresh env instance also means
    selection can never perturb the training env's RNG stream or window
    cursor (regression-tested: tests/test_training_validation.py).
    """
    val_cfg = copy.deepcopy(train_cfg.env_config)
    val_cfg.nyc_split = train_cfg.eval_split
    if (
        train_cfg.eval_window_limit is not None
        and hasattr(val_cfg, "nyc_window_limit")
    ):
        val_cfg.nyc_window_limit = train_cfg.eval_window_limit
    return RideGymAEC(config=val_cfg, seed=seed)


def selection_record(
    iteration: int,
    val_service_rate: float,
    eval_cfg: dict,
) -> dict:
    """Provenance block for a selected checkpoint: iteration + how it was picked.

    Written to ``best_checkpoint.json`` so every best checkpoint is auditable
    (which split, which seeds, how many episodes) without reading code.
    """
    return {
        "iteration": iteration,
        "val_service_rate": val_service_rate,
        "val_split": eval_cfg["val_split"],
        "eval_seed": eval_cfg["eval_seed"],
        "eval_episodes": eval_cfg["eval_episodes"],
    }


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
        # D22: only override the config default when the CLI flag is given;
        # passing None unconditionally used to clobber NYCConfig's default.
        if nyc_window_limit is not None:
            env_cfg = NYCConfig(nyc_window_limit=nyc_window_limit)
        else:
            env_cfg = NYCConfig()
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

    **Optimized**: all zone agents are batched into a single forward pass
    per turn instead of calling ``learner.act()`` once per zone. This keeps
    the GPU busy processing batch_size=N_zones instead of batch_size=1.

    **Sequential mode** (when env._sequential_mode is True): zones decide
    actions one at a time, each seeing the effects of prior zones' actions
    within the same turn. This improves coordination but disables batching.

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
    if total_turns <= 0:
        raise ValueError(f"collect_rollout requires total_turns > 0, got {total_turns}")
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
    episode_ended = False

    _sequential = getattr(env, "_sequential_mode", False)

    if _sequential:
        # Sequential decoding: zone-by-zone, no batching
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

            sel_order = [int(z) for z in env._agent_order]

            if not any(counts[s] < total_turns for s in sel_order):
                if not episode_ended:
                    kpi = env.kpi_summary()
                    served_total += int(kpi["served"])
                    cancelled_total += int(kpi["cancelled"])
                    empty_km_total += float(kpi["empty_km"])
                    if kpi["avg_wait"] > 0:
                        wait_times_all.append(kpi["avg_wait"])
                    episodes += 1
                for agent in list(env.agents):
                    if agent in pending:
                        s_slot, s_turn = pending.pop(agent)
                        buffer.close(
                            s_slot, s_turn,
                            float(env.rewards.get(agent, 0.0)),
                            done=bool(
                                env.terminations.get(agent, False)
                                or env.truncations.get(agent, False)
                            ),
                        )
                    env.step(None)
                episode_ended = False
                continue

            # Sequential zone-by-zone decoding
            for slot in sel_order:
                agent = f"zone_{slot}"
                if agent in pending:
                    s_slot, s_turn = pending.pop(agent)
                    buffer.close(
                        s_slot, s_turn,
                        float(env.rewards.get(agent, 0.0)),
                        done=bool(
                            env.terminations.get(agent, False)
                            or env.truncations.get(agent, False)
                        ),
                    )
                if (
                    counts[slot] >= total_turns
                    or env.terminations.get(agent, False)
                    or env.truncations.get(agent, False)
                ):
                    if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                        episode_ended = True
                    env.step(None)
                    continue

                obs_dict = env.observe(agent)
                obs_vec = obs_to_vector(obs_dict)
                mask = env.action_mask(agent)

                if use_state:
                    g_state = env.state()
                    state_vec = np.concatenate(
                        [g_state, np.asarray(
                            obs_dict["zone_location"].ravel(), dtype=np.float32,
                        )],
                    )
                    assert isinstance(learner, MAPPO)
                    _act_t, _logp_t, _ = learner.act(
                        obs_vec[None, :], mask[None, :],
                    )
                    _val_t = learner.value(state_vec[None, :])
                    act_val = int(_act_t[0])
                    logprob = float(_logp_t[0])
                    value = float(_val_t[0])
                else:
                    _act_t, _logp_t, _val_t = learner.act(
                        obs_vec[None, :], mask[None, :],
                    )
                    act_val = int(_act_t[0])
                    logprob = float(_logp_t[0])
                    value = float(_val_t[0])

                if act_val == 0:
                    acts_idle += 1
                elif 1 <= act_val <= env._max_pending:
                    acts_dispatch += 1
                else:
                    acts_relocate += 1

                turn = int(counts[slot])
                buffer.store(
                    slot, turn, obs_vec, mask,
                    act_val, logprob, value,
                    state=(state_vec if (
                        use_state and state_vec is not None) else None),
                )
                counts[slot] += 1
                pending[agent] = (slot, turn)
                env.step(act_val)
                # In sequential mode, apply shadow state update immediately
                # so the next zone's observe() sees the effects of this action
                if _sequential:
                    env._apply_action_sequentially(act_val, slot)
    else:

        # Per-agent observation/mask buffers for batch collection.
        # Each slot is a single observation vector (obs_dim,) not a full
        # episode rollout, so allocate (num_zones, obs_dim).
        _obs_dim = obs_dim  # for the closure below
        obs_batch = np.empty(
            (num_zones, buffer.obs.shape[2]), dtype=np.float32,
        )
        mask_batch = np.empty(
            (num_zones, buffer.action_mask.shape[2]), dtype=bool,
        )
        # MAPPO centralized-state batch (D18 fix): per-agent state = global
        # state + own zone_location, matching train.py's state_dim. IPPO leaves
        # this None and uses decentralized actor values.
        if use_state:
            assert buffer.states is not None, (
                "use_state=True requires a RolloutBuffer with state_dim (MAPPO)."
            )
            state_batch: np.ndarray | None = np.empty(
                (num_zones, buffer.states.shape[2]), dtype=np.float32,
            )
        else:
            state_batch = None

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

            # Selection-ordered turn (AEC correctness): env.step() records the
            # action for the CURRENTLY SELECTED agent, so a turn's 25 steps must
            # walk env._agent_order exactly. The pre-batching loop stepped
            # agent_selection directly; batching must preserve that walk, or zone
            # actions silently execute on the wrong zones (found by audit).
            sel_order = [int(z) for z in env._agent_order]

            if not any(counts[s] < total_turns for s in sel_order):
                # All turns collected but agents remain: drain without acting.
                # The episode's KPIs are accounted BEFORE the drain flush below
                # (an extra RideGym step) can pollute them. Pending delayed
                # rewards are closed from the per-agent reward tables directly.
                if not episode_ended:
                    kpi = env.kpi_summary()
                    served_total += int(kpi["served"])
                    cancelled_total += int(kpi["cancelled"])
                    empty_km_total += float(kpi["empty_km"])
                    if kpi["avg_wait"] > 0:
                        wait_times_all.append(kpi["avg_wait"])
                    episodes += 1
                for agent in list(env.agents):
                    if agent in pending:
                        s_slot, s_turn = pending.pop(agent)
                        buffer.close(
                            s_slot, s_turn,
                            float(env.rewards.get(agent, 0.0)),
                            done=bool(
                                env.terminations.get(agent, False)
                                or env.truncations.get(agent, False)
                            ),
                        )
                    env.step(None)
                episode_ended = False
                continue

            # Phase 1: close delayed rewards + observe. No env steps here, so the
            # Per-agent tables are read directly (D18 fix): env.last() reports
            # only the currently-selected agent, so calling it per zone
            # misattributes rewards and done-flags. Skipping the second
            # env.last() build also removes a redundant observation per agent.
            # MAPPO (use_state): values come from the centralized critic over
            # per-agent global states; the actor head's decentralized values are
            # discarded, per MAPPO.act's documented contract.
            g_state = env.state() if use_state else None
            dead: set[int] = set()
            live_idx: list[int] = []
            for i, slot in enumerate(sel_order):
                agent = f"zone_{slot}"
                if agent in pending:
                    s_slot, s_turn = pending.pop(agent)
                    buffer.close(
                        s_slot, s_turn,
                        float(env.rewards.get(agent, 0.0)),
                        done=bool(
                            env.terminations.get(agent, False)
                            or env.truncations.get(agent, False)
                        ),
                    )
                if (
                    counts[slot] >= total_turns
                    or env.terminations.get(agent, False)
                    or env.truncations.get(agent, False)
                ):
                    if env.terminations.get(agent, False) or env.truncations.get(agent, False):
                        episode_ended = True
                        dead.add(slot)
                    continue
                obs_dict = env.observe(agent)
                obs_batch[i] = obs_to_vector(obs_dict)
                mask_batch[i] = env.action_mask(agent)
                if use_state:
                    assert g_state is not None and state_batch is not None
                    state_batch[i] = np.concatenate(
                        [g_state, np.asarray(
                            obs_dict["zone_location"].ravel(), dtype=np.float32,
                        )],
                    )
                live_idx.append(i)

            # Single batched forward pass for the live zones (if any).
            # act_of maps zone slot -> (batch position, action, logprob, value).
            act_of: dict[int, tuple[int, int, float, float]] = {}
            if live_idx:
                _act_t, _logp_t, _val_t = learner.act(
                    obs_batch[live_idx], mask_batch[live_idx],
                )
                act_values = _act_t.astype(np.int64)
                logprobs = _logp_t.astype(np.float64)
                if use_state:
                    # Centralized critic values (D18 fix).
                    assert isinstance(learner, MAPPO)
                    assert state_batch is not None
                    values = learner.value(state_batch[live_idx]).astype(np.float64)
                else:
                    values = _val_t.astype(np.float64)
                for j, i in enumerate(live_idx):
                    act_of[sel_order[i]] = (
                        i, int(act_values[j]), float(logprobs[j]), float(values[j]),
                    )

            # Phase 2: walk the selection order, stepping each agent exactly
            # once. Dead/completed zones step None (no-op that keeps the walk
            # aligned); live zones step their batched action. This preserves the
            # pre-batching attribute-every-action-to-the-selected-agent semantics.
            for slot in sel_order:
                agent = f"zone_{slot}"
                if slot in act_of:
                    i, act_val, logprob, value = act_of[slot]
                    turn = int(counts[slot])

                    if act_val == 0:
                        acts_idle += 1
                    elif 1 <= act_val <= env._max_pending:
                        acts_dispatch += 1
                    else:
                        acts_relocate += 1

                    buffer.store(
                        slot, turn, obs_batch[i], mask_batch[i],
                        act_val, logprob, value,
                        state=(state_batch[i] if (
                            use_state and state_batch is not None) else None),
                    )
                    counts[slot] += 1
                    pending[agent] = (slot, turn)
                    env.step(act_val)
                else:
                    env.step(None)

        # Account final episode's KPIs
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


# ---------------------------------------------------------------------------
# Parallel rollout workers.
#
# Parallelism is across EPISODES, never across zones within a turn: zones in
# one turn share a single RideGym instance (claimed-order set, FIFO driver
# queues, one _flush_turn, zone reward attribution) and must stay sequential.
# Each worker owns an independent RideGymAEC copy, runs the standard
# sequential AEC loop (with the same per-turn zone batching as
# collect_rollout), and returns its buffer shard. The parent concatenates
# shards and runs the PPO update. Learning semantics are unchanged — same
# MDP, just more diverse samples per update.
# Workers run on CPU (spawn context; no CUDA in forks) and receive the
# current policy weights each iteration.
# ---------------------------------------------------------------------------

_WORKER_CTX: dict = {}


def _worker_init(
    env_name: str,
    env_cfg: EnvConfig,
    algo_name: str,
    algo_cfg: IPPOConfig | MAPPOConfig,
    observation_dim: int,
    action_dim: int,
    state_dim: int | None,
    use_state: bool,
) -> None:
    """Build one persistent env + CPU learner per worker process."""
    # One thread per worker: with N workers on N cores, torch's default
    # multi-threading oversubscribes and slows everything down.
    torch.set_num_threads(1)
    env, _, _ = _make_env(env_name, 0, None, base_env_cfg=env_cfg)
    cpu = torch.device("cpu")
    if use_state:
        assert isinstance(algo_cfg, MAPPOConfig)
        learner: IPPO | MAPPO = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim or 0,
            action_dim=action_dim,
            config=algo_cfg,
            device=cpu,
        )
    else:
        assert isinstance(algo_cfg, IPPOConfig)
        learner = IPPO(
            observation_dim=observation_dim,
            action_dim=action_dim,
            config=algo_cfg,
            device=cpu,
        )
    _WORKER_CTX.update(
        env=env,
        learner=learner,
        use_state=use_state,
        observation_dim=observation_dim,
        action_dim=action_dim,
        state_dim=state_dim,
    )


def _worker_collect(job: tuple[dict, int, int]) -> dict:
    """Run one rollout shard. Job = (policy state_dict, episodes, seed)."""
    state_dict, episodes, seed = job
    env: RideGymAEC = _WORKER_CTX["env"]
    learner = _WORKER_CTX["learner"]
    use_state: bool = _WORKER_CTX["use_state"]
    episode_turns = _episode_turns(env._cfg)
    total_turns = episodes * episode_turns
    if episodes <= 0:
        # Defensive: collect_rollout's loop assumes total_turns > 0
        # (empty env.agents with counts at 0 would spin forever).
        return {"arrays": {}, "kpis": _merge_kpis([]), "shard_turns": 0}
    learner.load_state_dict(state_dict)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    buffer = RolloutBuffer(
        num_agents=env._num_zones,
        num_turns=total_turns,
        obs_dim=_WORKER_CTX["observation_dim"],
        action_dim=_WORKER_CTX["action_dim"],
        state_dim=_WORKER_CTX["state_dim"],
    )
    env.reset(seed=seed)
    kpis = collect_rollout(env, learner, buffer, total_turns, use_state=use_state)
    arrays = {
        "obs": buffer.obs,
        "action_mask": buffer.action_mask,
        "actions": buffer.actions,
        "logprobs": buffer.logprobs,
        "values": buffer.values,
        "rewards": buffer.rewards,
        "dones": buffer.dones,
        "filled": buffer.filled,
        "states": buffer.states,
    }
    return {"arrays": arrays, "kpis": kpis, "shard_turns": total_turns}


def _split_episodes(n_episodes: int, n_workers: int) -> list[int]:
    """Split episodes as evenly as possible (first workers get the remainder)."""
    base, rem = divmod(n_episodes, n_workers)
    return [base + (1 if w < rem else 0) for w in range(n_workers)]


def _merge_kpis(shard_kpis: list[dict]) -> dict:
    """Combine per-worker KPI dicts, weighting averages by episode count."""
    episodes = sum(k["episodes"] for k in shard_kpis)
    ep = max(episodes, 1)
    served = sum(k["served_total"] for k in shard_kpis)
    cancelled = sum(k["cancelled_total"] for k in shard_kpis)
    empty = sum(k["empty_km_total"] for k in shard_kpis)
    avg_wait = sum(k["avg_wait"] * k["episodes"] for k in shard_kpis) / ep
    pct = lambda key: sum(k[key] * k["episodes"] for k in shard_kpis) / ep
    return {
        "episodes": episodes,
        "served_total": served,
        "cancelled_total": cancelled,
        "empty_km_total": empty,
        "avg_wait": avg_wait,
        "pct_dispatch": pct("pct_dispatch"),
        "pct_relocate": pct("pct_relocate"),
        "pct_idle": pct("pct_idle"),
    }


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
        state_dim_: int = state_dim if state_dim is not None else 0
        learner: IPPO | MAPPO = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim_,
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

    # ---- run manifest: config + seed + scenario (AGENTS rules) ----------
    manifest = {
        **train_cfg.to_dict(),
        "algo": algo_name,
        "torch_device": str(learner.device),
        "num_zones": env._num_zones,
        "num_drivers": env._num_drivers,
        "episode_turns": _episode_turns(env_cfg),
        "num_workers": train_cfg.num_workers,
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
          f"rollout_turns={total_turns} workers={train_cfg.num_workers}")
    print(f"[train] manifest: {json.dumps({k: manifest[k] for k in ('algo', 'scenario_id', 'seed')})}")

    pool = None
    best_val_sr = -1.0
    best_val_it = -1
    val_env = (
        _make_validation_env(train_cfg, seed=train_cfg.eval_seed)
        if train_cfg.eval_interval > 0
        else None
    )
    if train_cfg.num_workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=train_cfg.num_workers,
            initializer=_worker_init,
            initargs=(
                env_name, env_cfg, algo_name, train_cfg.algo_config,
                observation_dim, action_dim, state_dim, use_state,
            ),
        )

    try:
        for it in range(1, train_cfg.iterations + 1):
            t_roll0 = time.perf_counter()
            if pool is None:
                kpis = collect_rollout(env, learner, buffer, total_turns, use_state=use_state)
                t_merge1 = t_roll1 = time.perf_counter()
            else:
                shards = _split_episodes(
                    train_cfg.algo_config.rollout_episodes, train_cfg.num_workers,
                )
                state_dict = learner.state_dict()
                jobs = [
                    (state_dict, ep, train_cfg.seed + it * train_cfg.num_workers + w)
                    for w, ep in enumerate(shards)
                    if ep > 0
                ]
                results = pool.map(_worker_collect, jobs)
                t_roll1 = time.perf_counter()
                offset = 0
                for res in results:
                    t = res["shard_turns"]
                    arr = res["arrays"]
                    buffer.obs[:, offset:offset + t] = arr["obs"]
                    buffer.action_mask[:, offset:offset + t] = arr["action_mask"]
                    buffer.actions[:, offset:offset + t] = arr["actions"]
                    buffer.logprobs[:, offset:offset + t] = arr["logprobs"]
                    buffer.values[:, offset:offset + t] = arr["values"]
                    buffer.rewards[:, offset:offset + t] = arr["rewards"]
                    buffer.dones[:, offset:offset + t] = arr["dones"]
                    buffer.filled[:, offset:offset + t] = arr["filled"]
                    if buffer.states is not None and arr["states"] is not None:
                        buffer.states[:, offset:offset + t] = arr["states"]
                    offset += t
                kpis = _merge_kpis([res["kpis"] for res in results])
                t_merge1 = time.perf_counter()
                # Drop the multi-GB worker payloads BEFORE the update: the
                # merged buffer + update temporaries already fill RAM, and
                # paging here inflated update 3x in benchmarks (sys 50s).
                del results
                del jobs
            losses = learner.update(buffer)
            t_upd1 = time.perf_counter()
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
            print(
                f"[timing] rollout={t_roll1 - t_roll0:5.1f}s | "
                f"merge={t_merge1 - t_roll1:5.1f}s | "
                f"update={t_upd1 - t_merge1:5.1f}s"
            )

            if it % train_cfg.checkpoint_interval == 0 or it == train_cfg.iterations:
                torch.save(learner.state_dict(), run_dir / "checkpoint.pt")
                # Also keep a versioned copy: checkpoint.pt is overwritten on
                # every interval, so a policy that collapses late in training
                # (e.g. both PPO variants under D25's reward, see
                # docs/shift-results.md) would otherwise leave nothing behind
                # but its final, degraded weights.
                torch.save(
                    learner.state_dict(),
                    run_dir / f"checkpoint_it{it:04d}.pt",
                )

            # ---- validation-based best-checkpoint tracking ---------------
            if train_cfg.eval_interval > 0 and (
                it % train_cfg.eval_interval == 0 or it == train_cfg.iterations
            ):
                val = evaluate_deterministic(
                    val_env,
                    learner,
                    episodes=train_cfg.eval_episodes,
                    base_seed=train_cfg.eval_seed,
                )
                val_sr = val["service_rate"]
                record["val_service_rate"] = val_sr
                record["val_episodes"] = val["episodes"]
                # Rewrite the just-appended metrics line with the val fields
                # included, so metrics.jsonl stays one JSON object per line.
                metrics_lines = metrics_path.read_text().splitlines()
                metrics_lines[-1] = json.dumps(record)
                metrics_path.write_text("\n".join(metrics_lines) + "\n")

                if val_sr > best_val_sr:
                    best_val_sr = val_sr
                    best_val_it = it
                    torch.save(learner.state_dict(), run_dir / "checkpoint_best.pt")
                    with open(run_dir / "best_checkpoint.json", "w") as f:
                        json.dump(
                            selection_record(
                                iteration=it,
                                val_service_rate=val_sr,
                                eval_cfg={
                                    "val_split": train_cfg.eval_split,
                                    "eval_seed": train_cfg.eval_seed,
                                    "eval_episodes": train_cfg.eval_episodes,
                                },
                            ),
                            f, indent=2,
                        )
                    marker = " (new best)"
                else:
                    marker = ""
                print(
                    f"[val it {it:3d}] SR={val_sr*100:4.1f}% | "
                    f"empty={val['empty_km_per_ep']:5.1f}km | "
                    f"wait={val['avg_wait_min']:4.2f}m"
                    f" | best={best_val_sr*100:4.1f}% @ it{best_val_it}{marker}"
                )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if val_env is not None:
        val_env.close()

    if train_cfg.eval_interval > 0 and best_val_it > 0:
        print(
            f"[train] best validation SR {best_val_sr*100:.1f}% at iteration "
            f"{best_val_it} -> {run_dir / 'checkpoint_best.pt'}"
        )
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
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Deterministic validation eval + best-checkpoint "
                             "tracking every N iterations (0 = disabled)")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Validation episodes per eval point (default: 3)")
    parser.add_argument("--eval-split", type=str, default="val",
                        choices=["val", "test"],
                        help="Held-out split used for checkpoint selection "
                             "(default: val). Never 'train' — selecting on the "
                             "training split is not validation.")
    parser.add_argument("--eval-seed", type=int, default=9000,
                        help="Base seed for validation episodes (default: 9000, "
                             "disjoint from the training seed stream)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print internal RL statistics (entropy, value loss, EV)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight for waiting time penalty (default: 0.2 for NYC, 0.1 for Bandra)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Weight for empty distance penalty (default: 0.15 for NYC, 0.1 for Bandra)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Weight for served orders reward (default: 1.0 for NYC, 5.0 for Bandra)")
    parser.add_argument("--team-share", type=float, default=None,
                        help="Weight for shared global reward vs local credit (default: 0.2)")
    parser.add_argument("--cancel-penalty", type=float, default=None,
                        help="Penalty per cancelled order in zone (default: 1.0 for NYC, 0.5 for Bandra)")
    parser.add_argument("--prospective-bonus", type=float, default=None,
                        help="Relocation arrival bonus scale (D28; default: 2.0 for NYC, 0.0 for Bandra)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="PPO entropy bonus coefficient (default: 0.01)")
    parser.add_argument("--nyc-window-limit", type=int, default=None,
                        help="Per-window order cap for NYC (default: 500 for NYC)")
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
                        help="Number of active drivers in the fleet (default: 100 for NYC, 30 for Bandra)")
    parser.add_argument("--max-dispatch-radius", type=float, default=None,
                        help="Maximum pickup distance in km from zone centroid (default: 2.0 km)")
    parser.add_argument("--relocation-scope", choices=["anywhere", "neighbor"], default=None,
                        help="Relocation range (default: anywhere; 'neighbor' "
                             "restricts to adjacent zones)")
    parser.add_argument("--out-root", type=str, default="runs")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel rollout envs (default: 1 = serial). "
                             "Parallelism is across episodes; zones within a "
                             "turn stay sequential. Use ~n_cores-1.")
    parser.add_argument("--sequential-decoding", action="store_true", default=False,
                        help="Enable sequential zone-by-zone decoding within a turn "
                             "(each zone sees prior zones' actions). Disables batching.")
    parser.add_argument("--max-relocations-per-turn", type=int, default=0,
                        help="Max relocations per zone per turn (0 = unlimited)")
    args = parser.parse_args()

    algo_cfg: IPPOConfig | MAPPOConfig = (
        MAPPOConfig() if args.algo == "mappo" else IPPOConfig()
    )
    if args.rollout_episodes is not None:
        algo_cfg.rollout_episodes = args.rollout_episodes
    if args.entropy_coef is not None:
        algo_cfg.entropy_coef = args.entropy_coef

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
    if args.prospective_bonus is not None:
        env_cfg.prospective_bonus_scale = args.prospective_bonus
    if args.num_drivers is not None:
        env_cfg.num_drivers = args.num_drivers
    if args.max_dispatch_radius is not None:
        env_cfg.max_dispatch_radius_km = args.max_dispatch_radius
    if args.relocation_scope is not None:
        env_cfg.neighbor_relocation_only = (args.relocation_scope == "neighbor")
    if args.nyc_window_limit is not None:
        env_cfg.nyc_window_limit = args.nyc_window_limit
    if args.target_zone is not None:
        env_cfg.target_zone = args.target_zone
    if args.target_zone_demand_pct is not None:
        pct = args.target_zone_demand_pct
        if pct > 1.0:
            pct = pct / 100.0
        env_cfg.target_zone_demand_pct = pct
    if args.sequential_decoding:
        env_cfg.sequential_decoding = True
    if args.max_relocations_per_turn > 0:
        env_cfg.max_relocations_per_turn = args.max_relocations_per_turn

    train_cfg = TrainConfig(
        env_config=env_cfg,
        algo_config=algo_cfg,
        seed=args.seed,
        iterations=args.iterations,
        out_root=Path(args.out_root),
        run_name=args.name,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_split=args.eval_split,
        eval_seed=args.eval_seed,
        eval_window_limit=env_cfg.nyc_window_limit
        if hasattr(env_cfg, "nyc_window_limit") else None,
        verbose=args.verbose,
        num_workers=args.num_workers,
    )
    train(train_cfg, args.env, args.nyc_window_limit, args.algo,
          resume_from=args.resume)


if __name__ == "__main__":
    main()
