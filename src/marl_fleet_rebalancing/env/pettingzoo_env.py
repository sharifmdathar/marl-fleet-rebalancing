"""PettingZoo AEC wrapper around RideGym.

Exposes a per-driver flat observation Dict and a Discrete action space so that
IPPO/MAPPO can drive the simulation through the standard PettingZoo API.

All numeric constants come from EnvConfig — no magic numbers here.
Design decisions are documented in docs/pettingzoo-design-decisions.md.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv

from ride_gym import RidePoolEnv
from ride_gym.order_generator import OSMnxOrderGenerator
from ride_gym.osmnx_network import OSMnxNetwork

from marl_fleet_rebalancing.config import EnvConfig

STATUS_TO_IDX = {"idle": 0, "to_pickup": 1, "to_dropoff": 2, "relocating": 3}
NUM_STATUS = len(STATUS_TO_IDX)


def _to_float(x: Any) -> float:
    return float(x) if x is not None else 0.0


def _normalize_xy(
    x: float, y: float, area_min: tuple[float, float], area_range: tuple[float, float],
) -> tuple[float, float]:
    return (
        (x - area_min[0]) / max(area_range[0], 1e-9),
        (y - area_min[1]) / max(area_range[1], 1e-9),
    )


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


class RideGymAEC(AECEnv):
    """PettingZoo AEC environment wrapping a RidePoolEnv.

    Parameters
    ----------
    config:
        Experiment configuration (fleet size, horizon, reward weights, etc.)
    seed:
        RNG seed for the action-order shuffle.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = EnvConfig()

        self._cfg = config
        self._num_drivers = config.num_drivers
        self.possible_agents = [f"driver_{i}" for i in range(config.num_drivers)]

        network = OSMnxNetwork(
            graph_path=config.network_path, speed_kmh=config.speed_kmh,
        )
        order_gen = OSMnxOrderGenerator(
            network=network,
            horizon=config.horizon,
            num_orders=config.num_orders,
            arrival="poisson",
            max_party_size=config.max_party_size,
        )
        self.env = RidePoolEnv(
            num_drivers=config.num_drivers,
            driver_capacity=config.driver_capacity,
            dt=config.dt,
            horizon=config.horizon,
            order_timeout=config.order_timeout,
            order_generator=order_gen,
            road_network=network,
        )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        area = self.env.area
        self._area_min: tuple[float, float] = (area[0], area[2])
        if area[1] > area[0] and area[3] > area[2]:
            self._area_range: tuple[float, float] = (area[1] - area[0], area[3] - area[2])
        else:
            self._area_range = (100.0, 100.0)

        net_bounds = self.env.network.bounds
        self._geo_min: tuple[float, float] = (net_bounds[0], net_bounds[1])
        self._geo_range: tuple[float, float] = (
            net_bounds[2] - net_bounds[0],
            net_bounds[3] - net_bounds[1],
        )

        # Dynamic from RideGym — adapts to any map's zone count
        self._num_zones = len(self.env.relocation_points)

        # From config — should be measured from actual data for each map
        self._max_pending = config.max_pending

        # Action space: idle (1) + dispatch to pending order (max_pending) + relocate to zone (num_zones)
        self._action_dim = 1 + self._max_pending + self._num_zones

        self.observation_spaces: dict[str, Dict] = {
            a: self._make_obs_space() for a in self.possible_agents
        }
        self.action_spaces: dict[str, Discrete] = {
            a: Discrete(self._action_dim) for a in self.possible_agents
        }

        self._last_obs: dict[int, dict] | None = None
        self._current_pending: list[dict] = []
        self._agent_order = np.arange(self._num_drivers)
        self._actions_this_turn: dict[str, int] = {}
        self._turn_complete = False
        self._terminated = False
        self._cached_rewards: dict[str, float] = {}
        self._terminate_idx = 0

        # Reward tracking (for computing the custom reward)
        self._prev_served: int = 0
        self._turn_empty_distance: float = 0.0
        self._turn_waiting_time: float = 0.0

    def _make_obs_space(self) -> Dict:
        n = self._num_drivers
        z = self._num_zones
        p = self._max_pending
        return Dict({
            "self_location": Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "self_status": Box(0.0, 1.0, shape=(NUM_STATUS,), dtype=np.float32),
            "self_passengers_ratio": Box(0.0, 10.0, shape=(1,), dtype=np.float32),
            "self_capacity": Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "self_committed_ratio": Box(0.0, 10.0, shape=(1,), dtype=np.float32),
            "self_region": Box(0.0, 1.0, shape=(z,), dtype=np.float32),
            "all_locations": Box(0.0, 1.0, shape=(n * 2,), dtype=np.float32),
            "all_statuses": Box(0.0, 1.0, shape=(n * NUM_STATUS,), dtype=np.float32),
            "all_passengers": Box(0.0, 10.0, shape=(n,), dtype=np.float32),
            "pending_orders": Box(0.0, 1.0, shape=(p, 5), dtype=np.float32),
            "pending_mask": Box(0.0, 1.0, shape=(p,), dtype=np.float32),
            "pending_passengers": Box(0.0, 10.0, shape=(p,), dtype=np.float32),
            "time_frac": Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "relocation_points": Box(0.0, 1.0, shape=(z, 2), dtype=np.float32),
        })

    def observation_space(self, agent: str) -> Dict:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    def _extract_pending(self, obs: dict[int, dict]) -> list[dict]:
        any_did = next(iter(obs))
        return list(obs[any_did]["pending_orders"])

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        obs, _info = self.env.reset(seed=self._seed)
        self._last_obs = obs
        self._current_pending = self._extract_pending(obs)
        self._terminated = False

        # Reset reward tracking
        self._prev_served = 0
        self._turn_empty_distance = 0.0
        self._turn_waiting_time = 0.0

        self.agents = self.possible_agents[:]
        self._agent_order = self._rng.permutation(self._num_drivers)
        self.agent_selection = self.agents[int(self._agent_order[0])]
        self._actions_this_turn = {}
        self._turn_complete = False

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cached_rewards = {a: 0.0 for a in self.agents}

    def _driver_int_id(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        assert self._last_obs is not None
        did = self._driver_int_id(agent)
        dobs = self._last_obs[did]
        self_d = dobs["self"]

        capacity = max(int(self_d["capacity"]), 1)
        ox, oy = _normalize_xy(
            self_d["location"][0], self_d["location"][1],
            self._geo_min, self._geo_range,
        )
        status_idx = STATUS_TO_IDX.get(self_d["status"], 0)
        region_idx = int(self_d.get("current_region", 0))
        region_idx = max(0, min(region_idx, self._num_zones - 1))

        all_locs = np.zeros(self._num_drivers * 2, dtype=np.float32)
        all_statuses = np.zeros(self._num_drivers * NUM_STATUS, dtype=np.float32)
        all_pax = np.zeros(self._num_drivers, dtype=np.float32)
        for i, (other_did, other_d) in enumerate(dobs["all_drivers"].items()):
            ax, ay = _normalize_xy(
                other_d["location"][0], other_d["location"][1],
                self._geo_min, self._geo_range,
            )
            all_locs[2 * i] = ax
            all_locs[2 * i + 1] = ay
            other_status_idx = STATUS_TO_IDX.get(other_d["status"], 0)
            all_statuses[i * NUM_STATUS + other_status_idx] = 1.0
            all_pax[i] = float(other_d["onboard_passengers"]) / capacity

        # Pending orders: keep up to max_pending, sorted by urgency (closest to timeout first)
        sorted_pending = sorted(
            self._current_pending,
            key=lambda o: o.get("time_to_timeout", float("inf")),
        )
        pending = np.zeros((self._max_pending, 5), dtype=np.float32)
        pending_mask = np.zeros(self._max_pending, dtype=np.float32)
        pending_pax = np.zeros(self._max_pending, dtype=np.float32)
        for i, order in enumerate(sorted_pending[: self._max_pending]):
            oo_x, oo_y = _normalize_xy(
                order["origin"][0], order["origin"][1],
                self._geo_min, self._geo_range,
            )
            dd_x, dd_y = _normalize_xy(
                order["destination"][0], order["destination"][1],
                self._geo_min, self._geo_range,
            )
            pending[i] = (oo_x, oo_y, dd_x, dd_y, float(order["num_passengers"]) / capacity)
            pending_mask[i] = 1.0
            pending_pax[i] = float(order["num_passengers"]) / capacity

        reloc = np.zeros((self._num_zones, 2), dtype=np.float32)
        for i, (rx, ry) in enumerate(self.env.relocation_points):
            reloc[i] = (rx / 100.0, ry / 100.0)

        time_frac = np.array(
            [min(self.env.time / max(self.env.horizon, 1e-9), 1.0)],
            dtype=np.float32,
        )

        return {
            "self_location": np.array([ox, oy], dtype=np.float32),
            "self_status": _one_hot(status_idx, NUM_STATUS),
            "self_passengers_ratio": np.array(
                [float(self_d["onboard_passengers"]) / capacity], dtype=np.float32,
            ),
            "self_capacity": np.array([float(capacity) / 10.0], dtype=np.float32),
            "self_committed_ratio": np.array(
                [float(self_d["committed_passengers"]) / capacity], dtype=np.float32,
            ),
            "self_region": _one_hot(region_idx, self._num_zones),
            "all_locations": all_locs,
            "all_statuses": all_statuses,
            "all_passengers": all_pax,
            "pending_orders": pending,
            "pending_mask": pending_mask,
            "pending_passengers": pending_pax,
            "time_frac": time_frac,
            "relocation_points": reloc,
        }

    def action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(self._action_dim, dtype=bool)
        mask[0] = True  # idle is always valid
        if self._last_obs is None:
            return mask
        did = self._driver_int_id(agent)
        self_d = self._last_obs[did]["self"]
        capacity = max(int(self_d["capacity"]), 1)
        has_room = int(self_d["committed_passengers"]) < capacity
        if has_room:
            for i in range(min(len(self._current_pending), self._max_pending)):
                mask[1 + i] = True
        if self_d["status"] == "idle":
            for j in range(self._num_zones):
                mask[1 + self._max_pending + j] = True
        return mask

    def _decode_action(self, action: int) -> dict | None:
        if action is None:
            return None
        if action == 0:
            return None
        if 1 <= action <= self._max_pending:
            idx = action - 1
            if idx < len(self._current_pending):
                return {"orders": [self._current_pending[idx]["order_id"]]}
            return None
        zone_idx = action - self._max_pending - 1
        if 0 <= zone_idx < self._num_zones:
            return {"relocate": int(zone_idx)}
        return None

    def _advance_selection(self) -> None:
        if not self.agents:
            return
        if not self._turn_complete:
            current_idx_in_order = int(np.where(self._agent_order == self._driver_int_id(self.agent_selection))[0][0])
            if current_idx_in_order + 1 < self._num_drivers:
                next_int_id = int(self._agent_order[current_idx_in_order + 1])
                self.agent_selection = self.agents[next_int_id]
                return
        self._turn_complete = True
        self._flush_turn()

    def _compute_reward(self) -> float:
        """Compute the rebalancing reward: R = -alpha*wait - beta*empty_dist + gamma*served.

        Components:
        - waiting_time: average time orders have been waiting (from RideGym order data)
        - empty_distance: total distance drivers travel without passengers this turn
        - served_demand: number of orders completed this turn (delta from last turn)
        """
        from ride_gym.enums import OrderStatus

        cfg = self._cfg

        # Count newly served orders
        current_served = sum(
            1 for o in self.env.orders.values()
            if o.status == OrderStatus.COMPLETED
        )
        delta_served = current_served - self._prev_served
        self._prev_served = current_served

        # Average waiting time of pending orders (normalized by horizon)
        pending_wait = 0.0
        if self._current_pending:
            wait_times = [
                o.get("waiting_time", 0.0) for o in self._current_pending
            ]
            pending_wait = np.mean(wait_times) / max(cfg.horizon, 1e-9)

        # Empty distance: sum of distances traveled by idle drivers this turn
        # We approximate from driver status changes (idle → to_pickup = empty travel)
        # This is an approximation; exact tracking would require per-step distance logging
        empty_dist = 0.0
        for did in range(self._num_drivers):
            d = self._last_obs[did]["self"]
            if d["status"] == "to_pickup":
                # Driver is traveling to pickup — count as empty distance
                # Use the distance to pickup as a proxy (normalized by network size)
                dist = d.get("distance_to目的地", 0.0)
                empty_dist += dist

        # Normalize components to [0, 1] range approximately
        reward = (
            -cfg.alpha * pending_wait
            - cfg.beta * empty_dist
            + cfg.gamma * delta_served
        )
        return float(reward)

    def _flush_turn(self) -> None:
        if self._terminated or self._last_obs is None:
            return

        ridegym_actions: dict[int, dict | None] = {}
        claimed_orders: set[int] = set()
        for order_pos in self._agent_order:
            agent = self.agents[int(order_pos)]
            action = self._actions_this_turn.get(agent, 0)
            decoded = self._decode_action(action)
            if decoded is None:
                ridegym_actions[int(order_pos)] = None
                continue
            if "orders" in decoded:
                oid = decoded["orders"][0]
                if oid in claimed_orders:
                    ridegym_actions[int(order_pos)] = None
                else:
                    claimed_orders.add(oid)
                    ridegym_actions[int(order_pos)] = decoded
            elif "relocate" in decoded:
                driver_self = self._last_obs[int(order_pos)]["self"]
                if driver_self["status"] != "idle":
                    ridegym_actions[int(order_pos)] = None
                else:
                    ridegym_actions[int(order_pos)] = decoded
            else:
                ridegym_actions[int(order_pos)] = decoded

        obs, rewards, dones, info = self.env.step(ridegym_actions)
        self._last_obs = obs
        self._current_pending = self._extract_pending(obs)

        # Compute our custom reward
        custom_reward = self._compute_reward()

        # Distribute the same custom reward to all agents
        self._cached_rewards = {}
        for a in self.agents:
            self._cached_rewards[a] = custom_reward

        done_flag = bool(dones["__all__"])

        if done_flag:
            self._terminated = True
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {"time": info.get("time", self.env.time)} for a in self.agents}
            self.agent_selection = self.agents[0]
        else:
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {"time": info.get("time", self.env.time)} for a in self.agents}
            self._agent_order = self._rng.permutation(self._num_drivers)
            self.agent_selection = self.agents[int(self._agent_order[0])]
            self._actions_this_turn = {}
            self._turn_complete = False

    def step(self, action: int | None) -> None:
        if not self.agents:
            return
        if self._terminated:
            done_agent = self.agent_selection
            remaining = [a for a in self.agents if a != done_agent]
            self.agents = remaining
            self.agent_selection = remaining[0] if remaining else None
            self.rewards = {a: 0.0 for a in remaining}
            self.terminations = {a: True for a in remaining}
            self.truncations = {a: False for a in remaining}
            self.infos = {a: {} for a in remaining}
            return
        self.rewards = {a: 0.0 for a in self.agents}
        if (
            self.agent_selection in self._cached_rewards
            and not self.terminations.get(self.agent_selection, False)
        ):
            self.rewards[self.agent_selection] = self._cached_rewards[self.agent_selection]
        self._cumulative_rewards[self.agent_selection] = 0.0
        self._accumulate_rewards()
        self._actions_this_turn[self.agent_selection] = 0 if action is None else int(action)
        self._advance_selection()

    def state(self) -> np.ndarray:
        if self._last_obs is None:
            return np.zeros(0, dtype=np.float32)
        n = self._num_drivers
        flat: list[float] = []
        for did in range(n):
            d = self._last_obs[did]["self"]
            ax, ay = _normalize_xy(d["location"][0], d["location"][1], self._geo_min, self._geo_range)
            flat.extend([ax, ay])
            flat.append(float(STATUS_TO_IDX.get(d["status"], 0)))
            flat.append(float(d["onboard_passengers"]))
            flat.append(float(d["committed_passengers"]))
        for order in self._current_pending[: self._max_pending]:
            oo_x, oo_y = _normalize_xy(order["origin"][0], order["origin"][1], self._geo_min, self._geo_range)
            dd_x, dd_y = _normalize_xy(order["destination"][0], order["destination"][1], self._geo_min, self._geo_range)
            flat.extend([oo_x, oo_y, dd_x, dd_y, float(order["num_passengers"])])
        while len(flat) < n * 5 + self._max_pending * 5:
            flat.append(0.0)
        flat.append(min(self.env.time / max(self.env.horizon, 1e-9), 1.0))
        return np.asarray(flat, dtype=np.float32)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
