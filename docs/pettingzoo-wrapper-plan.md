# PettingZoo Wrapper — Implementation Plan

## Context

We are building a PettingZoo wrapper around RideGym so that IPPO and MAPPO can be trained using standard multi-agent RL tooling. The wrapper converts RideGym's per-driver dict interface into a PettingZoo AEC interface with flat, fixed-size observations and a discrete action space.

## Design Decisions (confirmed with user)

| Question | Answer |
|----------|--------|
| AEC vs Parallel | AEC (simpler, no practical difference for training) |
| Conflict policy | First-come-first-served, with **shuffled agent order each turn** (no bias) |
| Invalid actions | Action mask (PettingZoo standard pattern) |
| Mask API | `action_mask(agent)` method |

## File Layout

```
src/marl_fleet_rebalancing/env/
├── __init__.py              (create, empty)
└── pettingzoo_env.py        (create, ~250 lines)
```

## Step 0 — Add Dependencies

```bash
uv add pettingzoo gymnasium
```

## Step 1 — `src/marl_fleet_rebalancing/env/__init__.py`

Empty file. Makes the `env` directory a Python package.

## Step 2 — `src/marl_fleet_rebalancing/env/pettingzoo_env.py`

### 2a. Constants

```python
MAX_PENDING = 80            # max orders to pad to (saw max 70 in test)
STATUS_TO_IDX = {"idle": 0, "to_pickup": 1, "to_dropoff": 2, "relocating": 3}
NUM_STATUS = 4
```

### 2b. `class RideGymAEC(AECEnv)`

#### `__init__(self, num_drivers=30, horizon=30.0, ..., seed=None)`

- Create `self.env = RidePoolEnv(...)` with the given config
- `self.num_agents = num_drivers`
- `self.possible_agents = [f"driver_{i}" for i in range(num_drivers)]`
- `self._rng = np.random.default_rng(seed)`
- **Pre-compute** normalization constants from `self.env.area` (the 0-100 grid):
  - `self._area_min = (area[0], area[2])`  # (x_min, y_min)
  - `self._area_range = (area[1] - area[0], area[3] - area[2])`
- **Get num relocation zones** from `len(self.env.relocation_points)` → store as `self._num_zones`
- Define observation space and action space (see below)
- `self._pending_actions = {}`
- `self._last_obs = None`
- `self._current_pending_orders = []`  # cached for action decoding

#### Observation Space (per agent)

`gymnasium.spaces.Dict` with these keys:

| Key | Shape | Dtype | Source | Padding |
|-----|-------|-------|--------|---------|
| `self_location` | `(2,)` | float32 | normalized (x, y) of driver | n/a |
| `self_status` | `(NUM_STATUS,)` | float32 | one-hot | n/a |
| `self_passengers_ratio` | `(1,)` | float32 | onboard / capacity | n/a |
| `self_capacity` | `(1,)` | float32 | capacity (normalized) | n/a |
| `self_committed_ratio` | `(1,)` | float32 | committed / capacity | n/a |
| `self_region` | `(num_zones,)` | float32 | one-hot | n/a |
| `all_locations` | `(num_agents * 2,)` | float32 | all drivers' positions | n/a |
| `all_statuses` | `(num_agents * NUM_STATUS,)` | float32 | all drivers' statuses | n/a |
| `all_passengers` | `(num_agents,)` | float32 | all drivers' passenger counts | n/a |
| `pending_orders` | `(MAX_PENDING, 5)` | float32 | (ox, oy, dx, dy, party_size) | zero-pad |
| `pending_mask` | `(MAX_PENDING,)` | float32 | 1.0 if real order, else 0.0 | n/a |
| `pending_passengers` | `(MAX_PENDING,)` | float32 | party_size / capacity | zero-pad |
| `time_frac` | `(1,)` | float32 | env.time / horizon | n/a |
| `relocation_points` | `(num_zones, 2)` | float32 | zone centers (normalized) | n/a |

**Normalization rules:**
- Coordinates: `(x - area_min) / area_range` → [0, 1]
- Statuses: one-hot encoding
- Ratios: divide by capacity (avoid div-by-zero with `+ 1e-6`)
- Time: `env.time / env.horizon` → [0, 1]
- Pending orders: all components normalized to [0, 1]

**Computed shapes for num_drivers=30, num_zones=25, MAX_PENDING=80:**
- Per-agent obs: 2 + 4 + 1 + 1 + 1 + 25 + 60 + 120 + 30 + 80*5 + 80 + 80 + 1 + 50 = **933 floats**

#### Action Space (per agent)

`gymnasium.spaces.Discrete(1 + MAX_PENDING + num_zones)` = **1 + 80 + 25 = 106**

| Action range | Meaning |
|--------------|---------|
| `0` | Idle (do nothing) |
| `1 .. MAX_PENDING` | Bid on pending order `i - 1` |
| `MAX_PENDING + 1 .. MAX_PENDING + num_zones` | Relocate to zone `i - MAX_PENDING - 1` |

#### `reset(self, seed=None, options=None)`

1. If `seed` provided, set `self._rng = np.random.default_rng(seed)`
2. Call `self.env.reset(seed=seed)` → get `obs`
3. `self.agents = self.possible_agents[:]`
4. `self._last_obs = obs` (the dict from RideGym, keyed by int)
5. `self._current_pending_orders = self._extract_pending(obs)`
6. **Shuffle agent order** using `self._rng.permutation(num_agents)` → store as `self._agent_order`
7. `self.agent_selection = self.agents[self._agent_order[0]]`
8. `self.rewards = {a: 0.0 for a in self.agents}`
9. `self._cumulative_rewards = {a: 0.0 for a in self.agents}`
10. `self.terminations = {a: False for a in self.agents}`
11. `self.truncations = {a: False for a in self.agents}`
12. `self.dones = {a: False for a in self.agents}`
13. `self._pending_actions = {}`
14. `self._step_count = 0`
15. Return `(self.observe(self.agent_selection), {})`

#### `_extract_pending(self, obs)` (helper)

Pulls the list of pending order dicts from any driver's observation. Used for action decoding and masking. Returns a list of `{"order_id", "origin", "destination", "num_passengers"}`.

#### `observe(self, agent)`

1. `int_id = int(agent.split("_")[1])`
2. `dobs = self._last_obs[int_id]`
3. Build the flat dict described above
4. Return `{"observation": flat_dict}`

#### `_decode_action(self, agent, action)`

Converts a discrete action int into a RideGym action dict:

```python
if action == 0:
    return None  # idle
elif 1 <= action <= MAX_PENDING:
    idx = action - 1
    if idx < len(self._current_pending_orders):
        oid = self._current_pending_orders[idx]["order_id"]
        return {"orders": [oid]}
    return None  # order vanished, treat as idle
else:
    zone_idx = action - MAX_PENDING - 1
    if zone_idx < self._num_zones:
        return {"relocate": zone_idx}
    return None
```

#### `action_mask(self, agent)`

Returns `np.ndarray` of shape `(action_space.n,)` with `True` for valid actions:

| Action | Valid when |
|--------|-----------|
| `0` (idle) | always |
| `1..MAX_PENDING` (bid on order i) | `i < len(current_pending)` AND `i`'s order still exists AND driver has capacity AND driver is idle or to_dropoff (can accept more) |
| `MAX_PENDING+1..` (relocate to zone j) | driver status is IDLE |

**Implementation:**
```python
def action_mask(self, agent):
    mask = np.zeros(self.action_space(agent).n, dtype=bool)
    mask[0] = True  # idle always valid
    int_id = int(agent.split("_")[1])
    driver = self._last_obs[int_id]["self"]
    
    # Bids valid if: order exists AND driver has capacity room
    has_capacity = driver["committed_passengers"] < driver["capacity"]
    for i, order in enumerate(self._current_pending_orders):
        if has_capacity and i < MAX_PENDING:
            mask[1 + i] = True
    
    # Relocate only valid if driver is IDLE
    if driver["status"] == "idle":
        for j in range(self._num_zones):
            mask[MAX_PENDING + 1 + j] = True
    
    return mask
```

#### `step(self, action)`

Core AEC logic — called once per agent per turn:

1. `self._pending_actions[self.agent_selection] = action`
2. **Has every agent acted this turn?**
   - If NO: advance `self.agent_selection` to next agent in `self._agent_order`, return
   - If YES: continue below

3. **Shuffle the action collection order** (to resolve conflicts fairly):
   - We need to determine "first" vs "second" bidder
   - Use `self._agent_order` to decide priority (lower index = acted first)

4. **Build RideGym actions dict with conflict resolution:**
   - Iterate `self._agent_order` (low to high = first to last)
   - For each agent, get the decoded action
   - If it's a bid on an order that's already been claimed by a higher-priority agent, drop the bid (set to idle)
   - If it's a relocate and the agent isn't actually idle (race condition), drop it

5. Call `self.env.step(ridegym_actions_dict)` → get `obs, rewards, dones, info`
6. Update internal state:
   - `self._last_obs = obs`
   - `self._current_pending_orders = self._extract_pending(obs)`
   - `self.rewards = {f"driver_{did}": rewards[did] for did in range(self.num_agents)}`
   - `done_flag = dones["__all__"]`
   - `self.terminations = {a: done_flag for a in self.agents}`
   - `self.truncations = {a: False for a in self.agents}`  # we don't use truncation
   - `self.dones = {a: done_flag for a in self.agents}`
   - Add to cumulative rewards: `self._cumulative_rewards[a] += self.rewards[a]`
7. If done: clear `self.agents = []`
8. **Shuffle** `self._agent_order = self._rng.permutation(self.num_agents)` for next turn
9. Reset `self._pending_actions = {}`
10. `self._step_count += 1`
11. Advance `self.agent_selection` to `self.agents[self._agent_order[0]]`

#### `last(self, observe=True)`

Returns `(observe(agent_selection), reward(agent_selection), termination, truncation, info)`.

For PettingZoo 1.24+ API.

#### `close(self)`

Call `self.env.close()` if it exists.

## Step 3 — Smoke Test

Add this to a new file `src/marl_fleet_rebalancing/env/test_wrapper.py` or run inline:

```python
import numpy as np
from marl_fleet_rebalancing.env import RideGymAEC

env = RideGymAEC(num_drivers=5, horizon=30.0, seed=0)
obs, info = env.reset(seed=0)

print("Agents:", env.agents[:3], "...")
print("Obs keys:", list(obs.keys()))
print("Action space:", env.action_space(env.agent_selection))
print("Mask shape:", env.action_mask(env.agent_selection).shape)
print("Valid actions in mask:", env.action_mask(env.agent_selection).sum())

total_steps = 0
for agent in env.agent_iter(max_iter=3000):
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        action = None
    else:
        mask = env.action_mask(agent)
        valid = np.where(mask)[0]
        action = np.random.choice(valid)
    env.step(action)
    total_steps += 1

print(f"Total env.step() calls: {total_steps}")
print("Episode finished.")
```

Expected output: ~1500 step calls (5 drivers × 300 timesteps).

## Step 4 — Verify

```bash
uv run python -c "
from marl_fleet_rebalancing.env import RideGymAEC
import numpy as np
env = RideGymAEC(num_drivers=5, horizon=30.0, seed=0)
obs, info = env.reset(seed=0)
print('agents:', len(env.agents))
print('obs keys:', list(obs.keys()))
print('action_space:', env.action_space(env.agent_selection))
print('mask sum:', env.action_mask(env.agent_selection).sum())
"
```

## Step 5 — Update `main.py` (optional)

Add a `--pettingzoo` flag to run the wrapper instead of the raw env:

```python
parser.add_argument("--pettingzoo", action="store_true", help="Use PettingZoo wrapper")
```

Then in `main()`, if `args.pettingzoo`, instantiate `RideGymAEC` and run a smoke test with random actions (similar to the test above).

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `pyproject.toml` | Add deps | +2 |
| `src/marl_fleet_rebalancing/env/__init__.py` | Create | 1 |
| `src/marl_fleet_rebalancing/env/pettingzoo_env.py` | Create | ~280 |
| `main.py` | Add `--pettingzoo` flag | +15 |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Variable pending order count (up to 70) | Fixed `MAX_PENDING=80` padding with mask |
| Order vanishes between observe and step | `_decode_action` returns idle if idx out of range |
| Agent acts twice before step() | `_pending_actions` dict prevents this |
| `env.area` differs from driver locations | Normalize relative to area bounding box (uses 0-100 grid for both drivers and reloc zones — confirmed they share the same coordinate frame) |
| PettingZoo API version differences | Target 1.24+ (uses `last()` return signature) |
| Action mask edge cases (capacity=0) | Clamp to always allow idle |
