# Agent Mechanics, Architecture, and Training Flow

This document provides a comprehensive, human-readable guide to the Multi-Agent Reinforcement Learning (MARL) fleet rebalancing system: how the agents make decisions, how the environment prevents conflicts, how credit is assigned without penalizing cooperation, and how the training loop operates.

---

## 1. High-Level Architecture (Who are the Agents?)

Instead of treating each vehicle as an independent agent (which scales poorly when fleets have hundreds of cars), the system uses **Zone-Centric Decentralized Controllers**:

* **Geographic Zones ($Z$)**: The city map is partitioned into discrete zones ($Z = 25$ on Bandra West, $Z = 62$ on NYC).
* **One Agent per Zone**: Each zone is governed by its own reinforcement learning controller (`zone_0` to `zone_Z-1`).
* **Parameter Sharing**: All zone agents share the same actor and critic neural network weights. To give each agent its own spatial identity, each agent receives its own normalized GPS centroid in its observation.
* **Turn-Based Execution**: During each simulation turn (e.g. every $0.1$ minutes / 6 seconds), zone agents take turns in a randomly shuffled order to assign orders and relocate idle vehicles within their jurisdiction.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        City Road Network (OSMnx)                       │
│                                                                        │
│   Zone 0 Agent        Zone 1 Agent        ...       Zone 24 Agent      │
│   (Local Queue)       (Local Queue)                 (Local Queue)      │
│         │                   │                             │            │
│         ▼                   ▼                             ▼            │
│   [Pickups/Reloc]     [Pickups/Reloc]               [Pickups/Reloc]    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What Each Agent Sees (Observation Space)

Each zone agent receives a structured observation dictionary containing both local detail and citywide macro information.

| Component | Dimensions | Human Meaning |
| :--- | :--- | :--- |
| `zone_location` | `(2,)` | Normalized $(x, y)$ coordinate of this zone's centroid. Gives the shared neural network spatial awareness of which zone it is currently acting for. |
| `zone_state` | `(3,)` | Local conditions: `[fleet_fraction, idle_fraction, pending_demand_fraction]` inside this zone. |
| `all_zone_state` | `(Z, 3)` | **Citywide Macro Heatmap**: For all $Z$ zones, tracks: <br>1. Total cars in zone (fleet share) <br>2. Idle cars in zone (idle share) <br>3. **Unmet customer backlog** (`pending_per_zone`). <br>*This allows agents to detect distant demand surges and distribution shifts.* |
| `pending_orders` | `(max_pending, 6)` | Candidate passenger requests **within $1.2\text{ km}$**, sorted closest-first. Features: `[norm_ox, norm_oy, norm_dx, norm_dy, party_size/capacity, wait_time/timeout]`. |
| `pending_mask` | `(max_pending,)` | Binary mask (`1.0` if an actual passenger is waiting in slot $k$, `0.0` if empty padding). |
| `pending_passengers` | `(max_pending,)` | Party sizes for each candidate order. |
| `time_frac` | `(1,)` | Normalized episode clock ($t / T \in [0, 1]$). |
| `relocation_points` | `(Z, 2)` | Normalized GPS coordinates of all zone centroids in the city. |

> **Key Fix (D16)**: Originally, `all_zone_state` only included fleet counts `(Z, 2)`. When local orders were restricted to $1.2\text{ km}$, distant agents became blind to surges in other zones. Expanding this to `(Z, 3)` gives every agent live visibility into which zones have unserved customers.

---

## 3. What Each Agent Chooses to Do (Action Space)

The discrete action space has dimension $1 + \text{max\_pending} + Z$:

```
Action 0: [No-Op / Idle]
  └── Do nothing. Leave vehicles in their current state.

Action 1 .. max_pending: [Dispatch Candidate Order]
  ├── Action 1: Assign the CLOSEST valid customer to an idle vehicle in this zone.
  ├── Action 2: Assign the 2nd closest valid customer.
  └── Action k: Assign the k-th closest valid customer.

Action (max_pending + 1) .. (max_pending + Z): [Rebalance / Relocate]
  └── Command one idle vehicle to relocate to neighboring zone j.
```

### Action Filtering & Masking Rules
To prevent wasted exploration and illegal actions, the environment computes an exact boolean mask before the neural network selects an action:

1. **Zero-Fleet Masking**: If a zone has no available vehicles, all dispatch and relocation actions are masked out (`False`); only Action 0 (`No-Op`) is legal.
2. **Local Radius Cutoff ($1.2\text{ km}$)**: Customers located farther than $1.2\text{ km}$ from the zone centroid are excluded. This prevents agents from dispatching vehicles on massive cross-city pickup deadheads.
3. **Intra-Turn Collision Masking**: In each turn, agents act sequentially. If Zone 2 claims Customer #105, Customer #105 is dynamically masked out (`False`) for all remaining agents in that turn. This guarantees zero dispatch collisions.
4. **Neighbor-Only Relocation Pruning**: Vehicles cannot leap across the map in a single leap. Relocation is restricted to immediate topological graph neighbors (`region_neighbours`). This prunes 96% of irrelevant actions and enforces realistic street movement.
5. **Opportunistic Relocation Dispatch**: If an empty vehicle is currently in transit (`RELOCATING`) and passes near a newly spawned customer, the local zone agent can intercept and dispatch it immediately, canceling the empty trip.

---

## 4. Credit Assignment & The Cooperative Reward Engine

In multi-agent RL, naive rewards cause two major failure modes:
1. **Global Noise**: If everyone shares one global score, good actions get drowned out by bad actions.
2. **The Altruism Penalty**: If an agent is penalized for fuel when sending a car to help a neighbor, but gets nothing when the neighbor picks up the rider, the agent learns to hoard cars and never rebalance.

The reward engine balances localized accountability with cooperative incentives:

$$R_z = (1 - \text{team\_share}) \cdot R_{\text{local}, z} + \text{team\_share} \cdot \bar{R}_{\text{global}}$$

where $\text{team\_share} = 0.4$ (60% local performance + 40% citywide team health).

### Local Reward Breakdown ($R_{\text{local}, z}$)

| Term | Value | Rule & Credit Attribution |
| :--- | :--- | :--- |
| **Trip Completion** | $+\gamma = +5.0$ | **Rebalancing Assist Sharing**: <br>• **70% (+3.5)** goes to the zone that dispatched the trip.<br>• **30% (+1.5)** goes to the zone that previously relocated the car there.<br>*Sending an idle vehicle to assist a high-demand zone now generates direct profit for the sender.* |
| **Empty Travel Penalty** | $-\beta = -0.1\text{ / km}$ | **Shared Relocation Deadhead**: <br>• Pickup travel is charged to the dispatching zone.<br>• Relocation travel is **split 50/50** between sending zone and receiving zone.<br>*The sending zone is no longer punished 100% for being generous.* |
| **Passenger Wait Penalty** | $-\alpha = -1.0$ | **Volume-Scaled Waiting Cost**: Proportional to total passenger waiting time across unserved customers in the zone: $-0.1 \times \sum (\text{wait} / \text{timeout})$. |
| **Cancellation Penalty** | $-2.0$ | Charged directly to the zone origin when a customer gives up and cancels after exceeding the 2.0-minute timeout. |

---

## 5. The Training Loop (MAPPO & IPPO)

Training runs via Centralized Training with Decentralized Execution (CTDE):

```
                       TRAINING (Centralized Critic)
              ┌──────────────────────────────────────────────┐
              │ State: All Zone Fleets + Global Orders + Clock │
              │            Value Baseline V(s, zone_loc)     │
              └──────────────────────┬───────────────────────┘
                                     │ Advantage (GAE)
                                     ▼
                      EXECUTION (Decentralized Actor)
    ┌──────────────────────┬──────────────────────┬──────────────────────┐
    │     Zone 0 Actor     │     Zone 1 Actor     │    Zone 24 Actor     │
    │  Obs: Local + Heatmap│  Obs: Local + Heatmap│  Obs: Local + Heatmap│
    │      Action π(a|o)   │      Action π(a|o)   │      Action π(a|o)   │
    └──────────────────────┴──────────────────────┴──────────────────────┘
```

1. **Rollout Collection**:
   - The environment simulates full episodes ($T = 30\text{ minutes}$, $300\text{ turns}$).
   - Shuffled agent execution prevents the first zone from always having an unfair advantage.
   - Observations, actions, log-probabilities, rewards, and action masks are stored in rollout buffers.
2. **Advantage Estimation (GAE)**:
   - Generalized Advantage Estimation ($\gamma = 0.99, \lambda = 0.95$) computes temporal-difference advantages.
3. **PPO Clip Updates**:
   - Clipped surrogate objective ($\epsilon = 0.2$) with entropy bonus ($c_2 = 0.01$) ensures stable policy steps without policy collapse.
4. **Agent-Conditioned MAPPO Critic**:
   - In MAPPO, the centralized critic evaluates state values conditioned on both the global city state and the specific acting agent's location centroid (`[state, zone_location]`). This gives the critic spatial awareness of 25 distinct value trajectories.

---

## 6. Parameterized Demand & Distribution Shift Testing

To test how well the trained policy generalizes when passenger patterns shift out-of-distribution:

* **Training with Surge**: `--target-zone 19 --target-zone-demand-pct 95` forces 95% of trips to originate in Zone 19.
* **Testing Distribution Shift**: At evaluation time, flipping the surge to `--target-zone 4 --target-zone-demand-pct 80` tests whether agents can dynamically detect the shift via the macro-demand heatmap in `all_zone_state` and redirect the fleet across the city.

---

## 7. Command Reference (`justfile`)

| Action | Command | What it Does |
| :--- | :--- | :--- |
| **Train MAPPO** | `just train-mappo bandra 20 3 19 95 b_m_z_3` | Trains MAPPO on Bandra for 20 iterations, 3 rollouts/iter, targeting 95% demand in Zone 19. |
| **Train IPPO** | `just train-ippo bandra 20 3 19 95 b_i_z_3` | Trains decentralized IPPO baseline under identical scenario. |
| **Interactive Scrubber** | `just view-interactive runs/b_m_z_3/checkpoint.pt 4 80` | Opens interactive GUI to scrub through episode frames, inspect drivers, zoom, pan, and verify distribution shift. |
| **Render MP4 Video** | `just render-mp4 runs/b_m_z_3/checkpoint.pt 4 80 out="test.mp4"` | Exports high-resolution MP4 video of the evaluation rollout with on-screen stats. |
