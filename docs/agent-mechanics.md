# Agent Mechanics, Diagnostics, and Action Flow

This document provides a concise summary of the fleet-rebalancing reinforcement learning architecture: what the failure modes were, what each zone controller observes, what decisions it makes, and how rewards are assigned.

---

## 1. What the Issues Were (Root Cause Diagnostics)

During initial experiments on Bandra West, training plateaued at ~8–10% service rate with unstable policy gradients. Four core issues were diagnosed:

1. **Observation-Action Mismatch Bug**:
   - In `pettingzoo_env.py`, observations were generated using pending orders sorted by wait time, but `_decode_action` was looking up the raw, unsorted order list from the simulator. Action $1$ did not dispatch the order seen at Slot $1$.
2. **Spatial / Proximity Blindness**:
   - Candidate orders were global. An agent controlling Zone 24 (southern tip) saw orders originating in Zone 0 (northern tip). Choosing Action $1$ forced vehicles to deadhead across the entire city, burning empty kilometers without serving local demand.
3. **Credit Assignment Noise (Decentralized Dilemma)**:
   - A single scalar reward was computed globally and broadcasted equally to all 25 agents. If Zone 5 successfully dispatched rides while Zone 22 made poor relocation choices, both received the exact same reward. The resulting gradient updates cancelled each other out.
4. **Demand vs. Fleet Capacity Imbalance**:
   - Synthetic demand injected 400 orders into a 30-minute episode for a fleet of only 30 vehicles. A 30-car fleet can physically serve at most 40–60 trips in 30 minutes. Excess orders accumulated in timeout queues, overwhelming the waiting time penalty.

---

## 2. What Each Agent Sees (Observation Space)

Each RL agent controls **one geographic zone** ($Z = 25$ on Bandra, $Z = 62$ on NYC).

At every decision turn, `observe(agent)` provides:

| Observation Component | Shape | What it Contains |
| :--- | :--- | :--- |
| `zone_location` | `(2,)` | Normalized $(x, y)$ coordinate of this zone's centroid. |
| `zone_state` | `(3,)` | Local metrics: `[fleet_fraction, idle_ratio, local_pending_ratio]`. |
| `all_zone_state` | `(Z, 2)` | Citywide overview: fraction of total fleet and idle fleet in every zone. |
| `pending_orders` | `(max_pending, 6)` | Candidate orders **sorted by distance to this zone's centroid** (closest first). Each order contains: `[norm_ox, norm_oy, norm_dx, norm_dy, passengers/capacity, wait_time/timeout]`. |
| `pending_mask` | `(max_pending,)` | Binary mask: `1.0` if a candidate order exists in that slot, `0.0` if empty padding. |
| `pending_passengers` | `(max_pending,)` | Passenger party sizes for candidate orders. |
| `time_frac` | `(1,)` | Elapsed simulation time normalized by horizon ($t / T \in [0, 1]$). |
| `relocation_points` | `(Z, 2)` | Normalized coordinates of all zone centroids. |

---

## 3. What Each Agent Chooses to Do (Action Space)

Each zone agent makes decisions sequentially for idle vehicles in its zone.

The discrete action space has dimension $1 + \text{max\_pending} + Z$:

```
Action 0: [Idle]
  └── Hold idle vehicle in place (no action taken).

Action 1 .. max_pending: [Dispatch to Candidate Order]
  ├── Action 1: Dispatch to the CLOSEST available pending customer to this zone.
  ├── Action 2: Dispatch to the 2nd closest pending customer.
  └── Action k: Dispatch to the k-th closest pending customer.

Action (max_pending + 1) .. (max_pending + Z): [Rebalance / Relocate]
  └── Relocate an idle vehicle to another specific target zone (Z).
```

### Action Masking Rules
To prevent invalid actions and invalid gradients:
- If a zone has **0 idle vehicles**, all dispatch and relocate actions are masked out (`False`); only Action 0 (`Idle`) is permitted.
- An agent **cannot dispatch to empty candidate slots** (enforced by `pending_mask`).
- An agent **cannot relocate to its own zone** (same-zone relocation is a no-op).

---

## 4. How Rewards are Attributed (Local Credit Assignment)

To enable stable multi-agent reinforcement learning, rewards are attributed directly to the responsible zone controller:

$$R_z = (1 - \text{team\_share}) \cdot R_{\text{local}, z} + \text{team\_share} \cdot \bar{R}_{\text{global}}$$

- **Completed Order ($\gamma = +5.0$)**:
  Awarded directly to the specific zone controller that dispatched that order.
- **Empty Travel Penalty ($\beta = 0.1$)**:
  Charged directly to the zone controller that commanded the vehicle to move empty (pickup deadhead or relocation trip).
- **Waiting Time Penalty ($\alpha = 1.0$)**:
  Charged to the zone where the unserved customer is waiting (partitioned by nearest centroid).
- **Team Cooperation ($\text{team\_share} = 0.2$)**:
  $80\%$ of the reward comes from direct local performance, and $20\%$ is shared across all zones to incentivize collaborative rebalancing.

---

## 5. Summary of Tuned Parameters (Bandra West)

| Parameter | Previous Value | Tuned Value | Rationale |
| :--- | :--- | :--- | :--- |
| `num_orders` | 400 | **200** | Matches physical capacity of 30 vehicles in 30 minutes without runaway queue buildup. |
| `rollout_episodes` | 2 | **6** | 1,800 turns (45,000 steps) per PPO update to stabilize GAE advantage estimation. |
| `beta` (empty cost) | 0.05 / approx | **0.1** | Realistic deadhead penalty per empty kilometer. |
| `gamma` (trip reward) | 1.0 / 5.0 | **5.0** | Sufficient payoff incentive to encourage active dispatch over passive idling. |
| `team_share` | 1.0 (pure global) | **0.2** | 80% local credit assignment + 20% team cooperation. |
