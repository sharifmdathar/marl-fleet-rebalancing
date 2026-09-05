# MARL Fleet Rebalancing Under Demand Shift

**Research question:** How does a frozen fleet-rebalancing policy degrade when
passenger demand at deployment differs from the demand it was trained on — and do
independent (IPPO) and centralized-training (MAPPO) agents degrade differently,
relative to a tuned non-RL baseline?

## Stack

- Simulator: [RideGym](https://github.com/RS2002/RideGym) (`ride-gym` on PyPI)
- Road network: OSMnx graph of Bandra West, Mumbai (`data/bandra_west.gpickle`)
- Runner/env manager: `uv`
- Agents (planned): IPPO, MAPPO, tuned rolling-horizon rebalancing baseline

## Quick Start

```bash
uv sync                          # install deps
uv run python main.py            # random-policy rollout on Bandra West, writes episode.mp4
```

Current state: random policy smoke test. 30 drivers, capacity 4, 30-min horizon,
400 Poisson orders, dt = 0.1 min. Research question and scope: `INTENT.md`.