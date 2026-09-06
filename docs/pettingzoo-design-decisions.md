# PettingZoo Wrapper — Contested Design Decisions

This document captures the decisions where the right answer was not obvious, including the one issue the user caught that changed our approach.

---

## Decision 1: AEC vs Parallel

In reality, drivers act **concurrently** — so the natural reaction is to pick PettingZoo's Parallel API. We chose **AEC** instead.

The reasoning: in both APIs, training sees the same signal. All agents observe at time `t`, all act for time `t`, environment steps to `t+1`. The only difference is whether the wrapper collects actions one-by-one or all at once. For IPPO/MAPPO, the maths is identical. AEC matches RideGym's natural `env.step(dict)` interface, and debugging is dramatically easier. Parallel is only the right call when agents genuinely act asynchronously — different clocks, network delays, message-passing. Fleet rebalancing has none of that.

The tradeoff: AEC *can* introduce iteration-order bias. That was the next decision.

---

## Decision 2: Shuffled Agent Order (the bias the user caught)

The first version of the plan was **first-come-first-served conflict resolution** in a fixed agent order `0, 1, 2, ..., N-1`. The user caught the bug before any code was written:

> "wait wont this add baise in the loop as to the agent that WE iterate over first will be the one getting priority?"

They were right. In a fixed-order AEC wrapper, **agent 0 wins every conflict**. Agent 29 never wins. The trained policy would be biased toward low-indexed drivers and wouldn't generalise to a re-numbered fleet. This is the kind of bug that produces results that look fine but are silently broken.

**Fix:** the wrapper holds an `_agent_order` permutation and **shuffles it with a seeded RNG at the start of every turn**. Every agent has equal probability of being first across many turns. Training remains deterministic for a given seed. The policy has to learn to actually compete.

Alternatives rejected: "both bidders rejected" (wastes the turn's opportunity), "lowest ID always wins" (this was the bug we were trying to avoid).

---

## Decision 3: Action Masking vs Silent Replacement

A driver can't relocate unless idle, can't bid when full, can't bid on an order that vanished mid-step. Three ways to handle this:

1. Pass through and let RideGym raise `InvalidActionError`.
2. Silently replace illegal actions with idle.
3. Expose an `action_mask(agent)` method.

We chose **option 3**. The reasoning: the policy network wastes most of its exploration budget on illegal moves if we just hand it a 106-dim action space with 90 illegal entries. Silently replacing breaks credit assignment — the policy sampled "relocate to zone 5" but the environment executed "idle" and the gradient is on the wrong action. Action masking keeps the sampled action intact and just constrains which actions are legal, which is the standard PettingZoo pattern that stable-baselines3 and RLLib both support.

---

## Decision 4: Mask as a Method, Not an Observation Field

Two patterns exist: `env.action_mask(agent)` method vs baking `valid_actions` into the obs dict. We chose the **method**.

The reasoning: the observation is the state of the world (positions, orders, status). The action mask is a constraint on the policy, not on the world. They have different consumers and different lifetimes. If we bake the mask into the obs, the policy network might learn to predict actions from the mask (wrong thing). The mask is also stale-prone if the order is cancelled mid-decision, but a method called right before action sampling is always fresh.

---

## Cross-References

- Implementation plan: `docs/pettingzoo-wrapper-plan.md`
- Project rules: `AGENTS.md`
- Research intent: `INTENT.md`
