# Worked example — one real zone-turn of the shipped MAPPO checkpoint

Produced by `scripts/probe_one_zone_turn.py`, which replays the **real**
checkpoint on the **real** environment and dumps every intermediate tensor for
one decision. Nothing here is hand-written; re-run it to regenerate:

```
uv run python scripts/probe_one_zone_turn.py --seed 12002 --prefer relocate \
    --min-pending 1 --min-targets 2 --out results/probe_reloc_12002.json
```

## Setting

| | |
|---|---|
| checkpoint | `runs/d83-chain-mappo-s1/checkpoint_it0080.pt` (validation-selected) |
| scenario | 300 orders/h, 55 drivers, `max_pending=16`, `obs_inbound=True`, beta 0.12, patience 6.0 min |
| split | val (the loader forces val; the manifest said train) |
| eval seed | 12002 |
| zone | index 3, agent id `zone_3` |
| position in hour | `time_frac = 0.4333` → **26.0 min** in |

## 1. What the zone observes — 295 numbers

| slice | dims | field | this zone's value |
|---|---|---|---|
| `[0:2)` | 2 | `zone_location` (normalised lon, lat) | `0.786, 0.454` |
| `[2:6)` | 4 | `zone_state` (fleet share, idle share, pending share, deficit) | `0.833, 1.000, 0.167, −0.750` |
| `[6:70)` | 64 | `all_zone_state` — 16 zones × 4 channels | all zones |
| `[70:166)` | 96 | `pending_orders` — up to 16 slots × 6 features | **1 slot occupied** |
| `[166:182)` | 16 | `pending_mask` | 1.0 in slot 0, 0 elsewhere |
| `[182:198)` | 16 | `pending_passengers` | duplicate of `pending_orders[:,4]` |
| `[198:214)` | 16 | `neighbor_mask` — legal relocation targets | **10 of 16 set** |
| `[214:278)` | 64 | `demand_hist` — 16 zones × 4 turns (2.0 min) | own row all 0 |
| `[278:279)` | 1 | `time_frac` | `0.4333` |
| `[279:295)` | 16 | `inbound` — cars already flying here | own value `0.0` |

279 without `inbound`; 295 with it.

## 2. What the layers do

- **Order attention** — one scaled dot product, scale `1/sqrt(64) = 1/8`.
  All **1.0** of the attention mass lands on the single occupied slot.
- **Zone attention** — one scaled dot product, plus a learned scalar
  `neighbor_bias = 3.024` on the legal-target mask. **No mask on the softmax
  itself**, so it stays nearly flat: top zone `0.101`, uniform would be `0.0625`.
- **Context vector** = `zone_location(2) + zone_state(4) + order_pool(64) +
  zone_pool(64) + time_frac(1)` = **135 → 256**, two tanh layers, `|h| = 11.3`.

## 3. The action space — 33 = 1 + 16 + 16

| index | meaning |
|---|---|
| `0` | no zone action (wait) — always legal |
| `1 … 16` | dispatch the ranked candidate order `i` (nearest first) |
| `17 … 32` | relocate one idle car to zone `j = index − 17` |

**12 of 33 legal here**: index 0, dispatch index 1, and 10 relocations.

## 4. The scores, and the action taken

| action | meaning | logit | p |
|---|---|---|---|
| **26** | **relocate to zone 9** | **+7.899** | **0.505** |
| 1 | dispatch ranked candidate #1 | +7.837 | 0.474 |
| 21 | relocate to zone 4 | +4.611 | 0.019 |
| 24, 32, 23, … | further legal relocations | ≤ +1.6 | ≤ 0.001 |

**Chosen: action 26 — send one idle car to zone 9.** Critic value `−0.174`.
Every masked action has probability exactly 0 (masking is applied to the
logits before the softmax, not after sampling).

Note how close 26 and 1 are: 0.505 vs 0.474. The model is nearly indifferent
between serving the waiting rider and repositioning for the next one.

## 5. Where the relocation score comes from

Relocation logit = the pooled actor's contribution **+** four destination terms.

For the chosen zone 9:

| term | what it is | value |
|---|---|---|
| base actor | the pooled `Linear(256,33)` column | **+3.448** |
| A — key/query | `dest_key(zone_emb[9]) · self_query(self_ctx) / 16` | **+4.891** |
| B — raw bias | `dest_bias` on zone 9's own 8 features | +0.145 |
| C — travel offset | learned 2-d index embedding offset | −0.135 |
| D — real distance | `dist_cost(‖zone_xy[9] − own‖ × 10)`, `dist_norm = 0.4264` | −0.450 |
| | **total** | **+7.899** |

Three things worth saying out loud, because they are not what the code comments
imply:

1. **The pooled actor is not silent at convergence.** `encoder.py:228-230` zeroes
   `actor.weight[17:,:]` at initialisation so the destination head owns those
   columns, but 300 iterations of training moved it to **+3.448** — comparable to
   the destination head's own +4.451. "The destination head owns relocation" is
   true at init, not after training.
2. **Term A dominates.** It is +4.891 of a +4.451 destination sum; B and D pull
   the other way (−0.305 combined). The similarity term, not the demand or
   distance terms, is what put zone 9 on top here.
3. **Term D is a straight-line proxy**, not road distance — Euclidean on
   bbox-normalised centroids, per the `# ponytail` comment at `encoder.py:214`.
   The 2.08 km/move in the results table is *simulated road* distance, a
   different quantity.

## 6. Caveats that belong on the slide

- Single turn, single seed, single training seed. It illustrates the mechanism;
  it is not evidence that the policy is good.
- `neighbor_mask` is **not** adjacency. It is the acting zone's legal-target
  set, which under `neighbor_relocation_only=False` spans most of the map —
  which is why the zone attention came out nearly flat.
- `pending_passengers` duplicates `pending_orders[:,:,4]`, so 16 of the 295 dims
  are redundant and the order embedding sees that column twice.
- `demand_hist` at `hist_len=4, dt=0.5` covers **2.0 minutes**, i.e. 3.3% of the
  hour.
- IPPO's "decentralized" critic is a function of *all* zones via `zone_pool`,
  which feeds `h`, which feeds the critic. It is decentralized in input, not in
  receptive field.