# Literature — condensed notes & the gap

## The gap sentence (use this, verbatim, in the report)

Robust and distributionally robust rebalancing under demand uncertainty exists
(MIVR/Guo et al. 2021, Miao et al. 2017, PD-RSAC 2026), but treats uncertainty
as a TRAINING/DESIGN-time technique. The frozen-policy EVALUATION protocol —
train on one demand distribution, freeze, stress-test across a parameterized
family of demand shifts, and compare training paradigms (centralized-critic
MAPPO vs independent IPPO) against a tuned classical baseline — has not been
run for ride-hailing/fleet rebalancing.

## Adjacent work a reviewer will ask about (differentiate, don't ignore)

| Paper | What it does | Why it's not us |
|---|---|---|
| Fujimoto et al. 2024 (arXiv:2402.03590) | Post-training distribution-shift evaluation method | Atari + power grid, not fleets; we adopt the method |
| Yu et al. 2022 (MAPPO, arXiv:2103.01955) | MAPPO vs IPPO benchmark | In-distribution, games only |
| de Witt et al. 2020 (IPPO, arXiv:2011.09533) | Independent learning strong on SMAC | Non-stationarity during training ≠ demand shift after |
| Hu et al. 2021 (POMAC, arXiv:2106.14334) | Policy overfitting in cooperative training | Training-time fix; motivates (not concludes) our hypothesis |
| SMACv2 (Ellis et al., NeurIPS 2023) | Shows MARL memorizes scenarios; randomization fixes | Variation WITHIN training distribution, not frozen-policy shift |
| DG-MAPPO (arXiv:2601.17069, 2026) | Argues CTDE has train–test mismatch; proposes comm-based fix | Motivation + algorithm, not a measurement study |
| M3DDPG / robust MARL | Minimax robustness to adversarial dynamics | Training-time fix, adversarial ≠ natural demand shift |
| Melting Pot | Generalization to unseen co-players | Agent-population shift, different axis |
| Guo et al. 2021 (MIVR, TR-B) | Robust optimization rebalancing under demand uncertainty | Training-time robustness; also our baseline inspiration |
| Miao et al. 2017 | Distributionally robust vehicle balancing | Same as above |
| Wen, Zhao & Jaillet 2017 | RL rebalancing | No shift evaluation |
| Lee et al. 2024 (arXiv:2412.00276) | MADDPG rebalancing during metro disruptions | SUPPLY shock, not demand-pattern shift |
| PD-RSAC (2026) | Distributionally robust RL for EV dispatch | A fix, not a measurement |
| RLSCALE-BENCH (2026) | Tuned-baseline lesson (autoscaling) | Different domain; we import the methodology |
| Donatus et al. 2025 (survey, arXiv:2508.20315) | MARL in transportation survey | Confirms our question isn't a main topic |

## Open items

- [ ] Verify RLSCALE-BENCH and PD-RSAC arXiv IDs (couldn't be pulled during search).
- [ ] Re-run advisor's search terms on Scholar/Semantic Scholar before writing
      related work: "MARL out-of-distribution generalization", "CTDE distribution
      shift", "fleet rebalancing unseen demand", "ride-hailing domain shift RL".
- [ ] Find and re-read the paper we remember claiming "MAPPO less robust due to
      overfitting" — confirm what it actually varied before citing it.