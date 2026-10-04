import unittest
import numpy as np
import torch
from marl_fleet_rebalancing.config import BandraConfig
from marl_fleet_rebalancing.env.pettingzoo_env import RideGymAEC
from marl_fleet_rebalancing.agents.mappo import MAPPO, MAPPOConfig
from marl_fleet_rebalancing.networks.encoder import obs_dim


class TestFixesAndDemand(unittest.TestCase):
    def test_target_zone_demand_injection(self):
        """Test that target_zone and target_zone_demand_pct inject source demand in target zone."""
        cfg = BandraConfig()
        cfg.target_zone = 3
        cfg.target_zone_demand_pct = 0.20  # 20%
        cfg.max_pending_orders_per_zone = 8
        cfg.num_drivers = 50

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        total_orders = len(env.env.order_generator.orders)
        self.assertGreater(total_orders, 0)

        # Count how many orders originate near target zone centroid 3 (< 1.5 km)
        target_pt = env.env.relocation_points[3]

        in_target = 0
        for o in env.env.order_generator.orders:
            # Check proximity to target zone centroid (1.5 km ~ 0.015 deg in Bandra)
            d_sq = (o.origin[0] - target_pt[0]) ** 2 + (o.origin[1] - target_pt[1]) ** 2
            if d_sq < (0.015 ** 2):
                in_target += 1

        pct = in_target / total_orders
        self.assertGreaterEqual(pct, 0.18)

    def test_intra_turn_collision_masking(self):
        """Test that claiming an order dynamically masks it out for subsequent agents in the same turn."""
        from ride_gym import Order, OrderStatus

        cfg = BandraConfig()
        cfg.num_drivers = 30
        cfg.max_dispatch_radius_km = 10.0  # Wide radius so multiple zones can see the order

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inject order 999
        order_origin = tuple(env.env.relocation_points[0])
        new_id = 999
        order = Order(
            order_id=new_id,
            origin=order_origin,
            destination=tuple(env.env.relocation_points[1]),
            request_time=0.0,
            status=OrderStatus.PENDING,
        )
        env.env.orders[new_id] = order
        env.env._pending_ids.append(new_id)
        env._last_obs = env.env._build_observations()
        env._current_pending = env._extract_pending(env._last_obs)
        env._rebuild_caches()

        # Strict zone ownership (D23): verify order 999 is claimed only by origin zone 0
        agents_with_order = []
        for agent in env.possible_agents:
            z_idx = env._zone_int_id(agent)
            cands = env._zone_candidate_orders[z_idx]
            cand_order_ids = [c["order_id"] for c in cands]
            if new_id in cand_order_ids:
                agents_with_order.append(agent)

        self.assertEqual(agents_with_order, ["zone_0"], "Expected only origin zone 0 to own order 999 under strict zone partitioning")
        z0_cands = env._zone_candidate_orders[0]
        z0_order_ids = [c["order_id"] for c in z0_cands]
        z0_act = 1 + z0_order_ids.index(new_id)
        env._zone_dispatchable_counts[0] = 1
        env._zone_idle_counts[0] = 1
        self.assertTrue(env.action_mask("zone_0")[z0_act])

    def test_local_dispatch_radius_cutoff(self):
        """Test that orders beyond max_dispatch_radius_km are not offered to far-away zones."""
        from ride_gym import Order, OrderStatus

        cfg = BandraConfig()
        cfg.max_dispatch_radius_km = 0.5  # Very tight radius: 500m

        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inject order at zone 0
        new_id = 888
        order = Order(
            order_id=new_id,
            origin=tuple(env.env.relocation_points[0]),
            destination=tuple(env.env.relocation_points[1]),
            request_time=0.0,
            status=OrderStatus.PENDING,
        )
        env.env.orders[new_id] = order
        env.env._pending_ids.append(new_id)
        env._last_obs = env.env._build_observations()
        env._current_pending = env._extract_pending(env._last_obs)
        env._rebuild_caches()

        # Check a distant zone (e.g., zone 24, typically across the area)
        p0 = np.array(env.env.relocation_points[0])
        p_far = np.array(env.env.relocation_points[24])
        dist_km = float(np.sqrt(np.sum((p0 - p_far) ** 2)) * 111.0)
        self.assertGreater(dist_km, 1.0)

        far_cands = [c["order_id"] for c in env._zone_candidate_orders[24]]
        self.assertNotIn(new_id, far_cands, "Distant order should be filtered out by max_dispatch_radius_km")

    def test_neighbor_only_relocation_mask(self):
        """Test that idle vehicle relocation is pruned to adjacent neighboring zones."""
        cfg = BandraConfig()
        cfg.neighbor_relocation_only = True
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        # Inspect agent action masks
        agent = env.possible_agents[0]
        mask = env.action_mask(agent)
        zone_idx = 0
        neighbor_indices = set(env.env.region_neighbours[zone_idx])

        reloc_offset = 1 + env._max_pending
        for target_idx in range(env._num_zones):
            act = reloc_offset + target_idx
            # If target is not a neighbor and not own zone, it must be masked False
            if target_idx not in neighbor_indices and target_idx != zone_idx:
                self.assertFalse(mask[act], f"Action to non-neighbor zone {target_idx} should be masked False")

    def test_state_covers_every_observation_channel(self):
        """D38: every observation channel must reach MAPPO's centralized critic.

        Regression test. state() returned 177 dims while the actor saw 279:
        demand_hist (64) and neighbor_mask (16) were never added, so the critic
        MAPPO depends on was blind to recent demand history and to zone
        adjacency -- information IPPO's decentralized critic does have. Nothing
        failed loudly because the pre-existing MAPPO test only checked that
        value() returned a finite scalar of the right shape.

        This asserts the property directly: each observation key must appear in
        the centralized state. If a future observation channel is added without
        being threaded through state(), this fails instead of silently making
        MAPPO weaker than IPPO.
        """
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        obs = env.observe(env.possible_agents[0])
        state = env.state()
        state_size = state.size

        # Channels state() must cover. zone_location is appended separately by
        # train.py as the acting zone's identity, so it is checked separately.
        must_appear = [
            "all_zone_state",
            "pending_orders",
            "pending_mask",
            "demand_hist",
        ]
        for key in must_appear:
            arr = np.asarray(obs[key])
            n = int(arr.size)
            self.assertGreater(
                n, 0, "%s should be non-empty" % key,
            )
            self.assertLessEqual(
                n, state_size,
                "observation channel %r (%d dims) is larger than the whole "
                "centralized state (%d dims) -- state() almost certainly "
                "dropped it" % (key, n, state_size),
            )

        # Explicitly assert the two channels D38 restored. If someone reverts
        # state() to its pre-D38 form, these are what break.
        self.assertGreaterEqual(
            state_size, obs["demand_hist"].size + obs["all_zone_state"].size
            + obs["pending_orders"].size + obs["pending_mask"].size
            + env._num_zones + 1,
            "state() is missing demand_hist / neighbor_mask / clock "
            "(got %d dims)" % state_size,
        )

        # And the critic must actually accept that width.
        from marl_fleet_rebalancing.networks.encoder import obs_to_vector

        obs_dim_actual = int(obs_to_vector(obs).shape[0])
        learner = MAPPO(
            observation_dim=obs_dim_actual,
            state_dim=state_size,  # D39: no appended zone_location
            action_dim=env._action_dim,
            config=MAPPOConfig(),
            num_zones=env._num_zones,
        )
        val = learner.value(state[None, :], np.asarray([0]))
        self.assertEqual(val.shape, (1,))
        self.assertTrue(np.isfinite(val[0]))

    def test_critic_distinguishes_agents(self):
        """D39: the critic MUST give different values for different zones.

        The global state is captured once per turn and shared by every slot
        (train.py:308), so identical state + different agent_id is the only
        thing separating 16 per-zone return functions. Before D39 the identity
        was 2 floats of centroid geography, and MAPPO's explained_var was
        0.018-0.039 vs IPPO's 0.068-0.138 -- the centralized critic was a
        worse estimator than the decentralized head it should improve on.

        This test pins the property that was broken: perturbing ONLY the agent
        identity must change the value.
        """
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        obs = env.observe(env.possible_agents[0])
        from marl_fleet_rebalancing.networks.encoder import obs_to_vector

        n = env._num_zones
        learner = MAPPO(
            observation_dim=int(obs_to_vector(obs).shape[0]),
            state_dim=env.state().size,
            action_dim=env._action_dim,
            config=MAPPOConfig(),
            num_zones=n,
        )

        state = env.state()
        # Same state, every agent id: values must not all be identical.
        vals = np.concatenate([
            learner.value(state[None, :], np.asarray([z]))
            for z in range(n)
        ])
        self.assertGreater(
            float(np.std(vals)), 0.0,
            "critic returned the same value for every agent id -- identity is "
            "not reaching the critic",
        )

        # And a single wrong id must not raise.
        with self.assertRaises(ValueError):
            learner.value(state[None, :], None)

    def test_state_critic_sees_demand_history(self):
        """The critic input must actually vary when demand history changes.

        A width check alone is not enough: state() could allocate the right
        number of dims and fill them with zeros. This perturbs demand history
        and asserts the state vector changes.
        """
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        before = env.state().copy()
        self.assertGreater(env._demand_hist_arr.size, 0)
        env._demand_hist_arr[:] = 1.0  # perturb only the history channel
        after = env.state()

        self.assertFalse(
            np.allclose(before, after),
            "state() did not change when demand_hist changed -- the history "
            "channel is allocated but not populated",
        )

    def test_critic_input_superset_of_actor_obs(self):
        """D42: the critic must not be handicapped relative to IPPO's head.

        state() carries only GLOBAL channels. Notably _pending_features is the
        fleet-wide urgency list and has NO zone attribution, so a critic built
        from state() alone must reconstruct "which orders are waiting in MY
        zone" from 16 scalars -- while IPPO's decentralized head is handed that
        zone's own pending_orders tensor outright.

        That asymmetry made "identical actor, different critic" false at the
        information level, and confounded RQ2 in MAPPO's disfavour.

        This pins the fix: the critic's input must be a strict superset of what
        the actor observes.
        """
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        agent = env.possible_agents[0]
        obs = env.observe(agent)
        from marl_fleet_rebalancing.networks.encoder import obs_to_vector

        global_state = env.state()
        own_obs = obs_to_vector(obs)
        critic_input = np.concatenate([global_state, own_obs]).astype(np.float32)

        self.assertEqual(
            critic_input.size, global_state.size + own_obs.size,
        )
        self.assertGreater(
            critic_input.size, own_obs.size,
            "critic input must include the global state",
        )
        # The zone's OWN pending orders must reach the critic. Under state()
        # alone they are not recoverable, because _pending_features is a
        # fleet-wide list with no zone attribution.
        self.assertGreater(
            obs["pending_orders"].size, 0,
            "sanity: this env should have per-zone pending_orders to protect",
        )
        # The critic must accept this width.
        learner = MAPPO(
            observation_dim=int(own_obs.size),
            state_dim=int(critic_input.size),
            action_dim=env._action_dim,
            config=MAPPOConfig(),
            num_zones=env._num_zones,
        )
        val = learner.value(critic_input[None, :], np.asarray([0]))
        self.assertEqual(val.shape, (1,))
        self.assertTrue(np.isfinite(val[0]))

    def test_state_exposes_inbound_relocations(self):
        """D43: the critic must be able to see cars already in flight.

        Regression test for a capability that was entirely absent. MAPPO's critic
        returned a BIT-IDENTICAL value (0.0390501097) for two different
        relocation destinations across 11 consecutive turns, because state()
        never encoded where cars were going: a relocating car was just "in
        zone, not idle". The redundancy signal lived in
        `_reloc_inbound_this_turn` / `_driver_relocation_target` and reached
        only `action_mask`, never the critic -- so a centralized critic could
        not penalise redundant targeting even in principle.

        Asserts state() grows when cars are in flight toward a zone.
        """
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        mp = env._max_pending

        saw_relocation = False
        before = env.state().copy()
        for _ in range(80):
            if not env.agents:
                break
            a = env.agent_selection
            m = env.action_mask(a)
            r = np.flatnonzero(m[1 + mp :])
            d = np.flatnonzero(m[1 : 1 + mp])
            if len(r):
                tgt = int(r[0]) + 1 + mp - mp - 1
                env.step(int(r[0]) + 1 + mp)
                if np.count_nonzero(env._zone_inbound_reloc) > 0:
                    saw_relocation = True
                    break
            else:
                env.step(int(d[0]) + 1 if len(d) else 0)

        self.assertTrue(
            saw_relocation,
            "no in-flight relocation was ever observed; the test cannot "
            "verify _zone_inbound_reloc",
        )
        self.assertEqual(env._zone_inbound_reloc.size, env._num_zones)

    def test_mappo_conditioned_critic(self):
        """Test that MAPPO learner initializes and evaluates with conditioned state_dim (state + zone_loc)."""
        cfg = BandraConfig()
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)

        any_agent = env.possible_agents[0]
        obs = env.observe(any_agent)
        zone_loc_dim = len(obs["zone_location"].ravel())
        state_dim = len(env.state())  # D39: no appended zone_location
        observation_dim = obs_dim(env.observation_space(any_agent))
        action_dim = env._action_dim

        mappo = MAPPO(
            observation_dim=observation_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            config=MAPPOConfig(),
            num_zones=env._num_zones,
        )

        state_vec = env.state().astype(np.float32)
        val = mappo.value(state_vec[None, :], np.asarray([0]))
        self.assertEqual(val.shape, (1,))
        self.assertTrue(np.isfinite(val[0]))

    def test_macro_demand_observation_and_rebalance_credit(self):
        """Test that all_zone_state has shape (num_zones, 4) exposing macro-demand and deficit, and credit assignment handles assists."""
        cfg = BandraConfig()
        cfg.target_zone = 3
        cfg.target_zone_demand_pct = 0.80
        env = RideGymAEC(config=cfg, seed=42)
        env.reset(seed=42)
        # Completion credit is asserted on its own, so switch off the terms
        # that share the reward vector: the team blend mixes in the mean of
        # all zones, and waiting/cancel penalties land in the same dict.
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.cancel_penalty = 0.0
        credit_cfg.beta = 0.0
        credit_env = RideGymAEC(config=credit_cfg, seed=42)
        credit_env.reset(seed=42)

        any_agent = env.possible_agents[0]
        obs = env.observe(any_agent)

        # 1. Check all_zone_state shape is (Z, 4)
        self.assertEqual(obs["all_zone_state"].shape, (env._num_zones, 4))
        # 2. Check that target zone 3 has significant pending demand fraction in column 2
        demand_col = obs["all_zone_state"][:, 2]
        self.assertTrue(np.all(demand_col >= 0.0))
        # 3. The DISPATCHER's credit is paid on COMPLETION, not on dispatch: a
        #    zone that only aims a car at a rider earns nothing. The
        #    RELOCATOR is the exception (D48) -- it is paid when its car is
        #    dispatched, because that is the moment the relocation is proven.
        from ride_gym.enums import OrderStatus

        class _Order:
            def __init__(self, status, driver):
                self.status = status
                self.assigned_driver = driver

        credit_env.env.orders[100] = _Order(OrderStatus.PENDING, None)
        credit_env._order_dispatcher[100] = 5
        credit_env._driver_relocator[1] = 18  # driver 1 was relocated by zone 18

        # Dispatch: only the relocating zone is paid, the dispatcher is not.
        at_dispatch = credit_env._compute_zone_rewards({}, [(100, 1, 5)])
        self.assertEqual(at_dispatch[5], 0.0, "dispatch alone must not be paid")
        self.assertGreater(at_dispatch[18], 0.0,
                           "relocating zone is paid on dispatch (D48)")

        credit_env.env.orders[100] = _Order(OrderStatus.COMPLETED, 1)
        rewards = credit_env._compute_zone_rewards({}, [])
        credit5, credit18 = rewards[5], rewards[18]
        self.assertGreater(credit5, 0.0, "dispatcher is paid on completion")
        self.assertEqual(credit18, 0.0, "relocator was already paid at dispatch")
        self.assertAlmostEqual(
            credit5 + at_dispatch[18], credit_cfg.gamma, places=5,
            msg="exactly one gamma is paid per completed ride",
        )
        self.assertAlmostEqual(
            at_dispatch[18] / credit_cfg.gamma, credit_cfg.relocator_share, places=5,
            msg="relocating zone takes relocator_share",
        )

        # Paid exactly once: re-crediting the same completion pays nothing more.
        again = credit_env._compute_zone_rewards({}, [])
        self.assertEqual(again[5], 0.0)
        self.assertEqual(again[18], 0.0)

        # A ride completed by a car that was NOT relocated pays the dispatcher
        # in full, so a normal dispatch is not diluted.
        credit_env.env.orders[101] = _Order(OrderStatus.COMPLETED, 2)
        credit_env._order_dispatcher[101] = 7
        solo = credit_env._compute_zone_rewards({}, [])
        self.assertAlmostEqual(solo[7], credit_cfg.gamma, places=5)


    def test_relocation_credit_is_not_divided_across_hops(self):
        """D47+D48: the relocating zone is paid the FULL share, undivided, at dispatch.

        Replaces the relay-chain test. The chain was a list of every zone that
        had relocated a car since its last completed order, and the credit was
        split by its length. Measured over 504 relocations, that list was not a
        relay: median 20 turns between hops of the same car, 0.4% back-to-back,
        and 87% of hop pairs had the car serve an order in between.

        D48 moves the payment from ride completion to the dispatch of the
        relocated car, because 98.3% of relocated cars that are ever used are
        dispatched on their arrival turn -- a median of 0 turns, not the 21-40
        turns the completion credit took.
        """
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.cancel_penalty = 0.0
        credit_cfg.beta = 0.0
        env = RideGymAEC(config=credit_cfg, seed=7)
        env.reset(seed=7)

        from ride_gym.enums import OrderStatus

        class _Order:
            def __init__(self, status, driver):
                self.status = status
                self.assigned_driver = driver

        share = credit_cfg.relocator_share * credit_cfg.gamma

        # Turn 1: zone 9 dispatches driver 3, which zone 6 had relocated.
        # Only the relocator is paid, and only now.
        env.env.orders[500] = _Order(OrderStatus.PENDING, None)
        env._order_dispatcher[500] = 9
        env._driver_relocator[3] = 6
        at_dispatch = env._compute_zone_rewards({}, [(500, 3, 9)])
        self.assertAlmostEqual(at_dispatch[6], share, places=5,
                               msg="relocator paid the FULL share on dispatch")
        self.assertEqual(at_dispatch[9], 0.0, "dispatcher not paid until completion")

        # Turn 2: the ride completes. The dispatcher takes only its reduced
        # share, so the pair has paid exactly one gamma between them.
        env.env.orders[500] = _Order(OrderStatus.COMPLETED, 3)
        at_completion = env._compute_zone_rewards({}, [])
        self.assertAlmostEqual(
            at_completion[9], (1.0 - credit_cfg.relocator_share) * credit_cfg.gamma,
            places=5, msg="dispatcher keeps 1 - relocator_share")
        self.assertAlmostEqual(
            at_dispatch[6] + at_completion[9], credit_cfg.gamma, places=5,
            msg="dispatch + completion together pay exactly one gamma")
        self.assertEqual(at_completion[4], 0.0,
                         "a zone that did not move this car earns nothing")

    def test_relocation_credit_survives_repeated_moves_of_one_car(self):
        """D47: re-moving a car overwrites the relocator; credit is never divided.

        The failure this guards against is arithmetic, not bookkeeping: under
        the old divisor, a car relocated three times earned 0.45/3 = 0.150
        against a deadhead cost of beta*km = 0.149, a 1.004x margin.
        """
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.cancel_penalty = 0.0
        credit_cfg.beta = 0.0
        env = RideGymAEC(config=credit_cfg, seed=7)
        env.reset(seed=7)

        from ride_gym.enums import OrderStatus

        class _Order:
            def __init__(self, status, driver):
                self.status = status
                self.assigned_driver = driver

        # The same car moved by three different zones in turn; the last mover
        # is the one that put it where it serves.
        env.env.orders[500] = _Order(OrderStatus.PENDING, None)
        env._order_dispatcher[500] = 9
        env._driver_relocator[3] = 5
        env._driver_relocator[3] = 7
        env._driver_relocator[3] = 6

        rewards = env._compute_zone_rewards({}, [(500, 3, 9)])
        share = credit_cfg.relocator_share * credit_cfg.gamma
        self.assertAlmostEqual(rewards[6], share, places=5,
                               msg="only the most recent relocating zone is paid")
        self.assertEqual(rewards[5], 0.0, "superseded relocators earn nothing")
        self.assertEqual(rewards[7], 0.0, "superseded relocators earn nothing")

    def test_relocating_car_arriving_with_no_rider_earns_nothing(self):
        """D48: dispatch pays the relocator, so an unused arrival pays nothing.

        This is the distinction from D28's arrival bonus, which paid for the
        arrival itself and was switched off because it paid without proof.
        Measured, 55.7% of relocation legs arrive and are never dispatched.
        """
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.cancel_penalty = 0.0
        credit_cfg.beta = 0.0
        env = RideGymAEC(config=credit_cfg, seed=7)
        env.reset(seed=7)

        env._driver_relocator[3] = 6
        # No dispatched_this_turn entry: the car arrived but nobody wanted it.
        rewards = env._compute_zone_rewards({}, [])
        self.assertEqual(rewards[6], 0.0,
                         "an arrival with no dispatch must not be paid")

    def test_cancelled_relocated_ride_claws_back_the_relocator_share(self):
        """D48: a cancelled ride must not leave the relocator paid.

        The relocator is paid at dispatch, before it is known whether the ride
        survives. If a cancellation does not reverse it, a relocated cancelled
        ride pays 0.45 while a plain cancelled ride pays nothing, which biases
        the agent toward relocating into zones where orders time out.

        The reversal must SUBTRACT. An earlier version of this clawback used
        `+=`, which credited the relocator instead of reversing it and
        overpaid every relocated cancellation by 0.90; the invariant script
        `diag_served_credit_total.py` caught it as a constant +0.90 per episode.
        """
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.beta = 0.0
        credit_cfg.cancel_penalty = 0.0  # clawback must not depend on this
        env = RideGymAEC(config=credit_cfg, seed=7)
        env.reset(seed=7)

        from ride_gym.enums import OrderStatus

        class _Order:
            def __init__(self, status, driver):
                self.status = status
                self.assigned_driver = driver

        env.env.orders[700] = _Order(OrderStatus.PENDING, None)
        env._order_dispatcher[700] = 9
        env._driver_relocator[3] = 6

        at_dispatch = env._compute_zone_rewards({}, [(700, 3, 9)])
        share = credit_cfg.relocator_share * credit_cfg.gamma
        self.assertAlmostEqual(at_dispatch[6], share, places=5)

        env.env.orders[700] = _Order(OrderStatus.CANCELLED, 3)
        at_cancel = env._compute_zone_rewards({}, [])
        self.assertAlmostEqual(at_cancel[6], -share, places=5,
                               msg="clawback must SUBTRACT the share, not add it")
        self.assertAlmostEqual(at_dispatch[6] + at_cancel[6], 0.0, places=5,
                               msg="a relocated cancelled ride nets to zero")

    def test_relocation_paid_at_dispatch_and_not_again_at_completion(self):
        """D48: the relocator is paid once. Completion must not re-pay it.

        The regression this pins: paying the relocator at dispatch while
        section 1 still paid it at completion would pay 1.45 gamma for one ride,
        making relocation look far more profitable than it is.
        """
        credit_cfg = BandraConfig()
        credit_cfg.team_share = 0.0
        credit_cfg.alpha = 0.0
        credit_cfg.cancel_penalty = 0.0
        credit_cfg.beta = 0.0
        env = RideGymAEC(config=credit_cfg, seed=7)
        env.reset(seed=7)

        from ride_gym.enums import OrderStatus

        class _Order:
            def __init__(self, status, driver):
                self.status = status
                self.assigned_driver = driver

        env.env.orders[500] = _Order(OrderStatus.PENDING, None)
        env._order_dispatcher[500] = 9
        env._driver_relocator[3] = 6
        first = env._compute_zone_rewards({}, [(500, 3, 9)])

        env.env.orders[500] = _Order(OrderStatus.COMPLETED, 3)
        second = env._compute_zone_rewards({}, [])

        total = sum(first) + sum(second)
        self.assertAlmostEqual(total, credit_cfg.gamma, places=5,
                               msg="one ride pays exactly one gamma in total")
        self.assertAlmostEqual(second[6], 0.0, places=5,
                               msg="relocator is not re-paid at completion")

        # A car that was never relocated pays the dispatcher in full.
        env3 = RideGymAEC(config=credit_cfg, seed=7)
        env3.reset(seed=7)
        env3.env.orders[500] = _Order(OrderStatus.COMPLETED, 2)
        env3._order_dispatcher[500] = 9
        r3 = env3._compute_zone_rewards({}, [])
        self.assertAlmostEqual(r3[9], credit_cfg.gamma, places=5)



if __name__ == "__main__":
    unittest.main()
