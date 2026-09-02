"""Entry point: roll out a random policy on the Bandra network and render an MP4.

Run:
    uv run python main.py
"""
from ride_gym import RidePoolEnv
from ride_gym.order_generator import OSMnxOrderGenerator
from ride_gym.osmnx_network import OSMnxNetwork
from ride_gym.visualize import TrajectoryRecorder, render_animation
from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.agents.random_policy import random_actions

NETWORK_PATH = "data/bandra_west.gpickle"
SPEED_KMH = 30.0
NUM_DRIVERS = 30
DRIVER_CAPACITY = 4
DT = 0.1
HORIZON = 30.0
ORDER_TIMEOUT = 2.0
NUM_ORDERS = 400
MAX_PARTY_SIZE = 3
SEED = 0
OUT_PATH = "episode.mp4"
FPS = 10


def build_env() -> RidePoolEnv:
    network = OSMnxNetwork(graph_path=NETWORK_PATH, speed_kmh=SPEED_KMH)
    order_gen = OSMnxOrderGenerator(
        network=network,
        horizon=HORIZON,
        num_orders=NUM_ORDERS,
        arrival="poisson",
        max_party_size=MAX_PARTY_SIZE,
    )
    return RidePoolEnv(
        num_drivers=NUM_DRIVERS,
        driver_capacity=DRIVER_CAPACITY,
        dt=DT,
        horizon=HORIZON,
        order_timeout=ORDER_TIMEOUT,
        order_generator=order_gen,
        road_network=network,
    )


def main() -> None:
    env = build_env()
    rec = TrajectoryRecorder()
    obs, info = env.reset(seed=SEED)
    done = False
    step = 0
    while not done:
        actions = random_actions(env, obs)
        obs, rewards, dones, info = env.step(actions)
        rec.snapshot(env)
        step += 1
        done = dones["__all__"]
        if step % 30 == 0 or done:
            served = sum(o.status == OrderStatus.COMPLETED for o in env.orders.values())
            cancelled = sum(o.status == OrderStatus.CANCELLED for o in env.orders.values())
            print(
                f"\r  t = {env.time:5.1f} / {env.horizon:.0f} min   |   "
                f"step {step:3d}   |   served {served:3d}   cancelled {cancelled:3d}   "
                f"pending {len(env._pending_ids):3d}",
                end="",
                flush=True,
            )
    print()
    render_animation(rec, out_path=OUT_PATH, fps=FPS)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
