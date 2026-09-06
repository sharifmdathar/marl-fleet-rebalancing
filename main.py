"""Entry point: roll out a random policy on the Bandra network.

Run:
    uv run python main.py              # interactive viewer (default)
    uv run python main.py -i           # interactive viewer (explicit)
    uv run python main.py -o out.mp4   # save MP4 and exit
"""
import argparse

from ride_gym import RidePoolEnv
from ride_gym.order_generator import OSMnxOrderGenerator
from ride_gym.osmnx_network import OSMnxNetwork
from ride_gym.visualize import TrajectoryRecorder
from ride_gym.enums import OrderStatus

from marl_fleet_rebalancing.agents.random_policy import random_actions
from marl_fleet_rebalancing.visualization import interactive_view, render_animation

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
    parser = argparse.ArgumentParser(description="Run random policy on Bandra West")
    parser.add_argument("-i", "--interactive", action="store_true", default=False,
                        help="Open interactive viewer instead of saving MP4")
    parser.add_argument("-o", "--output", type=str, default="episode.mp4",
                        help="Output MP4 path (default: episode.mp4)")
    parser.add_argument("--detailed", action="store_true", default=True,
                        help="Show driver IDs and routes in output")
    args = parser.parse_args()

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
        done = env.time >= HORIZON
        if step % 30 == 0 or done:
            served = sum(o.status == OrderStatus.COMPLETED for o in env.orders.values())
            cancelled = sum(o.status == OrderStatus.CANCELLED for o in env.orders.values())
            print(
                f"\r  t = {env.time:5.1f} / {HORIZON:.0f} min   |   "
                f"step {step:3d}   |   served {served:3d}   cancelled {cancelled:3d}   "
                f"pending {len(env._pending_ids):3d}",
                end="",
                flush=True,
            )
    print()
    print()

    if args.interactive:
        print("Opening interactive viewer...")
        interactive_view(rec, detailed=args.detailed)
    else:
        print(f"Rendering {len(rec.frames)} frames to {args.output}...")
        render_animation(rec, out_path=args.output, detailed=args.detailed)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
