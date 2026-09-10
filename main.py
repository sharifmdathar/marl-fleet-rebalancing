"""Entry point: roll out a policy (--policy random|heuristic) on the Bandra network.

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
from marl_fleet_rebalancing.baselines.heuristic_baseline import heuristic_actions, reset_baseline
from marl_fleet_rebalancing.visualization import interactive_view, render_animation
from marl_fleet_rebalancing.config import BandraConfig


def build_env(config: BandraConfig | None = None) -> RidePoolEnv:
    if config is None:
        config = BandraConfig()

    network = OSMnxNetwork(graph_path=config.network_path, speed_kmh=config.speed_kmh)
    order_gen = OSMnxOrderGenerator(
        network=network,
        horizon=config.horizon,
        num_orders=config.num_orders,
        arrival="poisson",
        max_party_size=config.max_party_size,
    )
    return RidePoolEnv(
        num_drivers=config.num_drivers,
        driver_capacity=config.driver_capacity,
        dt=config.dt,
        horizon=config.horizon,
        order_timeout=config.order_timeout,
        order_generator=order_gen,
        road_network=network,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fleet policy on Bandra West")
    parser.add_argument("--policy", choices=["random", "heuristic"], default="random",
                        help="Policy to execute (default: random)")
    parser.add_argument("-i", "--interactive", action="store_true", default=False,
                        help="Open interactive viewer instead of saving MP4")
    parser.add_argument("-o", "--output", type=str, default="episode.mp4",
                        help="Output MP4 path (default: episode.mp4)")
    parser.add_argument("--detailed", action="store_true", default=True,
                        help="Show driver IDs and routes in output")
    args = parser.parse_args()

    cfg = BandraConfig()
    env = build_env(cfg)
    rec = TrajectoryRecorder()
    rec.relocation_points = list(getattr(env, "relocation_points", []))
    obs, info = env.reset(seed=cfg.seed)
    if args.policy == "heuristic":
        reset_baseline()

    done = False
    step = 0
    while not done:
        if args.policy == "heuristic":
            actions = heuristic_actions(env, obs)
        else:
            actions = random_actions(env, obs)
        obs, rewards, dones, info = env.step(actions)
        rec.snapshot(env)
        step += 1
        done = env.time >= cfg.horizon
        if step % 30 == 0 or done:
            served = sum(o.status == OrderStatus.COMPLETED for o in env.orders.values())
            cancelled = sum(o.status == OrderStatus.CANCELLED for o in env.orders.values())
            print(
                f"\r  t = {env.time:5.1f} / {cfg.horizon:.0f} min   |   "
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
