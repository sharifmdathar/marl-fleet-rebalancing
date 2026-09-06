"""Custom visualization for MARL fleet rebalancing.

Draws a clean, readable animation from a TrajectoryRecorder without modifying
any ride_gym internals.

Usage (MP4):
    from ride_gym.visualize import TrajectoryRecorder
    from marl_fleet_rebalancing.visualization import render_animation

    rec = TrajectoryRecorder()
    # ... run episode, calling rec.snapshot(env) each step ...
    render_animation(rec, out_path="episode.mp4", fps=10)

Usage (detailed MP4):
    render_animation(rec, out_path="debug.mp4", fps=10, detailed=True)

Usage (interactive viewer):
    from marl_fleet_rebalancing.visualization import interactive_view

    rec = TrajectoryRecorder()
    # ... run episode ...
    interactive_view(rec)            # opens a window
    # Space = play/pause, arrows = step, click = select driver, Esc = deselect
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from ride_gym.visualize import TrajectoryRecorder, STATUS_COLORS, STATUS_LABEL
from ride_gym.enums import DriverStatus

STATUS_COLORS_DARK = {
    DriverStatus.IDLE:       "#616161",
    DriverStatus.TO_PICKUP:  "#145a8c",
    DriverStatus.TO_DROPOFF: "#1d7a1d",
    DriverStatus.RELOCATING: "#cc6600",
}


def _precompute_network_segments(network) -> tuple[list, tuple]:
    """Extract road segments as line-coordinates for fast bulk drawing.

    Returns (segments, bounds) where segments is a list of [(x0,y0),(x1,y1)]
    pairs and bounds is (x0, y0, x1, y1) with 5% padding around the actual
    node extent (tighter than network.bounds).
    """
    g = network.graph
    segs = []
    xs, ys = [], []
    for u, v in g.edges():
        xu, yu = g.nodes[u]["x"], g.nodes[u]["y"]
        xv, yv = g.nodes[v]["x"], g.nodes[v]["y"]
        segs.append([(xu, yu), (xv, yv)])
        xs.extend([xu, xv])
        ys.extend([yu, yv])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    bounds = (x0 - dx, y0 - dy, x1 + dx, y1 + dy)
    return segs, bounds


def _route_segments(
    network,
    location: tuple[float, float],
    task_points: list[tuple],
) -> list[tuple]:
    """Return line-segments for the planned route from *location* through
    each task point, using the real street path."""
    if not task_points:
        return []
    stops = [location] + [tp[2] for tp in task_points]
    segments: list[tuple] = []
    for a, b in zip(stops[:-1], stops[1:]):
        try:
            path = network.node_path(network.snap(a), network.snap(b))
            coords = [network.node_coord(i) for i in path]
        except Exception:
            coords = [a, b]
        for c0, c1 in zip(coords[:-1], coords[1:]):
            segments.append((c0, c1))
    return segments


def _count_drivers(frame: dict) -> dict[str, int]:
    """Tally drivers by coarse category."""
    counts = {"moving": 0, "idle": 0}
    for d in frame["drivers"].values():
        if d["status"] == DriverStatus.IDLE:
            counts["idle"] += 1
        else:
            counts["moving"] += 1
    return counts


def _count_orders(frame: dict) -> dict[str, int]:
    """Tally order states from the snapshot."""
    onboard = sum(
        d["onboard_passengers"] for d in frame["drivers"].values()
    )
    return {
        "waiting": len(frame["pending"]),
        "on_trip": onboard,
        "completed": frame["served"],
        "cancelled": frame["cancelled"],
    }


_STATS_TEMPLATE = (
    "SIMULATION\n"
    "  Time       {time}\n"
    "\n"
    "DRIVERS\n"
    "  Moving     {moving}\n"
    "  Idle       {idle}\n"
    "\n"
    "ORDERS\n"
    "  Waiting    {waiting}\n"
    "  On trip    {on_trip}\n"
    "  Completed  {completed}\n"
    "  Cancelled  {cancelled}"
)


def _draw_stats(ax, frame: dict, horizon: float, selected: tuple | None = None) -> None:
    """Render the statistics panel on the right-side axes."""
    ax.clear()
    ax.set_axis_off()
    dc = _count_drivers(frame)
    oc = _count_orders(frame)
    txt = _STATS_TEMPLATE.format(
        time=f"{frame['time']:.1f} / {horizon:.0f} min",
        **dc, **oc,
    )
    if selected is not None:
        did, d = selected
        txt += "\n\n" + _driver_info_text(did, d, None)
    ax.text(
        0.05, 0.95, txt,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )


def _draw_frame(
    ax,
    stats_ax,
    frame: dict,
    recorder: TrajectoryRecorder,
    precomputed_segs: list,
    bounds: tuple,
    detailed: bool,
) -> None:
    """Draw one animation frame."""
    ax.clear()

    lon0, lat0, lon1, lat1 = bounds
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    lc = LineCollection(precomputed_segs, colors="#dddddd", linewidths=0.4, zorder=0)
    ax.add_collection(lc)

    if frame["pending"]:
        px, py = zip(*frame["pending"])
        ax.scatter(
            px, py, s=20, facecolors="none", edgecolors="#d62728",
            linewidths=0.8, zorder=2, label="pending",
        )

    if detailed:
        for oid, (orig, dest) in frame["order_od"].items():
            ax.scatter(*orig, s=14, c="#1f77b4", zorder=3, marker="o")
            ax.scatter(*dest, s=14, c="#2ca02c", zorder=3, marker="o")

    by_status: dict[DriverStatus, list] = {}
    for did, d in frame["drivers"].items():
        by_status.setdefault(d["status"], []).append((did, d))

    for status, items in by_status.items():
        xs = [d["location"][0] for _, d in items]
        ys = [d["location"][1] for _, d in items]
        color = STATUS_COLORS.get(status, "#333333")
        ax.scatter(
            xs, ys, s=40, c=color, edgecolors="white",
            linewidths=0.3, zorder=5, label=STATUS_LABEL.get(status, str(status)),
        )

    if detailed:
        network = recorder.network
        for did, d in frame["drivers"].items():
            segs = _route_segments(network, d["location"], d["task_points"])
            if segs:
                points = [segs[0][0]]
                for seg in segs:
                    points.append(seg[1])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                color = STATUS_COLORS.get(d["status"], "#888888")
                ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.6, zorder=1)
                ax.annotate(
                    str(did),
                    xy=d["location"],
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                    zorder=6,
                )

    ax.legend(
        loc="upper left", fontsize=7, framealpha=0.9,
        markerscale=1.0, ncol=2,
    )
    ax.set_title(
        f"t = {frame['time']:.1f} / {recorder.horizon:.0f} min",
        fontsize=10, pad=4,
    )

    _draw_stats(stats_ax, frame, recorder.horizon)


def render_animation(
    recorder: TrajectoryRecorder,
    out_path: str = "episode.mp4",
    fps: int = 10,
    detailed: bool = False,
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 100,
) -> str:
    """Render a recorded trajectory to an MP4 with a clean custom layout.

    Parameters
    ----------
    recorder:
        A :class:`TrajectoryRecorder` populated over one episode.
    out_path:
        Output file (.mp4 uses FFMpeg, .gif uses Pillow).
    fps:
        Frames per second.
    detailed:
        If True, draw driver IDs, street-level routes, and order
        pickup/dropoff locations (debug/presentation mode).
    figsize:
        Matplotlib figure size (width, height).
    dpi:
        Output resolution.

    Returns
    -------
    The path to the saved file.
    """
    if not recorder.frames:
        raise ValueError("recorder has no frames; call snapshot() during the run.")

    precomputed_segs, bounds = _precompute_network_segments(recorder.network)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_map = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    def _draw(frame_idx: int):
        _draw_frame(
            ax_map, ax_stats,
            recorder.frames[frame_idx],
            recorder,
            precomputed_segs,
            bounds,
            detailed,
        )

    anim = animation.FuncAnimation(
        fig, _draw, frames=len(recorder.frames), interval=1000 / max(fps, 1),
    )

    if out_path.lower().endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


_DRIVER_INFO_TEMPLATE = (
    "SELECTED DRIVER  {did}\n"
    "  Status         {status}\n"
    "  Passengers     {passengers}\n"
    "  Assigned       {assigned}\n"
    "  Current task   {current_task}\n"
    "{route_info}"
)


def _driver_info_text(did: str, d: dict, network) -> str:
    """Build the info string for one selected driver."""
    route_info = ""
    if d["task_points"]:
        kind, oid, coord = d["task_points"][0]
        label = {"pickup": "pickup", "dropoff": "dropoff", "relocate": "relocate"}.get(kind, kind)
        current_task = f"{label}(order {oid})" if oid else label
        route_info = f"  Destination    ({coord[0]:.4f}, {coord[1]:.4f})"
    else:
        current_task = "(idle)"

    return _DRIVER_INFO_TEMPLATE.format(
        did=did,
        status=STATUS_LABEL.get(d["status"], str(d["status"])),
        passengers=d["onboard_passengers"],
        assigned=len(d["assigned_orders"]),
        current_task=current_task,
        route_info=route_info,
    )


def interactive_view(
    recorder: TrajectoryRecorder,
    detailed: bool = True,
    figsize: Tuple[float, float] = (14, 7),
    dpi: int = 100,
) -> None:
    """Open an interactive matplotlib window to explore the episode.

    Controls
    --------
    Space       play / pause
    Right       step one frame forward
    Left        step one frame backward
    Click       select a driver (highlights route + shows info)
    Escape      deselect current driver
    r           reset to frame 0
    q / close   quit

    Parameters
    ----------
    recorder:
        A :class:`TrajectoryRecorder` populated over one episode.
    detailed:
        If True, show driver IDs and street-level routes for all drivers.
    figsize:
        Matplotlib figure size.
    dpi:
        Display resolution.
    """
    if not recorder.frames:
        raise ValueError("recorder has no frames; call snapshot() during the run.")

    frames = recorder.frames
    network = recorder.network
    precomputed_segs, bounds = _precompute_network_segments(network)
    lon0, lat0, lon1, lat1 = bounds

    state = {
        "idx": 0,
        "playing": False,
        "selected_id": None,
        "all_drivers": [],
        "scatter": None,
    }

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_map = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    def _draw(idx: int):
        """Redraw the entire frame."""
        ax_map.clear()
        ax_stats.clear()

        frame = frames[idx]

        ax_map.set_xlim(lon0, lon1)
        ax_map.set_ylim(lat0, lat1)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        lc = LineCollection(precomputed_segs, colors="#dddddd", linewidths=0.4, zorder=0)
        ax_map.add_collection(lc)

        if frame["pending"]:
            px, py = zip(*frame["pending"])
            ax_map.scatter(
                px, py, s=20, facecolors="none", edgecolors="#d62728",
                linewidths=0.8, zorder=2, label="pending",
            )

        if detailed:
            for oid, (orig, dest) in frame["order_od"].items():
                ax_map.scatter(*orig, s=14, c="#1f77b4", zorder=3, marker="o")
                ax_map.scatter(*dest, s=14, c="#2ca02c", zorder=3, marker="o")

        all_drivers = [(did, d) for did, d in frame["drivers"].items()]
        state["all_drivers"] = all_drivers

        if not all_drivers:
            ax_map.set_title(f"t = {frame['time']:.1f} / {recorder.horizon:.0f} min")
            _draw_stats(ax_stats, frame, recorder.horizon, None)
            return

        xs = [d["location"][0] for _, d in all_drivers]
        ys = [d["location"][1] for _, d in all_drivers]
        colors = [STATUS_COLORS_DARK.get(d["status"], "#333333") for _, d in all_drivers]

        sel_did = state["selected_id"]
        sel_d = frame["drivers"].get(sel_did) if sel_did else None
        if sel_did and sel_d is None:
            state["selected_id"] = None
        edge_colors = ["white"] * len(all_drivers)
        sizes = [80] * len(all_drivers)
        if sel_did is not None and sel_d is not None:
            for i, (did, d) in enumerate(all_drivers):
                if did == sel_did:
                    edge_colors[i] = "#ff00ff"
                    sizes[i] = 160
                    break

        scatter = ax_map.scatter(
            xs, ys, s=sizes, c=colors, edgecolors=edge_colors,
            linewidths=1.0, zorder=5, picker=True, pickradius=8,
        )
        state["scatter"] = scatter

        if sel_d is not None and sel_d["task_points"]:
            first_task = sel_d["task_points"][0]
            task_kind = first_task[0]
            task_coord = first_task[2]
            TASK_STYLE = {
                "pickup":  ("#1f77b4", "^"),
                "dropoff": ("#2ca02c", "v"),
                "relocate": ("#ff7f0e", "s"),
            }
            leg_color, dest_marker = TASK_STYLE.get(task_kind, ("#ff00ff", "o"))
            ax_map.plot(
                [sel_d["location"][0], task_coord[0]],
                [sel_d["location"][1], task_coord[1]],
                color=leg_color, linewidth=1.2, alpha=0.9, zorder=4,
                linestyle="--",
            )
            ax_map.scatter(
                *task_coord, s=90, c="#d62728", edgecolors="white",
                linewidths=1.0, zorder=6, marker="o",
            )

        if detailed:
            for did, d in all_drivers:
                ax_map.annotate(
                    str(did),
                    xy=d["location"],
                    fontsize=5,
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    zorder=6,
                )

        if detailed and sel_did is not None:
            for did, d in all_drivers:
                if did == sel_did:
                    continue
                if not d["task_points"]:
                    continue
                dest = d["task_points"][0][2]
                ax_map.plot(
                    [d["location"][0], dest[0]],
                    [d["location"][1], dest[1]],
                    color="#bbbbbb", linewidth=0.5, alpha=0.4, zorder=1,
                    linestyle="--",
                )

        ax_map.legend(loc="upper left", fontsize=7, framealpha=0.9, markerscale=1.0, ncol=2)
        play_state = "\u25b6" if state["playing"] else "\u275a\u275a"
        sel_label = f"  |  selected: {sel_did}" if sel_did is not None else ""
        ax_map.set_title(
            f"{play_state}  t = {frame['time']:.1f} / {recorder.horizon:.0f} min"
            f"  (frame {idx + 1}/{len(frames)}){sel_label}",
            fontsize=10, pad=4,
        )

        sel_tuple = (sel_did, sel_d) if sel_did and sel_d else None
        _draw_stats(ax_stats, frame, recorder.horizon, sel_tuple)

        fig.canvas.draw_idle()

    def _on_key(event):
        src = state.get("ani_event_source")
        if event.key == " ":
            state["playing"] = not state["playing"]
            if src is not None:
                if state["playing"]:
                    src.start()
                else:
                    src.stop()
        elif event.key == "right":
            state["playing"] = False
            if src is not None:
                src.stop()
            state["idx"] = min(state["idx"] + 1, len(frames) - 1)
            _draw(state["idx"])
        elif event.key == "left":
            state["playing"] = False
            if src is not None:
                src.stop()
            state["idx"] = max(state["idx"] - 1, 0)
            _draw(state["idx"])
        elif event.key == "escape":
            state["selected_id"] = None
            _draw(state["idx"])
        elif event.key == "r":
            state["idx"] = 0
            state["selected_id"] = None
            state["playing"] = False
            if src is not None:
                src.stop()
            _draw(0)

    def _on_click(event):
        if event.mouseevent.inaxes != ax_map:
            return
        if event.artist != state["scatter"]:
            return
        if len(event.ind) == 0:
            return
        ind = event.ind[0]
        if ind < len(state["all_drivers"]):
            did, _d = state["all_drivers"][ind]
            if state["selected_id"] == did:
                state["selected_id"] = None
            else:
                state["selected_id"] = did
            _draw(state["idx"])

    def _animate(_frame_idx):
        if state["playing"]:
            state["idx"] = min(state["idx"] + 1, len(frames) - 1)
            if state["idx"] >= len(frames) - 1:
                state["playing"] = False
                src = state.get("ani_event_source")
                if src is not None:
                    src.stop()
        _draw(state["idx"])

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("pick_event", _on_click)

    ani = animation.FuncAnimation(
        fig, _animate, frames=len(frames), interval=100, repeat=False,
    )
    state["ani_event_source"] = ani.event_source
    ani.event_source.stop()

    _draw(0)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)
