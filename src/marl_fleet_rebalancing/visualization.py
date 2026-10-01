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
    interactive_view(rec)            # opens an interactive control window
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Slider, Button

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


def _make_legend_handles(
    show_zones: bool = False,
    show_od: bool = True,
    has_target_zone: bool = False,
) -> list[Line2D]:
    """Build permanent proxy handles for the map legend with distinct shapes and colors."""
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Car (Idle)",
               markerfacecolor=STATUS_COLORS_DARK[DriverStatus.IDLE], markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Car (To Pickup)",
               markerfacecolor=STATUS_COLORS_DARK[DriverStatus.TO_PICKUP], markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Car (With Rider)",
               markerfacecolor=STATUS_COLORS_DARK[DriverStatus.TO_DROPOFF], markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Car (Relocating)",
               markerfacecolor=STATUS_COLORS_DARK[DriverStatus.RELOCATING], markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="o", color="#d62728", fillstyle="none",
               label="Rider Waiting", markersize=7, markeredgewidth=1.8),
    ]
    if show_od:
        handles.extend([
            Line2D([0], [0], marker="^", color="w", label="Pickup Point (▲)",
                   markerfacecolor="#00e5ff", markeredgecolor="#006064", markersize=7.5),
            Line2D([0], [0], marker="v", color="w", label="Dropoff Point (▼)",
                   markerfacecolor="#76ff03", markeredgecolor="#1b5e20", markersize=7.5),
        ])
    if show_zones:
        handles.append(
            Line2D([0], [0], marker="s", color="w", label="Zone Centroid (■)",
                   markerfacecolor="#9c27b0", markeredgecolor="white", markersize=6)
        )
    if has_target_zone:
        handles.append(
            Line2D([0], [0], marker="o", color="#ffd700", fillstyle="none",
                   label="Surge Zone (★)", markersize=8, markeredgewidth=2.0, linestyle="--")
        )
    return handles


_STATS_TEMPLATE = (
    "SIMULATION\n"
    "  Time         {time}\n"
    "\n"
    "DRIVERS\n"
    "  Moving       {moving}\n"
    "  Idle         {idle}\n"
    "\n"
    "ORDERS\n"
    "  Waiting      {waiting}\n"
    "  On trip      {on_trip}\n"
    "  Completed    {completed}\n"
    "  Cancelled    {cancelled}\n"
    "  Service rate {sr}\n"
    "\n"
    "CONTROLS\n"
    "  [Space]   Play / Pause\n"
    "  [← / →]   Step 1 frame\n"
    "  [↑ / ↓]   Jump 10 frames\n"
    "  [r]       Reset to start\n"
    "  [z]       Toggle zones\n"
    "  [m]       Toggle demand heatmap\n"
    "  [f]       Cycle speed\n"
    "  [Click]   Select driver\n"
    "  [Esc]     Deselect"
)


def _draw_stats(ax, frame: dict, horizon: float, selected: tuple | None = None) -> None:
    """Render the statistics panel on the right-side axes."""
    ax.clear()
    ax.set_axis_off()
    dc = _count_drivers(frame)
    oc = _count_orders(frame)
    tot = oc["completed"] + oc["cancelled"]
    sr_str = f"{oc['completed'] / tot * 100:.1f}%" if tot > 0 else "0.0%"
    txt = _STATS_TEMPLATE.format(
        time=f"{frame['time']:.1f} / {horizon:.0f} min",
        sr=sr_str,
        **dc, **oc,
    )
    if selected is not None:
        did, d = selected
        txt += "\n\n" + _driver_info_text(did, d, None)
    ax.text(
        0.05, 0.95, txt,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )


def _compute_zone_polygons(
    reloc_pts: list[tuple[float, float]] | None,
    bounds: tuple[float, float, float, float],
) -> list:
    """Compute Voronoi polygons bounded by the map bounds for each zone centroid."""
    if not reloc_pts or len(reloc_pts) < 2:
        return []
    try:
        import numpy as np
        from shapely.geometry import MultiPoint, Point, box
        from shapely.ops import voronoi_diagram

        lon0, lat0, lon1, lat1 = bounds
        bbox = box(lon0, lat0, lon1, lat1)
        mp = MultiPoint(reloc_pts)
        v_diagram = voronoi_diagram(mp, envelope=bbox)

        zone_polys = [None] * len(reloc_pts)
        cents = [Point(pt) for pt in reloc_pts]
        for geom in v_diagram.geoms:
            clipped = geom.intersection(bbox)
            if clipped.is_empty:
                continue
            for i, cpt in enumerate(cents):
                if clipped.contains(cpt) or clipped.distance(cpt) < 1e-5:
                    zone_polys[i] = clipped
                    break
        return zone_polys
    except Exception:
        return []


def _draw_zone_borders(ax, zone_polys) -> None:
    """Outline every zone region so the partition is readable without heat.

    The zone centroids alone do not show where one region ends and the next
    begins, which makes it impossible to tell whether a car is inside the
    zone whose demand it is serving.
    """
    import numpy as np

    for poly in zone_polys:
        if poly is None or poly.is_empty:
            continue
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            coords = np.asarray(g.exterior.coords)
            ax.plot(
                coords[:, 0], coords[:, 1],
                color="#1565c0", linewidth=1.1, alpha=0.75,
                linestyle="--", zorder=1,
            )


def _draw_zone_demand_heatmap(
    ax,
    frame: dict,
    reloc_pts: list[tuple[float, float]] | None,
    zone_polys: list,
    show_heatmap: bool = True,
) -> None:
    """Draw a zone outline-based heatmap colored by pending demand intensity."""
    if not show_heatmap or not reloc_pts or not zone_polys:
        return

    import numpy as np

    n_zones = len(reloc_pts)
    cents = np.asarray(reloc_pts, dtype=float)

    if frame.get("pending"):
        origins = np.asarray(frame["pending"], dtype=float)
        dists = ((origins[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        nearest = dists.argmin(axis=1)
        counts = np.bincount(nearest, minlength=n_zones)
    else:
        counts = np.zeros(n_zones, dtype=int)

    max_c = max(int(np.max(counts)), 1)
    cmap = plt.cm.YlOrRd

    for zi, poly in enumerate(zone_polys):
        if poly is None or poly.is_empty:
            continue
        c = counts[zi]
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            coords = np.asarray(g.exterior.coords)
            if c > 0:
                norm_val = min(1.0, 0.25 + 0.75 * (c / max(max_c, 3)))
                face_color = cmap(norm_val)
                patch = MplPolygon(
                    coords,
                    closed=True,
                    facecolor=face_color,
                    edgecolor="#e65100",
                    linewidth=1.2,
                    alpha=0.35,
                    zorder=1,
                )
            else:
                patch = MplPolygon(
                    coords,
                    closed=True,
                    facecolor="none",
                    edgecolor="#9c27b0",
                    linewidth=0.5,
                    linestyle=":",
                    alpha=0.25,
                    zorder=1,
                )
            ax.add_patch(patch)


def _draw_frame(
    ax,
    stats_ax,
    frame: dict,
    recorder: TrajectoryRecorder,
    precomputed_segs: list,
    bounds: tuple,
    detailed: bool,
    show_heatmap: bool = False,
    zone_polys: list | None = None,
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

    # Zones overlay if available
    reloc_pts = getattr(recorder, "relocation_points", None)
    if show_heatmap and reloc_pts and zone_polys:
        _draw_zone_demand_heatmap(ax, frame, reloc_pts, zone_polys, show_heatmap=True)
    elif reloc_pts and zone_polys:
        _draw_zone_borders(ax, zone_polys)

    if reloc_pts:
        zx = [p[0] for p in reloc_pts]
        zy = [p[1] for p in reloc_pts]
        ax.scatter(
            zx, zy, s=50, marker="s", c="#9c27b0", alpha=0.4,
            edgecolors="white", linewidths=0.6, zorder=2,
        )
        for zi, (x, y) in enumerate(reloc_pts):
            ax.text(
                x, y, str(zi), fontsize=6, ha="center", va="center",
                color="white", weight="bold", zorder=3,
            )
        target_zi = getattr(recorder, "target_zone", None)
        target_pct = getattr(recorder, "target_zone_pct", None)
        if target_zi is not None and 0 <= target_zi < len(reloc_pts):
            tx, ty = reloc_pts[target_zi]
            ax.scatter(
                [tx], [ty], s=350, marker="o", facecolors="none",
                edgecolors="#ffd700", linewidths=2.2, linestyle="--", zorder=3,
            )

    if frame["pending"]:
        px, py = zip(*frame["pending"])
        ax.scatter(
            px, py, s=40, facecolors="none", edgecolors="#d62728",
            linewidths=1.6, zorder=4,
        )

    if detailed:
        for oid, (orig, dest) in frame["order_od"].items():
            ax.scatter(*orig, s=32, c="#00e5ff", edgecolors="#006064", linewidths=0.8, zorder=3, marker="^")
            ax.scatter(*dest, s=32, c="#76ff03", edgecolors="#1b5e20", linewidths=0.8, zorder=3, marker="v")

    by_status: dict[DriverStatus, list] = {}
    for did, d in frame["drivers"].items():
        by_status.setdefault(d["status"], []).append((did, d))

    for status, items in by_status.items():
        xs = [d["location"][0] for _, d in items]
        ys = [d["location"][1] for _, d in items]
        color = STATUS_COLORS.get(status, "#333333")
        ax.scatter(
            xs, ys, s=55, c=color, edgecolors="white",
            linewidths=0.8, zorder=5, marker="o",
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

    has_target = bool(getattr(recorder, "target_zone", None) is not None)
    legend_handles = _make_legend_handles(
        show_zones=bool(reloc_pts),
        show_od=detailed,
        has_target_zone=has_target,
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        fontsize=6.8,
        framealpha=0.95,
        bbox_transform=ax.transAxes,
    )
    target_zi = getattr(recorder, "target_zone", None)
    target_pct = getattr(recorder, "target_zone_pct", None)
    surge_info = ""
    if target_zi is not None:
        pct_val = target_pct * 100 if (target_pct and target_pct <= 1.0) else (target_pct or 0.0)
        surge_info = f"  |  ★ Surge: Zone {target_zi} ({pct_val:.0f}% demand)"

    ax.set_title("")

    _draw_stats(stats_ax, frame, recorder.horizon)


def interpolate_recorder(
    recorder: TrajectoryRecorder,
    total_frames: int,
) -> TrajectoryRecorder:
    """Interpolate a TrajectoryRecorder to total_frames for silky-smooth high-FPS animation."""
    if len(recorder.frames) < 2 or total_frames <= len(recorder.frames):
        return recorder

    import copy
    import numpy as np

    new_rec = copy.copy(recorder)
    raw_frames = recorder.frames
    times = [f["time"] for f in raw_frames]
    t_min = times[0]
    t_max = times[-1]

    interp_frames = []
    target_times = np.linspace(t_min, t_max, total_frames)

    raw_idx = 0
    for t in target_times:
        while raw_idx < len(raw_frames) - 2 and times[raw_idx + 1] <= t:
            raw_idx += 1

        f0 = raw_frames[raw_idx]
        f1 = raw_frames[raw_idx + 1]
        t0 = f0["time"]
        t1 = f1["time"]

        dt = t1 - t0
        alpha = float((t - t0) / dt) if dt > 1e-6 else 0.0
        alpha = max(0.0, min(1.0, alpha))

        base_f = f0 if alpha < 0.5 else f1
        sub_drivers = {}
        for did, d0 in f0["drivers"].items():
            if did in f1["drivers"]:
                d1 = f1["drivers"][did]
                loc0 = d0["location"]
                loc1 = d1["location"]
                loc = (
                    float(loc0[0] + alpha * (loc1[0] - loc0[0])),
                    float(loc0[1] + alpha * (loc1[1] - loc0[1])),
                )
                target_d = d0 if alpha < 0.5 else d1
                sub_drivers[did] = {
                    "location": loc,
                    "status": target_d["status"],
                    "onboard_passengers": target_d["onboard_passengers"],
                    "assigned_orders": target_d["assigned_orders"],
                    "task_points": target_d["task_points"],
                }
            else:
                sub_drivers[did] = d0

        interp_frame = {
            "time": float(t),
            "drivers": sub_drivers,
            "pending": base_f["pending"],
            "order_od": base_f["order_od"],
            "served": int(round((1.0 - alpha) * f0["served"] + alpha * f1["served"])),
            "cancelled": int(round((1.0 - alpha) * f0["cancelled"] + alpha * f1["cancelled"])),
        }
        interp_frames.append(interp_frame)

    new_rec.frames = interp_frames
    return new_rec


def render_animation(
    recorder: TrajectoryRecorder,
    out_path: str = "episode.mp4",
    fps: int = 10,
    detailed: bool = True,
    figsize: Tuple[float, float] = (13.5, 8.5),
    dpi: int = 100,
    description: str = "",
    show_heatmap: bool = False,
    zoom_to_zones: bool = True,
    duration_sec: float | None = None,
) -> str:
    """Render a recorded trajectory to an MP4 with a clean custom layout."""
    if not recorder.frames:
        raise ValueError("recorder has no frames; call snapshot() during the run.")

    if duration_sec is None:
        duration_sec = getattr(recorder, "horizon", 60.0)

    target_total_frames = int(round(duration_sec * fps))
    if target_total_frames > len(recorder.frames):
        recorder = interpolate_recorder(recorder, target_total_frames)

    precomputed_segs, bounds = _precompute_network_segments(recorder.network)
    reloc_pts = getattr(recorder, "relocation_points", None)
    if reloc_pts and zoom_to_zones:
        # Zones cover ~65% of the network bbox. Cropping to them hides the
        # other 29% of road nodes and any car that drives there, so this
        # zoom may only EXPAND the view past the network bounds, never
        # shrink it below the area the fleet can actually reach.
        pad_lon = (max(p[0] for p in reloc_pts) - min(p[0] for p in reloc_pts)) * 0.12 + 1e-4
        pad_lat = (max(p[1] for p in reloc_pts) - min(p[1] for p in reloc_pts)) * 0.12 + 1e-4
        bounds = (
            min(bounds[0], min(p[0] for p in reloc_pts) - pad_lon),
            min(bounds[1], min(p[1] for p in reloc_pts) - pad_lat),
            max(bounds[2], max(p[0] for p in reloc_pts) + pad_lon),
            max(bounds[3], max(p[1] for p in reloc_pts) + pad_lat),
        )
    # Zone boundaries are drawn whenever zones are known, not only with the
    # heatmap: without them the regions are invisible and the frame reads as
    # one undifferentiated map.
    zone_polys = _compute_zone_polygons(reloc_pts, bounds) if reloc_pts else None

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.04, right=0.98)

    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08,
                          top=0.82, bottom=0.06, left=0.04, right=0.98)
    ax_map = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])

    # Persistent header strip: description text on top line, legend below it.
    # This stays inside the figure and is NOT cleared by _draw_frame (which only
    # touches ax_map / ax_stats), so the MP4 shows it on every frame.
    ax_legend = fig.add_axes([0.04, 0.805, 0.92, 0.10])
    ax_legend.axis("off")

    def _draw(frame_idx: int):
        _draw_frame(
            ax_map, ax_stats,
            recorder.frames[frame_idx],
            recorder,
            precomputed_segs,
            bounds,
            detailed,
            show_heatmap=show_heatmap,
            zone_polys=zone_polys,
        )

        # Persistent header strip: description text + legend, re-rendered every
        # frame so the MP4 shows both on all 601 frames.  ax_legend is not
        # touched by _draw_frame, so its artists survive across re-draws.
        ax_legend.clear()
        ax_legend.axis("off")

        # Description text (top line of the strip)
        if description:
            ax_legend.text(
                0.5, 0.88, description,
                ha="center", va="top", fontsize=8.5,
                fontfamily="monospace",
                transform=ax_legend.transAxes,
            )

        # Legend (bottom line of the strip, centered)
        has_target = bool(getattr(recorder, "target_zone", None) is not None)
        handles = _make_legend_handles(
            show_zones=bool(getattr(recorder, "relocation_points", None)),
            show_od=detailed,
            has_target_zone=has_target,
        )
        ax_legend.legend(
            handles=handles,
            loc="center",
            ncol=5,
            fontsize=7.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.35),
            bbox_transform=ax_legend.transAxes,
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
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 100,
    initial_heatmap: bool = False,
) -> None:
    """Open an interactive matplotlib window to explore the episode.

    Controls:
      Scroll-wheel      zoom in / zoom out map
      Right-click drag  pan map (or Shift + Left-click drag)
      Double-click      reset zoom to city overview
      0 / h             reset zoom
      Space             play / pause / replay
      Right/Left        step 1 frame forward/backward
      Up/Down           jump 10 frames forward/backward
      r                 reset to start (frame 0)
      z                 toggle zone overlay
      m                 toggle zone demand heatmap
      f / s             cycle playback speed (1x, 2x, 4x, 0.5x)
      Click vehicle     select vehicle to inspect trajectory & tasks
      Escape            deselect current vehicle
      q / close         quit
    """
    if not recorder.frames:
        raise ValueError("recorder has no frames; call snapshot() during the run.")

    frames = recorder.frames
    network = recorder.network
    precomputed_segs, bounds = _precompute_network_segments(network)
    lon0, lat0, lon1, lat1 = bounds
    reloc_pts = getattr(recorder, "relocation_points", None)
    zone_polys = _compute_zone_polygons(reloc_pts, bounds)

    state = {
        "idx": 0,
        "playing": False,
        "selected_id": None,
        "all_drivers": [],
        "scatter": None,
        "show_zones": bool(reloc_pts),
        "show_heatmap": initial_heatmap,
        "zone_polys": zone_polys,
        "speed_idx": 0,
        "speeds": [100, 50, 25, 200],
        "speed_labels": ["1x", "2x", "4x", "0.5x"],
        "updating_slider": False,
        "timer": None,
        "widgets": [],
        "xlim": (lon0, lon1),
        "ylim": (lat0, lat1),
        "pan_start": None,
    }

    fig = plt.figure(figsize=(figsize[0], max(figsize[1], 8.5)), dpi=dpi)

    # Main viewports (top margin reserved for title and legend outside map grid)
    ax_map = fig.add_axes([0.05, 0.16, 0.65, 0.70])
    ax_stats = fig.add_axes([0.72, 0.16, 0.25, 0.76])

    # Scrubber slider
    ax_slider = fig.add_axes([0.08, 0.085, 0.60, 0.03])
    slider = Slider(
        ax_slider, "Time", 0, len(frames) - 1,
        valinit=0, valstep=1, valfmt="%0.0f",
        color="#1f77b4",
    )

    # Control buttons
    ax_btn_reset = fig.add_axes([0.05, 0.025, 0.06, 0.04])
    ax_btn_prev = fig.add_axes([0.115, 0.025, 0.06, 0.04])
    ax_btn_play = fig.add_axes([0.18, 0.025, 0.07, 0.04])
    ax_btn_next = fig.add_axes([0.255, 0.06, 0.06, 0.04])
    ax_btn_next.set_position([0.255, 0.025, 0.06, 0.04])
    ax_btn_speed = fig.add_axes([0.32, 0.025, 0.07, 0.04])
    ax_btn_zones = fig.add_axes([0.395, 0.025, 0.075, 0.04])
    ax_btn_heatmap = fig.add_axes([0.475, 0.025, 0.085, 0.04])
    ax_btn_zoom = fig.add_axes([0.565, 0.025, 0.085, 0.04])

    btn_reset = Button(ax_btn_reset, "Reset")
    btn_prev = Button(ax_btn_prev, "« -10s")
    btn_play = Button(ax_btn_play, "Play")
    btn_next = Button(ax_btn_next, "» +10s")
    btn_speed = Button(ax_btn_speed, "Speed: 1x")
    btn_zones = Button(ax_btn_zones, f"Zones: {'ON' if state['show_zones'] else 'OFF'}")
    btn_heatmap = Button(ax_btn_heatmap, f"Heatmap: {'ON' if state['show_heatmap'] else 'OFF'}")
    btn_zoom = Button(ax_btn_zoom, "Reset Zoom")

    # Retain references on state so garbage collection never kills them
    state["widgets"] = [slider, btn_reset, btn_prev, btn_play, btn_next, btn_speed, btn_zones, btn_heatmap, btn_zoom]

    def _draw(idx: int, update_slider: bool = True):
        """Redraw map, drivers, zones, and statistics."""
        ax_map.clear()
        ax_stats.clear()

        frame = frames[idx]

        ax_map.set_xlim(state["xlim"])
        ax_map.set_ylim(state["ylim"])
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        target_zi = getattr(recorder, "target_zone", None)
        target_pct = getattr(recorder, "target_zone_pct", None)
        surge_info = ""
        if target_zi is not None:
            pct_val = target_pct * 100 if (target_pct and target_pct <= 1.0) else (target_pct or 0.0)
            surge_info = f"  |  ★ Surge: Zone {target_zi} ({pct_val:.0f}% demand)"

        # Road network
        lc = LineCollection(precomputed_segs, colors="#dddddd", linewidths=0.4, zorder=0)
        ax_map.add_collection(lc)

        # Zone demand heatmap
        if state["show_heatmap"] and reloc_pts and state["zone_polys"]:
            _draw_zone_demand_heatmap(ax_map, frame, reloc_pts, state["zone_polys"], show_heatmap=True)

        # Zones overlay
        if state["show_zones"] and reloc_pts:
            zx = [p[0] for p in reloc_pts]
            zy = [p[1] for p in reloc_pts]
            ax_map.scatter(
                zx, zy, s=50, marker="s", c="#9c27b0", alpha=0.45,
                edgecolors="white", linewidths=0.6, zorder=2,
            )
            for zi, (x, y) in enumerate(reloc_pts):
                ax_map.text(
                    x, y, str(zi), fontsize=6, ha="center", va="center",
                    color="white", weight="bold", zorder=3,
                )
            if target_zi is not None and 0 <= target_zi < len(reloc_pts):
                tx, ty = reloc_pts[target_zi]
                ax_map.scatter(
                    [tx], [ty], s=350, marker="o", facecolors="none",
                    edgecolors="#ffd700", linewidths=2.5, linestyle="--", zorder=3,
                )

        # Pending orders
        if frame["pending"]:
            px, py = zip(*frame["pending"])
            ax_map.scatter(
                px, py, s=45, facecolors="none", edgecolors="#d62728",
                linewidths=1.8, zorder=4,
            )

        if detailed:
            for oid, (orig, dest) in frame["order_od"].items():
                ax_map.scatter(*orig, s=36, c="#00e5ff", edgecolors="#006064", linewidths=0.8, zorder=3, marker="^")
                ax_map.scatter(*dest, s=36, c="#76ff03", edgecolors="#1b5e20", linewidths=0.8, zorder=3, marker="v")

        all_drivers = [(did, d) for did, d in frame["drivers"].items()]
        state["all_drivers"] = all_drivers

        if not all_drivers:
            ax_map.set_title(f"t = {frame['time']:.1f} / {recorder.horizon:.0f} min{surge_info}")
            _draw_stats(ax_stats, frame, recorder.horizon, None)
            fig.canvas.draw_idle()
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

        # Highlight selected driver path
        if sel_d is not None and sel_d["task_points"]:
            first_task = sel_d["task_points"][0]
            task_kind = first_task[0]
            task_coord = first_task[2]
            TASK_STYLE = {
                "pickup":   ("#1f77b4", "^"),
                "dropoff":  ("#2ca02c", "v"),
                "relocate": ("#ff7f0e", "s"),
            }
            leg_color, dest_marker = TASK_STYLE.get(task_kind, ("#ff00ff", "o"))
            ax_map.plot(
                [sel_d["location"][0], task_coord[0]],
                [sel_d["location"][1], task_coord[1]],
                color=leg_color, linewidth=1.5, alpha=0.9, zorder=6,
                linestyle="--",
            )
            ax_map.scatter(
                *task_coord, s=100, c="#d62728", edgecolors="white",
                linewidths=1.2, zorder=7, marker=dest_marker,
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

        has_target = bool(getattr(recorder, "target_zone", None) is not None)
        legend_handles = _make_legend_handles(
            show_zones=bool(state["show_zones"] and reloc_pts),
            show_od=detailed,
            has_target_zone=has_target,
        )
        ax_map.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            fontsize=7.0,
            framealpha=0.95,
        )

        status_str = "PLAY" if state["playing"] else "PAUSE"
        speed_lbl = state["speed_labels"][state["speed_idx"]]
        sel_label = f"  |  selected: {sel_did}" if sel_did is not None else ""
        ax_map.set_title(
            f"[{status_str}]  t = {frame['time']:.1f} / {recorder.horizon:.0f} min"
            f"  (frame {idx + 1}/{len(frames)})  [{speed_lbl}]{surge_info}{sel_label}",
            fontsize=10, y=1.13,
        )

        sel_tuple = (sel_did, sel_d) if sel_did and sel_d else None
        _draw_stats(ax_stats, frame, recorder.horizon, sel_tuple)

        if update_slider:
            state["updating_slider"] = True
            slider.set_val(idx)
            state["updating_slider"] = False

        fig.canvas.draw_idle()

    # Hardware timer via canvas: eliminates FuncAnimation generator exhaustion & GC drops
    timer = fig.canvas.new_timer(interval=state["speeds"][0])
    state["timer"] = timer

    def _on_timer():
        if not state["playing"]:
            return
        if state["idx"] < len(frames) - 1:
            state["idx"] += 1
            _draw(state["idx"], update_slider=True)
        else:
            # Reached end: stop playback cleanly
            state["playing"] = False
            timer.stop()
            btn_play.label.set_text("Play")
            _draw(state["idx"], update_slider=True)

    timer.add_callback(_on_timer)

    def _toggle_play(event=None):
        if state["idx"] >= len(frames) - 1:
            state["idx"] = 0
        state["playing"] = not state["playing"]
        if state["playing"]:
            btn_play.label.set_text("Pause")
            timer.start()
        else:
            btn_play.label.set_text("Play")
            timer.stop()
        _draw(state["idx"], update_slider=True)

    def _reset(event=None):
        state["playing"] = False
        timer.stop()
        btn_play.label.set_text("Play")
        state["idx"] = 0
        state["selected_id"] = None
        _draw(0, update_slider=True)

    def _step_prev(event=None, step: int = 10):
        state["playing"] = False
        timer.stop()
        btn_play.label.set_text("▶ Play")
        state["idx"] = max(0, state["idx"] - step)
        _draw(state["idx"], update_slider=True)

    def _step_next(event=None, step: int = 10):
        state["playing"] = False
        timer.stop()
        btn_play.label.set_text("▶ Play")
        state["idx"] = min(len(frames) - 1, state["idx"] + step)
        _draw(state["idx"], update_slider=True)

    def _toggle_speed(event=None):
        state["speed_idx"] = (state["speed_idx"] + 1) % len(state["speeds"])
        new_int = state["speeds"][state["speed_idx"]]
        lbl = state["speed_labels"][state["speed_idx"]]
        btn_speed.label.set_text(f"Speed: {lbl}")
        timer.interval = new_int
        _draw(state["idx"], update_slider=False)

    def _toggle_zones(event=None):
        state["show_zones"] = not state["show_zones"]
        btn_zones.label.set_text(f"Zones: {'ON' if state['show_zones'] else 'OFF'}")
        _draw(state["idx"], update_slider=False)

    def _toggle_heatmap(event=None):
        state["show_heatmap"] = not state["show_heatmap"]
        btn_heatmap.label.set_text(f"Heatmap: {'ON' if state['show_heatmap'] else 'OFF'}")
        _draw(state["idx"], update_slider=False)

    def _on_slider(val):
        if state["updating_slider"]:
            return
        target = int(round(val))
        target = max(0, min(target, len(frames) - 1))
        state["idx"] = target
        _draw(target, update_slider=False)

    def _on_key(event):
        if event.key == " ":
            _toggle_play()
        elif event.key == "right":
            _step_next(step=1)
        elif event.key == "left":
            _step_prev(step=1)
        elif event.key == "up":
            _step_next(step=10)
        elif event.key == "down":
            _step_prev(step=10)
        elif event.key == "r":
            _reset()
        elif event.key == "z":
            _toggle_zones()
        elif event.key in ("m", "d"):
            _toggle_heatmap()
        elif event.key in ("f", "s"):
            _toggle_speed()
        elif event.key in ("0", "h"):
            _reset_zoom()
        elif event.key == "escape":
            state["selected_id"] = None
            _draw(state["idx"], update_slider=False)

    def _reset_zoom(event=None):
        state["xlim"] = (lon0, lon1)
        state["ylim"] = (lat0, lat1)
        ax_map.set_xlim(state["xlim"])
        ax_map.set_ylim(state["ylim"])
        fig.canvas.draw_idle()

    def _on_scroll(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        if event.button == "up" or getattr(event, "step", 0) > 0:
            scale = 0.8
        elif event.button == "down" or getattr(event, "step", 0) < 0:
            scale = 1.25
        else:
            return

        cur_x0, cur_x1 = state["xlim"]
        cur_y0, cur_y1 = state["ylim"]
        xdata, ydata = event.xdata, event.ydata

        new_x0 = xdata - (xdata - cur_x0) * scale
        new_x1 = xdata + (cur_x1 - xdata) * scale
        new_y0 = ydata - (ydata - cur_y0) * scale
        new_y1 = ydata + (cur_y1 - ydata) * scale

        state["xlim"] = (new_x0, new_x1)
        state["ylim"] = (new_y0, new_y1)
        ax_map.set_xlim(state["xlim"])
        ax_map.set_ylim(state["ylim"])
        fig.canvas.draw_idle()

    def _on_mouse_down(event):
        if event.inaxes != ax_map:
            return
        if event.dblclick:
            _reset_zoom()
            return
        if event.button in (2, 3) or (event.button == 1 and event.key == "shift"):
            state["pan_start"] = (event.x, event.y, state["xlim"], state["ylim"])

    def _on_mouse_move(event):
        if state.get("pan_start") is None:
            return
        if event.x is None or event.y is None:
            return
        px0, py0, orig_xlim, orig_ylim = state["pan_start"]
        bbox = ax_map.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx = (orig_xlim[1] - orig_xlim[0]) * ((event.x - px0) / bbox.width)
        dy = (orig_ylim[1] - orig_ylim[0]) * ((event.y - py0) / bbox.height)
        new_xlim = (orig_xlim[0] - dx, orig_xlim[1] - dx)
        new_ylim = (orig_ylim[0] - dy, orig_ylim[1] - dy)
        state["xlim"] = new_xlim
        state["ylim"] = new_ylim
        ax_map.set_xlim(new_xlim)
        ax_map.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    def _on_mouse_up(event):
        state["pan_start"] = None

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
            _draw(state["idx"], update_slider=False)

    def _on_close(event):
        state["playing"] = False
        if timer is not None:
            timer.stop()

    slider.on_changed(_on_slider)
    btn_reset.on_clicked(_reset)
    btn_prev.on_clicked(lambda e: _step_prev(e, 10))
    btn_play.on_clicked(_toggle_play)
    btn_next.on_clicked(lambda e: _step_next(e, 10))
    btn_speed.on_clicked(_toggle_speed)
    btn_zones.on_clicked(_toggle_zones)
    btn_heatmap.on_clicked(_toggle_heatmap)
    btn_zoom.on_clicked(_reset_zoom)

    fig.canvas.mpl_connect("close_event", _on_close)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("pick_event", _on_click)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    fig.canvas.mpl_connect("button_press_event", _on_mouse_down)
    fig.canvas.mpl_connect("motion_notify_event", _on_mouse_move)
    fig.canvas.mpl_connect("button_release_event", _on_mouse_up)

    _draw(0, update_slider=True)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if timer is not None:
            timer.stop()
        plt.close(fig)
