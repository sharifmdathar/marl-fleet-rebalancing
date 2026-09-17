"""Standardized demand-shift definitions (S0–S6) and environment factory.

Defined per docs/demand-shift.md.
"""

from __future__ import annotations

import copy

from marl_fleet_rebalancing.config import NYCConfig
from marl_fleet_rebalancing.env import RideGymAEC

# Standardized shift scenarios:
# (label, nyc_window_limit, target_zone, target_zone_pct, shift_param, description)
SHIFTS: list[tuple[str, int, int | None, float | None, str | None, str]] = [
    ("S0", 500, None, None, None, "In-distribution base (train)"),
    ("S1", 500, None, 1.0, "permute", "Spatial permutation (rot offset = nz//2)"),
    ("S2a", 250, None, None, None, "Magnitude ×0.5 (surge/trough)"),
    ("S2b", 375, None, None, None, "Magnitude ×0.75"),
    ("S2d", 750, None, None, None, "Magnitude ×1.5"),
    ("S2e", 1000, None, None, None, "Magnitude ×2.0"),
    ("S3", 500, None, None, "temporal", "Temporal phase shift (val split)"),
    ("S4", 500, 1, 60.0, "concentrate", "Cold-zone concentration (60% to z1)"),
    ("S5", 500, 0, 20.0, "burst", "Event burst (20% in [20,30]m to z0)"),
    ("S6", 500, None, None, "real", "Real held-out demand (test split)"),
]

SHIFTS_DICT: dict[str, tuple[str, int, int | None, float | None, str | None, str]] = {
    s[0].upper(): s for s in SHIFTS
}


def get_shift_spec(shift_label: str) -> tuple[str, int, int | None, float | None, str | None, str]:
    """Retrieve shift definition tuple by label (case-insensitive)."""
    key = shift_label.strip().upper()
    if key not in SHIFTS_DICT:
        valid = list(SHIFTS_DICT.keys())
        raise ValueError(f"Unknown shift '{shift_label}'. Valid shifts: {valid}")
    return SHIFTS_DICT[key]


def make_shifted_env(
    shift_label: str,
    nyc_window_limit: int,
    target_zone: int | None,
    target_zone_pct: float | None,
    shift_param: str | None = None,
    seed: int = 0,
    base_cfg: NYCConfig | None = None,
) -> RideGymAEC:
    """Create an environment with shifted demand configuration."""
    if base_cfg is not None:
        cfg = copy.deepcopy(base_cfg)
    else:
        cfg = NYCConfig()

    cfg.nyc_window_limit = nyc_window_limit
    cfg.nyc_split = "train"
    cfg.order_source = "nyc"
    cfg.demand_shift_mode = "none"
    cfg.target_zone = None
    cfg.target_zone_demand_pct = None
    cfg.burst_zone = None
    cfg.burst_share_pct = None
    cfg.perm_strength = None
    cfg.perm_offset = None

    # Mechanism dispatch
    if shift_param is not None:
        if shift_param == "temporal":
            if shift_label.upper() == "S3":
                cfg.nyc_split = "val"
            elif shift_label.upper() == "S3B":
                cfg.nyc_split = "test"
        elif shift_param == "burst":
            cfg.demand_shift_mode = "burst"
            cfg.burst_zone = target_zone if target_zone is not None else 0
            cfg.burst_share_pct = target_zone_pct if target_zone_pct is not None else 20.0
            cfg.burst_start_min = 20.0
            cfg.burst_end_min = 30.0
        elif shift_param == "permute":
            cfg.demand_shift_mode = "permute"
            cfg.perm_strength = target_zone_pct if target_zone_pct is not None else 1.0
            cfg.perm_offset = target_zone if target_zone is not None else None
        elif shift_param == "concentrate":
            cfg.demand_shift_mode = "none"
            cfg.target_zone = target_zone
            cfg.target_zone_demand_pct = target_zone_pct
        elif shift_param == "real":
            cfg.nyc_split = "test"
        else:
            raise ValueError(f"Unknown shift_param: {shift_param!r}")

    if shift_param is None and target_zone is not None and target_zone_pct is not None:
        cfg.target_zone = target_zone
        cfg.target_zone_demand_pct = target_zone_pct

    env = RideGymAEC(config=cfg, seed=seed)
    env.reset(seed=seed)
    return env


def build_shifted_env(
    shift_label: str,
    seed: int = 0,
    base_cfg: NYCConfig | None = None,
) -> RideGymAEC:
    """Convenience helper to build a shifted environment given a shift ID."""
    label, win_lim, tgt_z, tgt_pct, shift_param, _desc = get_shift_spec(shift_label)
    return make_shifted_env(
        shift_label=label,
        nyc_window_limit=win_lim,
        target_zone=tgt_z,
        target_zone_pct=tgt_pct,
        shift_param=shift_param,
        seed=seed,
        base_cfg=base_cfg,
    )
