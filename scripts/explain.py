"""Six figures that walk somebody through the choices this project made.

Not a results pack. Each one stands at a fork the project came to, and shows
the measurement that settled it: why a drone must not hurry, why it cannot
simply report what it sees, how many of them it takes, why the coastline had
to be surveyed rather than drawn, and what the fleet does differently once it
has learned.

Every number is computed here from the configuration or from a recorded
episode; nothing is typed in.

    uv run python scripts/explain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from baltic_isr import config as config_module
from baltic_isr import geo
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet import power
from baltic_isr.render import figures
from baltic_isr.render.replay import Replay
from baltic_isr.sensing import acoustics
from baltic_isr.world import seabed as seabed_module
from baltic_isr.world import statics as statics_module

TRACES = Path("out/traces")

# Version controlled, like the loop in the README: a figure that only exists in
# an ignored output directory is a figure a clone does not have.
ASSETS = Path("assets")


def hearing_and_endurance(
    resolved: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return detection range and power, swept over speed."""
    cargo = next(item for item in resolved.traffic.classes if item.name == "cargo")
    speeds = np.linspace(0.0, resolved.auv.max_speed_ms, 60)
    hearing = [
        acoustics.detection_range_m(
            float(speed), cargo.source_level_db, resolved.acoustics, resolved.environment
        )
        / 1000.0
        for speed in speeds
    ]
    draw = [power.power_w(float(speed), resolved.auv) for speed in speeds]
    marks = power.regime_speeds(resolved.auv)
    return speeds, np.array(hearing), np.array(draw), marks


def loneliness(resolved: Config, trace: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Return the distance to the nearest other drone, over a whole episode."""
    reach = acoustics.modem_range_m(resolved.comms) / 1000.0
    xy = trace["vehicle_xy"]
    mode = trace["vehicle_mode"]
    at_sea = (mode == auv_module.MODE_DEPLOYED) | (mode == auv_module.MODE_RETURNING)
    gaps = []
    for frame in range(0, xy.shape[0], 20):
        live = xy[frame][at_sea[frame]]
        if len(live) < 2:
            continue
        spread = np.linalg.norm(live[:, None, :] - live[None, :, :], axis=-1)
        np.fill_diagonal(spread, np.inf)
        gaps.append(spread.min(axis=1) / 1000.0)
    return np.concatenate(gaps), reach


def coverage(
    resolved: Config, trace: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the share of water inside somebody's hearing, hour by hour.

    The UNION of the detection discs over the sea, sampled on a kilometre
    grid. Summing the discs instead counts the overlap twice, and forty discs
    of six kilometres sum to more than the whole gulf.
    """
    cargo = next(item for item in resolved.traffic.classes if item.name == "cargo")
    reach = acoustics.detection_range_m(
        resolved.auv.regimes.cruise_ms,
        cargo.source_level_db,
        resolved.acoustics,
        resolved.environment,
    )
    seabed = seabed_module.load()
    step = max(1, int(1000.0 / seabed.cell_m))
    water = seabed.depth_m[::step, ::step] > 0.0
    rows, columns = water.shape
    cell = seabed.cell_m * step
    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    sea_x = ((grid_x + 0.5) * cell)[water]
    sea_y = ((grid_y + 0.5) * cell)[water]

    xy = trace["vehicle_xy"]
    mode = trace["vehicle_mode"]
    live = (mode == auv_module.MODE_DEPLOYED) | (mode == auv_module.MODE_RETURNING)
    frames = range(0, xy.shape[0], 30)
    hours, share, afloat = [], [], []
    for frame in frames:
        here = xy[frame][live[frame]]
        hours.append(float(trace["times_s"][frame]) / 3600.0)
        afloat.append(float(live[frame].sum()))
        if len(here) == 0:
            share.append(0.0)
            continue
        near = np.zeros(sea_x.shape, dtype=bool)
        for spot in here:
            near |= (sea_x - spot[0]) ** 2 + (sea_y - spot[1]) ** 2 <= reach**2
        share.append(float(near.mean()))
    return np.array(hours), np.array(share), np.array(afloat), resolved.fleet.count


# The land model this project started with: two polylines, five vertices each,
# and a Tallinn base placed to suit them. Kept here so the figure can show what
# it looked like next to the survey that replaced it.
DRAWN_COAST = {
    "north": [(23.80, 60.02), (24.30, 60.05), (24.90, 60.13), (25.40, 60.20), (26.20, 60.25)],
    "south": [(23.80, 59.52), (24.30, 59.48), (24.80, 59.42), (25.40, 59.48), (26.20, 59.55)],
}
DRAWN_BASE = (24.75, 59.44)


def coastlines(
    resolved: Config,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return the drawn coast and the surveyed one, in the box's own metres."""
    frame = geo.build(resolved).frame
    fixed = statics_module.build(resolved, "cpu")
    origin = fixed.origin.numpy()
    drawn = {
        name: np.array([frame.to_xy(lon, lat) for lon, lat in line]) - origin
        for name, line in DRAWN_COAST.items()
    }
    surveyed = seabed_module.load().coast
    base_then = np.asarray(frame.to_xy(*DRAWN_BASE)) - origin
    base_now = fixed.bases.numpy()[1]
    return drawn, surveyed, base_then, base_now, fixed.extent.numpy()


def regimes(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> np.ndarray:
    """Return the share of decisions spent in each regime, before and after."""
    shares = []
    for trace in (before, after):
        mode = trace["vehicle_mode"]
        live = (mode == auv_module.MODE_DEPLOYED) | (mode == auv_module.MODE_RETURNING)
        regime = trace["vehicle_regime"][live]
        shares.append([float((regime == index).mean()) for index in range(3)])
    return np.array(shares)


def exposure(
    resolved: Config, before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Return hours since each cable stretch was last heard, over the episode."""
    out = []
    for trace in (before, after):
        ages = Replay(resolved, trace).__getattribute__("_exposure")()
        out.append(ages.reshape(-1) / 3600.0)
    return out


if __name__ == "__main__":
    resolved = config_module.load("baseline")
    before = dict(np.load(TRACES / "untrained.npz"))
    after = dict(np.load(TRACES / "learned.npz"))
    torch.set_grad_enabled(False)

    ASSETS.mkdir(parents=True, exist_ok=True)
    written = [
        figures.detection_range_against_speed(
            *hearing_and_endurance(resolved), path=ASSETS / "detection-range-against-speed.png"
        ),
        figures.distance_to_nearest_drone(
            *loneliness(resolved, after), path=ASSETS / "distance-to-nearest-drone.png"
        ),
        figures.gulf_within_detection_range(
            *coverage(resolved, after), path=ASSETS / "gulf-within-detection-range.png"
        ),
        figures.drawn_against_surveyed(
            *coastlines(resolved), path=ASSETS / "drawn-against-surveyed.png"
        ),
        figures.regime_share(
            regimes(before, after), path=ASSETS / "regime-share-before-and-after.png"
        ),
        figures.time_since_patrolled(
            *exposure(resolved, before, after),
            path=ASSETS / "time-since-cable-last-patrolled.png",
        ),
    ]
    print()
    for path in written:
        print(f"  wrote {path}")
