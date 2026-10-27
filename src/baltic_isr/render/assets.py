"""What the fleet is there to protect, as geometry.

The cables were four grey dashes. That is the wrong weight for the only reason
any of this exists: the scenario is the Gulf of Finland after Balticconnector
and Estlink 2, and what a dragging anchor severs is an *asset*. If the map does
not say which object matters, red on a vessel means nothing, because the viewer
has nothing to measure "serious" against.

So this module names the assets and, more usefully, finds the places where they
are actually at risk:

**A crossing** is where a cable corridor passes under a shipping lane. That is
where a vessel that drops an anchor is over a cable rather than over mud, and
there are only five of them in the whole box -- few enough to mark, important
enough that marking them organises the whole picture.

**A landfall** is where a cable leaves the box. Shallow, close inshore, and the
other place cables get cut.

Everything here is fixed geometry, computed once, in the same local frame as
the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baltic_isr import geo
from baltic_isr.config import Config


@dataclass(frozen=True)
class Crossing:
    """One place a cable passes under a shipping lane.

    Attributes
    ----------
    xy : ndarray
        Position in local metres, shape ``(2,)``.
    cable, lane : str
        What crosses what.
    """

    xy: np.ndarray
    cable: str
    lane: str

    @property
    def label(self) -> str:
        """Return a short name for the chokepoint."""
        return self.cable.replace("_", " ").title()


@dataclass(frozen=True)
class Assets:
    """The infrastructure layer, projected into local metres.

    Attributes
    ----------
    cables : list of tuple
        ``(name, vertices, half_width_m)`` per cable.
    crossings : list of Crossing
        Cable-under-lane chokepoints.
    landfalls : ndarray
        Cable ends, shape ``(k, 2)``.
    """

    cables: list[tuple[str, np.ndarray, float]]
    crossings: list[Crossing]
    landfalls: np.ndarray


def _intersections(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return every point where two polylines cross.

    The standard segment-pair parametric test, done as one outer product
    rather than a double loop: both polylines are short and this runs once at
    setup, but a loop here would be the only one in the render path.

    Parameters
    ----------
    first, second : ndarray
        Vertices, shape ``(k, 2)``.

    Returns
    -------
    ndarray
        Intersections, shape ``(n, 2)``.
    """
    start, run = first[:-1], first[1:] - first[:-1]
    other, reach = second[:-1], second[1:] - second[:-1]
    denominator = run[:, None, 0] * reach[None, :, 1] - run[:, None, 1] * reach[None, :, 0]
    offset = other[None, :, :] - start[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        along = (
            offset[..., 0] * reach[None, :, 1] - offset[..., 1] * reach[None, :, 0]
        ) / denominator
        across = (offset[..., 0] * run[:, None, 1] - offset[..., 1] * run[:, None, 0]) / denominator
    hit = np.isfinite(along) & (along >= 0.0) & (along <= 1.0) & (across >= 0.0) & (across <= 1.0)
    rows, columns = np.nonzero(hit)
    if rows.size == 0:
        return np.zeros((0, 2))
    found: np.ndarray = start[rows] + along[rows, columns, None] * run[rows]
    return found


def build_assets(config: Config) -> Assets:
    """Project the cables and find the places they are exposed.

    Parameters
    ----------
    config : Config
        Resolved configuration.

    Returns
    -------
    Assets
        Infrastructure geometry in local metres.
    """
    geometry = geo.build(config)
    x_min, y_min, _, _ = geometry.frame.bounds
    origin = np.array([x_min, y_min])

    cables = [(item.name, item.xy - origin, item.half_width_m) for item in geometry.corridors]
    crossings = [
        Crossing(point, cable.name, lane.name)
        for cable in geometry.corridors
        for lane in geometry.rails
        for point in _intersections(cable.xy, lane.xy) - origin
    ]
    # `cables` is already in the local frame, so subtracting the origin again
    # here put every landfall six million metres off the map -- which is why
    # the tick marks never appeared on it.
    landfalls = np.array([end for _, line, _ in cables for end in (line[0], line[-1])]).reshape(
        -1, 2
    )
    return Assets(cables=cables, crossings=crossings, landfalls=landfalls)


def over_a_cable(assets: Assets, xy: np.ndarray) -> np.ndarray:
    """Return whether each point lies inside a cable corridor.

    This is what turns a suspicion into a target. A vessel loitering in open
    water is odd; the same vessel loitering over Balticconnector is the thing
    the fleet exists to catch, and the map has to tell the two apart.

    Parameters
    ----------
    assets : Assets
        Infrastructure geometry.
    xy : ndarray
        Points in local metres, shape ``(n, 2)``.

    Returns
    -------
    ndarray
        Boolean, shape ``(n,)``.
    """
    inside = np.zeros(xy.shape[0], dtype=bool)
    for _, line, half_width in assets.cables:
        inside |= _distance_to_line(xy, line) <= half_width
    return inside


def _distance_to_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Return each point's distance to a polyline, in metres."""
    start, finish = line[:-1], line[1:]
    run = finish - start
    length = (run**2).sum(axis=1).clip(min=1e-9)
    offset = points[:, None, :] - start[None, :, :]
    along = ((offset * run[None, :, :]).sum(axis=2) / length[None, :]).clip(0.0, 1.0)
    closest = start[None, :, :] + along[..., None] * run[None, :, :]
    gap: np.ndarray = np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)
    return gap
