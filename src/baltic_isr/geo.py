"""Projection, and the fixed geometry of the box.

Everything is projected once, at setup, into metres. Batch 2 uploads the vertex
arrays this module produces straight to the GPU; the distance helpers here are
for setup and analysis, never for the training path.

Python's ``hash`` is not stable across processes, and neither is iteration over
a set. Nothing in this module uses either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from baltic_isr.config import Config, Corridor, Geo, Rail

_WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class Frame:
    """Projection between geographic degrees and local metres.

    Parameters
    ----------
    geo : Geo
        Box bounds and the projected CRS.
    """

    geo: Geo

    @property
    def _forward(self) -> Transformer:
        return Transformer.from_crs(_WGS84, self.geo.projected_crs, always_xy=True)

    @property
    def _inverse(self) -> Transformer:
        return Transformer.from_crs(self.geo.projected_crs, _WGS84, always_xy=True)

    def to_xy(self, lon: np.ndarray | float, lat: np.ndarray | float) -> np.ndarray:
        """Project degrees to metres.

        Parameters
        ----------
        lon, lat : ndarray or float
            Geographic coordinates.

        Returns
        -------
        ndarray
            Array of shape ``(..., 2)`` in projected metres.
        """
        x, y = self._forward.transform(lon, lat)
        return np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=-1)

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        """Unproject metres to degrees.

        Parameters
        ----------
        xy : ndarray
            Array of shape ``(..., 2)`` in projected metres.

        Returns
        -------
        ndarray
            Array of shape ``(..., 2)`` in degrees.
        """
        array = np.asarray(xy, dtype=float)
        lon, lat = self._inverse.transform(array[..., 0], array[..., 1])
        return np.stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)], axis=-1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the box as ``(x_min, y_min, x_max, y_max)`` in metres.

        The projected box is not a rectangle in degrees, so the bound is taken
        over all four corners rather than two.
        """
        corners = self.to_xy(
            np.array([self.geo.lon_min, self.geo.lon_max, self.geo.lon_min, self.geo.lon_max]),
            np.array([self.geo.lat_min, self.geo.lat_min, self.geo.lat_max, self.geo.lat_max]),
        )
        return (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )

    @property
    def extent_m(self) -> tuple[float, float]:
        """Return the box width and height in metres."""
        x_min, y_min, x_max, y_max = self.bounds
        return (x_max - x_min, y_max - y_min)

    @property
    def area_m2(self) -> float:
        """Return the box area in square metres.

        This is the bounding box, not the water. Land is subtracted in batch 2,
        where the bathymetry raster arrives.
        """
        width, height = self.extent_m
        return width * height


def _segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return distance from each point to each segment.

    Parameters
    ----------
    points : ndarray
        Shape ``(n, 2)``.
    starts, ends : ndarray
        Shape ``(m, 2)``.

    Returns
    -------
    ndarray
        Shape ``(n, m)``.
    """
    segment = ends - starts
    length_squared = np.einsum("md,md->m", segment, segment)
    length_squared = np.where(length_squared > 0.0, length_squared, 1.0)
    offset = points[:, None, :] - starts[None, :, :]
    t = np.einsum("nmd,md->nm", offset, segment) / length_squared
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * segment[None, :, :]
    delta = points[:, None, :] - closest
    distance: np.ndarray = np.sqrt(np.einsum("nmd,nmd->nm", delta, delta))
    return distance


@dataclass(frozen=True)
class Polyline:
    """A projected polyline with a corridor half-width.

    Parameters
    ----------
    name : str
        Identifier carried from configuration.
    xy : ndarray
        Vertices, shape ``(k, 2)``, in projected metres.
    half_width_m : float
        Half-width of the corridor around the line.
    """

    name: str
    xy: np.ndarray
    half_width_m: float

    @property
    def length_m(self) -> float:
        """Return the total length of the polyline."""
        delta = np.diff(self.xy, axis=0)
        return float(np.sqrt(np.einsum("kd,kd->k", delta, delta)).sum())

    def distance_to(self, points: np.ndarray) -> np.ndarray:
        """Return the distance from each point to the polyline.

        Parameters
        ----------
        points : ndarray
            Shape ``(n, 2)`` in projected metres.

        Returns
        -------
        ndarray
            Shape ``(n,)``.
        """
        array = np.atleast_2d(np.asarray(points, dtype=float))
        return _segment_distance(array, self.xy[:-1], self.xy[1:]).min(axis=1)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return whether each point lies inside the corridor.

        Parameters
        ----------
        points : ndarray
            Shape ``(n, 2)`` in projected metres.

        Returns
        -------
        ndarray
            Boolean array of shape ``(n,)``.
        """
        return self.distance_to(points) <= self.half_width_m

    def point_at(self, fraction: np.ndarray | float) -> np.ndarray:
        """Return positions a fraction of the way along the line.

        Used by batch 2 to spawn traffic on a rail.

        Parameters
        ----------
        fraction : ndarray or float
            Values in ``[0, 1]``.

        Returns
        -------
        ndarray
            Shape ``(..., 2)`` in projected metres.
        """
        delta = np.diff(self.xy, axis=0)
        lengths = np.sqrt(np.einsum("kd,kd->k", delta, delta))
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        target = np.asarray(fraction, dtype=float) * cumulative[-1]
        index = np.clip(np.searchsorted(cumulative, target, side="right") - 1, 0, len(delta) - 1)
        local = (target - cumulative[index]) / lengths[index]
        position: np.ndarray = self.xy[index] + local[..., None] * delta[index]
        return position


def project_rail(rail: Rail, frame: Frame) -> Polyline:
    """Project one navigation lane.

    Parameters
    ----------
    rail : Rail
        Configured lane.
    frame : Frame
        Projection.

    Returns
    -------
    Polyline
        Projected lane, half-width being half the configured width.
    """
    vertices = np.asarray(rail.vertices, dtype=float)
    return Polyline(
        name=rail.name,
        xy=frame.to_xy(vertices[:, 0], vertices[:, 1]),
        half_width_m=rail.width_m / 2.0,
    )


def project_corridor(corridor: Corridor, frame: Frame) -> Polyline:
    """Project one seabed asset.

    Parameters
    ----------
    corridor : Corridor
        Configured cable or pipe.
    frame : Frame
        Projection.

    Returns
    -------
    Polyline
        Projected corridor, carrying its own configured half-width.
    """
    vertices = np.asarray(corridor.vertices, dtype=float)
    return Polyline(
        name=corridor.name,
        xy=frame.to_xy(vertices[:, 0], vertices[:, 1]),
        half_width_m=corridor.half_width_m,
    )


@dataclass(frozen=True)
class Geometry:
    """Everything fixed about the box, projected once.

    Parameters
    ----------
    frame : Frame
        Projection.
    rails : tuple of Polyline
        Navigation lanes.
    corridors : tuple of Polyline
        Seabed assets.
    bases : ndarray
        Shore charging points, shape ``(b, 2)``, in projected metres.
    base_names : tuple of str
        Names in the same order as ``bases``.
    """

    frame: Frame
    rails: tuple[Polyline, ...]
    corridors: tuple[Polyline, ...]
    bases: np.ndarray
    base_names: tuple[str, ...]

    def rail(self, name: str) -> Polyline:
        """Return one lane by name.

        Parameters
        ----------
        name : str
            Lane name.

        Returns
        -------
        Polyline
            The lane.
        """
        for item in self.rails:
            if item.name == name:
                return item
        raise KeyError(f"unknown rail {name!r}")

    def distance_to_base(self, points: np.ndarray) -> np.ndarray:
        """Return the distance to the nearest base for each point.

        Parameters
        ----------
        points : ndarray
            Shape ``(n, 2)`` in projected metres.

        Returns
        -------
        ndarray
            Shape ``(n,)``.
        """
        array = np.atleast_2d(np.asarray(points, dtype=float))
        delta = array[:, None, :] - self.bases[None, :, :]
        nearest: np.ndarray = np.sqrt(np.einsum("nbd,nbd->nb", delta, delta)).min(axis=1)
        return nearest


def build(config: Config) -> Geometry:
    """Project every fixed feature of a configuration.

    Parameters
    ----------
    config : Config
        Resolved configuration.

    Returns
    -------
    Geometry
        Projected geometry.
    """
    frame = Frame(config.geo)
    bases = frame.to_xy(
        np.array([base.lon for base in config.bases]),
        np.array([base.lat for base in config.bases]),
    )
    return Geometry(
        frame=frame,
        rails=tuple(project_rail(rail, frame) for rail in config.rails),
        corridors=tuple(project_corridor(item, frame) for item in config.corridors),
        bases=bases,
        base_names=tuple(base.name for base in config.bases),
    )
