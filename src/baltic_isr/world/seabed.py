"""The measured seabed, read back from what ``scripts/ingest_geodata.py`` wrote.

One file is the single source of truth for where the water is, how deep it is
and where it stops. The simulation gathers from the rasters; the chart draws
the same rasters and the coastline extracted from them. They cannot disagree,
because there is nothing for them to disagree about.

The product is versioned rather than downloaded, so a clone runs offline and
every run of the project sees the same seabed. Rebuilding it needs the network
and the ``geodata`` dependency group; nothing here does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PRODUCT = Path(__file__).resolve().parents[3] / "data" / "gulf_of_finland.npz"


@dataclass(frozen=True)
class Seabed:
    """Depth, the waterline and where it came from, on one regular grid.

    Attributes
    ----------
    depth_m : ndarray
        Water depth in metres, ``(rows, columns)``, zero on land. Row zero is
        the southern edge, so the array indexes the same way the box does.
    coast_distance_m : ndarray
        Signed distance to the waterline, positive at sea, same shape.
    escape_sea, escape_berth : ndarray
        For each cell, the flat index of the nearest cell a vehicle may
        occupy, ``(rows, columns)``. A cell whose target is itself already
        qualifies. ``escape_sea`` uses the keel limit and ``escape_berth``
        the berthing clearance; which thresholds they were built for is in
        the metadata, and world.statics checks them.
    coast : tuple of ndarray
        The waterline as polylines in local metres, for drawing. Taken from
        the zero crossing of the terrain, so it is finer than the grid.
    cell_m : float
        Grid cell size in metres.
    origin : ndarray
        Projected coordinates of the grid's south-west corner, shape ``(2,)``.
    metadata : dict
        Provenance: source, licence, coverage, retrieval date, measured area.
    """

    depth_m: np.ndarray
    coast_distance_m: np.ndarray
    escape_sea: np.ndarray
    escape_berth: np.ndarray
    coast: tuple[np.ndarray, ...]
    cell_m: float
    origin: np.ndarray
    metadata: dict[str, object]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid shape as ``(rows, columns)``."""
        rows, columns = self.depth_m.shape
        return int(rows), int(columns)

    @property
    def extent_m(self) -> tuple[float, float]:
        """Return the grid's width and height in metres."""
        rows, columns = self.shape
        return (columns * self.cell_m, rows * self.cell_m)

    @property
    def sea_area_m2(self) -> float:
        """Return the water area the grid measures, in square metres."""
        return float((self.depth_m > 0.0).sum()) * self.cell_m * self.cell_m

    @property
    def credit(self) -> str:
        """Return the one-line attribution the licence requires on the chart."""
        return f"{self.metadata['source']} ({self.metadata['licence']})"


@lru_cache(maxsize=1)
def load(path: Path = PRODUCT) -> Seabed:
    """Return the versioned seabed product.

    Parameters
    ----------
    path : Path
        Where the product lives.

    Returns
    -------
    Seabed
        The rasters, the coastline and the provenance.

    Raises
    ------
    FileNotFoundError
        If the product has not been built. The message says how.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"the seabed product is missing at {path}. Build it with:\n"
            f"    uv sync --group geodata\n"
            f"    uv run python scripts/ingest_geodata.py"
        )
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        depth = archive["depth_dm"].astype(np.float32) * float(metadata["depth_quantum_m"])
        distance = archive["coast_distance_m"].astype(np.float32)
        escape_sea = archive["escape_sea"].astype(np.int64)
        escape_berth = archive["escape_berth"].astype(np.int64)
        vertices = archive["coast_vertices"]
        offsets = archive["coast_offsets"]
    coast = tuple(
        vertices[offsets[index] : offsets[index + 1]] for index in range(len(offsets) - 1)
    )
    return Seabed(
        depth_m=depth,
        coast_distance_m=distance,
        escape_sea=escape_sea,
        escape_berth=escape_berth,
        coast=coast,
        cell_m=float(metadata["cell_m"]),
        origin=np.array([metadata["origin_x_m"], metadata["origin_y_m"]], dtype=np.float64),
        metadata=metadata,
    )
