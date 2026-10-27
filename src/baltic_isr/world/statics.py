"""Fixed geometry, uploaded once as tensors.

Nothing here changes during an episode. Rails, corridors, coastlines and bases
are resampled at setup and live on the device for the rest of the run, so that
the step function only ever gathers from them.

**Coordinates are local.** EPSG:3067 puts the box near 6.6e6 m northing, where
float32 resolves about half a metre -- not good enough to integrate a vessel at
6 m/s over a 600 s step. Everything below is metres from the south-west corner
of the box, where float32 resolves a centimetre. ``origin`` converts back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import torch

from baltic_isr import geo
from baltic_isr.config import Config
from baltic_isr.world import seabed as seabed_module

CABLE_SAMPLES = 64
RAIL_SAMPLES = 256

# Distance to the nearest lane and the nearest corridor is wanted per vessel,
# per step. Measuring it against every segment would be a matrix of points by
# segments inside the hot loop, so it is baked once into a raster and read back
# with a bilinear gather. At 500 m the raster is 276 x 220 -- a quarter of a
# megabyte, and finer than the three kilometre corridors it describes.
RASTER_CELL_M = 500.0


def _resample(line: geo.Polyline, samples: int) -> np.ndarray:
    """Return ``samples`` points evenly spaced along a polyline."""
    return line.point_at(np.linspace(0.0, 1.0, samples))


def _tangents(points: np.ndarray) -> np.ndarray:
    """Return the unit tangent at each sample of a resampled line."""
    delta = np.gradient(points, axis=0)
    norm = np.hypot(delta[:, 0], delta[:, 1])
    unit: np.ndarray = delta / np.where(norm > 0.0, norm, 1.0)[:, None]
    return unit


@dataclass(frozen=True)
class Statics:
    """Every fixed feature of the box, as device tensors.

    Attributes
    ----------
    origin : Tensor
        Projected coordinates of the local origin, shape ``(2,)``, float64.
    extent : Tensor
        Box width and height in metres, shape ``(2,)``.
    rail_xy : Tensor
        Resampled rails, shape ``(rails, samples, 2)``, local metres.
    rail_tangent : Tensor
        Unit tangent at each rail sample, shape ``(rails, samples, 2)``.
    rail_length : Tensor
        Length of each rail, shape ``(rails,)``.
    rail_half_width : Tensor
        Half the lane width, shape ``(rails,)``. A vessel holds a constant
        offset within this, the way a ship holds its side of a lane.
    rail_over_corridor : Tensor
        Whether each rail sample sits over a seabed corridor, shape
        ``(rails, samples)``. This is where a loitering vessel becomes
        interesting, and batch 2 uses it to place deviations.
    depth_m : Tensor
        Measured water depth, shape ``(rows, columns)``, zero on land, on a
        regular raster of ``seabed_cell_m``. Row zero is the southern edge.
    coast_distance_m : Tensor
        Signed distance to the waterline, positive at sea, same raster.
    escape_sea, escape_berth : Tensor
        Flat index of the nearest cell a vehicle may occupy, shape
        ``(rows, columns)``. Precomputed, because this is a static fact about
        the seabed and computing it per step by walking uphill cost 85 % of a
        fleet step and still got stuck in bays.
    min_water_m, berth_clearance_m : float
        The two limits the escape fields were built for, carried here so that
        :func:`world.field.push_offshore` can ask whether a vehicle actually
        breaks one before it moves it.
    seabed_cell_m : float
        Cell size of the two rasters above. Finer than ``RASTER_CELL_M``
        because an archipelago needs it. A plain float, deliberately: reading
        a scalar off a device tensor inside the step synchronises the host,
        and this one is read on every depth lookup.
    cable_xy : Tensor
        Points along every seabed cable that lie at sea, shape ``(samples, 2)``.
        What the fleet is paid to keep watched, and what the map shades.
    rail_distance_m, corridor_distance_m : Tensor
        Distance to the nearest lane and the nearest seabed corridor, on a
        regular raster of ``RASTER_CELL_M``, shape ``(rows, columns)``.
    bases : Tensor
        Shore charging points, shape ``(bases, 2)``.
    """

    origin: torch.Tensor
    extent: torch.Tensor
    rail_xy: torch.Tensor
    rail_tangent: torch.Tensor
    rail_length: torch.Tensor
    rail_half_width: torch.Tensor
    rail_over_corridor: torch.Tensor
    depth_m: torch.Tensor
    coast_distance_m: torch.Tensor
    escape_sea: torch.Tensor
    escape_berth: torch.Tensor
    min_water_m: float
    berth_clearance_m: float
    seabed_cell_m: float
    rail_distance_m: torch.Tensor
    corridor_distance_m: torch.Tensor
    bases: torch.Tensor
    cable_xy: torch.Tensor

    @property
    def device(self) -> torch.device:
        """Return the device every tensor lives on."""
        return self.rail_xy.device

    @property
    def n_rails(self) -> int:
        """Return how many navigation lanes exist."""
        return int(self.rail_xy.shape[0])

    def to_projected(self, xy: torch.Tensor) -> torch.Tensor:
        """Convert local metres back to the projected CRS.

        Parameters
        ----------
        xy : Tensor
            Local coordinates, shape ``(..., 2)``.

        Returns
        -------
        Tensor
            Projected coordinates, float64.
        """
        return xy.to(torch.float64) + self.origin


def build(config: Config, device: torch.device | str = "cpu") -> Statics:
    """Project, resample and upload every fixed feature.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    device : torch.device or str
        Where the tensors are placed.

    Returns
    -------
    Statics
        Device-resident geometry.
    """
    geometry = geo.build(config)
    x_min, y_min, x_max, y_max = geometry.frame.bounds
    origin = np.array([x_min, y_min], dtype=np.float64)

    def local(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((array - origin).astype(np.float32)).to(device)

    rails = [_resample(rail, RAIL_SAMPLES) for rail in geometry.rails]
    over = np.stack(
        [
            np.any(
                np.stack([corridor.contains(points) for corridor in geometry.corridors]),
                axis=0,
            )
            for points in rails
        ]
    )
    # The measured seabed. Its grid was built against the same box bounds by
    # the ingestion script, so the two agree cell for cell; a test checks it
    # rather than trusting it.
    seabed = seabed_module.load()
    # The escape fields were baked against particular thresholds. If the
    # configuration has moved since, the fields are for a different fleet and
    # saying so is better than quietly projecting onto the wrong water.
    for key, configured in (
        ("min_water_m", config.fleet.min_water_m),
        ("berth_clearance_m", config.fleet.berth_clearance_m),
    ):
        baked = float(cast(float, seabed.metadata[key]))
        if abs(baked - configured) > 1e-6:
            raise ValueError(
                f"the seabed product was built with {key}={baked} but the "
                f"configuration says {configured}. Rebuild it:\n"
                f"    uv run python scripts/ingest_geodata.py"
            )

    columns = int(np.ceil((x_max - x_min) / RASTER_CELL_M)) + 1
    rows = int(np.ceil((y_max - y_min) / RASTER_CELL_M)) + 1
    grid_x, grid_y = np.meshgrid(
        np.arange(columns) * RASTER_CELL_M + x_min,
        np.arange(rows) * RASTER_CELL_M + y_min,
        indexing="xy",
    )
    cells = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    def nearest(lines: tuple[geo.Polyline, ...]) -> torch.Tensor:
        stacked = np.stack([line.distance_to(cells) for line in lines])
        return torch.from_numpy(stacked.min(axis=0).reshape(rows, columns).astype(np.float32)).to(
            device
        )

    fixed = Statics(
        origin=torch.from_numpy(origin).to(device),
        extent=torch.tensor([x_max - x_min, y_max - y_min], dtype=torch.float32, device=device),
        rail_xy=torch.stack([local(points) for points in rails]),
        rail_tangent=torch.stack(
            [torch.from_numpy(_tangents(points).astype(np.float32)).to(device) for points in rails]
        ),
        rail_length=torch.tensor(
            [rail.length_m for rail in geometry.rails], dtype=torch.float32, device=device
        ),
        rail_half_width=torch.tensor(
            [rail.half_width_m for rail in geometry.rails], dtype=torch.float32, device=device
        ),
        rail_over_corridor=torch.from_numpy(over).to(device),
        depth_m=torch.from_numpy(seabed.depth_m).to(device),
        coast_distance_m=torch.from_numpy(seabed.coast_distance_m).to(device),
        escape_sea=torch.from_numpy(seabed.escape_sea).to(device),
        escape_berth=torch.from_numpy(seabed.escape_berth).to(device),
        min_water_m=config.fleet.min_water_m,
        berth_clearance_m=config.fleet.berth_clearance_m,
        seabed_cell_m=seabed.cell_m,
        rail_distance_m=nearest(geometry.rails),
        corridor_distance_m=nearest(geometry.corridors),
        bases=local(geometry.bases),
        cable_xy=torch.zeros(0, 2, device=device),
    )

    # The cables as discrete points, so "has anyone heard this stretch lately"
    # is a question the reward can answer. The raster above gives a distance,
    # which cannot carry a per-stretch clock. Built in a second pass because
    # the sea test needs the coastlines, which only exist once Statics does.
    #
    # The renderer draws from these same samples: a map that shaded one set of
    # points while the policy was paid for another would be a lie told with
    # complete confidence.
    from baltic_isr.world import field

    samples = torch.cat(
        [local(_resample(cable, CABLE_SAMPLES)) for cable in geometry.corridors]
    ).reshape(-1, 2)
    return replace(fixed, cable_xy=samples[field.is_sea(fixed, samples)])


def sample_rail(statics: Statics, rail: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Return the position at a fractional progress along a rail.

    Linear interpolation between the two nearest resamples. Pure gather and
    arithmetic: the cost does not depend on how many vessels ask.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    rail : Tensor
        Rail index per vessel, shape ``(...)``, integer.
    progress : Tensor
        Fraction along the rail in ``[0, 1)``, same shape.

    Returns
    -------
    Tensor
        Positions, shape ``(..., 2)``.
    """
    return _gather_along(statics.rail_xy, rail, progress)


def sample_rail_tangent(
    statics: Statics, rail: torch.Tensor, progress: torch.Tensor
) -> torch.Tensor:
    """Return the unit tangent at a fractional progress along a rail.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    rail : Tensor
        Rail index per vessel, integer.
    progress : Tensor
        Fraction along the rail in ``[0, 1)``.

    Returns
    -------
    Tensor
        Unit tangents, shape ``(..., 2)``.
    """
    tangent = _gather_along(statics.rail_tangent, rail, progress)
    unit: torch.Tensor = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return unit


def _gather_along(table: torch.Tensor, rail: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Interpolate a ``(rails, samples, 2)`` table at fractional positions."""
    samples = table.shape[1]
    position = progress.clamp(0.0, 1.0) * (samples - 1)
    low = position.floor().to(torch.long).clamp(0, samples - 1)
    high = (low + 1).clamp(0, samples - 1)
    weight = (position - low.to(position.dtype)).unsqueeze(-1)
    index = rail.to(torch.long).clamp(0, table.shape[0] - 1)
    return torch.lerp(table[index, low], table[index, high], weight)


def sample_raster(
    statics: Statics,
    raster: torch.Tensor,
    xy: torch.Tensor,
    cell_m: float | None = None,
    centred: bool = False,
) -> torch.Tensor:
    """Read a setup-time raster at arbitrary positions, bilinearly.

    Parameters
    ----------
    statics : Statics
        Device geometry, for the cell size.
    raster : Tensor
        ``(rows, columns)``, indexed from the local origin.
    xy : Tensor
        Local metres, shape ``(..., 2)``.
    cell_m : float, optional
        Cell size of this raster. Defaults to the lane and corridor raster.
        The seabed rasters are finer, so they pass their own.
    centred : bool
        Whether the raster samples cell CENTRES rather than grid nodes. The
        lane and corridor rasters here are built on nodes starting at the
        origin; the ingested seabed is built on centres. Reading one with the
        other's convention shifts the whole field half a cell north-east,
        which at 100 m is seventy metres of coastline in the wrong place.

    Returns
    -------
    Tensor
        Values at those positions, shape ``(...)``.
    """
    rows, columns = raster.shape
    size = RASTER_CELL_M if cell_m is None else cell_m
    shift = 0.5 if centred else 0.0
    column = (xy[..., 0] / size - shift).clamp(0.0, columns - 1.001)
    row = (xy[..., 1] / size - shift).clamp(0.0, rows - 1.001)
    left, bottom = column.floor().to(torch.long), row.floor().to(torch.long)
    right, top = (left + 1).clamp(max=columns - 1), (bottom + 1).clamp(max=rows - 1)
    fx = (column - left.to(column.dtype)).unsqueeze(-1)
    fy = (row - bottom.to(row.dtype)).unsqueeze(-1)
    lower = torch.lerp(raster[bottom, left].unsqueeze(-1), raster[bottom, right].unsqueeze(-1), fx)
    upper = torch.lerp(raster[top, left].unsqueeze(-1), raster[top, right].unsqueeze(-1), fx)
    return torch.lerp(lower, upper, fy).squeeze(-1)
