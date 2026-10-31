"""Turn the published bathymetry of the Gulf of Finland into the rasters the box runs on.

Run once, offline, and commit what it writes. Everything downstream -- the land
model, the acoustic waveguide, the shore clearance and the chart -- reads the
single file this produces, so the simulation and the map can never disagree
about where the water is.

Until now the land model was two five-vertex polylines and the depth was
``70 (1 - exp(-x / 9000))``, a smooth shelf standing in for a chart. Both were
honest placeholders and both were labelled as such. This replaces them with
measurement: EMODnet's regional digital terrain model, which carries seabed and
land in one surface, so the coastline is simply where it crosses zero.

    uv sync --group geodata
    uv run python scripts/ingest_geodata.py

Source
------
EMODnet Bathymetry, Consolidated DTM for European sea regions, served as
coverage ``emodnet__mean`` from the project's own WCS endpoint. Licensed
CC-BY: the attribution is carried in the output file and printed on the chart.
Native resolution is 1/16 arc-minute, about 58 m of easting and 116 m of
northing at this latitude -- finer than the 100 m grid written here, so the
resampling below is an average rather than an invention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "config" / "world.yaml"
DATA = ROOT / "data"
PRODUCT = DATA / "gulf_of_finland.npz"

WCS = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE = "emodnet__mean"
LICENCE = "CC-BY 4.0"
CITATION = "EMODnet Bathymetry Consortium, Consolidated DTM for European sea regions"

# Fetched a little wider than the box, so that every target cell -- including
# the ones on the boundary, whose bilinear stencil reaches outside -- is
# sampled from real data rather than from an edge clamp.
MARGIN_DEG = 0.10

# 100 m. The decision is a trade between the archipelago and the file: the
# islands off Helsinki that matter to a chart are a few hundred metres across,
# so 100 m resolves them and 250 m dissolves them into the mainland. At this
# size the two rasters are 1.4 M cells each, which int16 and deflate bring to
# a couple of megabytes -- small enough to version, which is the point, because
# a world model that has to be downloaded is a world model that drifts.
CELL_M = 100.0

# Depth is stored in decimetres and the signed distance in metres, both int16.
# The quantisation is far below anything that reads them: the sonar wants the
# order of magnitude of the waveguide, and the shore clearance is a band
# hundreds of metres wide. Floats here would double the file and say nothing
# more. The distance saturates at 32 km, beyond which "far from shore" is the
# only fact anybody needs.
DEPTH_QUANTUM_M = 0.1
DISTANCE_LIMIT_M = 32000.0

# Shorter than this and it is a rock, not a feature of the chart. Dropping them
# keeps the coastline file honest about what 100 m can actually resolve.
MIN_COAST_RING_M = 400.0

# The escape fields. For every cell, the displacement to the nearest cell a
# vehicle is allowed to occupy -- one field for the open sea, where the limit
# is water under the keel, and one for a base approach, where it is distance
# off the quay. Baked here because they are static, exact, and otherwise cost
# the step dearly: computed by gradient ascent at run time they were 85 % of a
# fleet step, and they still got stuck in bays, because walking uphill from
# inside a shallow bay cannot find the deep water around the headland. A
# distance transform knows where it is.
#
# The thresholds come from the configuration and are written into the metadata,
# so world.statics can refuse to run against a product built for different
# ones rather than silently using the wrong field.

# The way home, as a field rather than a bearing.
#
# A recalled vehicle used to steer the straight bearing to its base, and an
# archipelago does not grant straight lines. Measured over a 48 h episode:
# vehicles recalled at a median 37.5 km made good 0.32 m/s against a 1.2 m/s
# cruise -- 27 % -- grinding along whatever island lay across the bearing, and
# 22 of 32 died doing it. They had been sent home with energy sized on the
# straight line times 1.4, while the passage actually cost 3.75x.
#
# One field fixes both halves. Geodesic distance to each berth over the water
# the fleet may use, solved once here by Dijkstra on the eight-connected grid:
# the distance sizes the recall margin honestly, and its descent direction IS
# the route. Sizing the old margin for the measured 3.75x instead would have
# kept the whole fleet alongside, which is not a fix, it is a surrender.
#
# Solved at the full 100 m and stored every other cell. 200 m still resolves
# the channels a 3.5 m vehicle needs -- and anything finer than that is
# push_offshore's job, not the route's.
HOME_STRIDE = 2
HOME_QUANTUM_M = 10.0
HOME_UNREACHABLE = 65535

# Four masked box blurs over the descent field. Pure eight-neighbour descent is
# correct and ugly: it can only ever point along one of eight bearings, so a
# 30 km passage comes out as a staircase and the film shows it. Smoothing mixes
# neighbouring descent directions, which stay descent directions, and the
# vehicle's own turn-rate limit does the rest.
HOME_SMOOTH_PASSES = 4

_STEPS = tuple(
    (dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
)


def _ways_home(
    masks: list[np.ndarray], cell_m: float, berths: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Return geodesic distance to each berth, and the route field that descends it.

    Parameters
    ----------
    masks : list of ndarray
        One boolean mask per base, True where a vehicle bound for THAT base may
        be. Per base because the rule is: water under the keel out in the gulf,
        clearance off the quay inside the approach to its own mooring. Each
        must already be eroded, so that a route never threads a cell the
        projection would push it out of.
    cell_m : float
        Cell size in metres.
    berths : list of tuple of int
        One ``(row, column)`` per base. Snapped to the nearest allowed cell if
        the berth itself falls outside its own mask.

    Returns
    -------
    tuple of ndarray
        Distance in metres, ``(bases, rows, columns)``, infinite where no
        route exists; and the unit route field, ``(bases, rows, columns, 2)``
        as east and north, zero where there is no route.
    """
    rows, columns = masks[0].shape
    distance = np.full((len(berths), rows, columns), np.inf)
    route = np.zeros((len(berths), rows, columns, 2))
    for index, ((row, column), allowed) in enumerate(zip(berths, masks, strict=True)):
        field = _solve(allowed, cell_m, row, column)
        distance[index] = field
        route[index] = _descend(field, allowed)
    return distance, route


def _solve(allowed: np.ndarray, cell_m: float, row: int, column: int) -> np.ndarray:
    """Return geodesic distance from one cell over one mask, by Dijkstra.

    Parameters
    ----------
    allowed : ndarray
        Boolean, True where a vehicle may be.
    cell_m : float
        Cell size in metres.
    row, column : int
        The source cell, snapped to the nearest allowed cell if it is outside.

    Returns
    -------
    ndarray
        Distance in metres, infinite where unreachable.
    """
    rows, columns = allowed.shape
    # Nodes are the allowed cells only. Over the whole box the graph would be
    # 1.4 M nodes and 11 M edges; on the water alone it is a little over half
    # that, which is the difference between comfortable and not.
    order = np.full(allowed.size, -1, dtype=np.int64)
    flat = np.flatnonzero(allowed.reshape(-1))
    order[flat] = np.arange(flat.size)
    node = order.reshape(rows, columns)

    sources, targets, lengths = [], [], []
    for dy, dx, span in _STEPS:
        y0, y1 = max(0, -dy), rows - max(0, dy)
        x0, x1 = max(0, -dx), columns - max(0, dx)
        here = node[y0:y1, x0:x1]
        there = node[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        both = (here >= 0) & (there >= 0)
        sources.append(here[both])
        targets.append(there[both])
        lengths.append(np.full(int(both.sum()), span * cell_m, dtype=np.float32))
    graph = csr_matrix(
        (np.concatenate(lengths), (np.concatenate(sources), np.concatenate(targets))),
        shape=(flat.size, flat.size),
    )

    if not allowed[row, column]:
        reachable = np.argwhere(allowed)
        gap = np.hypot(reachable[:, 0] - row, reachable[:, 1] - column)
        row, column = (int(value) for value in reachable[int(np.argmin(gap))])
    spread = dijkstra(graph, directed=False, indices=int(node[row, column]))
    field = np.full(allowed.size, np.inf)
    field[flat] = spread
    return np.asarray(field.reshape(rows, columns))


def _fill_from_reachable(
    distance: np.ndarray, route: np.ndarray, reachable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extend both fields over land, from the nearest cell that has a route.

    Both fields are read back bilinearly, over a two-by-two stencil that
    straddles the waterline wherever a vehicle is near one. Left as they come
    out of the solver, land carries an infinity and a zero vector, so a
    vehicle a single cell off the shore interpolates its distance home against
    four hundred kilometres and is recalled on the spot, and its course home
    shrinks towards nothing and comes out of the normalisation pointing
    anywhere.

    Filling each hole from the nearest cell that does have a route makes the
    rasters smooth across the waterline while changing nothing a vehicle can
    read: a vehicle is always in navigable water, which ``push_offshore``
    guarantees, so the value under it is the solved one either way.

    Parameters
    ----------
    distance : ndarray
        ``(bases, rows, columns)``, infinite where unsolved.
    route : ndarray
        ``(bases, rows, columns, 2)``, zero where unsolved.
    reachable : ndarray
        Boolean, ``(bases, rows, columns)``, True where a route was found.

    Returns
    -------
    tuple of ndarray
        The two fields, defined everywhere.
    """
    filled_distance = np.array(distance)
    filled_route = np.array(route)
    for index in range(distance.shape[0]):
        holes = ~reachable[index]
        if not holes.any():
            continue
        rows, columns = ndimage.distance_transform_edt(
            holes, return_distances=False, return_indices=True
        )
        filled_distance[index] = distance[index][rows, columns]
        filled_route[index] = route[index][rows, columns]
    return filled_distance, filled_route


def _descend(field: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Return the smoothed unit direction that walks ``field`` downhill.

    Parameters
    ----------
    field : ndarray
        Geodesic distance, infinite where unreachable.
    allowed : ndarray
        Boolean, True where a vehicle may be.

    Returns
    -------
    ndarray
        East and north components, ``(rows, columns, 2)``, zero where there is
        nowhere to go.
    """
    rows, columns = field.shape
    best = np.array(field, dtype=np.float64)
    step = np.zeros((rows, columns, 2))
    for dy, dx, _ in _STEPS:
        shifted = np.full_like(best, np.inf)
        y0, y1 = max(0, -dy), rows - max(0, dy)
        x0, x1 = max(0, -dx), columns - max(0, dx)
        shifted[y0:y1, x0:x1] = field[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        # Strictly better, so a plateau never sets a direction and the berth
        # cell itself -- the one minimum of the field -- keeps a zero.
        better = shifted < best
        best = np.where(better, shifted, best)
        span = math.hypot(dy, dx)
        step[..., 0] = np.where(better, dx / span, step[..., 0])
        step[..., 1] = np.where(better, dy / span, step[..., 1])

    live = allowed & np.isfinite(field)

    def normalise(vectors: np.ndarray) -> np.ndarray:
        span = np.hypot(vectors[..., 0], vectors[..., 1])[..., None]
        return np.divide(vectors, span, out=np.zeros_like(vectors), where=span > 1e-9)

    raw = normalise(step)
    smoothed = np.array(step)
    weight = live.astype(np.float64)
    for _ in range(HOME_SMOOTH_PASSES):
        mass = ndimage.uniform_filter(weight, size=3, mode="nearest")
        for axis in (0, 1):
            smoothed[..., axis] = np.divide(
                ndimage.uniform_filter(smoothed[..., axis] * weight, size=3, mode="nearest"),
                mass,
                out=np.zeros_like(mass),
                where=mass > 0.0,
            )
    smoothed = normalise(smoothed)

    # Check the smoothing, cell by cell, and keep only what survives.
    #
    # A smoothed course is a mixture of descent directions, which is usually a
    # descent direction and is not always one. In a channel a cell or two wide
    # the mixture points at the bank, and a vehicle following it grinds: one of
    # 192 sat 17.5 km from its mooring making 0.02 m/s of a 1.2 m/s cruise,
    # nominally "returning", until the battery went. It is not a margin
    # problem -- raising the slack to 1.5 did not save it -- it is a route that
    # does not go anywhere.
    #
    # So the stored course has to lead somewhere: one cell along it must be
    # legal water with a strictly smaller distance to the berth. Where the
    # smoothed course fails that, the raw eight-neighbour descent is kept,
    # staircase and all, because a staircase that gets home beats a smooth
    # curve that does not.
    rows_index, columns_index = np.mgrid[0:rows, 0:columns]
    ahead_x = np.clip(np.round(columns_index + smoothed[..., 0]).astype(int), 0, columns - 1)
    ahead_y = np.clip(np.round(rows_index + smoothed[..., 1]).astype(int), 0, rows - 1)
    downhill = live[ahead_y, ahead_x] & (field[ahead_y, ahead_x] < field)
    unit = np.where(downhill[..., None], smoothed, raw)
    return np.where(live[..., None], unit, 0.0)


def _fetch(bounds: tuple[float, float, float, float], cache: Path) -> np.ndarray:
    """Return the published terrain model over ``bounds`` as a north-up array.

    Parameters
    ----------
    bounds : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.
    cache : Path
        Where the downloaded GeoTIFF is kept, so a re-run costs nothing.

    Returns
    -------
    ndarray
        Elevation in metres, positive on land, shape ``(rows, columns)``.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    if not cache.exists():
        query = (
            f"{WCS}?service=WCS&version=2.0.1&request=GetCoverage"
            f"&coverageId={COVERAGE}"
            f"&subset=Long({lon_min},{lon_max})"
            f"&subset=Lat({lat_min},{lat_max})"
            f"&format=image/tiff"
        )
        print(f"  fetching {COVERAGE} over {lon_min}..{lon_max} E, {lat_min}..{lat_max} N")
        with urllib.request.urlopen(query, timeout=300) as response:
            payload = response.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        print(f"  cached {len(payload) / 1e6:.1f} MB at {cache}")
    else:
        print(f"  reusing cached download at {cache}")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(io.BytesIO(cache.read_bytes())) as image:
        if image.mode != "F":
            raise RuntimeError(f"expected a float coverage, the server sent mode {image.mode!r}")
        return np.asarray(image, dtype=np.float32)


def _sample(
    source: np.ndarray,
    bounds: tuple[float, float, float, float],
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Return the source raster sampled bilinearly at the given coordinates.

    The coverage is a plain north-up grid on a geographic axis -- the server
    honours the requested window exactly, so the corners given are the corners
    returned and no affine transform has to be read back out of the file.

    Parameters
    ----------
    source : ndarray
        Elevation, shape ``(rows, columns)``, row zero at the north edge.
    bounds : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` of the source raster.
    lon, lat : ndarray
        Where to sample, any matching shape.

    Returns
    -------
    ndarray
        Elevation at each coordinate, the shape of ``lon``.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    rows, columns = source.shape

    # Pixel centres, hence the half cell: the window edge is the outer edge of
    # the first pixel, not the middle of it.
    x = (lon - lon_min) / (lon_max - lon_min) * columns - 0.5
    y = (lat_max - lat) / (lat_max - lat_min) * rows - 0.5

    x0 = np.clip(np.floor(x).astype(np.int64), 0, columns - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, rows - 1)
    x1 = np.clip(x0 + 1, 0, columns - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)
    fx = np.clip(x - x0, 0.0, 1.0)
    fy = np.clip(y - y0, 0.0, 1.0)

    top = source[y0, x0] * (1.0 - fx) + source[y0, x1] * fx
    bottom = source[y1, x0] * (1.0 - fx) + source[y1, x1] * fx
    return np.asarray(top * (1.0 - fy) + bottom * fy, dtype=np.float64)


def _signed_distance(sea: np.ndarray, cell_m: float) -> np.ndarray:
    """Return distance to the coastline in metres, positive at sea.

    Two Euclidean transforms, one per side, differenced. Each measures to the
    nearest cell of the other kind, so the zero sits half a cell seaward of the
    true waterline; the half cell is subtracted back off to centre it.

    Parameters
    ----------
    sea : ndarray
        Boolean, True where there is water.
    cell_m : float
        Cell size in metres.

    Returns
    -------
    ndarray
        Signed distance in metres, the shape of ``sea``.
    """
    offshore = ndimage.distance_transform_edt(sea) * cell_m
    inland = ndimage.distance_transform_edt(~sea) * cell_m
    return np.asarray(offshore - inland, dtype=np.float64) - 0.5 * cell_m


def _escape(allowed: np.ndarray) -> np.ndarray:
    """Return, for each cell, the flat index of the nearest allowed cell.

    An **index**, not a displacement. A displacement has to be stored in
    metres, and the widest one here is 33.9 km -- from deep inside Finland --
    which does not fit in the int16 that a field of three million values
    wants to be. It silently wrapped, and a vehicle at the north-east corner
    of the box was projected two kilometres further inland. An index cannot
    overflow: the largest is 1.5 million, and int32 holds it with room to
    spare, in the same space the two int16 channels took.

    It also carries one extra fact for free. A cell whose target is itself is
    a cell that already qualifies, which is how the projection knows not to
    round a vehicle that is perfectly happy onto the nearest cell centre.

    Parameters
    ----------
    allowed : ndarray
        Boolean, True where a vehicle may be.

    Returns
    -------
    ndarray
        Flat indices into a raster of ``allowed``'s shape, int32.
    """
    # Eroded by one cell first, and this is what makes the guarantee exact
    # rather than approximate.
    #
    # The constraint is checked per CELL and read back per POINT: `depth_m`
    # interpolates bilinearly over a two-by-two stencil, so a point sitting in
    # a cell with 15.2 m under it can measure 14.9 m because its neighbour is
    # shallower. Measured, that left 47 points in 22 500 short of the limit,
    # the worst by 1.7 m.
    #
    # Requiring the whole three-by-three neighbourhood puts every corner of
    # every stencil inside the allowed set, whichever way the rounding falls,
    # so the interpolant cannot dip below the limit anywhere. It costs one
    # hundred-metre cell of navigable water along each edge, in the
    # conservative direction.
    allowed = ndimage.binary_erosion(allowed, structure=np.ones((3, 3), dtype=bool))
    if not allowed.any():
        raise RuntimeError("no cell in the box satisfies the constraint")
    # The transform runs on the COMPLEMENT: distance_transform_edt measures
    # from the zeros, so the zeros have to be the places worth reaching.
    rows, columns = ndimage.distance_transform_edt(
        ~allowed, return_distances=False, return_indices=True
    )
    return (rows * allowed.shape[1] + columns).astype(np.int32)


def _coastline(elevation: np.ndarray, cell_m: float) -> Iterator[np.ndarray]:
    """Yield the waterline as polylines in local metres.

    Taken as the zero contour of the terrain rather than as the edge of the
    land mask, so the line runs where the surface actually crosses sea level
    instead of stepping around cell corners. Rendering wants this; the physics
    uses the rasters.

    Parameters
    ----------
    elevation : ndarray
        Metres, positive on land, shape ``(rows, columns)``.
    cell_m : float
        Cell size in metres.

    Yields
    ------
    ndarray
        Vertices in local metres, shape ``(vertices, 2)``.
    """
    from contourpy import contour_generator

    rows, columns = elevation.shape
    # Row zero of the raster is the SOUTH edge: the grid is built northward
    # from the box origin, which is what every consumer expects.
    generate = contour_generator(
        x=(np.arange(columns) + 0.5) * cell_m,
        y=(np.arange(rows) + 0.5) * cell_m,
        z=elevation,
        line_type="SeparateCode",
    )
    for vertices in generate.lines(0.0)[0]:
        if len(vertices) < 3:
            continue
        span = np.hypot(*(vertices.max(axis=0) - vertices.min(axis=0)))
        if span < MIN_COAST_RING_M:
            continue
        yield np.asarray(vertices, dtype=np.float32)


def main() -> None:
    """Build the raster product and report what it measured."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-m", type=float, default=CELL_M)
    parser.add_argument("--cache", type=Path, default=DATA / "emodnet_raw.tif")
    arguments = parser.parse_args()

    world = yaml.safe_load(WORLD.read_text(encoding="utf-8"))
    geo = world["geo"]
    crs = geo["projected_crs"]

    # The box in projected metres, taken over all four corners, because a
    # rectangle in degrees is not one in TM35FIN. This mirrors geo.Frame.bounds
    # exactly; a test keeps the two from drifting apart.
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corner_lon = np.array([geo["lon_min"], geo["lon_max"], geo["lon_min"], geo["lon_max"]])
    corner_lat = np.array([geo["lat_min"], geo["lat_min"], geo["lat_max"], geo["lat_max"]])
    corner_x, corner_y = forward.transform(corner_lon, corner_lat)
    origin = (float(corner_x.min()), float(corner_y.min()))
    extent = (float(corner_x.max() - corner_x.min()), float(corner_y.max() - corner_y.min()))

    cell_m = float(arguments.cell_m)
    columns = int(np.ceil(extent[0] / cell_m))
    rows = int(np.ceil(extent[1] / cell_m))
    print(
        f"box {extent[0] / 1e3:.1f} x {extent[1] / 1e3:.1f} km"
        f" -> {columns} x {rows} at {cell_m:.0f} m"
    )

    source_bounds = (
        geo["lon_min"] - MARGIN_DEG,
        geo["lat_min"] - MARGIN_DEG,
        geo["lon_max"] + MARGIN_DEG,
        geo["lat_max"] + MARGIN_DEG,
    )
    source = _fetch(source_bounds, arguments.cache)
    print(f"  source grid {source.shape[1]} x {source.shape[0]}")

    # Cell centres of the target grid, unprojected one row at a time rather
    # than all at once: 1.4 M coordinate pairs through pyproj in one call is a
    # gigabyte of intermediate, and the loop costs a couple of seconds.
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    easting = origin[0] + (np.arange(columns) + 0.5) * cell_m
    elevation = np.empty((rows, columns), dtype=np.float64)
    for row in range(rows):
        northing = np.full(columns, origin[1] + (row + 0.5) * cell_m)
        lon, lat = inverse.transform(easting, northing)
        elevation[row] = _sample(source, source_bounds, lon, lat)

    sea = elevation < 0.0
    depth = np.where(sea, -elevation, 0.0)
    distance = _signed_distance(sea, cell_m)

    cell_area = cell_m * cell_m
    sea_area_m2 = float(sea.sum()) * cell_area
    coast = list(_coastline(elevation, cell_m))

    depth_dm = np.clip(np.round(depth / DEPTH_QUANTUM_M), 0, np.iinfo(np.int16).max)
    distance_m = np.clip(np.round(distance), -DISTANCE_LIMIT_M, DISTANCE_LIMIT_M)

    min_water_m = float(world["fleet"]["min_water_m"])
    berth_clearance_m = float(world["fleet"]["berth_clearance_m"])
    navigable = depth >= min_water_m
    escape_sea = _escape(navigable)
    # Report the eroded figure, since that is the water the fleet may use.
    usable = ndimage.binary_erosion(navigable, structure=np.ones((3, 3), dtype=bool))
    escape_berth = _escape(distance >= berth_clearance_m)
    here = np.arange(escape_sea.size, dtype=np.int64).reshape(escape_sea.shape)
    moved = escape_sea != here
    travel = (
        np.hypot(
            (escape_sea % columns - here % columns)[moved],
            (escape_sea // columns - here // columns)[moved],
        )
        * cell_m
    )
    print(
        f"  escape fields   at sea: {100 * float(usable.mean()):.1f} % of the box is"
        f" navigable, worst displacement {float(travel.max()) / 1e3:.1f} km"
    )

    # The way home, solved on the water the projection will actually leave a
    # vehicle on -- which is not the same mask everywhere.
    #
    # Out in the gulf it is the keel clearance, eroded. Inside the approach to
    # its own base a vehicle may come into shallow water, because that is what
    # a base is for, and `push_offshore` relaxes to the berthing clearance
    # there. Solving on the open-sea mask alone put the field's minimum 2.5 km
    # from the Naissaar mooring -- the solver snapped the berth out to the
    # nearest deep cell -- so the route led the fleet to a point near the base
    # instead of to the base, and eight vehicles of 192 sat 1.95 km off it
    # until the battery went. The mask has to say what the projection says.
    berths = []
    masks = []
    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    berthable = ndimage.binary_erosion(
        distance >= berth_clearance_m, structure=np.ones((3, 3), dtype=bool)
    )
    for base in world["bases"]:
        berth_x, berth_y = forward.transform(base["lon"], base["lat"])
        berth_row = int(np.clip((berth_y - origin[1]) / cell_m, 0, rows - 1))
        berth_column = int(np.clip((berth_x - origin[0]) / cell_m, 0, columns - 1))
        berths.append((berth_row, berth_column))
        # WHERE, not OR. Inside its own approach the berthing rule REPLACES the
        # keel rule -- `push_offshore` asks for clearance off the quay there and
        # stops asking about depth. Taking the union instead let the route
        # thread cells that are deep but within 400 m of the shore, which the
        # projection then refuses: a vehicle following such a route arrived
        # 1.97 km from its mooring, was pinned by the clearance it had been
        # routed into, and sat there for eight hours until the battery went.
        # Fourteen of 192 went that way. A route is only a route if the mask
        # that drew it is the mask that will be enforced on it.
        inside = np.hypot(grid_y - berth_row, grid_x - berth_column) * cell_m <= float(
            world["fleet"]["base_approach_m"]
        )
        masks.append(np.where(inside, berthable, usable))
    home_distance, home_route = _ways_home(masks, cell_m, berths)

    straight = np.zeros_like(home_distance)
    grid_y, grid_x = np.mgrid[0:rows, 0:columns]
    for index, (berth_row, berth_column) in enumerate(berths):
        straight[index] = np.hypot(grid_y - berth_row, grid_x - berth_column) * cell_m
    for index, base in enumerate(world["bases"]):
        reached = masks[index] & np.isfinite(home_distance[index])
        detour = home_distance[index][reached] / np.maximum(straight[index][reached], cell_m)
        sink = np.unravel_index(int(np.argmin(home_distance[index])), home_distance[index].shape)
        off = np.hypot(sink[0] - berths[index][0], sink[1] - berths[index][1]) * cell_m
        print(
            f"  way home        {base['name']}: {100 * reached.sum() / masks[index].sum():.1f} %"
            f" of navigable water reaches it, detour median {np.median(detour):.2f}x,"
            f" 95th {np.percentile(detour, 95):.2f}x, worst {detour.max():.2f}x,"
            f" field minimum {off:.0f} m off the berth"
        )

    solved = np.stack(masks) & np.isfinite(home_distance)

    # A pocket of navigable water with no route home is worse than shallow
    # water, and the fill below is what makes it dangerous rather than merely
    # useless: an unreachable cell inherits the distance AND the course of the
    # nearest cell that does have a route, so it stops being distinguishable
    # from a routed one. A vehicle reads a plausible course, that course
    # crosses the bank, and it grinds there until the battery goes -- 18.0 km
    # of path for 0.3 km of progress, measured on vehicle 23 of a forty-vehicle
    # episode, lost at 20.4 h.
    #
    # They exist because the router and the projection disagree about what a
    # channel is: this grid is eroded and strided to 200 m, `push_offshore`
    # runs at 100 m, so a vehicle can swim through a gap the router cannot see
    # and then cannot be routed back out of it.
    #
    # Excluded from the water the projection will leave a vehicle on, so the
    # vehicle is kept out instead of being rescued. Removing a cell that had no
    # route cannot strand one that had: no route ever passed through it.
    # Against the ROUTING masks and not against `navigable`: `navigable` is the
    # raw keel rule, un-eroded, so differencing it against a solve run on the
    # eroded mask condemns the whole one-cell shoreline band -- 4.2 % of the
    # water, which the clearance rules already handle. What stands a vehicle up
    # is a cell its own base's router was willing to put it in and cannot get
    # it out of.
    stranding = (np.stack(masks) & ~solved).any(axis=0)
    navigable = navigable & ~stranding
    escape_sea = _escape(navigable)
    usable = ndimage.binary_erosion(navigable, structure=np.ones((3, 3), dtype=bool))
    print(
        f"  no route home   {int(stranding.sum()):,} cells excluded,"
        f" {100 * stranding.sum() / max(int(np.stack(masks).any(axis=0).sum()), 1):.3f} %"
        f" of routable water, in {int(ndimage.label(stranding)[1]):,} pockets"
    )

    home_distance, home_route = _fill_from_reachable(home_distance, home_route, solved)
    home_distance_dam = np.clip(
        np.nan_to_num(home_distance / HOME_QUANTUM_M, posinf=HOME_UNREACHABLE),
        0,
        HOME_UNREACHABLE,
    )[:, ::HOME_STRIDE, ::HOME_STRIDE]
    home_route_i8 = np.round(home_route * 127.0)[:, ::HOME_STRIDE, ::HOME_STRIDE]

    # Polylines are ragged, so they travel as one vertex array plus the offsets
    # that cut it up. npz has no ragged array and object arrays would pickle.
    lengths = np.array([len(ring) for ring in coast], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    vertices = np.concatenate(coast, axis=0) if coast else np.zeros((0, 2), dtype=np.float32)

    metadata = {
        "source": CITATION,
        "coverage": COVERAGE,
        "endpoint": WCS,
        "licence": LICENCE,
        "retrieved": dt.date.today().isoformat(),
        "projected_crs": crs,
        "cell_m": cell_m,
        "origin_x_m": origin[0],
        "origin_y_m": origin[1],
        "rows": rows,
        "columns": columns,
        "depth_quantum_m": DEPTH_QUANTUM_M,
        "distance_limit_m": DISTANCE_LIMIT_M,
        "sea_area_m2": sea_area_m2,
        "min_water_m": min_water_m,
        "berth_clearance_m": berth_clearance_m,
        "home_stride": HOME_STRIDE,
        "home_quantum_m": HOME_QUANTUM_M,
        "home_unreachable": HOME_UNREACHABLE,
        "bases": [base["name"] for base in world["bases"]],
    }

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRODUCT,
        depth_dm=depth_dm.astype(np.int16),
        coast_distance_m=distance_m.astype(np.int16),
        escape_sea=escape_sea,
        escape_berth=escape_berth,
        home_distance_dam=home_distance_dam.astype(np.uint16),
        home_route_i8=home_route_i8.astype(np.int8),
        coast_vertices=vertices.astype(np.float32),
        coast_offsets=offsets,
        metadata=np.array(json.dumps(metadata, indent=2)),
    )

    water = depth[sea]
    box_area = extent[0] * extent[1]
    print()
    print(
        f"  water          {sea_area_m2 / 1e6:10.0f} km2"
        f"  ({100 * sea_area_m2 / box_area:.1f} % of the box)"
    )
    print(f"  mean depth     {water.mean():10.1f} m")
    print(f"  median depth   {np.median(water):10.1f} m")
    print(f"  deepest        {water.max():10.1f} m")
    print(f"  coastline      {len(coast):10d} rings, {len(vertices)} vertices")
    print(f"  written        {PRODUCT}  ({PRODUCT.stat().st_size / 1e6:.1f} MB)")
    print()
    print(f"  config/world.yaml wants  sea_area_m2: {sea_area_m2:.4e}")


if __name__ == "__main__":
    main()
