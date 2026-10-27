"""Water, land and the current that runs between them.

Every function takes local metres and returns a tensor of the same batch shape.
None of them loops, branches on data, or touches the host: they are written to
sit inside the step.

The land model is **measured**: EMODnet's terrain model for the Gulf of
Finland, ingested once onto a 100 m grid (``world.seabed``). Depth and the
signed distance to the waterline are bilinear gathers from that grid, so the
archipelago off Helsinki is where it really is and the shelf is the shelf.

It replaces two five-vertex polylines and an exponential shelf -- a caricature
that C-3-12 described honestly and that this supersedes. The practical
difference is that "there is land here" is now a fact rather than a smooth
approximation, which matters to a fleet that has to stay off it.
"""

from __future__ import annotations

import math

import torch

from baltic_isr.config import Environment
from baltic_isr.world.statics import Statics, sample_raster

# Transmission loss takes a logarithm of the depth, so a vehicle that strays
# into a metre of water must not divide by nothing. Three metres is shallower
# than anything the fleet is allowed in anyway.
_MIN_DEPTH_M = 3.0

# One cell of the seabed grid, used for the central difference that takes the
# gradient of the signed distance. Smaller than a cell would difference the
# bilinear interpolant instead of the field and return a staircase.
_NORMAL_STEP_M = 100.0


def _seabed(statics: Statics, raster: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Gather from a seabed raster, which is on its own finer, centred grid."""
    return sample_raster(statics, raster, xy, cell_m=statics.seabed_cell_m, centred=True)


def distance_to_coast(statics: Statics, xy: torch.Tensor) -> torch.Tensor:
    """Return signed distance to the nearer shore: positive at sea.

    A true Euclidean distance, read from the signed-distance field built once
    from the terrain model. The previous version measured straight across the
    gulf to one of two polylines, which was defensible for a basin treated as
    a channel and is simply wrong near an archipelago: an island two
    kilometres away is two kilometres away, whichever bearing it lies on.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.

    Returns
    -------
    Tensor
        Signed distance in metres, shape ``(...)``.
    """
    return _seabed(statics, statics.coast_distance_m, xy)


def is_sea(statics: Statics, xy: torch.Tensor) -> torch.Tensor:
    """Return whether each point is water.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.

    Returns
    -------
    Tensor
        Boolean, shape ``(...)``.
    """
    return distance_to_coast(statics, xy) > 0.0


def coast_normal(statics: Statics, xy: torch.Tensor) -> torch.Tensor:
    """Return the unit vector pointing from the nearer shore out to sea.

    The gradient of the signed distance field, by central difference over one
    cell. Nothing in the step calls it any more -- the projection onto legal
    water is a baked lookup now -- but it is how the shipped field is checked:
    a true distance field has ``|grad d| = 1`` everywhere except on the medial
    axis, and ``tests/unit/test_world.py`` holds the vendored product to that.
    If a future ingestion ever wrote something that merely resembled a
    distance field, this is what would notice.

    The closed form it replaced only existed because the old shore was a
    function ``y = coast(x)``. An archipelago is not a function of easting,
    so there is nothing to differentiate but the field itself.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.

    Returns
    -------
    Tensor
        Unit vectors, shape ``(..., 2)``.
    """
    step = _NORMAL_STEP_M
    east = torch.tensor([step, 0.0], dtype=xy.dtype, device=xy.device)
    north = torch.tensor([0.0, step], dtype=xy.dtype, device=xy.device)
    gradient = torch.stack(
        [
            distance_to_coast(statics, xy + east) - distance_to_coast(statics, xy - east),
            distance_to_coast(statics, xy + north) - distance_to_coast(statics, xy - north),
        ],
        dim=-1,
    )
    unit: torch.Tensor = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return unit


def has_clearance(statics: Statics, xy: torch.Tensor, min_water_m: float) -> torch.Tensor:
    """Return whether there is enough water under each point.

    The configuration states the fleet's shore limit as a depth, because a keel
    clearance is a physical thing and a setback from the waterline is not. With
    a measured seabed that statement can finally be taken literally: this reads
    the depth where the vehicle is, instead of converting the depth into a
    distance through a shelf model that no longer exists.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.
    min_water_m : float
        Water the vehicle needs under it, metres.

    Returns
    -------
    Tensor
        Boolean, shape ``(...)``.
    """
    return depth_m(statics, xy) >= min_water_m


def push_offshore(
    statics: Statics,
    xy: torch.Tensor,
    berthing: torch.Tensor | None = None,
    came_from: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return positions moved to the nearest water the fleet may occupy.

    A lookup, not a search. For every cell of the box, the INDEX of the
    nearest cell a vehicle is allowed to be in is computed once by a distance
    transform (``scripts/ingest_geodata.py``) and shipped with the seabed, so
    this is one gather and the answer is exact.

    An index rather than a displacement, and that is not a detail: stored as
    metres the widest of them -- 34 km, from deep inside Finland -- did not
    fit the int16 a field of three million values wants to be, wrapped, and
    projected a vehicle at the corner of the box further inland. An index
    cannot overflow, and it carries one fact for free: a cell whose target is
    itself is a cell that already qualifies.

    It replaces a gradient ascent, for two reasons and both were measured.

    **It was most of the step.** Climbing the distance field and then the
    depth field, four bounded steps each with a guard, came to 6.9 ms of an
    8.2 ms fleet step -- 85 % of it -- for a quantity that never changes.

    **It was also wrong.** Walking uphill from inside a shallow bay finds the
    middle of the bay, not the deep water round the headland: driven at the
    coast for a simulated day the fleet settled in 9 m of water against the
    15 m it asks for, because no local rule can see past the next ridge. A
    distance transform has looked at the whole box.

    What a caller gets is the *minimal* move onto legal water, which is what
    a projection should be. Where a vehicle already qualifies, its position
    is returned untouched -- not rounded to the cell centre, which would put
    the whole fleet on a hundred-metre lattice.

    **Two masks, for two different questions.** The field's targets come from
    a navigable mask eroded by one cell, which is what makes the landing exact:
    every corner of the landing point's interpolation stencil then clears the
    limit, so a vehicle cannot arrive at 14.9 m by sitting next to a shallow
    neighbour. But whether a vehicle *needs* moving is asked of the limit
    itself, at its own position.

    Using the eroded mask for both was a quiet disaster. The gulf's bottom is
    rough at a hundred metres, so the eroded set excludes a great deal of
    perfectly usable water: a vehicle in 17 m with one shallow cell nearby
    was shoved out every tick, swam back, and was shoved again. Measured, 18 %
    of all recorded positions were being "corrected", and the vehicles caught
    in it covered three kilometres an hour to move thirty metres, until the
    battery went. Thirty of forty were lost over an episode, against none
    before the measured seabed arrived.

    So: asked per point, answered per cell. A vehicle in legal water is never
    touched, however rough the bottom beside it.

    **A blocked step slides; it does not bounce.** The displacement to the
    nearest legal cell points straight back out of an obstacle, so applying
    it to a vehicle that swam head-on into a shoal returns it where it came
    from -- and the next tick it tries again. Measured, a vehicle returning to
    Helsinki stalled 6.5 km short of its base and held that radius for four
    hundred ticks until the battery went. With ``came_from``, only the inward
    component of the travel is removed and the along-shore component survives,
    so the hull slides around the obstacle the way a hull does. The projection
    is kept as the fallback for the step that lands somewhere the slide cannot
    rescue, which is what keeps the guarantee absolute.

    **The guarantee, exactly.** Afloat, and in water the fleet may use, at
    every point rather than on average. Checked over the whole box in
    `tests/unit/test_world.py`.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.
    berthing : Tensor, optional
        Boolean, shape ``(...)``. Where true, the berthing rule applies --
        distance off the quay rather than water under the keel -- because a
        vehicle alongside is allowed in shallow water and that is what a base
        is for. Defaults to the open-sea rule everywhere.
    came_from : Tensor, optional
        Where this position was a tick ago, shape ``(..., 2)``. Given it, a
        blocked step loses only the part of its travel that went into the
        obstacle and keeps the part that went along it, which is how a hull
        behaves against a shore. Without it the position is simply projected,
        which is right for a placement and wrong for a step.

    Returns
    -------
    Tensor
        Positions, shape ``(..., 2)``.
    """
    columns = statics.depth_m.shape[1]
    row, column = _cell(statics, xy)
    here = row * columns + column

    # Does this vehicle actually break its limit, where it actually is?
    adrift = depth_m(statics, xy) < statics.min_water_m
    target = statics.escape_sea[row, column]
    if berthing is not None:
        alongside = distance_to_coast(statics, xy) < statics.berth_clearance_m
        adrift = torch.where(berthing, alongside, adrift)
        target = torch.where(berthing, statics.escape_berth[row, column], target)

    # The DISPLACEMENT between the two cell centres, added to the position --
    # never the target centre itself.
    #
    # Snapping to the centre was tried and it stopped the fleet dead. A cruise
    # tick covers 72 m and a cell is 100 m, so a step that strays into an
    # illegal cell was answered by returning the vehicle to the centre of the
    # legal cell it had just left: the exact place it started. Net motion
    # zero, every tick, until the battery ran out. Thirty-one of forty
    # vehicles were lost that way over an episode, against none before.
    #
    # Adding the displacement keeps where the vehicle sat INSIDE its cell, and
    # that sub-cell remainder is the whole of its along-shore progress. A
    # vehicle pressed against an island now slides along it, which is what the
    # shore-normal projection this replaced did and what a hull does.
    #
    # The result is still legal, and that is what the eroded mask buys: the
    # landing point is within half a cell of a target whose whole 3x3
    # neighbourhood clears the limit, so every corner of its interpolation
    # stencil clears it too.
    cell = statics.seabed_cell_m
    offset = torch.stack(
        [
            ((target % columns) - (here % columns)).to(xy.dtype) * cell,
            (
                torch.div(target, columns, rounding_mode="floor")
                - torch.div(here, columns, rounding_mode="floor")
            ).to(xy.dtype)
            * cell,
        ],
        dim=-1,
    )
    projected = xy + offset
    if came_from is None:
        return torch.where(adrift.unsqueeze(-1), projected, xy)

    # The escape offset doubles as the outward normal: it is the shortest way
    # out of the obstacle, which is the direction the obstacle pushes back.
    reach = offset.norm(dim=-1, keepdim=True)
    outward = offset / reach.clamp_min(1e-6)
    travel = xy - came_from
    inward = (travel * outward).sum(dim=-1, keepdim=True).clamp(max=0.0)
    slid = xy - outward * inward

    if berthing is None:
        kept = depth_m(statics, slid) >= statics.min_water_m
    else:
        kept = torch.where(
            berthing,
            distance_to_coast(statics, slid) >= statics.berth_clearance_m,
            depth_m(statics, slid) >= statics.min_water_m,
        )
    moved = torch.where(kept.unsqueeze(-1), slid, projected)
    return torch.where(adrift.unsqueeze(-1), moved, xy)


def _cell(statics: Statics, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the row and column of the seabed cell each point falls in."""
    rows, columns = statics.depth_m.shape
    cell = statics.seabed_cell_m
    column = (xy[..., 0] / cell - 0.5).round().to(torch.long).clamp(0, columns - 1)
    row = (xy[..., 1] / cell - 0.5).round().to(torch.long).clamp(0, rows - 1)
    return row, column


def depth_m(statics: Statics, xy: torch.Tensor) -> torch.Tensor:
    """Return water depth, deepening away from either shore.

    Read from the terrain model, not modelled. It feeds transmission loss,
    which only cares about the order of magnitude of the waveguide -- but it
    also decides where the fleet may go, and there a smooth shelf and a real
    one are different places.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    xy : Tensor
        Local metres, shape ``(..., 2)``.

    Returns
    -------
    Tensor
        Depth in metres, shape ``(...)``, floored above zero on land.
    """
    return _seabed(statics, statics.depth_m, xy).clamp_min(_MIN_DEPTH_M)


def current(statics: Statics, xy: torch.Tensor, environment: Environment) -> torch.Tensor:
    """Return the current vector at each point.

    A uniform drift plus one gyre, taken as the curl of a stream function so
    the field is divergence-free -- water that flows in comes back out, which a
    sum of independent sinusoids would not guarantee.

    The vehicle does not know this field. That ignorance is the dominant term
    in its navigation error, and therefore in the false alarms of
    specification section 5.2.

    Parameters
    ----------
    statics : Statics
        Device geometry, for the box extent.
    xy : Tensor
        Local metres, shape ``(..., 2)``.
    environment : Environment
        Mean speed and direction.

    Returns
    -------
    Tensor
        Current vectors in m/s, shape ``(..., 2)``.
    """
    bearing = math.radians(environment.current_direction_deg)
    mean = environment.current_speed_ms * torch.tensor(
        [math.sin(bearing), math.cos(bearing)], dtype=xy.dtype, device=xy.device
    )
    kx = 2.0 * math.pi / statics.extent[0]
    ky = 2.0 * math.pi / statics.extent[1]
    amplitude = 0.5 * environment.current_speed_ms
    phase_x = kx * xy[..., 0]
    phase_y = ky * xy[..., 1]
    # psi = A sin(kx x) sin(ky y)  ->  u = dpsi/dy, v = -dpsi/dx.
    # The kx/ky on the second component is not decoration: without it the two
    # partial derivatives do not cancel, the field acquires sources and sinks,
    # and over a two-day episode vessels pile up in the corners.
    eddy = torch.stack(
        [
            amplitude * torch.sin(phase_x) * torch.cos(phase_y),
            -amplitude * (kx / ky) * torch.cos(phase_x) * torch.sin(phase_y),
        ],
        dim=-1,
    )
    return mean + eddy


def ambient_noise_db(
    statics: Statics,
    listeners: torch.Tensor,
    sources: torch.Tensor,
    source_level_db: torch.Tensor,
    active: torch.Tensor,
    environment: Environment,
    resolved_range_m: float,
) -> torch.Tensor:
    """Return the noise floor at each listener, raised by distant traffic.

    Dense traffic cuts both ways, and this is where the second edge lives: a
    busy lane lifts the floor for anyone listening near it.

    Only vessels **beyond** ``resolved_range_m`` count. Closer ones are targets,
    not background: folding them into the floor would have the detector compute
    a signal-to-noise ratio against a noise term containing its own signal. The
    cutoff is the best-case detection range, so everything the vehicle could
    resolve is excluded and everything it could not becomes floor.

    Parameters
    ----------
    statics : Statics
        Device geometry, for the depth used in spreading.
    listeners : Tensor
        Local metres, shape ``(worlds, listeners, 2)``.
    sources : Tensor
        Local metres, shape ``(worlds, vessels, 2)``.
    source_level_db : Tensor
        Source spectrum level per vessel, shape ``(worlds, vessels)``.
    active : Tensor
        Whether each vessel is in play, shape ``(worlds, vessels)``.
    environment : Environment
        Base ambient level and the loss terms.
    resolved_range_m : float
        Sources nearer than this are targets and do not count as floor.

    Returns
    -------
    Tensor
        Noise spectrum level in dB, shape ``(worlds, listeners)``.
    """
    distance = torch.cdist(listeners, sources).clamp_min(1.0)
    distant = distance > resolved_range_m
    depth = environment.water_depth_m
    linear = environment.absorption_db_per_km + environment.bottom_loss_db_per_km
    loss = (
        20.0 * math.log10(depth)
        + 10.0 * torch.log10((distance / depth).clamp_min(1.0))
        + linear * distance / 1000.0
    )
    received = source_level_db.unsqueeze(1) - loss
    counts = active.unsqueeze(1) & distant
    power = torch.where(counts, torch.pow(10.0, received / 10.0), 0.0).sum(dim=-1)
    base = 10.0 ** (environment.ambient_noise_db / 10.0)
    return 10.0 * torch.log10(power + base)


def sea_area_m2(statics: Statics, samples: int = 400) -> torch.Tensor:
    """Return the water area of the box, by quadrature on a regular grid.

    Called at setup to replace the guessed figure of C-3-03 with a measured
    one. Not part of the step.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    samples : int
        Grid resolution per axis.

    Returns
    -------
    Tensor
        Area in square metres, scalar.
    """
    axis_x = torch.linspace(0.0, float(statics.extent[0]), samples, device=statics.device)
    axis_y = torch.linspace(0.0, float(statics.extent[1]), samples, device=statics.device)
    grid = torch.stack(torch.meshgrid(axis_x, axis_y, indexing="ij"), dim=-1)
    cell = (statics.extent[0] / samples) * (statics.extent[1] / samples)
    return is_sea(statics, grid).sum() * cell
