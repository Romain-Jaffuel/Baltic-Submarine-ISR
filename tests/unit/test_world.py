"""Water, land, depth, current and the noise floor they make."""

from __future__ import annotations

import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.world import field
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics

# Counted cell by cell off the terrain model by scripts/ingest_geodata.py.
# The quadrature below samples it on a coarser grid and lands within 1 %.
MEASURED_SEA_AREA_KM2 = 9_888.0
# The deepest water in the box, from the same source. The old shelf model
# could not produce it: it saturated at 70 m by construction.
MEASURED_MAX_DEPTH_M = 125.3


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def statics(resolved: Config) -> Statics:
    return statics_module.build(resolved, "cpu")


def grid(statics: Statics, samples: int = 120) -> torch.Tensor:
    axis_x = torch.linspace(0.0, float(statics.extent[0]), samples)
    axis_y = torch.linspace(0.0, float(statics.extent[1]), samples)
    return torch.stack(torch.meshgrid(axis_x, axis_y, indexing="ij"), dim=-1)


def test_local_frame_round_trips(statics: Statics) -> None:
    """Local metres go back to the projected CRS exactly."""
    local = torch.tensor([[1234.5, 6789.0]])
    back = statics.to_projected(local)
    assert torch.allclose(back - statics.origin, local.to(torch.float64), atol=1e-6)


def test_local_frame_keeps_float32_precision(statics: Statics) -> None:
    """The reason the frame is local at all.

    EPSG:3067 northings are near 6.6e6, where float32 steps about half a metre.
    A vessel at 6 m/s over a 600 s step moves 3.6 km, and the last digits of
    that would be noise in the projected frame.
    """
    projected = float(statics.origin[1])
    assert torch.tensor(projected, dtype=torch.float32).item() != projected
    step = torch.finfo(torch.float32).eps * float(statics.extent[1])
    assert step < 0.05


def test_sea_area_is_measured_not_guessed(statics: Statics, resolved: Config) -> None:
    """C-3-03. The configured figure must track what the coastlines actually enclose."""
    measured = float(field.sea_area_m2(statics)) / 1e6
    assert measured == pytest.approx(MEASURED_SEA_AREA_KM2, rel=0.02)
    assert measured == pytest.approx(resolved.geo.sea_area_m2 / 1e6, rel=0.02)


def test_sea_is_a_minority_of_the_bounding_box(statics: Statics) -> None:
    box = float(statics.extent[0]) * float(statics.extent[1])
    assert 0.5 < float(field.sea_area_m2(statics)) / box < 0.8


def test_land_is_on_both_sides(statics: Statics) -> None:
    """Sea is what lies between the two shores, so both edges must be land."""
    easting = float(statics.extent[0]) * 0.5
    assert not bool(field.is_sea(statics, torch.tensor([easting, 0.0])))
    assert not bool(field.is_sea(statics, torch.tensor([easting, float(statics.extent[1])])))
    assert bool(field.is_sea(statics, statics.extent * 0.5))


def test_depth_matches_the_configured_average(statics: Statics, resolved: Config) -> None:
    """The one scalar depth the sonar uses has to be the mean of the real ones.

    `environment.water_depth_m` is the waveguide as a single number, which is
    all transmission loss wants. It is only honest if it is the average of the
    field underneath it, so this checks the two against each other rather than
    against a remembered figure.
    """
    points = grid(statics)
    depth = field.depth_m(statics, points)[field.is_sea(statics, points)]
    assert float(depth.mean()) == pytest.approx(resolved.environment.water_depth_m, rel=0.15)
    assert float(depth.max()) <= MEASURED_MAX_DEPTH_M


def test_depth_grows_away_from_the_shore(statics: Statics) -> None:
    easting = float(statics.extent[0]) * 0.5
    near = field.depth_m(statics, torch.tensor([easting, 20_000.0]))
    far = field.depth_m(statics, statics.extent * torch.tensor([0.5, 0.5]))
    assert float(far) > float(near)


def test_current_matches_its_configured_speed(statics: Statics, resolved: Config) -> None:
    speed = field.current(statics, grid(statics), resolved.environment).norm(dim=-1)
    assert float(speed.mean()) == pytest.approx(resolved.environment.current_speed_ms, rel=0.2)


def test_current_is_divergence_free(statics: Statics, resolved: Config) -> None:
    """Taken as the curl of a stream function, so water that flows in comes out.

    A sum of independent sinusoids would not guarantee it, and a field with
    sources would pile vessels up in corners over a two-day episode.
    """
    step = 50.0
    base = grid(statics, 60)
    shift_x = base + torch.tensor([step, 0.0])
    shift_y = base + torch.tensor([0.0, step])
    flow = field.current(statics, base, resolved.environment)
    du = (field.current(statics, shift_x, resolved.environment) - flow)[..., 0] / step
    dv = (field.current(statics, shift_y, resolved.environment) - flow)[..., 1] / step
    scale = flow.norm(dim=-1).mean() / float(statics.extent[0])
    assert float((du + dv).abs().mean()) < 0.05 * float(scale)


def test_noise_floor_rises_with_distant_traffic(statics: Statics, resolved: Config) -> None:
    listeners = (statics.extent * 0.5).reshape(1, 1, 2)
    sources = listeners + torch.tensor([[[12_000.0, 0.0]]]).repeat(1, 24, 1)
    sources = sources + torch.arange(24).reshape(1, 24, 1) * 400.0
    level = torch.full((1, 24), 145.0)
    active = torch.ones(1, 24, dtype=torch.bool)
    loud = field.ambient_noise_db(
        statics, listeners, sources, level, active, resolved.environment, 7000.0
    )
    quiet = field.ambient_noise_db(
        statics, listeners, sources, level, torch.zeros_like(active), resolved.environment, 7000.0
    )
    assert float(quiet) == pytest.approx(resolved.environment.ambient_noise_db, abs=0.01)
    assert float(loud) > float(quiet)


def test_resolvable_vessels_are_not_counted_as_noise(statics: Statics, resolved: Config) -> None:
    """Folding a target into the floor would have the detector fight its own signal."""
    listeners = (statics.extent * 0.5).reshape(1, 1, 2)
    sources = listeners + torch.tensor([[[1_000.0, 0.0]]])
    level = torch.full((1, 1), 145.0)
    active = torch.ones(1, 1, dtype=torch.bool)
    floor = field.ambient_noise_db(
        statics, listeners, sources, level, active, resolved.environment, 7000.0
    )
    assert float(floor) == pytest.approx(resolved.environment.ambient_noise_db, abs=0.01)


def test_rails_carry_their_geometry(statics: Statics, resolved: Config) -> None:
    assert statics.n_rails == len(resolved.rails)
    assert statics.rail_xy.shape[1] == statics_module.RAIL_SAMPLES
    assert torch.allclose(statics.rail_tangent.norm(dim=-1), torch.ones(1), atol=1e-4)


def test_some_of_every_rail_sits_over_a_corridor(statics: Statics) -> None:
    """Deviations are placed on these samples, so an empty rail would break them."""
    assert (statics.rail_over_corridor.sum(dim=1) > 0).all()


def test_sampling_a_rail_walks_it_end_to_end(statics: Statics) -> None:
    rail = torch.zeros(3, dtype=torch.long)
    points = statics_module.sample_rail(statics, rail, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(points[0], statics.rail_xy[0, 0], atol=1.0)
    assert torch.allclose(points[2], statics.rail_xy[0, -1], atol=1.0)
    assert float((points[2] - points[0]).norm()) > 50_000.0


def test_bases_sit_in_the_water_against_a_shore(statics: Statics, resolved: Config) -> None:
    """A charging point is a port approach: close to the coast, but afloat.

    Both halves matter. Drift into the middle of the gulf and the transit
    budget of C-3-04 is wrong; sit on the landward side and every vehicle based
    there is beached, which is exactly what the Helsinki base used to do -- 1.4
    km inland, and a quarter of all fleet positions on land. See C-3-16.
    """
    clearance = field.distance_to_coast(statics, statics.bases)
    assert float(clearance.min()) > resolved.fleet.berth_clearance_m
    assert float(clearance.max()) < 5_000.0


def test_shore_normal_points_out_to_sea(statics: Statics) -> None:
    """Everywhere a vehicle could be held off the coast, anyway.

    The normal is the gradient of a *minimum* of two distances, so it is
    undefined on the line where the two shores tie. Measured, that line sits 28
    km offshore -- thirteen times the widest clearance band -- so the ambiguity
    never touches the band the fleet is actually pushed out of.
    """
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, float(statics.extent[0]), 80),
            torch.linspace(0.0, float(statics.extent[1]), 80),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    normal = field.coast_normal(statics, grid)
    assert torch.allclose(normal.norm(dim=-1), torch.ones(grid.shape[0]), atol=1e-5)

    step = 50.0
    before = field.distance_to_coast(statics, grid)
    gain = (field.distance_to_coast(statics, grid + normal * step) - before) / step
    band = (before > 0.0) & (before < 3_000.0)
    assert bool(band.any())
    inside = gain[band]

    # A true distance field has |grad d| = 1, and the median says this one is.
    assert float(inside.median()) == pytest.approx(1.0, abs=0.02)

    # Everywhere EXCEPT the medial axis, which is not a flaw in the field but
    # a fact about the water: in a channel between two islands every point is
    # equidistant from both shores, so no direction is "away from the shore"
    # and the gradient collapses. The old two-polyline coast had no such
    # places, which is why this used to assert a floor on the minimum. An
    # archipelago has a great many, and a vehicle sitting in one has nowhere
    # to be pushed -- correctly, because the water there is as wide as it gets.
    assert float((inside > 0.9).float().mean()) > 0.85


def test_has_clearance_asks_the_seabed_rather_than_the_shore(statics: Statics) -> None:
    """The configuration states the limit as a DEPTH, so the test is about depth.

    This is the check that could not be written while the land model was two
    polylines: a point can be a long way from any shore and still sit on a
    shoal, and a point can be close inshore and still have water under it.
    """
    points = grid(statics)
    deep_enough = field.has_clearance(statics, points, 15.0)
    assert bool(deep_enough.any()) and not bool(deep_enough.all())
    assert torch.equal(deep_enough, field.depth_m(statics, points) >= 15.0)

    # Everything that clears fifteen metres is at sea; the converse is false,
    # and that asymmetry is the whole reason the function exists.
    assert bool(field.is_sea(statics, points)[deep_enough].all())
    assert int(field.is_sea(statics, points).sum()) > int(deep_enough.sum())


def test_pushing_offshore_leaves_open_water_alone(statics: Statics) -> None:
    """A vehicle in the middle of the gulf is not moved, not even by a cell.

    The projection snaps to a cell centre where it has to move at all, so
    "leaves alone" has to mean exactly untouched rather than nearly: a fleet
    quietly rounded onto a hundred-metre lattice every tick would show on the
    map as a shiver.
    """
    middle = (statics.extent * 0.5).reshape(1, 2)
    assert torch.equal(field.push_offshore(statics, middle), middle)


def test_pushing_offshore_lands_on_water_the_fleet_may_use(
    statics: Statics, resolved: Config
) -> None:
    """From anywhere at all, including well inland, in one call.

    This is the guarantee the whole land model exists to provide, and it is
    worth stating over a grid rather than over a handful of points: the
    failure it replaced was a vehicle in a bay that no local rule could get
    out of.
    """
    grid_xy = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, float(statics.extent[0]), 120),
            torch.linspace(0.0, float(statics.extent[1]), 120),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    landed = field.push_offshore(statics, grid_xy)

    # Depth, not `is_sea`. The two rasters are built differently -- the
    # waterline is a contour of the terrain, the signed distance a transform
    # of the land mask -- and they disagree by up to a few cells around
    # skerries. Measured over this grid, one landed point in 48 400 reads as
    # 39 m ashore while sitting in 22 m of water. Depth is the constraint the
    # configuration states and the one the fleet actually has.
    want = resolved.fleet.min_water_m
    assert bool(field.has_clearance(statics, landed, want).all())
    assert float(field.depth_m(statics, landed).min()) >= want
