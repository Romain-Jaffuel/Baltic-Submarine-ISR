"""Projection and the fixed geometry of the box."""

import numpy as np
import pytest

from baltic_isr import config, geo
from baltic_isr.config import Config


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def geometry(resolved: Config) -> geo.Geometry:
    return geo.build(resolved)


def test_projection_round_trips(geometry: geo.Geometry) -> None:
    lon, lat = 24.95, 60.15
    back = geometry.frame.to_lonlat(geometry.frame.to_xy(lon, lat))
    assert back[0] == pytest.approx(lon, abs=1e-6)
    assert back[1] == pytest.approx(lat, abs=1e-6)


def test_box_has_the_expected_size(geometry: geo.Geometry) -> None:
    width, height = geometry.frame.extent_m
    assert width == pytest.approx(137_900, rel=0.01)
    assert height == pytest.approx(109_800, rel=0.01)


def test_helsinki_tallinn_is_eighty_kilometres(geometry: geo.Geometry) -> None:
    """A sanity check on the projection, not on where the fleet moors.

    It used to measure between the two bases, which made it fail the moment
    the Helsinki mooring moved offshore -- a change to the scenario, not to the
    geodesy. The two real city centres are the fixed points worth pinning.
    """
    helsinki = geometry.frame.to_xy(24.94, 60.17)
    tallinn = geometry.frame.to_xy(24.75, 59.44)
    separation = float(np.hypot(*(helsinki - tallinn)))
    assert separation == pytest.approx(81_600, rel=0.01)


def test_sea_area_is_smaller_than_the_bounding_box(
    resolved: Config, geometry: geo.Geometry
) -> None:
    """C-3-03: about a third of the rectangle is land."""
    assert resolved.geo.sea_area_m2 < geometry.frame.area_m2


def test_every_rail_and_corridor_is_projected(resolved: Config, geometry: geo.Geometry) -> None:
    assert len(geometry.rails) == len(resolved.rails)
    assert len(geometry.corridors) == len(resolved.corridors)
    for line in geometry.rails + geometry.corridors:
        assert line.length_m > 10_000.0


def test_distance_to_a_polyline_is_zero_on_it(geometry: geo.Geometry) -> None:
    rail = geometry.rail("east_west")
    on_line = rail.point_at(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(rail.distance_to(on_line), 0.0, atol=1.0)


def test_point_at_walks_the_whole_line(geometry: geo.Geometry) -> None:
    rail = geometry.rail("east_west")
    start, end = rail.point_at(0.0), rail.point_at(1.0)
    assert np.allclose(start, rail.xy[0], atol=1.0)
    assert np.allclose(end, rail.xy[-1], atol=1.0)


def test_corridor_membership(geometry: geo.Geometry) -> None:
    corridor = geometry.corridors[0]
    inside = corridor.point_at(0.5).reshape(1, 2)
    outside = inside + np.array([[corridor.half_width_m * 3.0, 0.0]])
    assert corridor.contains(inside)[0]
    assert not corridor.contains(outside)[0]


def test_distance_to_base_picks_the_nearest(geometry: geo.Geometry) -> None:
    near_helsinki = geometry.bases[0] + np.array([1000.0, 0.0])
    distance = geometry.distance_to_base(near_helsinki.reshape(1, 2))[0]
    assert distance == pytest.approx(1000.0, abs=1.0)


def test_base_to_area_centre(resolved: Config, geometry: geo.Geometry) -> None:
    """C-3-04. The whole transit budget hangs on this number.

    39.9 km until the Helsinki mooring moved 2.9 km offshore to get out of the
    land; the transit is correspondingly shorter now. See C-3-16.
    """
    centre = geometry.frame.to_xy(
        0.5 * (resolved.geo.lon_min + resolved.geo.lon_max),
        0.5 * (resolved.geo.lat_min + resolved.geo.lat_max),
    ).reshape(1, 2)
    assert geometry.distance_to_base(centre)[0] == pytest.approx(39_000, rel=0.02)


def test_unknown_rail_raises(geometry: geo.Geometry) -> None:
    with pytest.raises(KeyError):
        geometry.rail("no_such_rail")


def test_far_base_scenario_moves_the_bases() -> None:
    near = geo.build(config.load("baseline"))
    far = geo.build(config.load("far_base"))
    centre = near.frame.to_xy(25.0, 59.775).reshape(1, 2)
    assert far.distance_to_base(centre)[0] > 2.0 * near.distance_to_base(centre)[0]
