"""The infrastructure layer: what the fleet is there to protect.

These are geometry tests, not rendering ones. They matter because the whole
colour scheme now rests on one predicate -- is this vessel over a cable? -- and
on five computed points. If either is wrong the map states something false with
complete confidence, which is worse than stating nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.render import assets as assets_module
from baltic_isr.render.assets import Assets, build_assets


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def assets(resolved: Config) -> Assets:
    return build_assets(resolved)


def test_every_cable_is_projected(assets: Assets, resolved: Config) -> None:
    assert len(assets.cables) == len(resolved.corridors)
    for _, line, half_width in assets.cables:
        assert line.shape[1] == 2
        assert half_width == pytest.approx(resolved.corridors[0].half_width_m)
        assert np.linalg.norm(np.diff(line, axis=0), axis=1).sum() > 50_000.0


def test_the_chokepoints_are_where_a_cable_meets_a_lane(assets: Assets) -> None:
    """Four of them, and each really is on both lines.

    This is the claim the map makes loudest -- a ring saying "here is where an
    anchor finds a cable" -- so it is checked against the geometry rather than
    trusted.

    Four, not five. The fifth was the ferry lane against c_lion_1, and it only
    ever existed because the caricature put the Finnish coast far enough south
    for the lane to begin north of that cable's landfall. Routed on the
    measured seabed the crossing starts seaward of it -- and the cables run
    north-south while the ferry does too, so the two are near parallel and do
    not meet. Every chokepoint left is where the east-west lane cuts a cable,
    which is the geometry this gulf actually has.
    """
    assert len(assets.crossings) == 4
    names = {(item.cable, item.lane) for item in assets.crossings}
    assert ("balticconnector", "east_west") in names
    assert ("c_lion_1", "east_west") in names
    assert {item.lane for item in assets.crossings} == {"east_west"}

    for crossing in assets.crossings:
        cable = next(line for name, line, _ in assets.cables if name == crossing.cable)
        on_cable = assets_module._distance_to_line(crossing.xy.reshape(1, 2), cable)[0]
        assert float(on_cable) < 1.0


def test_the_crossings_are_distinct_places(assets: Assets) -> None:
    points = np.array([item.xy for item in assets.crossings])
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    assert float(gaps.min()) > 3_000.0


def test_over_a_cable_agrees_with_the_corridor_width(assets: Assets) -> None:
    """The predicate that decides whether a vessel is drawn red.

    Dead on the cable is inside; a corridor-width away plus a metre is not.
    Nothing subtler is needed, and nothing less would do: this is the line
    between "odd" and "the thing we are here to stop".
    """
    # A cable may be a single segment, so take the direction of the first one
    # and a point halfway along it rather than indexing around a vertex.
    _, line, half_width = assets.cables[0]
    run = line[1] - line[0]
    middle = line[0] + run * 0.5
    normal = np.array([-run[1], run[0]]) / np.linalg.norm(run)

    probes = np.stack(
        [
            middle,
            middle + normal * (half_width * 0.5),
            middle + normal * (half_width + 500.0),
            middle + normal * (half_width * 6.0),
        ]
    )
    assert list(assets_module.over_a_cable(assets, probes)) == [True, True, False, False]


def test_nothing_in_the_middle_of_the_gulf_counts_as_over_a_cable(
    assets: Assets, resolved: Config
) -> None:
    """Only a minority of the gulf is over a cable.

    A sanity floor: if this ever passes for most of the box, the corridor
    width has been broken and every vessel becomes a target.
    """
    rng = np.random.default_rng(0)
    box = np.array([137_900.0, 109_800.0])
    scattered = rng.random((4_000, 2)) * box
    share = float(assets_module.over_a_cable(assets, scattered).mean())
    assert 0.02 < share < 0.30
    del resolved


def test_landfalls_are_the_ends_of_the_cables(assets: Assets) -> None:
    assert assets.landfalls.shape == (2 * len(assets.cables), 2)
    for _, line, _ in assets.cables:
        assert any(np.allclose(line[0], end) for end in assets.landfalls)
        assert any(np.allclose(line[-1], end) for end in assets.landfalls)
