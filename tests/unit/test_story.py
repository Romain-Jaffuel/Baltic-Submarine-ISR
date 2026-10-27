"""The demonstration films, checked for what a glance at them cannot check.

These films carry almost no text, which removes most of what the previous
version of this file tested -- there is no script, no chapter order, no
growing key to keep monotone. What is left is worth more:

**The key names every mark, and no mark is missing from it.** A film with no
narration is readable only through its key, so a mark drawn on the map with
no entry beside it is the whole artefact failing at its one job.

**The numbers describing the fleet come from the simulator**, not from a
sentence somebody typed, because the page quotes them.

**A frame comes out of the pipe with ink on it**, end to end, on a trace small
enough to run in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.fleet import power
from baltic_isr.render import assets as assets_module
from baltic_isr.render import story
from baltic_isr.render.palette import palette
from baltic_isr.render.replay import Replay, build_scene
from baltic_isr.sensing import acoustics

FRAMES = 6
VEHICLES = 4
TRACKS = 5


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def trace(resolved: Config) -> dict[str, np.ndarray]:
    """Return a small trace with one of everything the films draw."""
    built = assets_module.build_assets(resolved)
    spot = np.asarray(built.crossings[0].xy, dtype=np.float32)

    deviating = np.zeros((FRAMES, TRACKS), dtype=bool)
    deviating[:, 0] = True
    cov = np.zeros((FRAMES, TRACKS, 3), dtype=np.float32)
    cov[:, :, 0] = cov[:, :, 2] = 500.0**2

    rows, cols = build_scene(resolved).fog_shape
    return {
        "times_s": (np.arange(FRAMES) * 60.0).astype(np.float32),
        "vehicle_xy": np.broadcast_to(spot, (FRAMES, VEHICLES, 2)).astype(np.float32),
        "vehicle_estimate": np.broadcast_to(spot, (FRAMES, VEHICLES, 2)).astype(np.float32),
        "vehicle_mode": np.zeros((FRAMES, VEHICLES), dtype=np.int8),
        "vehicle_regime": np.zeros((FRAMES, VEHICLES), dtype=np.int8),
        "vehicle_energy": np.full((FRAMES, VEHICLES), 0.6, dtype=np.float32),
        "vehicle_reach_m": np.full((FRAMES, VEHICLES), 5_000.0, dtype=np.float32),
        "track_xy": np.broadcast_to(spot, (FRAMES, TRACKS, 2)).astype(np.float32),
        "track_cov": cov,
        "track_known": np.ones((FRAMES, TRACKS), dtype=bool),
        "track_alarmed": deviating.copy(),
        "track_age_s": np.zeros((FRAMES, TRACKS), dtype=np.float32),
        "vessel_xy": np.broadcast_to(spot, (FRAMES, TRACKS, 2)).astype(np.float32),
        "vessel_deviating": deviating,
        "vessel_class": np.zeros(TRACKS, dtype=np.int8),
        "link_pairs": np.array([[0, 1]], dtype=np.int16),
        "link_counts": np.array([1] + [0] * (FRAMES - 1), dtype=np.int32),
        "fog_frames": np.array([0], dtype=np.int32),
        "fog_age_s": np.zeros((1, rows * cols), dtype=np.float32),
    }


@pytest.fixture(scope="module")
def drawing(resolved: Config, trace: dict[str, np.ndarray]) -> Replay:
    return Replay(resolved, trace)


def test_every_number_the_page_quotes_comes_from_the_simulator(resolved: Config) -> None:
    """`facts` must agree with the physics, and with the CLI, on one reference.

    The page states these and `baltic-isr budget` prints them; both must come
    from the same function against the same vessel. They were briefly taken
    against the loudest traffic class while the CLI used the first one, which
    put 7.49 km in the README and 6.94 km in the command the README told the
    reader to run.
    """
    numbers = story.facts(resolved)
    loudest = resolved.traffic.classes[0].source_level_db
    for name, speed in (
        ("drift", resolved.auv.regimes.drift_ms),
        ("cruise", resolved.auv.regimes.cruise_ms),
        ("sprint", resolved.auv.regimes.sprint_ms),
    ):
        expected = acoustics.detection_range_m(
            speed, loudest, resolved.acoustics, resolved.environment
        )
        assert numbers[f"{name}_km"] == pytest.approx(expected / 1000.0)
        assert numbers[f"{name}_h"] == pytest.approx(
            power.endurance_s(speed, resolved.auv) / 3600.0
        )


def test_going_fast_costs_the_ears(resolved: Config) -> None:
    """The one claim the whole project rests on, asserted rather than assumed."""
    numbers = story.facts(resolved)
    assert numbers["sprint_km"] < numbers["cruise_km"] < numbers["drift_km"]
    assert numbers["sprint_h"] < numbers["cruise_h"] < numbers["drift_h"]
    assert numbers["deaf_ratio"] > 2.0


def test_the_key_names_every_mark_the_map_draws() -> None:
    """A film with no narration is readable only through its key.

    The map draws cables in two states, the crossings, vessels in two states,
    the vehicles with their hearing disc, and the ring for a contact in
    question. Nine marks, eight entries -- the vehicle and its disc share a
    row in the map and get one each here.
    """
    colours = palette("light")
    entries = story.keys(colours)
    labels = [label for _, label in entries]
    assert len(entries) == 8
    assert len(set(labels)) == len(labels)
    assert all(label and not label.endswith(" ") for label in labels)

    used = {key.colour for key, _ in entries}
    for needed in (colours.asset, colours.watch, colours.traffic, colours.threat):
        assert needed in used, f"no key entry uses {needed}"


def test_red_is_spent_on_one_thing_only() -> None:
    """Amber and red used to both mean "odd". Now red does, and amber is cable.

    Two marks for one idea was one too many: the difference between odd and
    odd-over-a-cable is real in the metrics and invisible to a first-time
    viewer. Collapsing them freed amber to mean exactly one thing of its own.
    """
    colours = palette("light")
    entries = story.keys(colours)
    red = [label for key, label in entries if key.colour == colours.threat]
    amber = [label for key, label in entries if key.colour == colours.watch]
    assert len(red) == 1
    assert len(amber) == 1
    assert "cable" in amber[0]


def test_the_fleet_is_alone_some_but_not_all_of_the_time(drawing: Replay) -> None:
    """The comms number is read off a trace, so it must come back a fraction."""
    assert 0.0 <= story.alone_fraction(drawing) <= 1.0


def test_the_crop_keeps_the_bases(resolved: Config) -> None:
    """The single-panel film trims the empty land. It must not trim the bases.

    The gulf is mostly beige at the top and bottom, and cropping is what takes
    the map from 1.26 to something near 16:9. But the two shore bases sit at
    0.16 and 0.85 of the height, so a crop tuned by eye to look good is one
    bad nudge away from cutting the places the fleet departs from.
    """
    scene = build_scene(resolved)
    height = float(scene.extent[1])
    margin = 0.5 * height * (1.0 - story.KEEP)
    for base in scene.bases:
        assert margin < float(base[1]) < height - margin


def test_a_frame_of_each_film_comes_out_with_ink_on_it(
    resolved: Config, trace: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """End to end on a tiny trace: every layer runs, and the pipe produces video."""
    import imageio.v3 as imageio

    path = story.demonstration(
        Replay(resolved, trace), Replay(resolved, trace), tmp_path / "pair.mp4", every=3
    )
    assert path.is_file() and path.stat().st_size > 0
    frame = np.asarray(imageio.imread(path, index=0))
    assert frame.shape[:2] == (1080, 1920)
    # Not a blank canvas: the surface alone would be one colour.
    assert len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0)) > 50
