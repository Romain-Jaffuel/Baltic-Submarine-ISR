"""The map, checked for the things a glance at it cannot check.

Rendering is the one part of this project whose acceptance criterion is "it
reads at a glance", and no test replaces looking. What a test *can* hold is the
scaffolding underneath: that the scene is built in the same local frame as the
simulation, that the fog index does not run ahead of the clock, that an expired
prediction is not drawn as a forty-kilometre circle, and that a frame comes out
of the pipe with ink on it.

So these are mostly invariants, plus one end-to-end render of a tiny synthetic
trace -- small enough to run in CI, real enough to catch an exception in any
layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.render import assets as assets_module
from baltic_isr.render import replay as replay_module
from baltic_isr.render.chart import Chart
from baltic_isr.render.palette import palette
from baltic_isr.render.replay import Replay, build_scene
from baltic_isr.world import field
from baltic_isr.world import statics as statics_module

FRAMES = 6
VEHICLES = 4
TRACKS = 5


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def trace(resolved: Config) -> dict[str, np.ndarray]:
    """Return a small synthetic trace with one of everything the map can draw."""
    scene = build_scene(resolved)
    width, height = float(scene.extent[0]), float(scene.extent[1])
    middle = np.array([width * 0.5, height * 0.5], dtype=np.float32)

    rng = np.random.default_rng(0)
    spread = rng.uniform(-8_000.0, 8_000.0, size=(FRAMES, VEHICLES, 2)).astype(np.float32)
    vehicle_xy = (middle + spread).astype(np.float32)

    # One live track, one so stale its ellipse would swamp the map, one unknown.
    cov = np.zeros((FRAMES, TRACKS, 3), dtype=np.float32)
    cov[:, :, 0] = cov[:, :, 2] = 500.0**2
    cov[:, 1, 0] = cov[:, 1, 2] = 40_000.0**2
    known = np.ones((FRAMES, TRACKS), dtype=bool)
    known[:, 2] = False
    alarmed = np.zeros((FRAMES, TRACKS), dtype=bool)
    alarmed[:, 0] = True
    alarmed[:, 1] = True

    fog_rows, fog_cols = scene.fog_shape
    return {
        "times_s": (np.arange(FRAMES) * 60.0).astype(np.float32),
        "vehicle_xy": vehicle_xy,
        "vehicle_estimate": vehicle_xy,
        "vehicle_mode": np.zeros((FRAMES, VEHICLES), dtype=np.int8),
        "vehicle_regime": (np.arange(FRAMES * VEHICLES) % 3)
        .reshape(FRAMES, VEHICLES)
        .astype(np.int8),
        "vehicle_energy": np.full((FRAMES, VEHICLES), 0.6, dtype=np.float32),
        "vehicle_reach_m": np.full((FRAMES, VEHICLES), 5_000.0, dtype=np.float32),
        "track_xy": np.broadcast_to(middle, (FRAMES, TRACKS, 2)).astype(np.float32),
        "track_cov": cov,
        "track_known": known,
        "track_alarmed": alarmed,
        "track_age_s": np.zeros((FRAMES, TRACKS), dtype=np.float32),
        "vessel_xy": np.broadcast_to(middle, (FRAMES, TRACKS, 2)).astype(np.float32),
        "vessel_deviating": alarmed.copy(),
        "vessel_class": np.zeros(TRACKS, dtype=np.int8),
        "link_pairs": np.array([[0, 1], [1, 2]], dtype=np.int16),
        "link_counts": np.array([2, 0, 0, 0, 0, 0], dtype=np.int32),
        "fog_frames": np.array([0, 3], dtype=np.int32),
        "fog_age_s": np.zeros((2, fog_rows * fog_cols), dtype=np.float32),
    }


def test_the_scene_is_in_the_same_local_frame_as_the_simulation(resolved: Config) -> None:
    """The map and the physics must share an origin.

    Both drop the projection origin. If they ever disagree the whole map is
    offset by six million metres and nothing else notices.
    """
    scene = build_scene(resolved)
    fixed = statics_module.build(resolved, "cpu")
    assert np.allclose(scene.extent, fixed.extent.numpy(), atol=1.0)
    assert np.allclose(scene.bases, fixed.bases.numpy(), atol=1.0)
    # The coastline is no longer projected twice -- there is one of it, in
    # the seabed product, and the chart and the simulation both read it. What
    # is left to check is that the chart's grid is laid on the same origin,
    # which is the error that used to be possible and invisible.
    chart = Chart.build(resolved, palette("light"))
    assert np.allclose(chart.origin, fixed.origin.numpy(), atol=1.0)
    assert np.allclose(chart.extent, scene.extent, atol=150.0)


def test_the_coastline_is_the_one_the_simulation_uses(resolved: Config) -> None:
    """Drawn land and simulated land are the same land.

    They cannot drift, because there is only one of them -- but "cannot" is
    worth a test, since the failure would be a fleet swimming through islands
    the viewer can see and the physics cannot. Every vertex of the drawn
    coastline should sit on the simulation's own waterline.
    """
    chart = Chart.build(resolved, palette("light"))
    fixed = statics_module.build(resolved, "cpu")

    vertices = np.concatenate([ring[::7] for ring in chart.coast])
    assert len(vertices) > 1_000
    distance = np.abs(field.distance_to_coast(fixed, torch.from_numpy(vertices).float()).numpy())

    # Checked as a distribution, because the two are built differently and
    # agreeing to the metre everywhere is not a thing they can do. The drawn
    # line is the zero crossing of the terrain, interpolated between cells;
    # the simulated one is a distance transform of the land mask, which
    # measures to the nearest land cell CENTRE. Where the coast is a smooth
    # shore they agree to a fraction of a cell. Where it is a scatter of
    # single-cell skerries the centre of the nearest one can be a couple of
    # cells from the waterline that runs round it, and that is the tail.
    cell = fixed.seabed_cell_m
    assert float(np.median(distance)) < 0.6 * cell
    assert float(np.percentile(distance, 99)) < 2.0 * cell
    assert float(distance.max()) < 5.0 * cell


def test_the_navigable_edge_is_the_one_the_physics_enforces(resolved: Config) -> None:
    """The drawn channel wall is the limit the fleet actually obeys.

    Otherwise the fleet turns away from empty water and the viewer sees no
    reason for it. It used to be drawn as a constant setback from the shore,
    which was the best a shelf model could do; now it is the fifteen-metre
    contour, and every point of it should measure fifteen metres.
    """
    chart = Chart.build(resolved, palette("light"))
    fixed = statics_module.build(resolved, "cpu")
    assert chart.navigable

    vertices = np.concatenate([line[::5] for line in chart.navigable])
    depth = field.depth_m(fixed, torch.from_numpy(vertices).float()).numpy()

    # Generously, because the drawn line is contoured from a smoothed copy --
    # a chart generalises -- while the depth here is read from the measured
    # raster. The tolerance is the price of that generalisation, and it is
    # well inside the margin the fleet keeps.
    assert float(np.median(depth)) == pytest.approx(resolved.fleet.min_water_m, abs=2.0)
    assert float(np.abs(depth - resolved.fleet.min_water_m).mean()) < 4.0


def test_the_fog_never_runs_ahead_of_the_clock(
    resolved: Config, trace: dict[str, np.ndarray]
) -> None:
    """A frame shows the freshest picture at or before it, never a later one.

    The search picture is recorded every perception tick and the map is drawn
    every physics tick, so the two cadences have to be reconciled somewhere.
    """
    drawing = Replay(resolved, trace)
    for frame in range(FRAMES):
        row = max((v for k, v in drawing._fog_index.items() if k <= frame), default=0)
        assert row == (1 if frame >= 3 else 0)
        assert drawing.fog_at(frame).shape == drawing._scene.fog_shape


def test_links_are_sliced_per_frame(resolved: Config, trace: dict[str, np.ndarray]) -> None:
    drawing = Replay(resolved, trace)
    assert len(drawing.links_at(0)) == 2
    assert len(drawing.links_at(1)) == 0


def test_an_expired_prediction_is_not_drawn(resolved: Config, trace: dict[str, np.ndarray]) -> None:
    """An expired prediction is not drawn.

    At its true size a stale track is a 40 km circle swamping the map, and it
    is also a lie: an expired prediction is not a belief, it is the absence of
    one.
    """
    major = replay_module._major_axis(trace["track_cov"][0])
    live = trace["track_known"][0] & (major < resolved.fleet.gate_m)
    assert bool(live[0])
    assert not bool(live[1])  # the 40 km one
    assert not bool(live[2])  # the unknown one


def test_a_belief_and_a_vessel_are_not_drawn_the_same_way(
    resolved: Config, trace: dict[str, np.ndarray]
) -> None:
    """What the fleet suspects must not look like what is actually there.

    Both were red outlines once, and the first viewer of the map could not tell
    them apart -- which collapses the only question the project asks. A belief
    is now a wash, a vessel is an opaque marker.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    drawing = Replay(resolved, trace)
    cov = trace["track_cov"][0][:1]
    centres = trace["track_xy"][0][:1]

    drawing._ellipses(axes, centres, cov, "#d03b3b", 1.2, 0.9, 10)
    assert len(axes.collections) == 1  # an outline alone
    drawing._ellipses(axes, centres, cov, "#d03b3b", 1.2, 0.9, 10, fill=0.18)
    assert len(axes.collections) == 3  # plus a wash and its edge
    washed = axes.collections[1]
    assert np.asarray(washed.get_facecolor()).shape[0] > 0
    plt.close(figure)


def test_the_caption_says_what_is_true_as_well_as_what_is_believed(
    resolved: Config, trace: dict[str, np.ndarray]
) -> None:
    """The caption carries the truth beside the belief.

    Thirteen alarms shown next to two real deviations is the project's
    question, and it reads without knowing anything about the model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    Replay(resolved, trace)._caption(axes, 0)

    # The headline is the claim, in words, and it leads with the asset.
    assert "cable" in axes.get_title(loc="left")

    # A grid of runs names its panel on a line of its own. On one line the
    # two were clipped, and what got cut was the end of the sentence -- the
    # part saying whether the fleet saw anything.
    Replay(resolved, trace)._caption(axes, 0, prefix="lane-by-lane sweep")
    lines = axes.get_title(loc="left").splitlines()
    assert lines[0] == "lane-by-lane sweep"
    assert "cable" in lines[1]
    # The counts underneath are its evidence.
    written = " ".join(item.get_text() for item in axes.texts)
    assert "chokepoints" in written
    assert "open questions" in written
    plt.close(figure)


def test_the_verdict_leads_with_the_cable() -> None:
    """Every branch of the sentence, including the ones a good policy reaches.

    The wording is the only part of the map a viewer reads literally, so a
    plural in the wrong place is a real defect, not a typo. And the subject of
    every branch is the cable: an earlier version led with deviations, which
    made the detector the subject of the film instead of the thing being
    protected.
    """
    assert replay_module._verdict(0, 0, 0) == "No vessel over a cable."
    assert "1 more behaving oddly" in replay_module._verdict(0, 0, 1)
    assert "2 more behaving oddly" in replay_module._verdict(0, 0, 2)
    assert "1 vessel sitting over a cable" in replay_module._verdict(1, 0, 0)
    assert "2 vessels sitting over a cable" in replay_module._verdict(2, 0, 0)
    assert "cannot see it" in replay_module._verdict(1, 0, 0)
    assert "cannot see them" in replay_module._verdict(2, 0, 0)
    assert "the fleet is tracking 1" in replay_module._verdict(2, 1, 0)
    assert "the fleet has it" in replay_module._verdict(1, 1, 0)
    assert "the fleet has them" in replay_module._verdict(3, 3, 0)
    for branch in (0, 1, 2):
        assert replay_module._verdict(branch, 0, 0).count("cable") == 1


def test_being_over_a_cable_agrees_with_the_geometry(
    resolved: Config, trace: dict[str, np.ndarray]
) -> None:
    """The headline sentence counts vessels over a cable, so it has to be right.

    The map no longer spends a colour on this: amber and red said "odd" and
    "odd, over a cable", a distinction that is real in the metrics and
    invisible to a first-time viewer, so the two marks collapsed into one red
    one meaning simply "not behaving". The *sentence* still makes the
    distinction, and `_on_cable` is what it counts with -- so it still has to
    agree with the corridor geometry or the headline is a lie.
    """
    assets = assets_module.build_assets(resolved)
    drawing = Replay(resolved, trace)
    for frame in range(FRAMES):
        here = drawing._on_cable(frame)
        assert here.shape == (TRACKS,)
        assert np.array_equal(here, assets_module.over_a_cable(assets, trace["vessel_xy"][frame]))
        for index, inside in enumerate(here):
            point = trace["vessel_xy"][frame][index : index + 1]
            nearest = min(
                float(assets_module._distance_to_line(point, line)[0])
                for _, line, _ in assets.cables
            )
            widest = max(width for _, _, width in assets.cables)
            assert bool(inside) == (nearest <= widest)


def test_a_still_comes_out_with_ink_on_it(
    resolved: Config, trace: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """A still renders, end to end through every layer.

    Catches an exception anywhere in the stack, and a canvas that comes out
    empty.
    """
    import matplotlib.image as mpimg

    destination = Replay(resolved, trace).still(0, tmp_path / "still.png")
    assert destination.is_file()
    pixels = mpimg.imread(destination)[:, :, :3]
    assert pixels.shape[0] > 400
    assert float(pixels.std()) > 0.02


def test_the_film_is_sized_for_the_encoder(
    resolved: Config, trace: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """The canvas is a size the encoder takes without resampling.

    ffmpeg resizes anything not divisible by 16, which resamples every hairline
    on the map. Cheaper to pick a canvas it accepts.
    """
    import imageio.v3 as imageio

    destination = Replay(resolved, trace).animate(tmp_path / "film.mp4", every=2, fps=4)
    frames = imageio.imread(destination, index=None)
    assert frames.shape[0] == 3
    assert frames.shape[1] % 16 == 0
    assert frames.shape[2] % 16 == 0
