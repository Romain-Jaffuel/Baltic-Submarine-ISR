"""The picture: tracks, plausibility, search.

Three things are pinned here above all others, because the demonstration of
specification section 0.1 rests on them:

* the covariance algebra is self-consistent -- an absolute floor on the
  determinant once made every fusion look five times better than it is, and
  only the ``sqrt(2)`` check caught it;
* **converging on an alarmed contact pays more than dispersing**, which is what
  makes the fleet visibly close on a threat;
* a slow trawler is released and a slow cargo is not, which is what stops the
  fleet crying wolf.
"""

from __future__ import annotations

import math

import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.picture import tracks as tracks_module
from baltic_isr.picture.plausibility import Plausibility
from baltic_isr.picture.presence import Search
from baltic_isr.picture.tracks import Picture, Tracks
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def statics(resolved: Config) -> Statics:
    return statics_module.build(resolved, "cpu")


def fix_from(bearing_deg: float, range_m: float, resolved: Config) -> torch.Tensor:
    """Return the covariance of a fix on a contact at the origin."""
    angle = math.radians(bearing_deg)
    listener = torch.tensor([-range_m * math.sin(angle), -range_m * math.cos(angle)])
    return tracks_module.measurement_cov(
        listener,
        torch.zeros(2),
        resolved.acoustics.range_error_fraction,
        math.radians(resolved.acoustics.bearing_sigma_deg),
    )


def fuse(covariances: list[torch.Tensor]) -> torch.Tensor:
    """Fuse fixes through the information filter."""
    total = tracks_module.inverse(covariances[0])
    for item in covariances[1:]:
        total = total + tracks_module.inverse(item)
    return tracks_module.inverse(total)


# --- the covariance algebra ------------------------------------------------


def test_inverting_twice_returns_the_original() -> None:
    """The check that caught the bug.

    An absolute floor of 1e-9 on the determinant silently replaced the real
    determinant of an information matrix, which is around 1e-11. Every fusion
    came out five times better than it is.
    """
    for cov in (
        torch.tensor([30450.0, 0.0, 1562500.0]),
        torch.tensor([1.0e6, 2.0e5, 4.0e5]),
        torch.tensor([9.0, -1.0, 4.0]),
    ):
        assert torch.allclose(tracks_module.inverse(tracks_module.inverse(cov)), cov, rtol=1e-3)


def test_major_axis_reads_the_long_side() -> None:
    assert float(tracks_module.major_axis_m(torch.tensor([100.0, 0.0, 400.0]))) == pytest.approx(
        20.0
    )
    rotated = torch.tensor([250.0, 150.0, 250.0])
    assert float(tracks_module.major_axis_m(rotated)) == pytest.approx(20.0)


def test_a_passive_fix_is_a_smear_pointing_at_the_listener(resolved: Config) -> None:
    cov = fix_from(0.0, 5000.0, resolved)
    assert float(tracks_module.major_axis_m(cov)) == pytest.approx(1250.0, rel=0.01)
    assert float(cov[2]) > 40.0 * float(cov[0])


def test_mahalanobis_is_scaled_by_the_ellipse() -> None:
    cov = torch.tensor([100.0, 0.0, 10_000.0])
    assert float(tracks_module.mahalanobis(torch.tensor([10.0, 0.0]), cov)) == pytest.approx(1.0)
    assert float(tracks_module.mahalanobis(torch.tensor([0.0, 100.0]), cov)) == pytest.approx(1.0)


# --- why the fleet converges ----------------------------------------------


def test_two_fixes_from_the_same_bearing_give_root_two(resolved: Config) -> None:
    """Theory says exactly sqrt(2). Anything else means the filter is wrong."""
    one = fix_from(0.0, 5000.0, resolved)
    both = fuse([one, fix_from(0.0, 5000.0, resolved)])
    ratio = float(tracks_module.major_axis_m(one)) / float(tracks_module.major_axis_m(both))
    assert ratio == pytest.approx(math.sqrt(2.0), rel=0.01)


def test_triangulating_beats_repeating(resolved: Config) -> None:
    """Each listener's good axis covers the other's bad one."""
    alone = float(tracks_module.major_axis_m(fix_from(0.0, 5000.0, resolved)))
    repeated = float(tracks_module.major_axis_m(fuse([fix_from(0.0, 5000.0, resolved)] * 2)))
    crossed = float(
        tracks_module.major_axis_m(
            fuse([fix_from(0.0, 5000.0, resolved), fix_from(90.0, 5000.0, resolved)])
        )
    )
    assert crossed < repeated < alone
    assert (alone - crossed) / (alone - repeated) == pytest.approx(2.9, rel=0.1)


def test_separation_helps_monotonically(resolved: Config) -> None:
    axes = [
        float(
            tracks_module.major_axis_m(
                fuse([fix_from(0.0, 5000.0, resolved), fix_from(angle, 5000.0, resolved)])
            )
        )
        for angle in (0.0, 30.0, 60.0, 90.0)
    ]
    assert axes == sorted(axes, reverse=True)


def test_converging_on_a_threat_outpays_finding_a_stale_track(resolved: Config) -> None:
    """The economics the demonstration depends on.

    Against the traffic gate, joining a contact someone already holds is worth
    less than going to find a stale one, so the fleet disperses -- correct for
    traffic, and useless as a demonstration. The tightened alarm gate is what
    reverses it.
    """
    gate, alarm = resolved.fleet.gate_m, resolved.prediction.alarm_gate_m
    alone = float(tracks_module.major_axis_m(fix_from(0.0, 5000.0, resolved)))
    crossed = float(
        tracks_module.major_axis_m(
            fuse([fix_from(0.0, 5000.0, resolved), fix_from(90.0, 5000.0, resolved)])
        )
    )
    finding_stale = (gate - alone) / gate
    joining_traffic = (alone - crossed) / gate
    joining_threat = (alone - crossed) / alarm
    assert joining_traffic < finding_stale
    assert joining_threat > 5.0 * finding_stale


# --- prediction and staleness ---------------------------------------------


def seeded(resolved: Config, statics: Statics, plausible: bool, on_lane: bool) -> Tracks:
    """Return a one-track table, held now, at the middle of the box."""
    table = tracks_module.empty(1, 1, 1, statics.device)
    return Tracks(
        last_seen_s=torch.zeros(1, 1, 1),
        xy=(statics.extent * 0.5).reshape(1, 1, 1, 2).clone(),
        velocity=torch.tensor([[[[0.0, 6.2]]]]),
        cov=torch.tensor([[[[10_000.0, 0.0, 10_000.0]]]]),
        rail=torch.full((1, 1, 1), 0 if on_lane else -1, dtype=torch.int8),
        progress=torch.full((1, 1, 1), 0.5),
        direction=torch.ones(1, 1, 1),
        growth_m_per_s=torch.full((1, 1, 1), resolved.traffic.classes[0].growth_m_per_s),
        issuer=table.issuer,
        plausible=torch.full((1, 1, 1), plausible, dtype=torch.bool),
        alarmed=torch.zeros(1, 1, 1, dtype=torch.bool),
    )


def test_uncertainty_grows_with_time(resolved: Config, statics: Statics) -> None:
    picture = Picture(resolved, statics)
    table = seeded(resolved, statics, plausible=True, on_lane=True)
    growth = [float(picture.uncertainty_m(table, t)) for t in (0.0, 1800.0, 7200.0)]
    assert growth == sorted(growth)


def test_a_known_lane_holds_a_prediction_far_longer(resolved: Config, statics: Statics) -> None:
    """The measured factor of thirty-two, now inside the track table."""
    picture = Picture(resolved, statics)
    on_lane = picture.uncertainty_m(seeded(resolved, statics, True, True), 3600.0)
    blind = picture.uncertainty_m(seeded(resolved, statics, False, False), 3600.0)
    assert float(blind) > 10.0 * float(on_lane)


def test_a_track_goes_stale_at_the_gate(resolved: Config, statics: Statics) -> None:
    picture = Picture(resolved, statics)
    table = seeded(resolved, statics, plausible=False, on_lane=False)
    assert not bool(picture.stale(table, 0.0).any())
    assert bool(picture.stale(table, 24.0 * 3600.0).any())


def test_an_unheard_contact_is_never_stale_and_never_certain(statics: Statics) -> None:
    """Zero covariance would mean certainty, which is the opposite of unheard."""
    table = tracks_module.empty(2, 3, 5, statics.device)
    assert not bool(table.known.any())
    assert float(tracks_module.major_axis_m(table.cov).min()) > 100_000.0


def test_an_alarmed_track_is_held_to_a_tighter_standard(resolved: Config, statics: Statics) -> None:
    from dataclasses import replace

    picture = Picture(resolved, statics)
    calm = seeded(resolved, statics, plausible=True, on_lane=True)
    alarmed = replace(calm, alarmed=torch.ones(1, 1, 1, dtype=torch.bool))
    assert float(picture.gate_m(calm)) == resolved.fleet.gate_m
    assert float(picture.gate_m(alarmed)) == resolved.prediction.alarm_gate_m
    assert float(picture.gate_m(alarmed)) < float(picture.gate_m(calm))


# --- plausibility ----------------------------------------------------------


@pytest.fixture(scope="module")
def judge(resolved: Config, statics: Statics) -> Plausibility:
    return Plausibility(resolved, statics)


def on_corridor(statics: Statics) -> torch.Tensor:
    return statics_module.sample_rail(
        statics, torch.zeros(1, dtype=torch.long), torch.tensor([0.30])
    )[0]


def judged(judge: Plausibility, position: torch.Tensor, speed: float, vessel_class: int) -> float:
    verdict = judge.assess(
        position.reshape(1, 2), torch.tensor([[0.0, speed]]), torch.tensor([vessel_class])
    )
    return float(verdict.score)


def test_a_loitering_cargo_over_a_cable_is_held(judge: Plausibility, statics: Statics) -> None:
    assert judged(judge, on_corridor(statics), 0.6, 0) < 0.2


def test_a_loitering_cargo_in_open_water_is_also_held(
    judge: Plausibility, statics: Statics
) -> None:
    """Three deviations in ten happen off-corridor, so this one has to bite too."""
    assert judged(judge, statics.extent * 0.5, 0.6, 0) < 0.5


def test_a_cargo_making_its_usual_speed_is_released(
    judge: Plausibility, statics: Statics, resolved: Config
) -> None:
    speed = resolved.traffic.classes[0].speed_mean_ms
    assert judged(judge, on_corridor(statics), speed, 0) > 0.9


def test_a_slow_trawler_is_fishing_not_threatening(judge: Plausibility, statics: Statics) -> None:
    """A slow trawler is fishing, not threatening.

    Surprise is measured against the class's own distribution, or the fleet
    would flag every trawler that slowed down.
    """
    fishing = next(
        index
        for index, item in enumerate(config.load("baseline").traffic.classes)
        if item.name == "fishing"
    )
    assert judged(judge, statics.extent * 0.5, 0.6, fishing) > 0.5
    assert judged(judge, statics.extent * 0.5, 0.6, 0) < 0.5


def test_the_verdict_explains_itself(judge: Plausibility, statics: Statics) -> None:
    """Specification section 12.5: why a vessel was released must be readable."""
    verdict = judge.assess(
        on_corridor(statics).reshape(1, 2), torch.tensor([[0.0, 0.6]]), torch.tensor([0])
    )
    sentence = judge.explain(verdict, (0,))
    assert "held" in sentence
    assert "loitering" in sentence
    assert "corridor" in sentence


# --- the search picture ----------------------------------------------------


@pytest.fixture(scope="module")
def search(resolved: Config, statics: Statics) -> Search:
    return Search(resolved, statics, 2, 3)


def test_the_prior_is_zero_on_land(search: Search) -> None:
    assert float((search.prior == 0.0).float().mean()) > 0.2


def test_the_prior_favours_the_lanes(search: Search, statics: Statics) -> None:
    """A flat prior would send the fleet to look at the emptiest water."""
    to_lane = statics_module.sample_raster(statics, statics.rail_distance_m, search.centres)
    at_sea = search.prior > 0.0
    near = search.prior[at_sea & (to_lane < 5000.0)].mean()
    far = search.prior[at_sea & (to_lane > 30_000.0)].mean()
    assert float(near) > 1.5 * float(far)


def test_sweeping_removes_belief(search: Search, statics: Statics) -> None:
    picture = search.empty()
    before = float(picture.probability.sum())
    positions = (statics.extent * 0.5).reshape(1, 1, 2).expand(2, 3, 2)
    picture, gain = search.step(
        picture, positions, torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 5910.0), 900.0
    )
    assert float(picture.probability.sum()) < before
    assert float(gain.mean()) > 0.5


def test_sweeping_the_same_water_again_pays_much_less(search: Search, statics: Statics) -> None:
    """Which is what sends a vehicle somewhere else."""
    picture = search.empty()
    positions = (statics.extent * 0.5).reshape(1, 1, 2).expand(2, 3, 2)
    listening, reach = torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 5910.0)
    gain = torch.zeros(1)
    for _ in range(20):
        picture, gain = search.step(picture, positions, listening, reach, 900.0)
    stale, elsewhere = float(gain.mean()), 0.0
    _, fresh = search.step(
        picture, positions + torch.tensor([40_000.0, 0.0]), listening, reach, 900.0
    )
    elsewhere = float(fresh.mean())
    assert elsewhere > 5.0 * stale


def test_belief_recovers_when_nobody_looks(search: Search, statics: Statics) -> None:
    picture = search.empty()
    positions = (statics.extent * 0.5).reshape(1, 1, 2).expand(2, 3, 2)
    listening, reach = torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 5910.0)
    for _ in range(10):
        picture, _ = search.step(picture, positions, listening, reach, 900.0)
    swept = float(picture.probability.sum())
    for _ in range(40):
        picture, _ = search.step(
            picture, positions, torch.zeros(2, 3, dtype=torch.bool), reach, 900.0
        )
    assert float(picture.probability.sum()) > swept


def test_a_surfaced_vehicle_sweeps_nothing(search: Search, statics: Statics) -> None:
    picture = search.empty()
    positions = (statics.extent * 0.5).reshape(1, 1, 2).expand(2, 3, 2)
    _, gain = search.step(
        picture, positions, torch.zeros(2, 3, dtype=torch.bool), torch.full((2, 3), 5910.0), 900.0
    )
    assert float(gain.abs().max()) == 0.0


def test_a_sprinting_vehicle_sweeps_far_less(search: Search, statics: Statics) -> None:
    """Deaf is also blind to the search picture, which is the same coupling."""
    positions = (statics.extent * 0.5).reshape(1, 1, 2).expand(2, 3, 2)
    listening = torch.ones(2, 3, dtype=torch.bool)
    _, cruising = search.step(
        search.empty(), positions, listening, torch.full((2, 3), 5910.0), 900.0
    )
    _, sprinting = search.step(
        search.empty(), positions, listening, torch.full((2, 3), 2170.0), 900.0
    )
    assert float(sprinting.mean()) < 0.3 * float(cruising.mean())
