"""The speed-endurance law, and the calibration the whole project rests on."""

import math

import pytest

from baltic_isr import config
from baltic_isr.config import Auv
from baltic_isr.fleet import power

PUBLISHED_SPEED_MS = 1.543
PUBLISHED_ENDURANCE_H = 24.0


@pytest.fixture(scope="module")
def auv() -> Auv:
    return config.load("baseline").auv


def test_matches_published_endurance(auv: Auv) -> None:
    """The batch 1 acceptance criterion: k_prop reproduces 24 h at 3 kt."""
    hours = power.endurance_s(PUBLISHED_SPEED_MS, auv) / 3600.0
    assert hours == pytest.approx(PUBLISHED_ENDURANCE_H, rel=0.01)


def test_power_is_cubic_in_speed(auv: Auv) -> None:
    """Doubling speed multiplies the propulsion term by eight, not by two."""
    hotel = auv.hotel_power_w
    low = power.power_w(1.0, auv) - hotel
    high = power.power_w(2.0, auv) - hotel
    assert high == pytest.approx(8.0 * low, rel=1e-9)


def test_drift_draws_hotel_load_only(auv: Auv) -> None:
    assert power.power_w(0.0, auv) == pytest.approx(auv.hotel_power_w)


def test_regime_endurances(auv: Auv) -> None:
    """The three regimes span two and a half orders of magnitude."""
    speeds = power.regime_speeds(auv)
    drift = power.endurance_s(speeds["drift"], auv) / 3600.0
    cruise = power.endurance_s(speeds["cruise"], auv) / 3600.0
    sprint = power.endurance_s(speeds["sprint"], auv) / 3600.0
    assert drift > 300.0
    assert 40.0 < cruise < 55.0
    assert sprint < 7.0
    assert drift / sprint > 50.0


def test_range_peaks_below_cruise(auv: Auv) -> None:
    """Distance per battery peaks at about one knot, well under the cruise regime."""
    best = power.most_efficient_speed_ms(auv)
    assert best < auv.regimes.cruise_ms
    assert power.range_m(best, auv) > power.range_m(auv.regimes.cruise_ms, auv)
    for speed in (best * 0.5, best * 0.9, best * 1.1, best * 2.0):
        assert power.range_m(speed, auv) <= power.range_m(best, auv) + 1e-6


def test_drift_cannot_travel(auv: Auv) -> None:
    assert power.energy_to_travel_j(1000.0, 0.0, auv) == math.inf


def test_return_margin_grows_with_distance(auv: Auv) -> None:
    near = power.return_margin_j(1000.0, auv)
    far = power.return_margin_j(40_000.0, auv)
    assert far > near > auv.energy_reserve_fraction * auv.battery_j


def test_sprint_cannot_make_the_round_trip(auv: Auv) -> None:
    """Specification section 2.4, checked rather than asserted in prose."""
    distance = 39_900.0
    sprint = power.energy_to_travel_j(2.0 * distance, auv.regimes.sprint_ms, auv)
    cruise = power.energy_to_travel_j(2.0 * distance, auv.regimes.cruise_ms, auv)
    assert sprint > auv.battery_j
    assert cruise < 0.5 * auv.battery_j
