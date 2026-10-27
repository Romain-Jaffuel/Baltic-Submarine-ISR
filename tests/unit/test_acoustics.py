"""The passive sonar budget, and the ranges it derives."""

import pytest

from baltic_isr import config
from baltic_isr.config import Config, VesselClass
from baltic_isr.fleet import power
from baltic_isr.sensing import acoustics


@pytest.fixture(scope="module")
def setup() -> tuple[Config, VesselClass]:
    resolved = config.load("baseline")
    cargo = next(item for item in resolved.traffic.classes if item.name == "cargo")
    return resolved, cargo


def ranges(resolved: Config, cargo: VesselClass) -> dict[str, float]:
    return {
        name: acoustics.detection_range_m(
            speed, cargo.source_level_db, resolved.acoustics, resolved.environment
        )
        for name, speed in power.regime_speeds(resolved.auv).items()
    }


def test_transmission_loss_is_monotonic(setup: tuple[Config, VesselClass]) -> None:
    resolved, _ = setup
    values = [acoustics.transmission_loss_db(r, resolved.environment) for r in (100, 1000, 10_000)]
    assert values == sorted(values)


def test_bottom_loss_dominates_absorption(setup: tuple[Config, VesselClass]) -> None:
    """Brackish water barely absorbs; soft clay does the damage. C-5-02."""
    resolved, _ = setup
    environment = resolved.environment
    assert environment.bottom_loss_db_per_km > 30.0 * environment.absorption_db_per_km


def test_array_gain_needs_a_kilohertz(setup: tuple[Config, VesselClass]) -> None:
    """A 2 m aperture has no directivity low down, which is why the band is high."""
    resolved, _ = setup
    low = resolved.acoustics.model_copy(update={"band_centre_hz": 300.0})
    assert acoustics.array_gain_db(low) == 0.0
    assert acoustics.array_gain_db(resolved.acoustics) > 5.0


def test_drift_makes_no_flow_noise(setup: tuple[Config, VesselClass]) -> None:
    resolved, _ = setup
    level = acoustics.self_noise_db(
        0.0, resolved.acoustics.band_centre_hz, resolved.acoustics, resolved.environment
    )
    assert level == float("-inf")


def test_self_noise_rises_steeply_with_speed(setup: tuple[Config, VesselClass]) -> None:
    resolved, _ = setup
    sound, environment = resolved.acoustics, resolved.environment
    quiet = acoustics.self_noise_db(1.0, sound.band_centre_hz, sound, environment)
    loud = acoustics.self_noise_db(2.0, sound.band_centre_hz, sound, environment)
    assert loud - quiet > 12.0


def test_detection_range_falls_with_speed(setup: tuple[Config, VesselClass]) -> None:
    resolved, cargo = setup
    derived = ranges(resolved, cargo)
    assert derived["drift"] > derived["cruise"] > derived["sprint"]


def test_hearing_is_a_threshold_not_a_slope(setup: tuple[Config, VesselClass]) -> None:
    """The batch 1 finding. Cruise hears nearly as far as drift; sprint does not.

    Specification v2 postulated a smooth 2x per regime. The derivation says the
    collapse happens between cruise and sprint and nowhere else. If this test
    ever fails, docs/constants.md section 7.1 needs rewriting.
    """
    resolved, cargo = setup
    derived = ranges(resolved, cargo)
    assert derived["drift"] / derived["cruise"] < 1.4
    assert derived["cruise"] / derived["sprint"] > 2.0


def test_flow_noise_does_not_reach_the_modem_band(setup: tuple[Config, VesselClass]) -> None:
    """C-5-06: at 25 kHz the vehicle's own flow noise sits below ambient."""
    resolved, _ = setup
    sound, environment = resolved.acoustics, resolved.environment
    at_band = acoustics.self_noise_db(
        resolved.auv.regimes.sprint_ms, sound.band_centre_hz, sound, environment
    )
    at_modem = acoustics.self_noise_db(
        resolved.auv.regimes.sprint_ms, resolved.comms.frequency_hz, sound, environment
    )
    assert at_band > environment.ambient_noise_db
    assert at_modem < environment.ambient_noise_db


def test_modem_range_is_multipath_limited_not_level_limited(
    setup: tuple[Config, VesselClass],
) -> None:
    """C-6-02: the level budget would allow far more than the channel does."""
    resolved, _ = setup
    level = acoustics.level_limited_modem_range_m(
        0.0, resolved.acoustics, resolved.comms, resolved.environment
    )
    assert level > 3.0 * acoustics.modem_range_m(resolved.comms)


def test_doppler_is_what_speed_costs_the_link(setup: tuple[Config, VesselClass]) -> None:
    """C-6-03: a sprinting vehicle is mute through lost packets, not lost range."""
    resolved, _ = setup
    speeds = power.regime_speeds(resolved.auv)
    drift = acoustics.doppler_success(speeds["drift"], resolved.comms)
    cruise = acoustics.doppler_success(speeds["cruise"], resolved.comms)
    sprint = acoustics.doppler_success(speeds["sprint"], resolved.comms)
    assert drift == pytest.approx(1.0)
    assert 0.4 < cruise < 0.7
    assert sprint < 0.1


def test_quiet_scenario_blinds_the_sprint() -> None:
    resolved = config.load("quiet")
    cargo = next(item for item in resolved.traffic.classes if item.name == "cargo")
    sprint = acoustics.detection_range_m(
        resolved.auv.regimes.sprint_ms,
        cargo.source_level_db,
        resolved.acoustics,
        resolved.environment,
    )
    assert sprint < 1000.0


def test_louder_target_is_heard_further(setup: tuple[Config, VesselClass]) -> None:
    resolved, cargo = setup
    near = acoustics.detection_range_m(
        0.0, cargo.source_level_db, resolved.acoustics, resolved.environment
    )
    far = acoustics.detection_range_m(
        0.0, cargo.source_level_db + 12.0, resolved.acoustics, resolved.environment
    )
    assert far > near
