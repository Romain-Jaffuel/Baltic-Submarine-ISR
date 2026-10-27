"""Configuration loading, overlay and fingerprinting."""

import pytest
from pydantic import ValidationError

from baltic_isr import config

SCENARIOS = ("baseline", "dense", "quiet", "noisy", "far_base")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_resolves(scenario: str) -> None:
    resolved = config.load(scenario)
    assert resolved.scenario == scenario
    assert resolved.schema_version == config.SCHEMA_VERSION


def test_scenario_overlays_the_base() -> None:
    base = config.load("baseline")
    dense = config.load("dense")
    assert dense.traffic.n_vessels > base.traffic.n_vessels
    assert dense.environment.ambient_noise_db > base.environment.ambient_noise_db
    assert dense.auv == base.auv


def test_fingerprints_differ_between_scenarios() -> None:
    seen = {config.load(name).fingerprint for name in SCENARIOS}
    assert len(seen) == len(SCENARIOS)


def test_fingerprint_is_stable() -> None:
    assert config.load("baseline").fingerprint == config.load("baseline").fingerprint


def test_payload_is_the_design_constraint() -> None:
    """250 bytes per exchange. Specification section 5.4 hangs on this."""
    assert config.load("baseline").comms.payload_bytes == 250


def test_unknown_scenario_lists_the_known_ones() -> None:
    with pytest.raises(FileNotFoundError, match="baseline"):
        config.load("does_not_exist")


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ValidationError):
        config.Simulation(decision_interval_s=1.0, episode_s=2.0, seed=0, typo=3)  # type: ignore[call-arg]


def test_config_is_frozen() -> None:
    resolved = config.load("baseline")
    with pytest.raises(ValidationError):
        resolved.simulation.seed = 1  # type: ignore[misc]


def test_regimes_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        config.Regimes(drift_ms=0.0, cruise_ms=3.0, sprint_ms=1.0)


def test_vessel_shares_must_sum_to_one() -> None:
    classes = [
        {
            "name": "a",
            "share": 0.5,
            "speed_mean_ms": 1.0,
            "speed_sigma_ms": 0.0,
            "source_level_db": 100.0,
            "growth_m_per_s": 1.0,
        },
    ]
    with pytest.raises(ValidationError, match="sum to 1"):
        config.Traffic(n_vessels=1, classes=classes)


def test_box_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        config.Geo(
            projected_crs="EPSG:3067",
            lon_min=26.0,
            lon_max=24.0,
            lat_min=59.0,
            lat_max=60.0,
            sea_area_m2=1.0,
        )


def test_merge_is_deep_and_does_not_mutate() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 20}}
    merged = config._merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}
