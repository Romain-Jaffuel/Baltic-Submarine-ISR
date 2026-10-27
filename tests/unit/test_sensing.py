"""Detection, measurement, and how long a prediction is worth holding.

The batch 4 acceptance criterion is that the useful prediction horizon of each
vessel class is *measured* rather than tabulated. The tests below pin the shape
of that measurement -- knowing the lanes has to be worth a large factor on a
cargo, and a slow erratic vessel has to come out predictable for its own reason.
"""

from __future__ import annotations

import pytest
import torch

from baltic_isr import config, seeds
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.picture import prediction
from baltic_isr.sensing.detector import Detector
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics
from baltic_isr.world.traffic import Traffic


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def statics(resolved: Config) -> Statics:
    return statics_module.build(resolved, "cpu")


@pytest.fixture(scope="module")
def detector(resolved: Config, statics: Statics) -> Detector:
    return Detector(resolved, statics, seeds.torch_generator(7, "detector", "cpu"))


def scene(
    detector: Detector,
    statics: Statics,
    separation_m: float,
    regime: int = auv_module.REGIME_CRUISE,
    listening: bool = True,
    level_db: float = 145.0,
    drift_m: float = 0.0,
    worlds: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put one listener and one vessel at a fixed separation, many times over."""
    here = (statics.extent * 0.5).reshape(1, 1, 2).repeat(worlds, 1, 1)
    there = here + torch.tensor([[[separation_m, 0.0]]])
    contacts = detector.listen(
        listeners=here,
        believed=here + torch.tensor([[[drift_m, 0.0]]]),
        regime=torch.full((worlds, 1), regime),
        listening=torch.full((worlds, 1), listening),
        sources=there,
        source_level_db=torch.full((worlds, 1), level_db),
        vessel_class=torch.zeros((worlds, 1), dtype=torch.long),
        active=torch.ones((worlds, 1), dtype=torch.bool),
    )
    return contacts.detected[:, 0, 0], contacts.position[:, 0, 0]


# --- detection -------------------------------------------------------------


def test_close_contacts_are_heard_and_distant_ones_are_not(
    detector: Detector, statics: Statics
) -> None:
    near, _ = scene(detector, statics, 500.0)
    far, _ = scene(detector, statics, 40_000.0)
    assert float(near.float().mean()) > 0.8
    assert float(far.float().mean()) == 0.0


def test_a_sprinting_vehicle_is_nearly_deaf(detector: Detector, statics: Statics) -> None:
    """The couple of specification section 1. At 4 km it is the whole story."""
    cruising, _ = scene(detector, statics, 4000.0, regime=auv_module.REGIME_CRUISE)
    sprinting, _ = scene(detector, statics, 4000.0, regime=auv_module.REGIME_SPRINT)
    drifting, _ = scene(detector, statics, 4000.0, regime=auv_module.REGIME_DRIFT)
    assert float(sprinting.float().mean()) < 0.1
    assert float(cruising.float().mean()) > 0.4
    assert float(drifting.float().mean()) >= float(cruising.float().mean())


def test_drift_and_cruise_hear_alike(detector: Detector, statics: Statics) -> None:
    """The threshold, not the slope. docs/constants.md section 7.1."""
    cruising, _ = scene(detector, statics, 3000.0, regime=auv_module.REGIME_CRUISE)
    drifting, _ = scene(detector, statics, 3000.0, regime=auv_module.REGIME_DRIFT)
    assert abs(float(drifting.float().mean()) - float(cruising.float().mean())) < 0.2


def test_a_quiet_vessel_is_heard_less_far(detector: Detector, statics: Statics) -> None:
    loud, _ = scene(detector, statics, 4000.0, level_db=145.0)
    quiet, _ = scene(detector, statics, 4000.0, level_db=128.0)
    assert float(quiet.float().mean()) < float(loud.float().mean())


def test_a_surfaced_vehicle_hears_nothing(detector: Detector, statics: Statics) -> None:
    detected, _ = scene(detector, statics, 500.0, listening=False)
    assert not bool(detected.any())


def test_the_range_table_inverts_the_loss(detector: Detector, resolved: Config) -> None:
    from baltic_isr.sensing import acoustics

    for budget in (60.0, 80.0, 100.0):
        reach = float(detector.range_for_budget(torch.tensor(budget)))
        assert acoustics.transmission_loss_db(reach, resolved.environment) == pytest.approx(
            budget, abs=1.5
        )


def test_traffic_noise_shortens_everyone_else_s_reach(
    detector: Detector, statics: Statics, resolved: Config
) -> None:
    """Dense traffic cuts both ways: more targets, and a louder floor."""
    here = (statics.extent * 0.5).reshape(1, 1, 2)
    crowd = here + torch.arange(40).reshape(1, 40, 1) * torch.tensor([[[900.0, 300.0]]]) + 12_000.0
    level = torch.full((1, 40), 147.0)
    busy = detector.noise_floor_db(
        here,
        torch.tensor([[auv_module.REGIME_CRUISE]]),
        crowd,
        level,
        torch.ones(1, 40, dtype=torch.bool),
    )
    empty = detector.noise_floor_db(
        here,
        torch.tensor([[auv_module.REGIME_CRUISE]]),
        crowd,
        level,
        torch.zeros(1, 40, dtype=torch.bool),
    )
    assert float(busy) > float(empty)


# --- measurement -----------------------------------------------------------


def test_range_error_dominates_bearing_error(detector: Detector, statics: Statics) -> None:
    """A passive contact is a smear pointing at the listener, not a blob."""
    detected, position = scene(detector, statics, 5000.0, regime=auv_module.REGIME_DRIFT)
    here = statics.extent * 0.5
    offset = position[detected] - here
    along = (offset[:, 0] - 5000.0).std()
    across = offset[:, 1].std()
    assert float(along) > 3.0 * float(across)


def test_measurement_error_grows_with_range(detector: Detector, statics: Statics) -> None:
    spreads = []
    for separation in (1000.0, 4000.0):
        detected, position = scene(detector, statics, separation, regime=auv_module.REGIME_DRIFT)
        offset = position[detected] - statics.extent * 0.5
        spreads.append(float((offset[:, 0] - separation).std()))
    assert spreads[1] > 2.0 * spreads[0]


def test_a_mis_navigated_vehicle_mis_places_its_contacts(
    detector: Detector, statics: Statics
) -> None:
    """The mechanism of specification section 5.2, in one assertion.

    The listener's own position error goes straight into every contact it
    reports, and therefore into every prediction it issues.
    """
    honest, clean = scene(detector, statics, 2000.0, regime=auv_module.REGIME_DRIFT)
    lost, shifted = scene(detector, statics, 2000.0, regime=auv_module.REGIME_DRIFT, drift_m=2500.0)
    bias = float(shifted[lost][:, 0].mean()) - float(clean[honest][:, 0].mean())
    assert bias == pytest.approx(2500.0, rel=0.2)


# --- how long a prediction lasts ------------------------------------------


def test_knowing_the_lane_is_worth_an_order_of_magnitude(
    resolved: Config, statics: Statics
) -> None:
    """Batch 4 acceptance. It is why a cargo can be released instead of escorted."""
    horizons = prediction.measure_horizons(resolved, statics, worlds=8, horizon_s=7200.0)
    by_rule = {(item.name, item.rule): item for item in horizons}
    straight = by_rule[("cargo", "constant_velocity")]
    lane = by_rule[("cargo", "rail")]
    assert lane.error_at_hour_m < straight.error_at_hour_m / 10.0
    assert lane.horizon_s > 5.0 * straight.horizon_s


def test_a_slow_erratic_vessel_is_predictable_too(resolved: Config, statics: Statics) -> None:
    """For its own reason: the horizon is a distance budget, and it is slow.

    Counterintuitive, and worth a test so it does not get optimised away: a
    fishing boat wandering at 3 m/s stays inside the gate longer than a cargo
    running straight at 6 m/s whose lane you do not know.
    """
    horizons = prediction.measure_horizons(resolved, statics, worlds=8, horizon_s=7200.0)
    by_rule = {(item.name, item.rule): item for item in horizons}
    fishing = by_rule[("fishing", "constant_velocity")]
    cargo = by_rule[("cargo", "constant_velocity")]
    assert fishing.error_at_hour_m < cargo.error_at_hour_m
    assert fishing.horizon_s > cargo.horizon_s


def test_no_rule_is_offered_for_a_vessel_with_no_lane(resolved: Config, statics: Statics) -> None:
    horizons = prediction.measure_horizons(resolved, statics, worlds=4, horizon_s=3600.0)
    assert not [item for item in horizons if item.name == "fishing" and item.rule == "rail"]


def test_a_prediction_is_exact_at_zero_elapsed(statics: Statics) -> None:
    position = torch.tensor([[1000.0, 2000.0]])
    velocity = torch.tensor([[3.0, -4.0]])
    zero = torch.zeros(1)
    assert torch.allclose(prediction.constant_velocity(position, velocity, zero), position)


def test_rail_prediction_stops_at_the_end_of_the_lane(statics: Statics) -> None:
    """It cannot see past the end either, and pretending otherwise flatters it."""
    rail = torch.zeros(1, dtype=torch.long)
    far = prediction.along_rail(
        statics,
        rail,
        torch.tensor([0.9]),
        torch.tensor([20.0]),
        torch.zeros(1),
        torch.ones(1),
        torch.tensor([100_000.0]),
    )
    assert torch.allclose(far[0], statics.rail_xy[0, -1], atol=200.0)


def test_lane_bound_vessels_turn_round_at_the_end(resolved: Config, statics: Statics) -> None:
    """They used to wrap to the far end, which made every long prediction nonsense."""
    generator = Traffic(resolved, statics, 4)
    state = generator.reset()
    jumps = []
    for index in range(400):
        previous = state.xy.clone()
        state = generator.step(state, (index + 1) * 120.0, 120.0)
        jumps.append(float((state.xy - previous).norm(dim=-1).max()))
    assert max(jumps) < 3000.0
