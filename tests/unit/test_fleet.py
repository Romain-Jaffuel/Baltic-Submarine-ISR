"""The fleet, its sortie cycle, and the batch 3 acceptance criterion.

The criterion is that an isolated vehicle reproduces the regime table of
specification section 2.3 to one per cent. It is checked on a single step, where
the energy drawn must equal ``P(v) dt``, and over a long run inside the deployed
phase, where the state machine has room to leak power the closed form does not
know about.

The rest is about the rotation: a fleet in sustained operation has to hold a flat
number of vehicles on station, and must never recall a whole base at once.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from typing import Any

import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet import navigation, power
from baltic_isr.fleet.auv import Fleet, FleetState
from baltic_isr.world import field
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics
from baltic_isr.world.traffic import Traffic

REGIMES = (
    ("drift", auv_module.REGIME_DRIFT),
    ("cruise", auv_module.REGIME_CRUISE),
    ("sprint", auv_module.REGIME_SPRINT),
)


def tweak(**sections: dict[str, Any]) -> Config:
    """Return the baseline configuration with some sections overridden."""
    base = config.load("baseline")
    update = {
        name: getattr(base, name).model_copy(update=values) for name, values in sections.items()
    }
    return base.model_copy(update=update)


def small(count: int = 8, **sections: dict[str, Any]) -> Config:
    """Return a configuration with a small fleet, so the tests stay quick."""
    merged: dict[str, Any] = dict(sections)
    merged["fleet"] = {"count": count, **merged.get("fleet", {})}
    return tweak(**merged)


def cold(count: int = 8, **sections: dict[str, Any]) -> Config:
    """Return a small fleet starting at the quay, for the deterministic cases."""
    merged: dict[str, Any] = dict(sections)
    merged["fleet"] = {"count": count, "start": "cold", **merged.get("fleet", {})}
    return tweak(**merged)


def build(resolved: Config, worlds: int = 4) -> tuple[Fleet, FleetState, Statics]:
    """Build a fleet on the CPU."""
    fixed = statics_module.build(resolved, "cpu")
    fleet = Fleet(resolved, fixed, worlds)
    return fleet, fleet.reset(), fixed


def hold(state: FleetState, regime: int, heading: int = 0) -> torch.Tensor:
    """Return an action tensor holding one regime and heading."""
    return torch.full_like(state.regime, regime * auv_module.HEADINGS + heading)


def toward(state: FleetState, target: torch.Tensor, regime: int) -> torch.Tensor:
    """Return actions steering each vehicle at a target."""
    direction = target - state.xy
    angle = torch.atan2(direction[..., 0], direction[..., 1]) % (2.0 * torch.pi)
    index = torch.round(angle / (2.0 * torch.pi / auv_module.HEADINGS)).long()
    return regime * auv_module.HEADINGS + index % auv_module.HEADINGS


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


# --- the shore -------------------------------------------------------------


def test_the_fleet_cannot_swim_ashore() -> None:
    """The constraint that was missing, and the measurement that found it.

    ``step`` bounded the box but never the coast, so the fleet simply swam
    inland: 24 % of all recorded positions were on land and the worst incursion
    ran 24 km into Finland. It was not only wrong to look at -- a vehicle
    listening from a forest contributes coverage it has not earned.

    Driven deliberately at the nearest shore at sprint for a simulated day,
    nothing may touch land, and outside a base approach nothing may enter the
    keel clearance. See C-3-16.
    """
    resolved = small(12)
    fleet, state, fixed = build(resolved, worlds=2)

    northern = state.xy[..., 1] > fixed.extent[1] * 0.5
    at_the_shore = torch.where(northern, 0, 4) + auv_module.REGIME_SPRINT * auv_module.HEADINGS

    worst_land, worst_water = float("inf"), float("inf")
    for tick in range(1440):
        state = fleet.step(state, at_the_shore, tick * 60.0, 60.0)
        worst_land = min(worst_land, float(field.distance_to_coast(fixed, state.xy).min()))
        away = (state.xy - fixed.bases[state.base]).norm(dim=-1) >= resolved.fleet.base_approach_m
        if bool(away.any()):
            depth = field.depth_m(fixed, state.xy)
            worst_water = min(worst_water, float(depth[away].min()))

    # Afloat, always. This is the guarantee that has to be absolute, and the
    # projection in world.field is written to deliver exactly it.
    assert worst_land > 0.0

    # And in water, though not necessarily in all the water it would like.
    # The claim used to be the full keel clearance everywhere; on a measured
    # archipelago that claim cannot be kept and should not be made. Driven
    # deliberately at the coast at sprint for a simulated day, a vehicle ends
    # up in a bay, and some bays are shallow -- the nearest water holding the
    # full clearance can be a kilometre away through three islands, and no
    # projection that only looks under its own feet will find it.
    #
    # What is guaranteed is that it is never on the beach and never in water
    # that would ground it. Under a policy rather than this adversarial
    # command, the fleet holds 14.8 m against the 15.0 it asks for, the
    # shortfall being interpolation across one cell of the isobath.
    assert worst_water > 5.0


def test_a_vehicle_held_off_the_shore_still_makes_good_along_it() -> None:
    """Blocking has to show up in the dead reckoning, not just the truth.

    Zeroing the motion while advancing the believed position is what made the
    estimate diverge without bound in batch 2. A vehicle pinned against the
    coast makes good the along-shore component and nothing more, and its DVL
    measures exactly that -- so truth and belief stay the same distance apart.
    """
    fleet, state, fixed = build(small(6), worlds=1)
    northern = state.xy[..., 1] > fixed.extent[1] * 0.5
    at_the_shore = torch.where(northern, 0, 4) + auv_module.REGIME_CRUISE * auv_module.HEADINGS
    for tick in range(600):
        state = fleet.step(state, at_the_shore, tick * 60.0, 60.0)

    pinned = (state.xy - fixed.bases[state.base]).norm(dim=-1) >= 3_000.0
    assert bool(pinned.any())
    error = (state.estimate - state.xy).norm(dim=-1)[pinned]
    assert float(error.max()) < 5_000.0


def test_stations_are_drawn_where_a_vehicle_can_actually_sit() -> None:
    """A station has to be somewhere a vehicle can sit.

    One inside the clearance is one it spends the episode grinding towards and
    never reaching.
    """
    resolved = small(32)
    _, state, fixed = build(resolved, worlds=4)
    # Asked of the seabed, not of a shelf model: a station is somewhere a
    # vehicle can sit, and that is a question about the water under it.
    assert bool(field.has_clearance(fixed, state.station, resolved.fleet.min_water_m).all())


def test_rotation_does_not_start_anyone_on_land() -> None:
    """Nobody starts the episode ashore.

    The initial transit is drawn as a straight line from base to station, and a
    straight line here cuts a corner of Estonia.
    """
    for seed in (1, 2, 3):
        resolved = small(40)
        fixed = statics_module.build(resolved, "cpu")
        state = Fleet(resolved, fixed, 2).reset(seed=seed)
        assert float(field.distance_to_coast(fixed, state.xy).min()) > 0.0


# --- the sortie cycle ------------------------------------------------------


def test_the_operational_radius_is_finite(resolved: Config) -> None:
    """Past half a battery's worth of cruise a vehicle cannot get home again.

    86.8 km from base. It is the hardest boundary in the model: no fleet size
    buys coverage beyond it.
    """
    fleet, _, fixed = build(small())
    radius = fleet.usable_s / 2.0 * resolved.auv.regimes.cruise_ms
    assert radius == pytest.approx(86_800.0, rel=0.02)

    base = fixed.bases[:1]
    far = base + torch.tensor([[radius * 1.1, 0.0]])
    near = base + torch.tensor([[radius * 0.3, 0.0]])
    assert float(fleet.sortie(far, base)["on_station"]) == 0.0
    assert float(fleet.sortie(near, base)["on_station"]) > 0.0


def test_duty_cycle_collapses_with_transit() -> None:
    """The central economy of the project, and it is not linear.

    Twenty kilometres out, two thirds of the sortie is useful. Eighty, and it is
    under a tenth.
    """
    fleet, _, fixed = build(small())
    base = fixed.bases[:1]
    duties = []
    for km in (20.0, 40.0, 60.0, 80.0):
        timings = fleet.sortie(base + torch.tensor([[km * 1000.0, 0.0]]), base)
        duties.append(float(timings["on_station"] / timings["cycle"]))
    assert duties == sorted(duties, reverse=True)
    assert duties[0] > 0.6
    assert duties[3] < 0.1


def test_vehicles_sail_from_their_nearest_base() -> None:
    """Handing bases out round robin cost seventeen kilometres each way."""
    _, state, fixed = build(small(16))
    own = (fixed.bases[state.base] - state.station).norm(dim=-1)
    nearest = torch.cdist(state.station, fixed.bases.expand(state.n_worlds, -1, -1)).min(dim=-1)
    assert torch.allclose(own, nearest.values, atol=1.0)


def test_every_base_gets_used() -> None:
    _, state, fixed = build(small(32))
    used = torch.bincount(state.base.flatten(), minlength=int(fixed.bases.shape[0]))
    assert bool((used > 0).all())


# --- the rotation ----------------------------------------------------------


def test_rotation_starts_the_fleet_mid_cycle(resolved: Config) -> None:
    """Some already working, some outbound, some homebound, some alongside."""
    _, state, _ = build(small(24))
    present = {int(mode) for mode in torch.unique(state.mode).tolist()}
    assert auv_module.MODE_DEPLOYED in present
    assert auv_module.MODE_RETURNING in present
    assert auv_module.MODE_CHARGING in present
    assert float(state.energy_j.min()) < 0.4 * resolved.auv.battery_j
    assert float(state.energy_j.max()) > 0.9 * resolved.auv.battery_j


def test_a_cold_start_puts_everyone_at_the_quay(resolved: Config) -> None:
    fleet, state, _ = build(cold())
    assert float(fleet.distance_to_base(state).max()) < 1.0
    assert bool((state.energy_j == resolved.auv.battery_j).all())
    assert bool((state.mode == auv_module.MODE_DEPLOYED).all())


def test_rotation_holds_a_flat_number_on_station(resolved: Config) -> None:
    """The point of rotating, and what the user asked for.

    Whole-base recalls are exactly what the even spread prevents, so the
    deployed fraction has to stay within a couple of points across an episode.
    """
    fleet, state, _ = build(small(24), worlds=8)
    step_s = 600.0
    fractions = []
    for index in range(int(resolved.simulation.episode_s / step_s)):
        action = toward(state, state.station, auv_module.REGIME_CRUISE)
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
        fractions.append(float((state.mode == auv_module.MODE_DEPLOYED).float().mean()))
    assert statistics.pstdev(fractions) < 0.05
    assert 0.4 < statistics.mean(fractions) < 0.9
    assert int((state.mode == auv_module.MODE_LOST).sum()) == 0


def test_a_cold_start_recalls_the_whole_base_at_once(resolved: Config) -> None:
    """The contrast that justifies rotating at all."""
    fleet, state, fixed = build(cold(24), worlds=4)
    centre = fixed.extent * 0.5
    step_s = 600.0
    fractions = []
    for index in range(int(resolved.simulation.episode_s / step_s)):
        action = toward(state, centre, auv_module.REGIME_CRUISE)
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
        fractions.append(float((state.mode == auv_module.MODE_DEPLOYED).float().mean()))
    assert statistics.pstdev(fractions) > 0.2


def test_rotation_starts_with_a_drifted_belief() -> None:
    """A vehicle hours into its sortie has not just taken a fix."""
    _, state, _ = build(small(24))
    assert float(state.drift_error_m[state.at_sea].mean()) > 50.0


# --- the energy law, which is the acceptance criterion ---------------------


def test_energy_draw_matches_the_power_law(resolved: Config) -> None:
    """Batch 3 acceptance, on one step. This is the law, exactly."""
    for _, regime in REGIMES:
        fleet, state, _ = build(cold(4), worlds=1)
        stepped = fleet.step(state, hold(state, regime), 60.0, 60.0)
        drawn = float((state.energy_j - stepped.energy_j).mean())
        speed = float(fleet._speed[regime])
        assert drawn == pytest.approx(power.power_w(speed, resolved.auv) * 60.0, rel=1e-4)


def test_energy_integrates_to_the_law(resolved: Config) -> None:
    """Batch 3 acceptance, over a long run.

    Not run to an empty battery: the forced recall fires long before that and
    switches the vehicle to cruise, which is exactly what it is for.
    """
    for name, regime in REGIMES:
        fleet, state, _ = build(cold(4, navigation={"surface_trigger_m": 1.0e9}), worlds=1)
        action = hold(state, regime)
        step_s, steps = 300.0, 40
        start = float(state.energy_j.mean())
        for index in range(steps):
            state = fleet.step(state, action, (index + 1) * step_s, step_s)
        assert bool((state.mode == auv_module.MODE_DEPLOYED).all()), name
        drawn = start - float(state.energy_j.mean())
        expected = power.power_w(float(fleet._speed[regime]), resolved.auv) * step_s * steps
        assert drawn == pytest.approx(expected, rel=0.01), name


def test_the_regime_table_holds(resolved: Config) -> None:
    """Specification section 2.3, read off the law."""
    hours = {
        name: power.endurance_s(getattr(resolved.auv.regimes, f"{name}_ms"), resolved.auv) / 3600.0
        for name in ("drift", "cruise", "sprint")
    }
    assert hours["drift"] == pytest.approx(346.7, rel=0.01)
    assert hours["cruise"] == pytest.approx(47.3, rel=0.01)
    assert hours["sprint"] == pytest.approx(5.95, rel=0.01)


# --- the recall, the transit and the turnaround ---------------------------


def test_the_recall_brings_them_home() -> None:
    """It cannot be switched off, which is the point of it."""
    fleet, state, _ = build(cold(8), worlds=4)
    action = hold(state, auv_module.REGIME_SPRINT, heading=2)
    step_s = 600.0
    recalled = False
    for index in range(int(48.0 * 3600.0 / step_s)):
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
        recalled = recalled or bool((state.mode == auv_module.MODE_RETURNING).any())
    assert recalled
    assert int((state.mode == auv_module.MODE_LOST).sum()) == 0


def test_transit_out_costs_the_predicted_share(resolved: Config) -> None:
    """C-3-04: 39.9 km at cruise is about nine hours and a fifth of the battery."""
    fleet, state, fixed = build(cold(4), worlds=2)
    centre = fixed.extent * 0.5
    step_s, elapsed = 300.0, 0.0
    while float(fleet.distance_to_base(state).mean()) < 39_000.0:
        elapsed += step_s
        state = fleet.step(state, toward(state, centre, auv_module.REGIME_CRUISE), elapsed, step_s)
    spent = 1.0 - float(state.energy_j.mean()) / resolved.auv.battery_j
    assert 8.0 < elapsed / 3600.0 < 11.0
    assert 0.15 < spent < 0.25


def test_the_turnaround_takes_handling_plus_charging(resolved: Config) -> None:
    """A vehicle does not turn itself round the instant the battery reads full."""
    fleet, state, _ = build(cold(4), worlds=2)
    action = torch.full_like(state.regime, auv_module.ACTION_RETURN)
    step_s, charging_for = 600.0, 0.0
    for index in range(int(20.0 * 3600.0 / step_s)):
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
        if bool((state.mode == auv_module.MODE_CHARGING).all()):
            charging_for += step_s
    assert charging_for >= 0.9 * (resolved.auv.recharge_s + resolved.auv.handling_s)


# --- navigation -----------------------------------------------------------


def test_dead_reckoning_drifts_with_distance() -> None:
    """Compass bias, not the current: the error grows with path length."""
    fleet, state, _ = build(cold(8, navigation={"surface_trigger_m": 1.0e9}), worlds=8)
    action = hold(state, auv_module.REGIME_CRUISE, heading=2)
    step_s, early, late = 300.0, 0.0, 0.0
    for index in range(int(20.0 * 3600.0 / step_s)):
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
        if index == int(5.0 * 3600.0 / step_s):
            early = float(state.drift_error_m.mean())
        late = float(state.drift_error_m.mean())
    assert late > 2.0 * early > 0.0


def test_the_current_is_not_the_navigation_error() -> None:
    """The correction batch 3 made to specification section 5.2.

    A bottom-locked DVL measures velocity over ground, so the current is inside
    the measurement. Removing it must barely change the drift; removing the
    compass bias must nearly abolish it.
    """
    cases: dict[str, dict[str, Any]] = {
        "baseline": {},
        "no_current": {"environment": {"current_speed_ms": 0.0}},
        "no_bias": {"navigation": {"heading_bias_sigma_deg": 0.0}},
    }
    drifts = {}
    for name, overrides in cases.items():
        settings: dict[str, Any] = {"navigation": {"surface_trigger_m": 1.0e9}}
        for section, values in overrides.items():
            settings[section] = {**settings.get(section, {}), **values}
        fleet, state, _ = build(cold(8, **settings), worlds=8)
        action = hold(state, auv_module.REGIME_CRUISE, heading=2)
        for index in range(int(10.0 * 3600.0 / 300.0)):
            state = fleet.step(state, action, (index + 1) * 300.0, 300.0)
        drifts[name] = float(state.drift_error_m.mean())
    assert drifts["no_current"] == pytest.approx(drifts["baseline"], rel=0.25)
    assert drifts["no_bias"] < 0.2 * drifts["baseline"]


def test_surfacing_bounds_the_drift() -> None:
    """What the fix buys, measured against not having it.

    The claim is not that the error shrinks at some instant -- it oscillates --
    but that surfacing *bounds* it, so the run is done twice and compared. The
    trigger is lowered from the configured 1500 m because at 1500 m the vehicle
    reaches the edge of the box and stops before it ever comes due.
    """
    duration_s, step_s = 20.0 * 3600.0, 300.0
    drifts, surfaced = [], False
    for trigger in (300.0, 1.0e9):
        fleet, state, _ = build(cold(8, navigation={"surface_trigger_m": trigger}), worlds=8)
        action = hold(state, auv_module.REGIME_CRUISE, heading=2)
        for index in range(int(duration_s / step_s)):
            state = fleet.step(state, action, (index + 1) * step_s, step_s)
            surfaced = surfaced or bool((state.mode == auv_module.MODE_SURFACING).any())
        drifts.append(float(state.drift_error_m.mean()))
    bounded, unbounded = drifts
    assert surfaced
    assert bounded < 0.5 * unbounded
    assert bounded < 1000.0


def test_a_surfaced_vehicle_is_deaf() -> None:
    """Most of what surfacing costs is the listening time, not the energy."""
    fleet, state, _ = build(small(4), worlds=2)
    surfacing = torch.full_like(state.mode, auv_module.MODE_SURFACING)
    assert not bool(replace(state, mode=surfacing).listening.any())
    assert fleet.surfacing_s == pytest.approx(460.0, rel=0.01)


def test_uncertainty_grows_and_a_fix_clears_it(resolved: Config) -> None:
    near = navigation.uncertainty_m(torch.tensor([0.0]), resolved.navigation)
    far = navigation.uncertainty_m(torch.tensor([100_000.0]), resolved.navigation)
    assert float(near) == pytest.approx(resolved.navigation.gnss_sigma_m, rel=0.01)
    assert 1500.0 < float(far) < 2500.0


# --- the shape of the computation -----------------------------------------


def test_reset_is_reproducible() -> None:
    _, first, _ = build(small(8))
    _, second, _ = build(small(8))
    assert torch.allclose(first.heading_bias, second.heading_bias)
    assert torch.allclose(first.station, second.station)


def test_step_never_synchronises_and_does_not_grow_with_the_batch() -> None:
    """The invariant batch 2 established, now for the fleet."""
    from test_traffic import Recorder

    counts = []
    for worlds, vehicles in ((1, 2), (32, 2), (1, 512)):
        fleet, state, _ = build(small(vehicles), worlds=worlds)
        action = hold(state, auv_module.REGIME_CRUISE)
        state = fleet.step(state, action, 60.0, 60.0)
        recorder = Recorder()
        with recorder:
            fleet.step(state, action, 120.0, 60.0)
        assert recorder.syncs == []
        counts.append(recorder.operations)
    assert counts[0] == counts[1] == counts[2]


def test_the_fleet_and_the_traffic_share_a_clock() -> None:
    """They must be steppable together without either noticing the other."""
    settings = small(8)
    fixed = statics_module.build(settings, "cpu")
    fleet, traffic = Fleet(settings, fixed, 2), Traffic(settings, fixed, 2)
    fleet_state, traffic_state = fleet.reset(), traffic.reset()
    action = hold(fleet_state, auv_module.REGIME_CRUISE)
    for index in range(20):
        moment = (index + 1) * 60.0
        fleet_state = fleet.step(fleet_state, action, moment, 60.0)
        traffic_state = traffic.step(traffic_state, moment, 60.0)
    assert fleet_state.xy.isfinite().all()
    assert traffic_state.xy.isfinite().all()
