"""Talking under water, and the batch 5 acceptance criterion.

The criterion was that the 250-byte budget be verified by a test. It is, below
-- and the test says the budget is **not** the binding constraint, which is the
finding of the batch. What binds is range: a vehicle is alone most of the time,
so the network is delay-tolerant and information travels by being carried.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from baltic_isr import config, seeds
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet.auv import Fleet
from baltic_isr.fleet.comms import Comms, message_bits
from baltic_isr.picture import tracks as tracks_module
from baltic_isr.picture.tracks import Tracks
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics


@pytest.fixture(scope="module")
def resolved() -> Config:
    return config.load("baseline")


@pytest.fixture(scope="module")
def statics(resolved: Config) -> Statics:
    return statics_module.build(resolved, "cpu")


@pytest.fixture(scope="module")
def comms(resolved: Config, statics: Statics) -> Comms:
    return Comms(resolved, statics, seeds.torch_generator(5, "comms", "cpu"))


def line(statics: Statics, count: int, spacing_m: float) -> torch.Tensor:
    """Return vehicles in a row, a fixed distance apart."""
    start = statics.extent * 0.5
    offsets = torch.arange(count, dtype=torch.float32).unsqueeze(-1) * torch.tensor(
        [spacing_m, 0.0]
    )
    return (start + offsets).unsqueeze(0)


def seeded(worlds: int, vehicles: int, vessels: int, device: str = "cpu") -> Tracks:
    """Return a table where nobody knows anything."""
    return tracks_module.empty(worlds, vehicles, vessels, device)


# --- the budget: the stated criterion, and what it actually says ------------


def test_one_exchange_carries_two_hundred_and_fifty_bytes(resolved: Config) -> None:
    """Batch 5 acceptance, as written in specification section 5.4."""
    assert resolved.comms.payload_bytes == 250


def test_the_message_of_the_specification_fits(comms: Comms) -> None:
    """Header, own position, state, three predictions, four presence cells."""
    assert message_bits(predictions=3, presence_cells=4) == 374
    assert message_bits(predictions=3, presence_cells=4) <= comms.payload_bits


def test_the_budget_is_not_the_binding_constraint(comms: Comms, resolved: Config) -> None:
    """The batch 5 finding, pinned so the claim cannot quietly stop being true.

    A packet holds 27 predictions and the whole table is 180. Two exchanges and
    a bit, against roughly a hundred available per decision. The pipe is wide.
    """
    assert comms.predictions_per_message >= 25
    per_decision = resolved.simulation.decision_interval_s / comms.slot_s()
    whole_table = resolved.traffic.n_vessels / comms.predictions_per_message
    assert per_decision > 20.0 * whole_table


def test_a_belief_grid_would_never_fit(comms: Comms) -> None:
    """Which is why the fleet passes predictions and not maps."""
    from baltic_isr.picture.presence import PRESENCE_CELL_M

    del PRESENCE_CELL_M
    cells = 437
    assert message_bits(predictions=0, presence_cells=cells) > 4 * comms.payload_bits


def test_the_guard_interval_is_larger_than_the_slot(comms: Comms, resolved: Config) -> None:
    """Sound is slow. Waiting for it costs more than speaking does."""
    assert comms.slot_s() > 2.0 * resolved.comms.slot_s


# --- what actually binds: range ---------------------------------------------


def test_nobody_hears_anybody_at_fleet_spacing(comms: Comms, statics: Statics) -> None:
    """Mean spacing is 23.7 km against a 4 km modem."""
    positions = line(statics, 6, 23_700.0)
    links = comms.links(positions, torch.zeros_like(positions), torch.ones(1, 6, dtype=torch.bool))
    assert int(links.degree.sum()) == 0
    assert bool(links.isolated.all())


def test_close_vehicles_hear_each_other(comms: Comms, statics: Statics) -> None:
    positions = line(statics, 4, 1500.0)
    links = comms.links(positions, torch.zeros_like(positions), torch.ones(1, 4, dtype=torch.bool))
    assert int(links.degree[0, 0]) >= 2
    assert not bool(links.reachable[0, 0, 0])


def test_the_fleet_is_isolated_most_of_the_time(
    resolved: Config, statics: Statics, comms: Comms
) -> None:
    """The batch 5 finding, measured on a real episode rather than asserted.

    This is what makes the network delay-tolerant: predictions move because a
    vehicle carries them until it meets someone, not because it can call.
    """
    fleet = Fleet(resolved, statics, 8)
    state = fleet.reset()
    step_s = 600.0
    alone = total = 0
    for index in range(int(24.0 * 3600.0 / step_s)):
        links = comms.links(state.xy, torch.zeros_like(state.xy), state.listening)
        degree = links.degree[state.listening]
        alone += int((degree == 0).sum())
        total += int(degree.numel())
        action = torch.full_like(state.regime, auv_module.REGIME_CRUISE * auv_module.HEADINGS)
        state = fleet.step(state, action, (index + 1) * step_s, step_s)
    assert alone / total > 0.5


def test_a_surfaced_vehicle_is_off_the_acoustic_net(comms: Comms, statics: Statics) -> None:
    positions = line(statics, 3, 1000.0)
    listening = torch.tensor([[True, False, True]])
    links = comms.links(positions, torch.zeros_like(positions), listening)
    assert int(links.degree[0, 1]) == 0
    assert not bool(links.reachable[0, 0, 1])


def test_sprinting_costs_packets(comms: Comms, statics: Statics) -> None:
    """The Doppler of docs/constants.md C-6-03, now on the link itself."""
    positions = line(statics, 2, 1000.0)
    still = torch.zeros_like(positions)
    parting = torch.tensor([[[0.0, 2.5], [0.0, -2.5]]])
    quiet = comms.links(positions, still, torch.ones(1, 2, dtype=torch.bool))
    fast = comms.links(positions, parting, torch.ones(1, 2, dtype=torch.bool))
    assert float(fast.success[0, 0, 1]) < 0.1 * float(quiet.success[0, 0, 1])
    assert bool(fast.reachable[0, 0, 1])


def test_range_loss_grows_with_distance(comms: Comms, statics: Statics) -> None:
    still = torch.zeros(1, 2, 2)
    ones = torch.ones(1, 2, dtype=torch.bool)
    near = comms.links(line(statics, 2, 500.0), still, ones).success[0, 0, 1]
    far = comms.links(line(statics, 2, 3500.0), still, ones).success[0, 0, 1]
    assert float(near) > float(far) > 0.0


# --- what an exchange does ---------------------------------------------------


def two_vehicles_one_knowing(statics: Statics) -> Tracks:
    """Vehicle 0 holds a track at t=100; vehicle 1 holds nothing."""
    table = seeded(1, 2, 3)
    last_seen = table.last_seen_s.clone()
    last_seen[0, 0, 1] = 100.0
    xy = table.xy.clone()
    xy[0, 0, 1] = statics.extent * 0.5
    issuer = table.issuer.clone()
    issuer[0, 0, 1] = 0
    cov = table.cov.clone()
    cov[0, 0, 1] = torch.tensor([10_000.0, 0.0, 10_000.0])
    return replace(table, last_seen_s=last_seen, xy=xy, issuer=issuer, cov=cov)


def test_a_neighbour_learns_what_you_know(comms: Comms, statics: Statics) -> None:
    tracks = two_vehicles_one_knowing(statics)
    positions = line(statics, 2, 300.0)
    links = comms.links(positions, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    merged, taken = comms.exchange(tracks, links)
    assert bool(taken[0, 1, 1])
    assert float(merged.last_seen_s[0, 1, 1]) == 100.0
    assert int(merged.issuer[0, 1, 1]) == 0


def test_a_relayed_prediction_keeps_its_author(comms: Comms, statics: Statics) -> None:
    """Verifying someone else's prediction is the only thing that pays for talking."""
    tracks = two_vehicles_one_knowing(statics)
    positions = line(statics, 2, 300.0)
    links = comms.links(positions, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    merged, _ = comms.exchange(tracks, links)
    assert int(merged.issuer[0, 1, 1]) != 1


def test_nothing_moves_when_nobody_is_in_range(comms: Comms, statics: Statics) -> None:
    tracks = two_vehicles_one_knowing(statics)
    positions = line(statics, 2, 30_000.0)
    links = comms.links(positions, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    merged, taken = comms.exchange(tracks, links)
    assert not bool(taken.any())
    assert not bool(merged.known[0, 1].any())


def test_a_stale_neighbour_teaches_you_nothing(comms: Comms, statics: Statics) -> None:
    """Merging takes the fresher of the two, never simply the other one's."""
    tracks = two_vehicles_one_knowing(statics)
    mine = tracks.last_seen_s.clone()
    mine[0, 1, 1] = 500.0
    tracks = replace(tracks, last_seen_s=mine)
    positions = line(statics, 2, 300.0)
    links = comms.links(positions, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    merged, _ = comms.exchange(tracks, links)
    assert float(merged.last_seen_s[0, 1, 1]) == 500.0


# --- the surface burst -------------------------------------------------------


def test_surfacing_pools_the_whole_fleet(comms: Comms, statics: Statics) -> None:
    """One hop instead of a chain of chance meetings."""
    tracks = seeded(1, 4, 3)
    last_seen = tracks.last_seen_s.clone()
    last_seen[0, 0, 2] = 900.0
    tracks = replace(tracks, last_seen_s=last_seen)
    surfaced = torch.tensor([[True, True, True, False]])
    merged, _, _ = comms.surface_exchange(
        tracks, comms.shore(1, 3), surfaced, torch.ones(1, 3, dtype=torch.bool)
    )
    assert float(merged.last_seen_s[0, 1, 2]) == 900.0
    assert float(merged.last_seen_s[0, 2, 2]) == 900.0
    assert not math.isfinite(float(merged.last_seen_s[0, 3, 2]))


def test_the_shore_reaches_further_than_a_neighbour(comms: Comms, statics: Statics) -> None:
    """Two vehicles 80 km apart share nothing acoustically and everything by radio."""
    tracks = two_vehicles_one_knowing(statics)
    positions = line(statics, 2, 80_000.0)
    links = comms.links(positions, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    acoustic, _ = comms.exchange(tracks, links)
    assert not bool(acoustic.known[0, 1].any())
    radio, _, _ = comms.surface_exchange(
        tracks,
        comms.shore(1, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert bool(radio.known[0, 1, 1])


def test_the_shore_brings_the_ais_picture(comms: Comms, statics: Statics) -> None:
    """C-12-05: AIS is VHF, so a submerged vehicle cannot have it.

    A vessel that has gone dark stops being releasable. Not an alarm -- an
    alarm needs an observation contradicting a prediction -- but it is what
    sends someone to get one.
    """
    tracks = two_vehicles_one_knowing(statics)
    tracks = replace(tracks, plausible=torch.ones_like(tracks.plausible))
    dark = torch.tensor([[True, False, True]])
    merged, _, _ = comms.surface_exchange(
        tracks, comms.shore(1, 3), torch.ones(1, 2, dtype=torch.bool), dark
    )
    assert not bool(merged.plausible[0, 0, 1])
    assert bool(merged.plausible[0, 0, 2])


def test_ais_never_raises_an_alarm_on_its_own(comms: Comms, statics: Statics) -> None:
    tracks = two_vehicles_one_knowing(statics)
    merged, _, _ = comms.surface_exchange(
        tracks,
        comms.shore(1, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert not bool(merged.alarmed.any())


def test_a_submerged_vehicle_learns_nothing_from_the_shore(comms: Comms, statics: Statics) -> None:
    tracks = replace(
        two_vehicles_one_knowing(statics), plausible=torch.ones(1, 2, 3, dtype=torch.bool)
    )
    merged, _, taken = comms.surface_exchange(
        tracks,
        comms.shore(1, 3),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert not bool(taken.any())
    assert bool(merged.plausible.all())


# --- the shape of the computation -------------------------------------------


def test_exchange_never_synchronises_and_does_not_grow_with_the_batch(
    resolved: Config, statics: Statics
) -> None:
    from test_traffic import Recorder

    counts = []
    for worlds, vehicles, vessels in ((1, 2, 4), (32, 2, 4), (1, 2, 512)):
        comms = Comms(resolved, statics, seeds.torch_generator(1, "comms", "cpu"))
        tracks = seeded(worlds, vehicles, vessels)
        positions = statics.extent.reshape(1, 1, 2).expand(worlds, vehicles, 2) * 0.5
        links = comms.links(
            positions, torch.zeros_like(positions), torch.ones(worlds, vehicles, dtype=torch.bool)
        )
        comms.exchange(tracks, links)
        recorder = Recorder()
        with recorder:
            comms.exchange(tracks, links)
        assert recorder.syncs == []
        counts.append(recorder.operations)
    assert counts[0] == counts[1] == counts[2]


def test_the_shore_remembers_between_visits(comms: Comms, statics: Statics) -> None:
    """The correction of batch 5, and it is worth 45 points of fleet knowledge.

    Pooling only among vehicles surfaced at the same instant lifted fleet-wide
    knowledge of one prediction from 55 % to 57 % over an episode: vehicles
    surface every twenty hours, independently, and almost never coincide. A
    shore station keeps a database, so a vehicle that comes up alone and late
    still gets everything. Measured over the same episode: 99.5 %.
    """
    tracks = two_vehicles_one_knowing(statics)
    tracks = replace(
        tracks, last_seen_s=torch.cat([tracks.last_seen_s, tracks.last_seen_s[:, :1]], dim=1)
    )
    tracks = tracks_module.empty(1, 3, 3)
    seen = tracks.last_seen_s.clone()
    seen[0, 0, 1] = 100.0
    tracks = replace(tracks, last_seen_s=seen)
    shore = comms.shore(1, 3)
    everyone_visible = torch.ones(1, 3, dtype=torch.bool)

    tracks, shore, _ = comms.surface_exchange(
        tracks, shore, torch.tensor([[True, False, False]]), everyone_visible
    )
    assert bool(torch.isfinite(shore.last_seen_s[0, 0, 1]))

    # Much later, somebody else surfaces alone. Nobody was up with them.
    tracks, shore, taken = comms.surface_exchange(
        tracks, shore, torch.tensor([[False, False, True]]), everyone_visible
    )
    assert bool(taken[0, 2, 1])
    assert float(tracks.last_seen_s[0, 2, 1]) == 100.0
    assert not bool(tracks.known[0, 1].any())
