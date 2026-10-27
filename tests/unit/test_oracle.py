"""The upper bound, and what makes it one.

The oracle is only a meaningful bound if two things hold. It must genuinely
see the truth -- a deviating vessel has to show up in its observation whether
or not anybody heard it -- and it must differ from the decentralised policy in
*nothing else*: same network, same PPO, same reward, same evaluation. If a
second difference creeps in, the measured gap stops being the price of
decentralisation and becomes the price of decentralisation plus whatever else
changed, and nobody will notice.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from baltic_isr import config
from baltic_isr.config import Config
from baltic_isr.learn.env import (
    NEAREST_TRACKS,
    OBSERVATION,
    ORACLE_CABLE,
    ORACLE_CELLS,
    ORACLE_GLOBAL,
    TRACK_FEATURES,
    Environment,
)


@pytest.fixture(scope="module")
def resolved() -> Config:
    base = config.load("baseline")
    return base.model_copy(update={"fleet": base.fleet.model_copy(update={"count": 8})})


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown observation"):
        Environment(config.load("baseline"), 1, "cpu", sees="telepathy")


def test_the_two_observations_have_the_widths_they_claim(resolved: Config) -> None:
    for sees, expected in (
        ("local", OBSERVATION),
        ("oracle", OBSERVATION + NEAREST_TRACKS * TRACK_FEATURES + ORACLE_GLOBAL),
    ):
        environment = Environment(resolved, 2, "cpu", sees=sees)
        state, observation = environment.reset(seed=1)
        assert environment.observation_width == expected
        assert observation.shape == (2, resolved.fleet.count, expected)
        assert bool(torch.isfinite(observation).all())
        del state


def test_the_oracle_sees_a_deviation_nobody_heard(resolved: Config) -> None:
    """The whole point of the bound.

    The decentralised policy learns about a deviation only if a vehicle hears
    it; the oracle is told. If this ever stopped being true the oracle would
    be a second decentralised policy with a wider input, and the gap between
    them would measure nothing.
    """
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, _ = environment.reset(seed=4)

    quiet = environment.observe_oracle(state)
    deviating = torch.zeros_like(state.traffic.deviating)
    deviating[:, 0] = True
    loud = environment.observe_oracle(
        replace(state, traffic=replace(state.traffic, deviating=deviating))
    )
    # The deviating-vessel grid is the second raster after the local block.
    start = OBSERVATION + NEAREST_TRACKS * TRACK_FEATURES + ORACLE_CELLS
    assert not torch.allclose(
        quiet[..., start : start + ORACLE_CELLS], loud[..., start : start + ORACLE_CELLS]
    )


def test_the_oracle_sees_everything_the_local_policy_could(resolved: Config) -> None:
    """A bound that does not dominate is not a bound.

    The first version gave the truth on a raster of 17 by 18 km cells -- three
    times coarser than the detection range -- while the decentralised policy
    got six-kilometre cells, its six nearest contacts with their uncertainty,
    and a bearing to the most neglected cable. It knew more and could aim
    less, and it **lost to the policy it was meant to bound, on every
    metric**. The oracle now carries the whole local observation, computed
    from truth, as its first block, so domination is a property of the
    construction rather than a hope.
    """
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, observation = environment.reset(seed=5)
    assert observation.shape[-1] > OBSERVATION
    assert bool(torch.isfinite(observation[..., :OBSERVATION]).all())

    perfect = environment._perfect(state)
    assert bool(perfect.known.all())
    assert float((state.t - perfect.last_seen_s).abs().max()) == 0.0
    assert torch.allclose(perfect.xy[:, 0], state.traffic.xy)
    assert bool((perfect.alarmed[:, 0] == state.traffic.deviating).all())


def test_the_oracle_reads_the_fleet_picture_as_well_as_the_truth(resolved: Config) -> None:
    """It must see what it is paid to fix, which is the picture's staleness.

    This test used to assert the opposite, and that assertion was the bug.
    "Not limited by the picture" and "does not see the picture" are different
    claims and I had merged them. Verification is 75 % of the reward and pays
    for reducing sigma on the fleet's **own** table; an oracle shown a perfect
    picture reads sigma as zero everywhere and is blind to the dominant term.
    Measured, three of its six track features were dead constants while the
    real sigma on held tracks sat at a median of 1542 m.
    """
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, _ = environment.reset(seed=6)
    for _ in range(6):
        actions = torch.randint(0, 25, state.fleet.regime.shape)
        state, _ = environment.step(state, actions)
    assert bool(state.tracks.known.any()), "nothing held, the test would be vacuous"

    before = environment.observe_oracle(state)
    blind = replace(state.tracks, last_seen_s=torch.full_like(state.tracks.last_seen_s, -torch.inf))
    assert not torch.allclose(before, environment.observe_oracle(replace(state, tracks=blind)))


def test_the_truth_block_carries_what_makes_a_vessel_a_target(resolved: Config) -> None:
    """Deviating, and over a cable, ranked above everything else.

    Reusing the ordinary nearest-contacts block on a perfect picture was
    tried: that block ranks by urgency, every urgency on a perfect picture is
    zero, so it returned six vessels chosen by a tie-break and carried no
    deviation flag at all. Truth has to be asked a different question.
    """
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, _ = environment.reset(seed=11)

    deviating = torch.zeros_like(state.traffic.deviating)
    deviating[:, 3] = True
    loud = replace(state, traffic=replace(state.traffic, deviating=deviating))
    block = environment._nearest_threats(loud).reshape(
        2, resolved.fleet.count, NEAREST_TRACKS, TRACK_FEATURES
    )
    # The flagged vessel takes the first slot for every vehicle, whatever the
    # distance, because deviating outranks near.
    assert bool((block[..., 0, 3] == 1.0).all())
    assert bool((block[..., 1:, 3] == 0.0).all())


def test_the_cable_block_is_the_real_clock(resolved: Config) -> None:
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, _ = environment.reset(seed=7)
    observation = environment.observe_oracle(state)
    start = OBSERVATION + NEAREST_TRACKS * TRACK_FEATURES + 3 * ORACLE_CELLS
    block = observation[0, 0, start : start + ORACLE_CABLE]
    assert block.shape == (ORACLE_CABLE,)
    # At reset nothing has been heard, so every stretch is at its floor.
    assert float(block.min()) > 0.0
    assert float(block.max()) <= 1.0


def test_each_vehicle_gets_the_same_global_block_and_its_own_tail(resolved: Config) -> None:
    """Assembled per vehicle so the same network serves both policies.

    The shared part must be identical across vehicles, and the tail must not
    be -- otherwise the oracle cannot tell its vehicles apart and commands
    forty identical actions.
    """
    environment = Environment(resolved, 2, "cpu", sees="oracle")
    state, _ = environment.reset(seed=8)
    observation = environment.observe_oracle(state)

    # The rasters are the fleet's shared view; everything before them -- the
    # local block and the truth block -- is a bearing from this vehicle.
    shared = observation[..., OBSERVATION + NEAREST_TRACKS * TRACK_FEATURES :]
    assert torch.allclose(shared[:, 0], shared[:, 1])

    own = observation[..., : OBSERVATION + NEAREST_TRACKS * TRACK_FEATURES]
    assert not torch.allclose(own[:, 0], own[:, 1])


def test_both_modes_share_everything_downstream(resolved: Config) -> None:
    """Same network, same reward, same step. Only the input differs."""
    from baltic_isr.learn.policy import Decentralised

    for sees in ("local", "oracle"):
        environment = Environment(resolved, 2, "cpu", sees=sees)
        policy = Decentralised(environment.observation_width)
        state, observation = environment.reset(seed=9)
        actions, _, _ = policy.act(observation)
        state, result = environment.step(state, actions)
        assert result.reward.shape == (2, resolved.fleet.count)
        assert bool(torch.isfinite(result.reward).all())
