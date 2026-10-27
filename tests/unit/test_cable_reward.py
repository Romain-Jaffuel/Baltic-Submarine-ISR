"""The reward term that finally mentions what the fleet is guarding.

The reward said nothing about the cables at all. A vehicle could see where they
were -- the corridor distance is in its observation -- and was paid nothing for
watching one, while the map the project is judged on shades exactly that. A
policy trained on it would have optimised one quantity while the film measured
another, and nobody could have said whether it had learned.

These tests hold the two halves of that fix: the clock is honest, and the
credit lands on the vehicle that earned it. The last one is the real check --
a policy that patrols cables must score higher than one that ignores them, or
the term measures something other than what it claims.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from baltic_isr import config
from baltic_isr.baselines.cable_patrol import CablePatrol
from baltic_isr.baselines.lawnmower import Lawnmower
from baltic_isr.config import Config
from baltic_isr.learn.env import Environment
from baltic_isr.world import field
from baltic_isr.world import statics as statics_module

DECISIONS = 24


def tweak(**sections: dict[str, Any]) -> Config:
    base = config.load("baseline")
    return base.model_copy(
        update={
            name: getattr(base, name).model_copy(update=values) for name, values in sections.items()
        }
    )


@pytest.fixture(scope="module")
def resolved() -> Config:
    return tweak(fleet={"count": 12})


def walk(resolved: Config, policy_class: type, decisions: int = DECISIONS) -> tuple[Any, float]:
    """Run a scripted policy and return the final state and mean reward."""
    env = Environment(resolved, 2, "cpu")
    policy = policy_class(resolved, env.statics)
    state, _ = env.reset(seed=11)
    total = 0.0
    for _ in range(decisions):
        state, out = env.step(state, policy.act(state.fleet, state.t))
        total += float(out.reward.mean())
    return state, total / decisions


def test_the_cable_samples_are_at_sea(resolved: Config) -> None:
    """A sample ashore would be credited to whoever sailed past the beach."""
    fixed = statics_module.build(resolved, "cpu")
    assert fixed.cable_xy.shape[0] > 100
    assert bool((field.distance_to_coast(fixed, fixed.cable_xy) > 0.0).all())


def test_the_clock_starts_fully_stale(resolved: Config) -> None:
    """Nothing has been heard before the episode begins.

    Starting the clock at zero would hand the fleet a calm first hours it has
    not earned, and hide the cost of a cold start.
    """
    env = Environment(resolved, 2, "cpu")
    state, _ = env.reset(seed=1)
    age = state.t - state.cable_heard_s
    assert float(age.min()) == pytest.approx(resolved.reward.cable_horizon_s)


def test_hearing_a_cable_resets_its_clock(resolved: Config) -> None:
    env = Environment(resolved, 2, "cpu")
    state, _ = env.reset(seed=1)
    mower = Lawnmower(resolved, env.statics)
    before = float((state.t - state.cable_heard_s).mean())
    for _ in range(DECISIONS):
        state, _ = env.step(state, mower.act(state.fleet, state.t))
    age = state.t - state.cable_heard_s
    assert float(age.min()) < before
    assert bool((age >= 0.0).all())


def test_the_credit_goes_to_whoever_is_near_the_cable(resolved: Config) -> None:
    """Per vehicle, not shared.

    A penalty on total exposure would tell forty vehicles the same thing
    whatever each of them did, which is exactly the credit-assignment problem
    PPO is worst at.
    """
    env = Environment(resolved, 1, "cpu")
    state, _ = env.reset(seed=3)
    samples = env.statics.cable_xy

    # The far vehicles go to whichever point of the box is furthest from every
    # cable, found rather than guessed. A fixed offset from the cables' centre
    # used to serve: it stopped serving the moment the measured coastline put
    # more of EstLink 2 in the water, which moved that centre and dropped the
    # "far" vehicles straight onto the cable they were meant to be nowhere near.
    axis_x = torch.linspace(0.0, float(env.statics.extent[0]), 64)
    axis_y = torch.linspace(0.0, float(env.statics.extent[1]), 64)
    grid = torch.stack(torch.meshgrid(axis_x, axis_y, indexing="xy"), dim=-1).reshape(-1, 2)
    remotest = grid[torch.cdist(grid, samples).min(dim=-1).values.argmax()]

    fleet = state.fleet
    moved = fleet.xy.clone()
    moved[0, 0] = samples[0]  # one vehicle sat exactly on a cable
    moved[0, 1:] = remotest
    placed = type(fleet)(**{**vars(fleet), "xy": moved})

    _, credit = env._watch_cables(state.cable_heard_s, placed, state.t, 300.0)
    assert float(credit[0, 0]) > 0.0
    assert float(credit[0, 1:].max()) == pytest.approx(0.0)


def test_a_fleet_that_guards_cables_scores_higher(resolved: Config) -> None:
    """The claim the whole term makes, checked against a policy that acts.

    If patrolling cables did not beat sweeping the box here, the term would be
    measuring something other than what it says, and training would have
    learned that something else instead -- silently.
    """
    _, swept = walk(resolved, Lawnmower)
    _, guarded = walk(resolved, CablePatrol)
    assert guarded > swept


def test_guarding_cables_lowers_their_exposure() -> None:
    """The same claim measured on the film's own quantity, not the reward's.

    Needs a longer horizon than the reward does: the patrol spends its first
    hours transiting to its slice of cable, so over six simulated hours the
    lawnmower's incidental crossings are still ahead. Over a day it is not
    close -- 5.96 h of mean exposure against 9.42 over a full episode.
    """
    resolved = tweak(fleet={"count": 12})
    patrol, _ = walk(resolved, CablePatrol, decisions=96)
    sweep, _ = walk(resolved, Lawnmower, decisions=96)
    assert float((patrol.t - patrol.cable_heard_s).mean()) < float(
        (sweep.t - sweep.cable_heard_s).mean()
    )


def test_turning_the_term_off_restores_the_old_reward(resolved: Config) -> None:
    """Zero has to mean zero, or the two cannot be told apart."""
    off = resolved.model_copy(update={"reward": resolved.reward.model_copy(update={"cable": 0.0})})
    _, with_term = walk(resolved, CablePatrol)
    _, without = walk(off, CablePatrol)
    assert with_term > without


def test_the_verification_cost_is_actually_produced(resolved: Config) -> None:
    """A metric that silently reports zero is worse than a missing one.

    `Scorer.add_contact_time` existed from lot 6 and nothing ever called it,
    so `verification_cost` read 0.0 for every policy ever scored -- including
    in the lot table. The two things that can go wrong again are that nobody
    feeds it, and that the numerator sums over worlds while the denominator
    does not, which would make the number depend on the batch size.
    """
    from baltic_isr.baselines.lawnmower import Lawnmower as Mower
    from baltic_isr.learn.train import evaluate

    small = resolved.model_copy(
        update={"simulation": resolved.simulation.model_copy(update={"episode_s": 7200.0})}
    )
    costs = [
        evaluate(small, None, Mower, 60_001, worlds, "cpu").outcome.verification_cost
        for worlds in (2, 4)
    ]
    assert all(cost > 0.0 for cost in costs), "nobody is feeding the metric again"
    assert costs[0] == pytest.approx(costs[1], rel=0.25), "it scales with the batch"


def test_the_patrol_uses_more_than_one_regime(resolved: Config) -> None:
    """The floor that exercises the project's actual subject.

    The lawnmower cruises for the whole episode, so the speed-endurance trade
    -- the thing the project is about -- is never touched by any baseline.
    """
    env = Environment(resolved, 2, "cpu")
    patrol = CablePatrol(resolved, env.statics)
    state, _ = env.reset(seed=5)
    seen = torch.zeros(3)
    for _ in range(DECISIONS):
        state, _ = env.step(state, patrol.act(state.fleet, state.t))
        seen += torch.bincount(state.fleet.regime.flatten(), minlength=3).float()
    assert int((seen > 0).sum()) >= 2
