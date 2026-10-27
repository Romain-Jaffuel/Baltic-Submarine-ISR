"""The environment, the floor policy, and the batch 6 acceptance criterion.

The criterion is that the lawnmower produces a finite, reproducible alarm delay.
It does -- and the numbers it produces are a floor with room above it, which is
what a floor is for: it fails two of the three targets of specification section
7 and passes the third.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from baltic_isr import config
from baltic_isr.baselines.lawnmower import Lawnmower
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.learn.env import OBSERVATION, Environment
from baltic_isr.metrics.outcome import Scorer


def small(**sections: dict[str, Any]) -> Config:
    """Return a configuration small enough to run many times in a test."""
    base = config.load("baseline")
    merged: dict[str, Any] = {
        "fleet": {"count": 6},
        "traffic": {"n_vessels": 30},
        **sections,
    }
    update = {
        name: getattr(base, name).model_copy(update=values) for name, values in merged.items()
    }
    return base.model_copy(update=update)


def rollout(env: Environment, policy: Lawnmower, decisions: int) -> tuple[Any, list[torch.Tensor]]:
    """Run the lawnmower for a while and keep the rewards."""
    state, _ = env.reset(seed=4242)
    rewards = []
    for _ in range(decisions):
        state, result = env.step(state, policy.act(state.fleet, state.t))
        rewards.append(result.reward)
    return state, rewards


@pytest.fixture(scope="module")
def resolved() -> Config:
    return small()


@pytest.fixture(scope="module")
def env(resolved: Config) -> Environment:
    return Environment(resolved, 4, "cpu")


# --- the observation -------------------------------------------------------


def test_the_observation_has_the_declared_shape(env: Environment) -> None:
    _, observation = env.reset()
    assert observation.shape == (4, env.n_vehicles, OBSERVATION)


def test_the_observation_is_finite_and_bounded(env: Environment) -> None:
    """An infinity here poisons the policy on the first forward pass."""
    state, observation = env.reset()
    for _ in range(4):
        state, result = env.step(state, torch.randint(0, 25, (4, env.n_vehicles)))
        observation = result.observation
        assert bool(torch.isfinite(observation).all())
        assert float(observation.abs().max()) <= 10.0


def test_vehicles_see_different_things(env: Environment) -> None:
    """Each one carries its own picture. A global state would make them equal."""
    state, _ = env.reset()
    for _ in range(6):
        state, result = env.step(state, torch.randint(0, 25, (4, env.n_vehicles)))
    assert not torch.allclose(result.observation[0, 0], result.observation[0, 1])


def test_worlds_are_independent(env: Environment) -> None:
    state, _ = env.reset()
    for _ in range(4):
        state, result = env.step(state, torch.zeros(4, env.n_vehicles, dtype=torch.long))
    assert not torch.allclose(result.observation[0], result.observation[1])


# --- the two cadences ------------------------------------------------------


def test_physics_runs_finer_than_perception(resolved: Config) -> None:
    """85 % of a step is perception, so it runs at 300 s while hulls run at 60."""
    assert resolved.simulation.perception_interval_s > 60.0
    assert resolved.simulation.decision_interval_s > resolved.simulation.perception_interval_s
    ratio = resolved.simulation.perception_interval_s / 60.0
    assert ratio == int(ratio)


def test_a_coarser_cadence_keeps_the_same_detection_process(resolved: Config) -> None:
    """Bernoulli trials compound: five chances at p is one chance at 1-(1-p)^5."""
    from baltic_isr import seeds
    from baltic_isr.sensing.detector import Detector
    from baltic_isr.world import statics as statics_module

    statics = statics_module.build(resolved, "cpu")
    detector = Detector(resolved, statics, seeds.torch_generator(1, "d", "cpu"))
    here = (statics.extent * 0.5).reshape(1, 1, 2).repeat(2000, 1, 1)
    there = here + torch.tensor([[[6000.0, 0.0]]])

    def rate(trials: int) -> float:
        contacts = detector.listen(
            here,
            here,
            torch.full((2000, 1), auv_module.REGIME_DRIFT),
            torch.ones(2000, 1, dtype=torch.bool),
            there,
            torch.full((2000, 1), 145.0),
            torch.zeros((2000, 1), dtype=torch.long),
            torch.ones((2000, 1), dtype=torch.bool),
            trials=trials,
        )
        return float(contacts.detected.float().mean())

    single, five = rate(1), rate(5)
    assert five > single
    assert five == pytest.approx(1.0 - (1.0 - single) ** 5, abs=0.05)


# --- the reward ------------------------------------------------------------


def test_the_reward_is_finite_and_not_dominated_by_one_term(env: Environment) -> None:
    """The staleness penalty once reached -1800 over an episode.

    It was a tally of expired tracks rather than a fraction of them, so it grew
    with how much the fleet knew and buried the signal it was meant to nudge.
    """
    mower = Lawnmower(env._config, env.statics)
    _, rewards = rollout(env, mower, 20)
    stacked = torch.stack(rewards)
    assert bool(torch.isfinite(stacked).all())
    assert float(stacked.mean().abs()) < 5.0


def test_sprinting_everywhere_costs_more_than_cruising(env: Environment) -> None:
    """Energy is in the reward, and the cube law makes sprint expensive."""
    spent = {}
    for name, regime in (
        ("cruise", auv_module.REGIME_CRUISE),
        ("sprint", auv_module.REGIME_SPRINT),
    ):
        state, _ = env.reset(seed=7)
        start = float(state.fleet.energy_j.mean())
        action = torch.full((4, env.n_vehicles), regime * auv_module.HEADINGS)
        for _ in range(6):
            state, _ = env.step(state, action)
        spent[name] = start - float(state.fleet.energy_j.mean())
    assert spent["sprint"] > 4.0 * spent["cruise"]


# --- the lawnmower, which is the acceptance criterion ----------------------


def test_the_lawnmower_partitions_the_box(env: Environment, resolved: Config) -> None:
    """One strip each, so the sweep covers ground rather than piling up."""
    mower = Lawnmower(resolved, env.statics)
    target = mower.target(0.0, resolved.fleet.count)
    eastings = sorted(float(value) for value in target[:, 0])
    assert len(set(eastings)) == resolved.fleet.count
    assert eastings[0] > 0.0
    assert eastings[-1] < float(env.statics.extent[0])


def test_the_lawnmower_walks_up_and_down(env: Environment, resolved: Config) -> None:
    height = float(env.statics.extent[1])
    leg = height / resolved.auv.regimes.cruise_ms
    low = mower_northing(env, resolved, 0.0)
    middle = mower_northing(env, resolved, leg * 0.5)
    assert abs(middle - low) > 0.2 * height


def mower_northing(env: Environment, resolved: Config, t: float) -> float:
    mower = Lawnmower(resolved, env.statics)
    return float(mower.target(t, resolved.fleet.count)[0, 1])


def test_the_lawnmower_only_ever_cruises(env: Environment, resolved: Config) -> None:
    """It makes no tactical choices, which is the point of it being the floor."""
    mower = Lawnmower(resolved, env.statics)
    state, _ = env.reset()
    for _ in range(5):
        action = mower.act(state.fleet, state.t)
        assert bool((action // auv_module.HEADINGS == auv_module.REGIME_CRUISE).all())
        state, _ = env.step(state, action)


def test_the_lawnmower_is_reproducible(resolved: Config) -> None:
    """Batch 6 acceptance, half one."""
    runs = []
    for _ in range(2):
        env = Environment(resolved, 2, "cpu")
        mower = Lawnmower(resolved, env.statics)
        _, rewards = rollout(env, mower, 8)
        runs.append(torch.stack(rewards))
    assert torch.allclose(runs[0], runs[1])


def test_the_lawnmower_raises_a_finite_alarm_delay(resolved: Config) -> None:
    """Batch 6 acceptance, half two.

    Measured on the full configuration rather than this cut-down one, the
    lawnmower reaches a median of 52.7 min and a P90 of 208.8, misses 71 % of
    deviations and holds 41 % picture accuracy. It fails two of the three
    targets of section 7 and passes false alarms -- which is what a floor with
    room above it looks like.
    """
    env = Environment(resolved, 16, "cpu")
    mower = Lawnmower(resolved, env.statics)
    scorer = Scorer(resolved, env._picture)
    state, _ = env.reset(seed=999_999)
    decisions = int(resolved.simulation.episode_s / resolved.simulation.decision_interval_s)
    for _ in range(decisions):
        state, _ = env.step(state, mower.act(state.fleet, state.t))
        scorer.observe(state, resolved.simulation.decision_interval_s)
    outcome = scorer.score(state)

    assert 0.0 < outcome.alarm_delay_median_s < resolved.simulation.episode_s
    assert outcome.alarm_delay_p90_s >= outcome.alarm_delay_median_s
    assert 0.0 <= outcome.deviations_missed <= 1.0
    assert 0.0 < outcome.picture_accuracy < 1.0


def test_the_floor_leaves_room_above_it(resolved: Config) -> None:
    """A floor that already met the targets would make the project pointless."""
    env = Environment(resolved, 16, "cpu")
    mower = Lawnmower(resolved, env.statics)
    scorer = Scorer(resolved, env._picture)
    state, _ = env.reset(seed=999_999)
    for _ in range(int(resolved.simulation.episode_s / resolved.simulation.decision_interval_s)):
        state, _ = env.step(state, mower.act(state.fleet, state.t))
        scorer.observe(state, resolved.simulation.decision_interval_s)
    outcome = scorer.score(state)
    assert outcome.picture_accuracy < 0.8
    assert outcome.deviations_missed > 0.2


# --- the metrics themselves ------------------------------------------------


def test_a_false_alarm_is_counted_once(resolved: Config) -> None:
    """It is an event, not a state.

    Counting the standing flag every decision turned a handful of false alarms
    into eighty per vessel-day.
    """
    env = Environment(resolved, 8, "cpu")
    mower = Lawnmower(resolved, env.statics)
    scorer = Scorer(resolved, env._picture)
    state, _ = env.reset(seed=11)
    for _ in range(40):
        state, _ = env.step(state, mower.act(state.fleet, state.t))
        scorer.observe(state, resolved.simulation.decision_interval_s)
    assert scorer.score(state).false_alarm_rate < 5.0


def test_picture_accuracy_is_satisfiable_by_prediction(resolved: Config) -> None:
    """A cargo correctly predicted and never revisited counts as known.

    That is the whole difference from the 'time in sensor range' metric of
    specification v1, which rewarded escorting it.
    """
    env = Environment(resolved, 8, "cpu")
    mower = Lawnmower(resolved, env.statics)
    scorer = Scorer(resolved, env._picture)
    state, _ = env.reset(seed=3)
    for _ in range(30):
        state, _ = env.step(state, mower.act(state.fleet, state.t))
        scorer.observe(state, resolved.simulation.decision_interval_s)
    outcome = scorer.score(state)
    assert outcome.picture_accuracy > 0.0
    assert outcome.verification_cost >= 0.0
