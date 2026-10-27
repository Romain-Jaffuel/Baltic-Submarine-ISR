"""The learning machinery, checked where it is easy to be silently wrong.

Reinforcement learning fails quietly. A sign error in the advantage, a
discount applied twice, a minibatch that reuses stale log probabilities --
none of these raise, and all of them produce a run that trains for an hour and
learns nothing. So the pieces are tested against closed forms and against
contrived cases where the right answer is known by hand, and the loop itself
is tested on a toy problem it must be able to solve.
"""

from __future__ import annotations

import torch

from baltic_isr import config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.learn import ppo
from baltic_isr.learn.env import OBSERVATION
from baltic_isr.learn.policy import Decentralised


def test_the_policy_starts_almost_uniform() -> None:
    """The policy starts with no opinion.

    One that starts opinionated spends its first updates unlearning an opinion
    that came from the initialiser rather than from the world.
    """
    policy = Decentralised(OBSERVATION)
    with torch.no_grad():
        logits, _ = policy(torch.randn(64, OBSERVATION))
        spread = float(logits.std(dim=-1).mean())
        spread_of = torch.distributions.Categorical(logits=logits)
        entropy = float(spread_of.entropy().mean())  # type: ignore[no-untyped-call]
    assert spread < 0.1
    assert entropy > 0.98 * float(torch.log(torch.tensor(float(auv_module.N_ACTIONS))))


def test_every_action_is_reachable() -> None:
    policy = Decentralised(OBSERVATION)
    actions, log_probability, values = policy.act(torch.randn(8, 40, OBSERVATION))
    assert actions.shape == (8, 40)
    assert int(actions.min()) >= 0
    assert int(actions.max()) < auv_module.N_ACTIONS
    assert log_probability.shape == actions.shape
    assert values.shape == actions.shape
    assert bool(torch.isfinite(log_probability).all())


def test_advantage_matches_the_closed_form_without_a_trace() -> None:
    """GAE with no trace is the one-step TD error.

    Which can be written down. If the discount were applied twice, this would
    be off by a factor of gamma at every step.
    """
    settings = ppo.Settings(discount=0.9, trace=0.0)
    reward = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    value = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])
    last = torch.tensor([[2.0]])

    advantage, target = ppo.advantages(reward, value, last, settings)
    expected = torch.tensor(
        [
            [[1.0 + 0.9 * 1.0 - 0.5]],
            [[2.0 + 0.9 * 1.5 - 1.0]],
            [[3.0 + 0.9 * 2.0 - 1.5]],
        ]
    )
    assert torch.allclose(advantage, expected, atol=1e-6)
    assert torch.allclose(target, advantage + value, atol=1e-6)


def test_advantage_discounts_the_future_once_per_step() -> None:
    """A constant reward with a zero critic sums the geometric series."""
    settings = ppo.Settings(discount=0.5, trace=1.0)
    steps = 12
    reward = torch.ones(steps, 1, 1)
    value = torch.zeros(steps, 1, 1)
    advantage, _ = ppo.advantages(reward, value, torch.zeros(1, 1), settings)
    # From the first step: 1 + g + g^2 + ... + g^(steps-1).
    total = sum(settings.discount**power for power in range(steps))
    assert float(advantage[0, 0, 0]) == torch.tensor(total).item()


def test_a_perfect_critic_leaves_no_advantage() -> None:
    """A critic that is already right leaves nothing to learn.

    The update must then be a no-op rather than noise.
    """
    settings = ppo.Settings(discount=0.9, trace=0.95)
    steps = 8
    reward = torch.zeros(steps, 2, 3)
    value = torch.zeros(steps, 2, 3)
    advantage, _ = ppo.advantages(reward, value, torch.zeros(2, 3), settings)
    assert float(advantage.abs().max()) == 0.0


def _toy(rows: int, width: int) -> ppo.Rollout:
    """Return a rollout where one action is plainly better than the rest."""
    observation = torch.randn(4, rows, 1, width)
    action = torch.zeros(4, rows, 1, dtype=torch.long)
    advantage = torch.ones(4, rows, 1)
    return ppo.Rollout(
        observation=observation,
        action=action,
        log_probability=torch.full((4, rows, 1), -torch.log(torch.tensor(25.0)).item()),
        value=torch.zeros(4, rows, 1),
        reward=torch.ones(4, rows, 1),
        advantage=advantage,
        target=torch.ones(4, rows, 1),
    )


def test_the_update_moves_the_policy_toward_what_paid() -> None:
    """The one end-to-end check of the objective.

    Every transition says action zero carried a positive advantage, so after a
    few epochs action zero must be more likely than it was. A sign error in the
    surrogate passes every other test in this file and fails this one.
    """
    torch.manual_seed(0)
    width = 16
    policy = Decentralised(width, hidden=32, depth=1)
    rollout = _toy(rows=64, width=width)
    sample = rollout.observation.reshape(-1, width)

    with torch.no_grad():
        before = policy(sample)[0].softmax(dim=-1)[:, 0].mean()
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-2)
    # The advantage is constant, so normalising it would divide by zero: this
    # test drives `update` with a rollout it is not meant to normalise.
    rollout.advantage = rollout.advantage + torch.randn_like(rollout.advantage) * 1e-3
    ppo.update(policy, optimiser, rollout, ppo.Settings(epochs=8, minibatches=2))
    with torch.no_grad():
        after = policy(sample)[0].softmax(dim=-1)[:, 0].mean()

    assert float(after) > float(before)


def test_the_clip_bounds_how_far_one_update_can_go() -> None:
    """One update cannot go arbitrarily far.

    PPO's whole premise. Without the clip, a large advantage and a small
    probability send the ratio to infinity and the run never recovers.
    """
    torch.manual_seed(1)
    width = 16
    policy = Decentralised(width, hidden=32, depth=1)
    rollout = _toy(rows=128, width=width)
    rollout.advantage = rollout.advantage * 50.0 + torch.randn_like(rollout.advantage) * 1e-3
    optimiser = torch.optim.Adam(policy.parameters(), lr=1e-2)

    losses = ppo.update(policy, optimiser, rollout, ppo.Settings(epochs=4, minibatches=2))
    # The clipped objective cannot reward a ratio past 1 + clip, so the loss
    # bottoms out near -(1 + clip) times the normalised advantage.
    assert losses["policy"] > -10.0
    assert abs(losses["kl"]) < 5.0


def test_a_rollout_flattens_to_one_row_per_transition() -> None:
    rollout = _toy(rows=8, width=OBSERVATION)
    observation, action, log_probability, advantage, target = rollout.flatten()
    assert observation.shape == (4 * 8, OBSERVATION)
    for item in (action, log_probability, advantage, target):
        assert item.shape == (4 * 8,)


def test_a_learned_policy_can_be_filmed() -> None:
    """The one adapter between the learner and the renderer.

    The renderer drives the episode and hands a policy the fleet state; a
    learned policy needs the whole observation, which only the environment can
    assemble. If this breaks there is no before-and-after film, which is the
    project's acceptance criterion.
    """
    from baltic_isr.learn.env import Environment
    from baltic_isr.learn.train import Greedy
    from baltic_isr.render import trace as trace_module

    resolved = config.load("baseline").model_copy(
        update={
            "fleet": config.load("baseline").fleet.model_copy(update={"count": 6}),
            "simulation": config.load("baseline").simulation.model_copy(
                update={"episode_s": 3600.0}
            ),
        }
    )
    environment = Environment(resolved, 1, "cpu")
    greedy = Greedy(Decentralised(OBSERVATION), environment)
    assert greedy.wants_state
    recorded = trace_module.record(environment, greedy)
    assert recorded.frames > 0


def test_the_discount_is_a_distance() -> None:
    """The horizon has to be the scale of the problem, not a habit.

    0.99 is the reflex and would ask the critic to value a day and a half of
    future from one sample of it. The chosen discount has to see further than
    the sensor and less far than a whole sortie.
    """
    resolved = config.load("baseline")
    settings = ppo.Settings()
    decisions = 1.0 / (1.0 - settings.discount)
    metres = decisions * resolved.auv.regimes.cruise_ms * resolved.simulation.decision_interval_s
    detection = 5_900.0
    assert metres > 3.0 * detection
    assert metres < 0.6 * float(resolved.geo.sea_area_m2**0.5)
