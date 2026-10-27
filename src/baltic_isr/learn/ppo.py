"""PPO over a batch of worlds, with every vehicle an agent.

The shape of the problem decides the shape of the algorithm. One decision of
one world yields forty transitions, all from the same shared policy, so the
natural batch is ``worlds x vehicles`` and the natural rollout is a handful of
decisions rather than a whole episode: storing an episode's observations at
W=1024 would want six terabytes, while thirty-two decisions at W=256 want two
hundred and seventy megabytes.

**The discount is a distance.** A vehicle at cruise makes 1 080 m between
decisions, so a discount of 0.95 -- a horizon of twenty decisions -- is about
twenty kilometres of lookahead. That is the right scale for this problem: it is
three times the detection range and half the transit to base, so the policy can
learn to set off towards something it cannot yet hear, and cannot learn to plan
a whole sortie, which the forced recall handles anyway. Picking the discount by
habit (0.99, a hundred-decision horizon) would be asking the critic to value a
day and a half of future from one sample of it.

Nothing here synchronises with the host inside the rollout. The whole point of
a vectorised simulator is lost if the training loop reads a tensor back every
decision to decide what to do next.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baltic_isr.learn.env import Environment, EnvState
from baltic_isr.learn.policy import Decentralised


@dataclass(frozen=True)
class Settings:
    """How the optimiser is run.

    Attributes
    ----------
    rollout : int
        Decisions collected before each update.
    epochs : int
        Passes over each rollout.
    minibatches : int
        How many pieces each pass is cut into.
    discount : float
        ``gamma``. See the module docstring: it is a distance.
    trace : float
        GAE ``lambda``.
    clip : float
        PPO's trust region.
    value_weight, entropy_weight : float
        Loss mixing.
    learning_rate, max_grad_norm : float
        Optimiser.
    """

    rollout: int = 32
    epochs: int = 4
    minibatches: int = 8
    discount: float = 0.95
    trace: float = 0.95
    clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass
class Rollout:
    """One batch of experience, laid out ``(steps, worlds, vehicles, ...)``."""

    observation: torch.Tensor
    action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    advantage: torch.Tensor
    target: torch.Tensor

    def flatten(self) -> tuple[torch.Tensor, ...]:
        """Return the batch as flat rows, one per transition."""
        width = self.observation.shape[-1]
        return (
            self.observation.reshape(-1, width),
            self.action.reshape(-1),
            self.log_probability.reshape(-1),
            self.advantage.reshape(-1),
            self.target.reshape(-1),
        )


def advantages(
    reward: torch.Tensor, value: torch.Tensor, last: torch.Tensor, settings: Settings
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GAE advantages and value targets.

    No ``done`` mask. The episode is a fixed 48 h and the rollout is truncated
    well inside it, so every step bootstraps from the next; the one boundary
    per episode is handled by resetting between rollouts rather than by
    carrying a flag that is false 99.5 % of the time.

    Parameters
    ----------
    reward, value : Tensor
        ``(steps, worlds, vehicles)``.
    last : Tensor
        Value of the state after the final step, ``(worlds, vehicles)``.
    settings : Settings
        Discount and trace.

    Returns
    -------
    tuple
        Advantages and targets, both ``(steps, worlds, vehicles)``.
    """
    steps = reward.shape[0]
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(last)
    following = last
    for step in range(steps - 1, -1, -1):
        delta = reward[step] + settings.discount * following - value[step]
        running = delta + settings.discount * settings.trace * running
        advantage[step] = running
        following = value[step]
    return advantage, advantage + value


def collect(
    environment: Environment,
    policy: Decentralised,
    state: EnvState,
    observation: torch.Tensor,
    settings: Settings,
) -> tuple[EnvState, torch.Tensor, Rollout]:
    """Run the policy for ``settings.rollout`` decisions and score the result.

    Parameters
    ----------
    environment : Environment
        The simulator.
    policy : Decentralised
        The current policy.
    state : EnvState
        Where to carry on from.
    observation : Tensor
        The observation of ``state``.
    settings : Settings
        Rollout length and discount.

    Returns
    -------
    tuple
        The state and observation to carry on from, and the rollout.
    """
    observations, actions, log_probabilities, values, rewards = [], [], [], [], []
    for _ in range(settings.rollout):
        action, log_probability, value = policy.act(observation)
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        state, result = environment.step(state, action)
        observation = result.observation
        rewards.append(result.reward)

    with torch.no_grad():
        _, last = policy(observation)

    reward = torch.stack(rewards)
    value = torch.stack(values)
    advantage, target = advantages(reward, value, last, settings)
    return (
        state,
        observation,
        Rollout(
            observation=torch.stack(observations),
            action=torch.stack(actions),
            log_probability=torch.stack(log_probabilities),
            value=value,
            reward=reward,
            advantage=advantage,
            target=target,
        ),
    )


def update(
    policy: Decentralised,
    optimiser: torch.optim.Optimizer,
    rollout: Rollout,
    settings: Settings,
) -> dict[str, float]:
    """Apply the clipped surrogate objective to one rollout.

    Returns
    -------
    dict
        Mean losses and the approximate KL, for watching the run.
    """
    observation, action, old_log, advantage, target = rollout.flatten()
    # Normalised per update, not per minibatch: a minibatch of a few thousand
    # transitions from one moment of one policy has a mean that says more about
    # the moment than about the action.
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-8)

    rows = observation.shape[0]
    size = max(rows // settings.minibatches, 1)
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
    batches = 0

    for _ in range(settings.epochs):
        order = torch.randperm(rows, device=observation.device)
        for start in range(0, rows, size):
            index = order[start : start + size]
            log_probability, entropy, value = policy.evaluate(observation[index], action[index])
            ratio = (log_probability - old_log[index]).exp()
            unclipped = ratio * advantage[index]
            clipped = ratio.clamp(1.0 - settings.clip, 1.0 + settings.clip) * advantage[index]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (value - target[index]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + settings.value_weight * value_loss
                - settings.entropy_weight * entropy_loss
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]  # torch stubs
            torch.nn.utils.clip_grad_norm_(policy.parameters(), settings.max_grad_norm)
            optimiser.step()

            with torch.no_grad():
                totals["policy"] += float(policy_loss)
                totals["value"] += float(value_loss)
                totals["entropy"] += float(entropy_loss)
                totals["kl"] += float((old_log[index] - log_probability).mean())
            batches += 1

    return {name: total / max(batches, 1) for name, total in totals.items()}
