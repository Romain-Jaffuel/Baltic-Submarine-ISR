"""The network each vehicle carries, and the one that would command them all.

**Parameter sharing, not one brain per vehicle.** Forty vehicles with forty
networks would each see a fortieth of the experience and learn the same thing
forty times over, badly. They are interchangeable -- same hull, same sensor,
same job -- so they share weights and differ only by what they observe. The
fleet's behaviour is then entirely a function of the picture each one holds,
which is exactly the claim the project wants to test: that coordination can
come out of a shared policy and a shared reward rather than out of a protocol
anyone wrote down.

The critic is separate from the actor rather than a head on a shared trunk.
Shared trunks are cheaper and usually fine, but here the value of a state is
dominated by how much stale picture the fleet is sitting on, while the action
depends on what is close -- two different readings of the same 209 numbers, and
letting them fight over one trunk costs more than the parameters save.
"""

from __future__ import annotations

import torch
from torch import nn

from baltic_isr.fleet import auv as auv_module


def _trunk(width: int, depth: int, hidden: int) -> nn.Sequential:
    """Return a plain MLP of the given shape, tanh throughout.

    Tanh rather than ReLU: PPO's trust region assumes the policy moves smoothly
    with its parameters, and a bounded, smooth activation keeps the logits from
    sliding far on a single update.
    """
    layers: list[nn.Module] = []
    size = width
    for _ in range(depth):
        layers += [nn.Linear(size, hidden), nn.Tanh()]
        size = hidden
    return nn.Sequential(*layers)


def _orthogonal(module: nn.Module, gain: float) -> nn.Module:
    """Initialise a linear layer orthogonally, which PPO is sensitive to."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)
    return module


class Decentralised(nn.Module):
    """One shared policy, run on every vehicle's own observation.

    Parameters
    ----------
    observation : int
        Width of a single vehicle's observation.
    hidden : int
        Width of the hidden layers.
    depth : int
        How many hidden layers.
    """

    def __init__(self, observation: int, hidden: int = 256, depth: int = 2) -> None:
        super().__init__()
        self.actions = auv_module.N_ACTIONS
        self.actor = _trunk(observation, depth, hidden)
        self.critic = _trunk(observation, depth, hidden)
        # A small gain on the action head so the policy starts near uniform.
        # Starting opinionated wastes the first updates unlearning an opinion
        # that came from the initialiser rather than from the world.
        self.head = _orthogonal(nn.Linear(hidden, self.actions), 0.01)
        self.value = _orthogonal(nn.Linear(hidden, 1), 1.0)
        for layer in list(self.actor) + list(self.critic):
            _orthogonal(layer, 2.0**0.5)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and state values.

        Parameters
        ----------
        observation : Tensor
            ``(..., observation)``.

        Returns
        -------
        tuple
            Logits ``(..., actions)`` and values ``(...)``.
        """
        return self.head(self.actor(observation)), self.value(self.critic(observation)).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one action per vehicle.

        Parameters
        ----------
        observation : Tensor
            ``(worlds, vehicles, observation)``.

        Returns
        -------
        tuple
            Actions, their log probabilities, and the critic's values.
        """
        logits, values = self(observation)
        # torch ships Categorical without annotations, so every call below
        # is untyped as far as mypy is concerned. The alternative is to
        # hand-roll a sampler, which would be worse code for no gain.
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()  # type: ignore[no-untyped-call]
        return actions, distribution.log_prob(actions), values  # type: ignore[no-untyped-call]

    def evaluate(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score actions already taken, for the update.

        Parameters
        ----------
        observation : Tensor
            ``(batch, observation)``.
        actions : Tensor
            ``(batch,)``.

        Returns
        -------
        tuple
            Log probabilities, entropies and values.
        """
        logits, values = self(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        return (
            distribution.log_prob(actions),  # type: ignore[no-untyped-call]
            distribution.entropy(),  # type: ignore[no-untyped-call]
            values,
        )
