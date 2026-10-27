"""The floor beneath the floor: a fleet that decides nothing.

Every comparison needs a zero. The lawnmower already sweeps the box sensibly
and the cable patrol already walks the right ground, so both are *competent*
floors -- beating them means winning on coordination rather than on having any
plan at all. Neither answers the simpler question a reader asks first: how
much of what I am looking at is the policy, and how much is the simulator?

This answers it. Uniform over all twenty-five actions, every decision, every
vehicle. Whatever the fleet still achieves here is what the world does on its
own: the traffic runs in lanes, the sensors hear, the shore database collects
what drifts past it. That is the number everything else must be measured from.

It is not the untrained network, which starts near-uniform but not uniform and
drifts the moment it is touched. Chance is chance.
"""

from __future__ import annotations

import torch

from baltic_isr import seeds
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet.auv import FleetState
from baltic_isr.world.statics import Statics


class Chance:
    """Uniform random actions, reproducibly.

    Parameters
    ----------
    config : Config
        Resolved configuration, for the seed.
    statics : Statics
        Device geometry, for the device only.
    """

    def __init__(self, config: Config, statics: Statics) -> None:
        self._device = statics.device
        self._generator = seeds.torch_generator(config.simulation.seed, "chance", self._device)

    def act(self, state: FleetState, t: float) -> torch.Tensor:
        """Return one uniformly random action per vehicle.

        Parameters
        ----------
        state : FleetState
            Current fleet state, for its shape.
        t : float
            Simulation time. Unused: chance has no memory and no clock.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)`` in ``[0, N_ACTIONS)``.
        """
        del t
        return torch.randint(
            auv_module.N_ACTIONS,
            state.regime.shape,
            generator=self._generator,
            device=self._device,
        )
