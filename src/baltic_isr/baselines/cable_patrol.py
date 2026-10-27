"""A floor that knows what it is guarding.

The lawnmower sweeps the box. This sweeps the **cables**, and it exists for two
reasons.

First, as a test of the reward rather than of a policy. A reward term for
keeping cables listened to is a claim, and the way to check a claim like that
is to run a policy that does the thing and see whether the number moves. If a
fleet that patrols cables does not score higher on the cable term than one that
ignores them, the term is broken, and no amount of training would have revealed
it -- the policy would simply have learned whatever the term actually measures.

Second, as a harder floor. Beating a fleet that sweeps the box at random is
easy and proves little; beating one that already walks the right ground means
winning on *when* and *with whom*, which is the part that needs coordination.

Scripted, stateless and reproducible, like the lawnmower: the slice of cable
comes from the vehicle index and the phase from the clock.
"""

from __future__ import annotations

import math

import torch

from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet.auv import FleetState
from baltic_isr.world.statics import Statics


class CablePatrol:
    """Each vehicle walks its own slice of the cable network.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, for the cable samples.
    """

    def __init__(self, config: Config, statics: Statics) -> None:
        self._config = config
        self._statics = statics
        self._device = statics.device
        count = config.fleet.count

        # The cable network as one path, cut into one slice per vehicle. The
        # samples are already ordered along each cable and the cables are
        # concatenated, so consecutive indices are neighbours except at the
        # four joins -- which costs a transit nobody has to model.
        samples = statics.cable_xy
        edges = torch.linspace(0.0, float(samples.shape[0]), count + 1, device=self._device)
        self._first = edges[:-1].floor().to(torch.long)
        self._last = (edges[1:].ceil().to(torch.long) - 1).clamp_min(self._first)

        # One full out-and-back per sortie, staggered, so neighbouring slices
        # are not walked in step.
        span = (samples.max(dim=0).values - samples.min(dim=0).values).norm()
        self._leg_s = float(span) / count / config.auv.regimes.cruise_ms

    def target(self, t: float, vehicles: int) -> torch.Tensor:
        """Return the cable sample each vehicle is walking toward.

        Parameters
        ----------
        t : float
            Simulation time.
        vehicles : int
            How many vehicles.

        Returns
        -------
        Tensor
            ``(vehicles, 2)`` in local metres.
        """
        phase = (
            t / max(self._leg_s, 1.0) + torch.arange(vehicles, device=self._device) / vehicles
        ) % 2.0
        along = torch.where(phase < 1.0, phase, 2.0 - phase)
        first, last = self._first[:vehicles], self._last[:vehicles]
        index = (first + along * (last - first).to(along.dtype)).round().to(torch.long)
        return self._statics.cable_xy[index.clamp(0, self._statics.cable_xy.shape[0] - 1)]

    def act(self, state: FleetState, t: float) -> torch.Tensor:
        """Return one action per vehicle.

        Parameters
        ----------
        state : FleetState
            Current fleet state.
        t : float
            Simulation time.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)`` actions.
        """
        target = self.target(t, state.xy.shape[1]).unsqueeze(0)
        direction = target - state.estimate
        angle = torch.atan2(direction[..., 0], direction[..., 1]) % (2.0 * math.pi)
        heading = torch.round(angle / (2.0 * math.pi / auv_module.HEADINGS)).long()
        heading = heading % auv_module.HEADINGS

        # Cruise on the way, and **drift once it is close enough to listen**.
        # The lawnmower never touches the regime; this one makes the single
        # choice its job implies, which is to stop driving when sitting still
        # hears better. It is still not a tactical policy -- it reacts to
        # nothing, and converges on nothing.
        reach = self._config.fleet.gate_m
        arrived = direction.norm(dim=-1) < reach
        regime = torch.where(
            arrived,
            torch.full_like(heading, auv_module.REGIME_DRIFT),
            torch.full_like(heading, auv_module.REGIME_CRUISE),
        )
        return regime * auv_module.HEADINGS + heading
