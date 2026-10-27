"""The floor: a fleet that sweeps its own patch and talks to nobody.

Each vehicle owns a strip of the box and walks it end to end, turning round at
the edges. It never reacts to a contact, never converges on an alarm, never
goes to meet anybody. It is not a straw man -- a sector search is what you
actually do when you have no coordination worth the name, and it covers ground
efficiently.

That is exactly why it is the right floor. It gets the **coverage** part of the
task for free, so a learned policy that beats it has to be winning on the part
that needs coordination: verifying before a prediction expires, and converging
on what has gone wrong. A policy that only learns to spread out will tie.

Scripted, stateless and reproducible: the strip assignment comes from the
vehicle index, and the phase from the clock.
"""

from __future__ import annotations

import math

import torch

from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet.auv import FleetState
from baltic_isr.world import field
from baltic_isr.world.statics import Statics


class Lawnmower:
    """A sector search, one strip per vehicle.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, for the box.
    """

    def __init__(self, config: Config, statics: Statics) -> None:
        self._config = config
        self._statics = statics
        self._device = statics.device
        count = config.fleet.count
        # Strips across the long axis, each vehicle on the centreline of its
        # own. Two vehicles never share a strip, so the sweep is a partition.
        index = torch.arange(count, dtype=torch.float32, device=self._device)
        self._strip_x = (index + 0.5) / count * float(statics.extent[0])
        self._headings = torch.arange(auv_module.HEADINGS, device=self._device) * (
            2.0 * math.pi / auv_module.HEADINGS
        )

        # Each strip ends where the water does. Sweeping the full height of the
        # box instead sent every vehicle at the coast twice a leg and left it
        # grinding along the clearance line until the phase turned it round:
        # 31 % of all fleet positions were pinned against the shore, and the
        # worst vehicle spent half the episode there. That is not a dumb
        # baseline, it is a broken one, and it would have flattered the learned
        # policy for the wrong reason.
        # Where the navigable water starts and stops along each strip, read
        # off the measured seabed one column at a time. The old version took
        # the signed distance at the box edge and added a shelf-derived
        # clearance, which assumed the shore was a function of easting. It is
        # not: the north side of this box is an archipelago, and a strip that
        # ends at "the coastline" there ends in the middle of the islands.
        height = float(statics.extent[1])
        northings = torch.arange(0.0, height, statics.seabed_cell_m, device=statics.device)
        column = torch.stack(
            [
                self._strip_x.unsqueeze(-1).expand(-1, northings.numel()),
                northings.unsqueeze(0).expand(self._strip_x.numel(), -1),
            ],
            dim=-1,
        )
        navigable = field.has_clearance(statics, column, config.fleet.min_water_m)
        # A strip with no navigable water at all collapses to the box centre,
        # which the clamp below produces on its own: first > last.
        index = torch.arange(northings.numel(), device=statics.device)
        far = torch.full_like(index, northings.numel())
        first = torch.where(navigable, index, far).min(dim=-1).values
        last = torch.where(navigable, index, torch.zeros_like(index)).max(dim=-1).values
        self._strip_south = northings[first.clamp(max=northings.numel() - 1)]
        self._strip_north = northings[last]

    def target(self, t: float, vehicles: int) -> torch.Tensor:
        """Return the point each vehicle is walking toward.

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
        south = self._strip_south[:vehicles]
        north = self._strip_north[:vehicles]
        # One full up-and-down per leg time, staggered so neighbouring strips
        # are not in step -- a fleet marching in line abreast covers the same
        # ground as a fleet spread along its strips, and looks sillier.
        leg_s = float((north - south).mean()) / self._config.auv.regimes.cruise_ms
        phase = (t / leg_s + torch.arange(vehicles, device=self._device) / vehicles) % 2.0
        along = torch.where(phase < 1.0, phase, 2.0 - phase)
        return torch.stack([self._strip_x[:vehicles], torch.lerp(south, north, along)], dim=-1)

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

        # Cruise, always. A lawnmower has no reason to sprint and no reason to
        # drift: both are tactical choices and it makes none.
        return auv_module.REGIME_CRUISE * auv_module.HEADINGS + heading
