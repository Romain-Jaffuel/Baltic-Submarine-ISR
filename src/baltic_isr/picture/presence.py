"""Where a vehicle has not looked lately.

The track table (``tracks.py``) is about vessels somebody has heard. This is the
other half: the belief that something is out there that nobody has heard yet. It
is what sends the fleet to look at empty water, and without it a policy paid
only on tracks would never search.

Belief is **per vehicle and stale by construction**, like everything else the
fleet holds. Sweeping a cell knocks its probability down; time lets it recover
toward the prior, because a patch of sea checked eleven hours ago is not checked.

The prior is not uniform. Traffic runs in lanes, so the chance of finding
something is higher near one, and a fleet that spread itself evenly over the
water would be spreading itself evenly over mostly nothing.

Cells are six kilometres, which is the cruise detection range: one sweep covers
about one cell. Finer would be a bigger tensor describing the same knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baltic_isr.config import Config
from baltic_isr.world import field
from baltic_isr.world.statics import Statics, sample_raster

PRESENCE_CELL_M = 6000.0


@dataclass(frozen=True)
class Presence:
    """One search picture per vehicle.

    Attributes
    ----------
    probability : Tensor
        ``(worlds, vehicles, cells)``, chance of an unheard vessel being there.
    age_s : Tensor
        ``(worlds, vehicles, cells)``, how long since this vehicle last swept it.
    """

    probability: torch.Tensor
    age_s: torch.Tensor


class Search:
    """The presence grid, its decay and its sweeps.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry.
    worlds : int
        How many worlds run in parallel.
    vehicles : int
        How many vehicles each world holds.
    """

    def __init__(self, config: Config, statics: Statics, worlds: int, vehicles: int) -> None:
        self._config = config
        self._statics = statics
        self._worlds = worlds
        self._vehicles = vehicles
        device = statics.device

        width, height = float(statics.extent[0]), float(statics.extent[1])
        self._columns = int(width // PRESENCE_CELL_M) + 1
        self._rows = int(height // PRESENCE_CELL_M) + 1
        axis_x = (torch.arange(self._columns, device=device) + 0.5) * PRESENCE_CELL_M
        axis_y = (torch.arange(self._rows, device=device) + 0.5) * PRESENCE_CELL_M
        grid = torch.stack(torch.meshgrid(axis_x, axis_y, indexing="xy"), dim=-1)
        self._centres = grid.reshape(-1, 2)

        # The prior. Water only, and weighted toward the lanes, because that is
        # where the traffic is and a flat prior would send the fleet to look at
        # the emptiest parts of the box.
        # The prior. Water only, weighted toward the lanes -- that is where the
        # traffic is, and a flat prior would send the fleet to look at the
        # emptiest parts of the box -- and, if asked, toward the cables, which
        # is what the traffic might be there to damage. The lane term alone
        # pointed the fleet at where ships *are*; the cable term points it at
        # where it would matter. See docs/constants.md C-12-04.
        at_sea = field.is_sea(statics, self._centres)
        settings = config.prediction
        scale = 3.0 * PRESENCE_CELL_M
        to_lane = sample_raster(statics, statics.rail_distance_m, self._centres)
        to_cable = sample_raster(statics, statics.corridor_distance_m, self._centres)
        weight = (
            settings.search_floor
            + settings.search_lane_weight * torch.exp(-to_lane / scale)
            + settings.search_cable_weight * torch.exp(-to_cable / scale)
        )
        self._prior = torch.where(at_sea, weight, torch.zeros_like(weight))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid as ``(rows, columns)``."""
        return (self._rows, self._columns)

    @property
    def centres(self) -> torch.Tensor:
        """Return the centre of every cell, ``(cells, 2)`` in local metres."""
        return self._centres

    @property
    def prior(self) -> torch.Tensor:
        """Return the resting belief of every cell, ``(cells,)``."""
        return self._prior

    def empty(self) -> Presence:
        """Return a picture that has never been swept.

        Returns
        -------
        Presence
            Every cell at its prior, every cell infinitely stale.
        """
        shape = (self._worlds, self._vehicles, self._centres.shape[0])
        return Presence(
            probability=self._prior.expand(shape).clone(),
            age_s=torch.zeros(shape, device=self._statics.device),
        )

    def step(
        self,
        presence: Presence,
        positions: torch.Tensor,
        listening: torch.Tensor,
        reach_m: torch.Tensor,
        dt_s: float,
    ) -> tuple[Presence, torch.Tensor]:
        """Sweep, then let the unswept recover.

        Parameters
        ----------
        presence : Presence
            The current pictures.
        positions : Tensor
            Where each vehicle believes it is, ``(worlds, vehicles, 2)``. Its own
            navigation error goes into its search picture too.
        listening : Tensor
            ``(worlds, vehicles)``; a surfaced vehicle sweeps nothing.
        reach_m : Tensor
            Detection range of each vehicle right now, ``(worlds, vehicles)``.
            It depends on the regime, so sprinting sweeps a much smaller disc.
        dt_s : float
            Step length.

        Returns
        -------
        tuple
            The updated pictures, and the belief each vehicle removed -- which is
            the information-gain term of specification section 6.4.
        """
        distance = torch.cdist(positions, self._centres.expand(positions.shape[0], -1, -1))
        inside = (distance <= reach_m.unsqueeze(-1)) & listening.unsqueeze(-1)

        # What a sweep is worth falls off across the disc the same way detection
        # does, so the edge of the range is not treated as a certainty.
        confidence = torch.exp(-((distance / reach_m.unsqueeze(-1).clamp_min(1.0)) ** 2))
        removed = torch.where(inside, presence.probability * confidence, 0.0)
        probability = presence.probability - removed

        recovery = 1.0 - torch.exp(
            torch.tensor(
                -dt_s / self._config.prediction.presence_recovery_s, device=distance.device
            )
        )
        probability = probability + (self._prior - probability) * recovery
        age = torch.where(inside, torch.zeros_like(presence.age_s), presence.age_s + dt_s)
        return Presence(probability=probability, age_s=age), removed.sum(dim=-1)
