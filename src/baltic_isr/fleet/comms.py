"""Talking under water, which mostly means not talking.

Specification v6 §5.4 said the binding constraint was the 250 bytes an acoustic
exchange carries. **Measured, it is not.** A prediction is 74 bits, so a packet
holds 27 of them, and at one exchange every five seconds a vehicle can push its
entire 180-track table through the pipe several times per decision. The pipe is
not the problem.

The problem is the **range**. Four kilometres, against a fleet whose vehicles sit
an average of 23.7 km apart: measured over a full episode, a vehicle has nobody
within earshot **61 % of the time**, and its mean number of neighbours is 0.53.

So this is not a network. It is a **delay-tolerant** one: information moves
because a vehicle physically carries it until it happens to pass someone. That
makes the surface burst worth its 7.7 minutes of deafness -- it is the only way
to reach the whole fleet at once -- and it makes *where you go* a communications
decision as much as a search one.

Two things the shore knows and a submerged vehicle cannot: AIS is VHF and does
not reach sixty metres down, so the two AIS signatures of specification §4.3
arrive only here, at the surface (docs/constants.md C-12-05). A vehicle that
surfaces learns which vessels have gone dark, and stops releasing them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from baltic_isr.config import Config
from baltic_isr.picture import tracks as tracks_module
from baltic_isr.picture.tracks import Tracks
from baltic_isr.sensing import acoustics
from baltic_isr.world.statics import Statics

# What one exchange carries. Each field is what it costs to say the thing, not
# a guess: a position quantised to 50 m over a 138 km box needs 12 bits an axis,
# a sigma to 10 % needs 6, a validity time to a minute over 48 h needs 10.
HEADER_BITS = 32
POSITION_BITS = 32
STATE_BITS = 8
PREDICTION_BITS = 74
PRESENCE_CELL_BITS = 20


@dataclass(frozen=True)
class Links:
    """Who can hear whom, right now.

    Attributes
    ----------
    reachable : Tensor
        ``(worlds, vehicles, vehicles)``, within modem range and both listening.
    success : Tensor
        ``(worlds, vehicles, vehicles)``, probability a packet gets through,
        range loss times the Doppler penalty.
    degree : Tensor
        ``(worlds, vehicles)``, how many neighbours each vehicle has.
    """

    reachable: torch.Tensor
    success: torch.Tensor
    degree: torch.Tensor

    @property
    def isolated(self) -> torch.Tensor:
        """Return which vehicles have nobody to talk to."""
        return self.degree == 0


def message_bits(predictions: int, presence_cells: int) -> int:
    """Return what a message of this shape costs.

    Parameters
    ----------
    predictions : int
        How many track predictions it carries.
    presence_cells : int
        How many search-grid updates it carries.

    Returns
    -------
    int
        Bits.
    """
    return (
        HEADER_BITS
        + POSITION_BITS
        + STATE_BITS
        + predictions * PREDICTION_BITS
        + presence_cells * PRESENCE_CELL_BITS
    )


class Comms:
    """The acoustic link and the surface burst.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry.
    generator : torch.Generator
        Seeded stream for packet loss.
    """

    def __init__(self, config: Config, statics: Statics, generator: torch.Generator) -> None:
        self._config = config
        self._statics = statics
        self._generator = generator
        self._range = acoustics.modem_range_m(config.comms)

    @property
    def payload_bits(self) -> int:
        """Return how many bits one exchange carries."""
        return int(self._config.comms.bitrate_bps * self._config.comms.slot_s)

    @property
    def predictions_per_message(self) -> int:
        """Return how many predictions fit in one exchange."""
        spare = self.payload_bits - HEADER_BITS - POSITION_BITS - STATE_BITS
        return int(spare // PREDICTION_BITS)

    def slot_s(self) -> float:
        """Return what one turn on the channel costs, guard included.

        Sound crosses the modem range in nearly three seconds, so a shared
        channel has to leave that long between speakers. The guard is larger
        than the slot.
        """
        return self._config.comms.slot_s + self._range / self._config.acoustics.sound_speed_ms

    def links(
        self, positions: torch.Tensor, velocity: torch.Tensor, listening: torch.Tensor
    ) -> Links:
        """Return who can hear whom.

        Parameters
        ----------
        positions : Tensor
            True positions, ``(worlds, vehicles, 2)``. Physics uses the truth
            even though the picture uses the estimate.
        velocity : Tensor
            Velocity over ground, ``(worlds, vehicles, 2)``, for the Doppler.
        listening : Tensor
            ``(worlds, vehicles)``; a surfaced or charging vehicle is off the
            acoustic net.

        Returns
        -------
        Links
            Reachability, packet success and degree.
        """
        comms = self._config.comms
        distance = torch.cdist(positions, positions)
        both = listening.unsqueeze(2) & listening.unsqueeze(1)
        vehicles = positions.shape[1]
        self_pair = torch.eye(vehicles, dtype=torch.bool, device=positions.device)
        reachable = (distance < self._range) & both & ~self_pair

        loss = (1.0 - (distance / self._range).clamp(0.0, 1.0) ** 3).clamp_min(0.0)
        closing = (velocity.unsqueeze(2) - velocity.unsqueeze(1)).norm(dim=-1)
        doppler = torch.exp(-((closing / comms.doppler_tolerance_ms) ** 2))
        success = torch.where(reachable, loss * doppler, torch.zeros_like(loss))
        return Links(reachable=reachable, success=success, degree=reachable.sum(-1))

    def exchange(self, tracks: Tracks, links: Links) -> tuple[Tracks, torch.Tensor]:
        """Take on what one neighbour knows, if anyone is in earshot.

        **One neighbour, not all of them.** With a mean degree of 0.53 the
        difference is small, and merging from every neighbour at once would need
        a tensor of worlds by vehicles by vehicles by vessels -- 590 MB at full
        batch, to serve a graph that is empty most of the time. The best link is
        taken instead, and the rest arrive on later steps.

        Propagation delay is ignored: 3.8 s across the modem range, against a
        900 s decision.

        Parameters
        ----------
        tracks : Tracks
            Every vehicle's table.
        links : Links
            The current graph.

        Returns
        -------
        tuple
            The merged tables, and which entries were taken from someone else.
        """
        draw = torch.rand(
            links.success.shape, generator=self._generator, device=links.success.device
        )
        delivered = torch.where(draw < links.success, links.success, torch.zeros_like(draw))
        quality, best = delivered.max(dim=-1)
        heard = quality > 0.0

        vessels = tracks.last_seen_s.shape[-1]
        source = best.unsqueeze(-1).expand(-1, -1, vessels)
        pair = source.unsqueeze(-1).expand(-1, -1, -1, 2)
        triple = source.unsqueeze(-1).expand(-1, -1, -1, 3)

        theirs = tracks.last_seen_s.gather(1, source)
        fresher = heard.unsqueeze(-1) & (theirs > tracks.last_seen_s)

        # A prediction someone else issued is the only thing in the model that
        # pays for talking: verifying it later is a relay (specification 6.4).
        return (
            replace(
                tracks,
                last_seen_s=torch.where(fresher, theirs, tracks.last_seen_s),
                xy=torch.where(fresher.unsqueeze(-1), tracks.xy.gather(1, pair), tracks.xy),
                velocity=torch.where(
                    fresher.unsqueeze(-1), tracks.velocity.gather(1, pair), tracks.velocity
                ),
                cov=torch.where(fresher.unsqueeze(-1), tracks.cov.gather(1, triple), tracks.cov),
                rail=torch.where(fresher, tracks.rail.gather(1, source), tracks.rail),
                progress=torch.where(fresher, tracks.progress.gather(1, source), tracks.progress),
                direction=torch.where(
                    fresher, tracks.direction.gather(1, source), tracks.direction
                ),
                growth_m_per_s=torch.where(
                    fresher, tracks.growth_m_per_s.gather(1, source), tracks.growth_m_per_s
                ),
                issuer=torch.where(fresher, tracks.issuer.gather(1, source), tracks.issuer),
                plausible=torch.where(
                    fresher, tracks.plausible.gather(1, source), tracks.plausible
                ),
                alarmed=tracks.alarmed | (fresher & tracks.alarmed.gather(1, source)),
            ),
            fresher,
        )

    def shore(self, worlds: int, vessels: int) -> Tracks:
        """Return an empty shore database.

        A base station is not a relay that has to be listening at the same
        moment: it is a **database**. Pooling only among vehicles that happen to
        be surfaced together is almost worthless, because they surface every
        twenty hours and independently -- measured, it lifted fleet-wide
        knowledge from 54 % to 57 %. Persisting what the shore has been told
        lifts it far higher, and it is also how a shore station actually works.

        Parameters
        ----------
        worlds : int
            How many worlds run in parallel.
        vessels : int
            How many vessels the world holds.

        Returns
        -------
        Tracks
            A one-row-per-world table, shaped like a single vehicle's.
        """
        return tracks_module.empty(worlds, 1, vessels, self._statics.device)

    def surface_exchange(
        self, tracks: Tracks, shore: Tracks, surfaced: torch.Tensor, ais_on: torch.Tensor
    ) -> tuple[Tracks, Tracks, torch.Tensor]:
        """Upload to the shore, download everything it has, and take its AIS.

        The one moment a vehicle is on the same page as the rest of the fleet.
        It uploads its table, the shore keeps it, and it downloads whatever the
        shore has accumulated from everybody since -- so a prediction made
        eleven hours ago on the far side of the box arrives in one hop instead
        of being carried across.

        The shore also has AIS receivers, which a submerged vehicle does not and
        cannot. A vessel that has gone dark stops being releasable: the vehicle
        will go and look rather than trust a prediction. It is not an alarm --
        an alarm needs an observation that contradicts a prediction -- but it is
        what sends someone to get one.

        Parameters
        ----------
        tracks : Tracks
            Every vehicle's table.
        surfaced : Tensor
            ``(worlds, vehicles)``, who is in the radio window.
        ais_on : Tensor
            ``(worlds, vessels)``, what the shore hears on AIS.

        Returns
        -------
        tuple
            The merged tables, and which entries came from the shore.
        """
        hidden = torch.tensor(-torch.inf, device=tracks.last_seen_s.device)
        vehicles = tracks.last_seen_s.shape[1]

        # Up: the shore takes the freshest thing any surfacing vehicle brings,
        # and keeps it whether or not anyone else is up at the same moment.
        offered = torch.where(surfaced.unsqueeze(-1), tracks.last_seen_s, hidden)
        newest, owner = offered.max(dim=1, keepdim=True)
        better = (newest > shore.last_seen_s) & torch.isfinite(newest)
        landed = _adopt(shore, tracks, owner.expand(-1, 1, -1), better)

        # Down: every surfacing vehicle takes whatever the shore now holds.
        held = landed.last_seen_s.expand(-1, vehicles, -1)
        fresher = surfaced.unsqueeze(-1) & (held > tracks.last_seen_s) & torch.isfinite(held)
        zero = torch.zeros_like(tracks.last_seen_s, dtype=torch.long)
        merged = _adopt(tracks, landed, zero, fresher)

        gone_dark = surfaced.unsqueeze(-1) & ~ais_on.unsqueeze(1) & merged.known
        return replace(merged, plausible=merged.plausible & ~gone_dark), landed, fresher


def _adopt(into: Tracks, source: Tracks, index: torch.Tensor, take: torch.Tensor) -> Tracks:
    """Copy selected entries of one table into another.

    Parameters
    ----------
    into : Tracks
        The table being updated.
    source : Tracks
        Where the entries come from.
    index : Tensor
        Which row of ``source`` each entry of ``into`` reads, same shape as
        ``take``.
    take : Tensor
        Which entries to replace.

    Returns
    -------
    Tracks
        The updated table.
    """
    pair = index.unsqueeze(-1).expand(-1, -1, -1, 2)
    triple = index.unsqueeze(-1).expand(-1, -1, -1, 3)
    wide = take.unsqueeze(-1)
    return replace(
        into,
        last_seen_s=torch.where(take, source.last_seen_s.gather(1, index), into.last_seen_s),
        xy=torch.where(wide, source.xy.gather(1, pair), into.xy),
        velocity=torch.where(wide, source.velocity.gather(1, pair), into.velocity),
        cov=torch.where(wide, source.cov.gather(1, triple), into.cov),
        rail=torch.where(take, source.rail.gather(1, index), into.rail),
        progress=torch.where(take, source.progress.gather(1, index), into.progress),
        direction=torch.where(take, source.direction.gather(1, index), into.direction),
        growth_m_per_s=torch.where(
            take, source.growth_m_per_s.gather(1, index), into.growth_m_per_s
        ),
        issuer=torch.where(take, source.issuer.gather(1, index), into.issuer),
        plausible=torch.where(take, source.plausible.gather(1, index), into.plausible),
        alarmed=into.alarmed | (take & source.alarmed.gather(1, index)),
    )
