"""The picture each vehicle holds, and why two listeners beat one twice over.

A track is not a position, it is a **covariance**. That distinction is the whole
of this module, and it is what makes coordination fall out of the physics rather
than out of a reward term someone bolted on.

A passive contact at five kilometres is not a blob. Bearing is good to two
degrees, which is 175 m across the line of sight; range is good to a quarter,
which is 1250 m along it. So one listener produces an ellipse seven times longer
than it is wide, pointing straight at itself. Observing the same contact again
from the *same* place fuses two ellipses of the same shape and barely helps.
Observing it from ninety degrees away fuses two ellipses at right angles, and
each one's good axis covers the other's bad one.

```
one listener          two listeners, same bearing     two at ninety degrees
                                                              |
  ---------                  --------                       -----
  (1250 x 175)               (884 x 124)                   (173 x 173)
```

**Major axis: 1250 m, then 884 m, then 173 m.** Triangulating is worth 2.9 times
what repeating is worth, and that is arithmetic rather than policy.

On its own it is not enough to make a fleet converge: joining a contact someone
already holds earns 0.36 of a staleness gate, while going to find a stale one
earns a whole gate. Dispersing still wins, which is correct for traffic and
useless as a demonstration.

What settles it is :func:`Picture.gate_m`. A contact that has contradicted its
prediction stops being traffic and becomes a problem, and a problem is localised
to a different standard -- three hundred metres, not three kilometres. Against
that gate the same triangulation is worth ten times more, and converging
dominates. The fleet closes on what it has found and disperses again once the
ellipse is tight, which is what coordination looks like from above.

Covariances are carried packed as ``(xx, xy, yy)``: they are symmetric, and a
fourth number would be 15 MB of duplication at full batch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from baltic_isr.config import Config
from baltic_isr.picture import prediction
from baltic_isr.world.statics import Statics, sample_rail, sample_rail_tangent

NO_RAIL = -1
NO_ISSUER = -1

# Wider than the box: what a vehicle believes about a contact it has never
# heard. Not zero -- zero is a singular covariance, which means certainty.
UNKNOWN_SIGMA_M = 1.0e6


def inverse(cov: torch.Tensor) -> torch.Tensor:
    """Invert a packed symmetric 2x2.

    Parameters
    ----------
    cov : Tensor
        ``(..., 3)`` holding ``xx, xy, yy``.

    Returns
    -------
    Tensor
        The inverse, packed the same way.
    """
    xx, xy, yy = cov[..., 0], cov[..., 1], cov[..., 2]
    # The floor has to be RELATIVE. An information matrix is the inverse of a
    # covariance in square metres, so its determinant is around 1e-11; an
    # absolute floor of 1e-9 silently replaced it and made every fusion look
    # five times better than it is. Guard against a degenerate ellipse instead,
    # which is what the floor is actually for.
    product = (xx * yy).clamp_min(0.0)
    determinant = (xx * yy - xy * xy).clamp_min(1e-6 * product + 1e-300)
    return torch.stack([yy, -xy, xx], dim=-1) / determinant.unsqueeze(-1)


def apply(cov: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Multiply a packed symmetric 2x2 by a vector.

    Parameters
    ----------
    cov : Tensor
        ``(..., 3)``.
    vector : Tensor
        ``(..., 2)``.

    Returns
    -------
    Tensor
        ``(..., 2)``.
    """
    xx, xy, yy = cov[..., 0], cov[..., 1], cov[..., 2]
    return torch.stack(
        [xx * vector[..., 0] + xy * vector[..., 1], xy * vector[..., 0] + yy * vector[..., 1]],
        dim=-1,
    )


def major_axis_m(cov: torch.Tensor) -> torch.Tensor:
    """Return the longest semi-axis of the uncertainty ellipse.

    The honest scalar for "how well do I know where this is": it is the error in
    the *worst* direction, which is the one that decides whether anyone sent to
    look will find anything. Using a trace instead would let a long thin ellipse
    masquerade as a good fix.

    Parameters
    ----------
    cov : Tensor
        ``(..., 3)``.

    Returns
    -------
    Tensor
        ``(...)`` in metres.
    """
    xx, xy, yy = cov[..., 0], cov[..., 1], cov[..., 2]
    half = 0.5 * (xx + yy)
    spread = torch.sqrt((0.5 * (xx - yy)) ** 2 + xy * xy)
    return torch.sqrt((half + spread).clamp_min(0.0))


def mahalanobis(offset: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Return the normalised distance of an offset under a covariance.

    Parameters
    ----------
    offset : Tensor
        ``(..., 2)``.
    cov : Tensor
        ``(..., 3)``.

    Returns
    -------
    Tensor
        ``(...)``, in sigmas.
    """
    weighted = apply(inverse(cov), offset)
    return torch.sqrt((offset * weighted).sum(dim=-1).clamp_min(0.0))


def measurement_cov(
    listener: torch.Tensor, contact: torch.Tensor, along_fraction: float, across_rad: float
) -> torch.Tensor:
    """Return the covariance of one passive fix, in the listener's geometry.

    Long along the line of sight, short across it. The orientation is what makes
    triangulation pay.

    Parameters
    ----------
    listener : Tensor
        ``(..., 2)``.
    contact : Tensor
        ``(..., 2)``.
    along_fraction : float
        Range error as a fraction of range.
    across_rad : float
        Bearing error in radians.

    Returns
    -------
    Tensor
        ``(..., 3)`` packed covariance.
    """
    offset = contact - listener
    distance = offset.norm(dim=-1).clamp_min(1.0)
    unit = offset / distance.unsqueeze(-1)
    along = (distance * along_fraction) ** 2
    across = (distance * across_rad) ** 2
    ux, uy = unit[..., 0], unit[..., 1]
    return torch.stack(
        [
            along * ux * ux + across * uy * uy,
            (along - across) * ux * uy,
            along * uy * uy + across * ux * ux,
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class Tracks:
    """One table per vehicle, indexed by vessel.

    Association is by gate, not by hypothesis: an observation that falls inside
    a contact's own ellipse updates it, and one that falls outside is a
    deviation. Full data association is out of scope -- specification section 0
    keeps fine tracking out of the project -- and folding it in would cost a
    hypothesis tree per vehicle for no gain to the question being asked.

    All tensors lead with ``(worlds, vehicles, vessels)``.

    Attributes
    ----------
    last_seen_s : Tensor
        When this vehicle last had a fix. ``-inf`` if never.
    xy, velocity : Tensor
        The state at that moment, ``(..., 2)``.
    cov : Tensor
        Position covariance at that moment, packed ``(..., 3)``.
    rail, progress, direction : Tensor
        Where on a lane the contact was, for the lane predictor. ``rail`` is
        ``-1`` for a contact that is not on one.
    growth_m_per_s : Tensor
        How fast this contact's ellipse grows, from its observed class.
    issuer : Tensor
        Which vehicle produced the prediction currently held. A verification
        performed by anyone else is a relay, and that is the only thing in the
        model that pays for talking.
    plausible : Tensor
        Whether the plausibility model cleared it for release.
    alarmed : Tensor
        Whether this vehicle has seen this contact contradict its own prediction.
    """

    last_seen_s: torch.Tensor
    xy: torch.Tensor
    velocity: torch.Tensor
    cov: torch.Tensor
    rail: torch.Tensor
    progress: torch.Tensor
    direction: torch.Tensor
    growth_m_per_s: torch.Tensor
    issuer: torch.Tensor
    plausible: torch.Tensor
    alarmed: torch.Tensor

    @property
    def known(self) -> torch.Tensor:
        """Return which contacts this vehicle has ever held."""
        return torch.isfinite(self.last_seen_s)


def empty(worlds: int, vehicles: int, vessels: int, device: torch.device | str = "cpu") -> Tracks:
    """Return a table holding nothing.

    Parameters
    ----------
    worlds, vehicles, vessels : int
        Table shape.
    device : torch.device or str
        Where to place it.

    Returns
    -------
    Tracks
        An empty table.
    """
    shape = (worlds, vehicles, vessels)
    zeros = torch.zeros(shape, device=device)
    # An unheld contact is not known perfectly, it is not known at all, so its
    # covariance is enormous rather than zero. Zero is singular, and a singular
    # matrix in the fusion produces infinities that leak through the masks.
    ignorance = torch.full(shape, UNKNOWN_SIGMA_M**2, device=device)
    return Tracks(
        last_seen_s=torch.full(shape, -torch.inf, device=device),
        xy=torch.zeros(*shape, 2, device=device),
        velocity=torch.zeros(*shape, 2, device=device),
        cov=torch.stack([ignorance, torch.zeros_like(ignorance), ignorance], dim=-1),
        rail=torch.full(shape, NO_RAIL, dtype=torch.int8, device=device),
        progress=zeros.clone(),
        direction=torch.ones(shape, device=device),
        growth_m_per_s=torch.ones(shape, device=device),
        issuer=torch.full(shape, NO_ISSUER, dtype=torch.int16, device=device),
        plausible=torch.zeros(shape, dtype=torch.bool, device=device),
        alarmed=torch.zeros(shape, dtype=torch.bool, device=device),
    )


class Picture:
    """Prediction, fusion, staleness and deviation, batched.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, for the lane predictor.
    """

    def __init__(self, config: Config, statics: Statics) -> None:
        self._config = config
        self._statics = statics
        self._rail_growth = config.prediction.rail_growth_m_per_s

    def predict(self, tracks: Tracks, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Push every held contact forward to now.

        Lane-bound contacts cleared as plausible run along their lane; everything
        else runs in a straight line. The ellipse grows at the rate measured for
        that contact's class, and it grows **anisotropically** for a lane-bound
        one: along the lane, where the speed is uncertain, not across it.

        Parameters
        ----------
        tracks : Tracks
            The table.
        t : float
            Now.

        Returns
        -------
        tuple of Tensor
            Predicted positions ``(..., 2)`` and covariances ``(..., 3)``.
        """
        elapsed = (t - tracks.last_seen_s).clamp_min(0.0)
        finite = torch.where(torch.isfinite(elapsed), elapsed, torch.zeros_like(elapsed))
        on_lane = (tracks.rail >= 0) & tracks.plausible

        straight = prediction.constant_velocity(tracks.xy, tracks.velocity, finite)
        speed = tracks.velocity.norm(dim=-1)

        # Advance along the lane once, then read both the point and the tangent
        # off it. Getting the tangent by predicting again a second later and
        # differencing cost a whole second pass, and this is the hot path: it
        # was 54 % of a simulation step.
        rail = tracks.rail.to(torch.long)
        index = rail.clamp_min(0)
        advanced = (
            tracks.progress + tracks.direction * speed * finite / self._statics.rail_length[index]
        ).clamp(0.0, 1.0)
        lane = sample_rail(self._statics, index, advanced)
        heading = sample_rail_tangent(self._statics, index, advanced) * tracks.direction.unsqueeze(
            -1
        )
        position = torch.where(on_lane.unsqueeze(-1), lane, straight)
        lane_sigma = self._rail_growth * finite
        blind_sigma = tracks.growth_m_per_s * finite

        hx, hy = heading[..., 0], heading[..., 1]
        along = lane_sigma**2
        across = (0.15 * lane_sigma) ** 2
        lane_cov = torch.stack(
            [
                along * hx * hx + across * hy * hy,
                (along - across) * hx * hy,
                along * hy * hy + across * hx * hx,
            ],
            dim=-1,
        )
        round_cov = torch.stack(
            [blind_sigma**2, torch.zeros_like(blind_sigma), blind_sigma**2], dim=-1
        )
        grown = torch.where(on_lane.unsqueeze(-1), lane_cov, round_cov)
        return position, tracks.cov + grown

    def update(
        self,
        tracks: Tracks,
        t: float,
        detected: torch.Tensor,
        measured: torch.Tensor,
        listener: torch.Tensor,
        plausible: torch.Tensor,
        growth: torch.Tensor,
        rail: torch.Tensor,
        progress: torch.Tensor,
        direction: torch.Tensor,
        velocity: torch.Tensor,
        listener_sigma_m: torch.Tensor,
    ) -> tuple[Tracks, torch.Tensor]:
        """Fuse this step's fixes, and flag the contacts that contradicted them.

        Fusion is an information filter, which is what makes triangulation pay:
        ``P_new^-1 = P_pred^-1 + R^-1``. Two fixes taken from ninety degrees
        apart have inverses whose strong directions are orthogonal, so they add
        where each is weak.

        Parameters
        ----------
        tracks : Tracks
            The table.
        t : float
            Now.
        detected : Tensor
            ``(worlds, vehicles, vessels)``.
        measured : Tensor
            Where each vehicle places each contact, ``(..., 2)``.
        listener : Tensor
            Each vehicle's believed position, ``(worlds, vehicles, 2)``.
        plausible : Tensor
            What the plausibility model concluded, ``(worlds, vehicles, vessels)``.
        growth : Tensor
            Ellipse growth rate implied by the observed class.
        rail, progress, direction : Tensor
            Lane fix, for contacts found on one.
        velocity : Tensor
            Observed velocity, ``(..., 2)``.
        listener_sigma_m : Tensor
            How far out the observer believes its own position to be,
            ``(worlds, vehicles)``. A fix is only as good as the place it was
            taken from, and leaving this out made the ellipse claim an accuracy
            the vehicle had no right to: 96 % of tracks ended up contradicted,
            because every observation from a drifted vehicle looked like a
            deviation.

        Returns
        -------
        tuple
            The updated table, and which contacts deviated on this step.
        """
        sound = self._config.acoustics
        predicted, grown = self.predict(tracks, t)

        own = (listener_sigma_m**2).unsqueeze(-1)
        zero = torch.zeros_like(own)
        fix_cov = measurement_cov(
            listener.unsqueeze(2),
            measured,
            sound.range_error_fraction,
            torch.pi * sound.bearing_sigma_deg / 180.0,
        ) + torch.cat([own, zero, own], dim=-1).unsqueeze(2)

        # A contact that turns up outside its own ellipse is not the contact the
        # prediction described. That is the deviation of section 4.2, and it is
        # the only alarm in the model.
        #
        # **But only while the prediction was still worth something.** Section
        # 4.2 lists three outcomes and an expired one is its own: a prediction
        # past its gate has already failed, and an observation disagreeing with
        # it says nothing about the vessel. Letting expiry raise alarms made
        # 99 % of them false -- the median held prediction was 29 km out against
        # a declared sigma of 3.7, because the lane predictor cannot see a
        # vessel turn round at the end of its lane. The right answer to an
        # expired track is to re-acquire it, which is what happens below.
        residual = measured - predicted
        expired = major_axis_m(grown) > self._gate_for(tracks)
        held = tracks.known & detected & ~expired
        distance = mahalanobis(residual, grown + fix_cov)
        deviated = held & (distance > self._config.prediction.gate_mahalanobis)

        # An alarm is a condition, not a scar. A contact that has been
        # re-acquired and agrees with its prediction again is behaving, and a
        # flag that only ever latched left 96 % of the fleet's tracks red by the
        # end of an episode -- which would both wreck the picture and spend the
        # tightened gate of C-10-03 on everything at once.
        # Any observation that does not contradict clears it, including one that
        # re-acquires an expired track. Requiring the track to be live first
        # left alarms latched forever, because expiry is the common case.
        agreed = detected & ~deviated

        # Fresh contacts and deviating ones restart from the fix alone; agreeing
        # ones fuse. Restarting is what stops a wrong prediction poisoning the
        # track it was wrong about.
        restart = detected & (~tracks.known | expired | deviated)
        information = inverse(grown) + inverse(fix_cov)
        fused_cov = inverse(information)
        fused_xy = apply(
            fused_cov, apply(inverse(grown), predicted) + apply(inverse(fix_cov), measured)
        )
        new_xy = torch.where(restart.unsqueeze(-1), measured, fused_xy)
        new_cov = torch.where(restart.unsqueeze(-1), fix_cov, fused_cov)

        keep = detected.unsqueeze(-1)
        vehicles = tracks.last_seen_s.shape[1]
        me = torch.arange(vehicles, device=tracks.issuer.device, dtype=torch.int16)
        me = me.reshape(1, vehicles, 1).expand_as(tracks.issuer)

        return (
            replace(
                tracks,
                last_seen_s=torch.where(
                    detected, torch.full_like(tracks.last_seen_s, t), tracks.last_seen_s
                ),
                xy=torch.where(keep, new_xy, tracks.xy),
                velocity=torch.where(keep, velocity, tracks.velocity),
                cov=torch.where(keep, new_cov, tracks.cov),
                rail=torch.where(detected, rail.to(torch.int8), tracks.rail),
                progress=torch.where(detected, progress, tracks.progress),
                direction=torch.where(detected, direction, tracks.direction),
                growth_m_per_s=torch.where(
                    detected,
                    self._growth_for(tracks, t, growth, velocity, new_xy),
                    tracks.growth_m_per_s,
                ),
                issuer=torch.where(detected, me, tracks.issuer),
                plausible=torch.where(detected, plausible & ~deviated, tracks.plausible),
                alarmed=(tracks.alarmed | deviated) & ~agreed,
            ),
            deviated,
        )

    def _growth_for(
        self,
        tracks: Tracks,
        t: float,
        class_growth: torch.Tensor,
        velocity: torch.Tensor,
        new_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return how fast a contact's ellipse should grow once off its lane.

        The default was the **class** rate: 7.10 m/s for a cargo, which is a
        cargo's transit speed. It is the right number for a cargo in transit
        and badly wrong for the only contacts that matter. Measured on this
        simulator, a deviating vessel moves a median of 584 m in half an hour
        and 266 m in an hour -- it loiters, that is what deviating *means*
        here -- while the class rate declares 12 780 m at thirty minutes, **22
        times too much**. The track crosses the 3 km staleness gate in seven
        minutes and is dropped, although it is still within 600 m half an hour
        later.

        Worth 6.2 points of missed deviations on the floor, and no more than
        that: the binding constraint is the **revisit interval**, 338 minutes
        against a deviation lasting 180. This widens the window a policy is
        given by a factor of twenty-two; it cannot help one that never comes
        back. See docs/constants.md C-10-04 and section 15.

        ``"class"`` keeps that behaviour, for comparison. The others use the
        velocity actually observed, which is the honest first-order model: a
        contact last seen doing ``v`` with an uncertain heading is somewhere on
        a circle of radius ``v t``, so the growth *is* its speed.

        ``"observed"`` takes that literally, floored so a contact seen dead
        still is not claimed to be perfectly known, and capped by the class
        rate so the model can never be more confident than the old one.
        ``"blended"`` adds the floor in quadrature instead of clamping, which
        makes the transition smooth rather than kinked.

        Both of those turned out to measure the wrong thing. A vessel
        deviating here **circles at three hundred metres**, so its
        instantaneous speed is its full transit speed and the prediction stays
        nearly as pessimistic: mean sigma on a deviating contact fell only from
        51 km to 44 km. What collapses when a vessel loiters is not its speed,
        it is the ground it makes good.

        ``"drift"`` measures that instead -- the net displacement between two
        fixes over the time between them -- and smooths it into the rate
        already stored, so one unlucky interval cannot swing it. That is the
        quantity the prediction error actually follows.
        """
        model = self._config.prediction.growth_model
        if model == "class":
            return class_growth
        settings = self._config.prediction
        floor = settings.growth_floor_ms
        if model == "drift":
            elapsed = (t - tracks.last_seen_s).clamp_min(settings.growth_floor_s)
            made_good = (new_xy - tracks.xy).norm(dim=-1) / elapsed
            rate = (made_good * settings.growth_manoeuvre).clamp(min=floor)
            smoothed = torch.lerp(tracks.growth_m_per_s, rate, settings.growth_memory)
            # A contact seen for the first time has no previous fix to drift
            # from, so it starts on the only estimate available: its class.
            return torch.where(tracks.known, smoothed, class_growth).minimum(class_growth)
        observed = velocity.norm(dim=-1) * settings.growth_manoeuvre
        if model == "observed":
            clamped: torch.Tensor = observed.clamp(min=floor)
            return clamped.minimum(class_growth)
        return torch.sqrt(observed**2 + floor**2).minimum(class_growth)

    def uncertainty_m(self, tracks: Tracks, t: float) -> torch.Tensor:
        """Return each held contact's worst-direction error, right now.

        This is what the verification reward of specification section 6.4 is
        paid on, and it is why a second vehicle arriving from a different
        bearing earns so much more than the first one staying put.

        Parameters
        ----------
        tracks : Tracks
            The table.
        t : float
            Now.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, vessels)`` in metres, ``inf`` where unknown.
        """
        _, grown = self.predict(tracks, t)
        axis = major_axis_m(grown)
        return torch.where(tracks.known, axis, torch.full_like(axis, torch.inf))

    def _gate_for(self, tracks: Tracks) -> torch.Tensor:
        """Return the accuracy each contact is held to, without the docstring."""
        loose = torch.full_like(tracks.progress, self._config.fleet.gate_m)
        tight = torch.full_like(tracks.progress, self._config.prediction.alarm_gate_m)
        return torch.where(tracks.alarmed, tight, loose)

    def gate_m(self, tracks: Tracks) -> torch.Tensor:
        """Return the accuracy each contact is held to.

        Wide for traffic behaving itself, tight for anything that has
        contradicted its own prediction. The difference is what turns a
        dispersing fleet into a converging one, and it is an operational fact
        rather than a device: you do not localise a problem to the standard you
        were happy with for a merchant.

        Parameters
        ----------
        tracks : Tracks
            The table.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, vessels)`` in metres.
        """
        loose = torch.full_like(tracks.progress, self._config.fleet.gate_m)
        tight = torch.full_like(tracks.progress, self._config.prediction.alarm_gate_m)
        return torch.where(tracks.alarmed, tight, loose)

    def stale(self, tracks: Tracks, t: float) -> torch.Tensor:
        """Return which held contacts have grown past the staleness gate.

        Parameters
        ----------
        tracks : Tracks
            The table.
        t : float
            Now.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, vessels)``.
        """
        return tracks.known & (self.uncertainty_m(tracks, t) > self.gate_m(tracks))
