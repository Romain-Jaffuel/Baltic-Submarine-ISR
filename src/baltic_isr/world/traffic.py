"""Surface traffic, batched over worlds.

One state object of tensors, one step function. There is no Python loop over
vessels or over worlds anywhere in this module, and nothing reads a tensor back
to the host: the operation count of :func:`step` is the same for one vessel as
for a hundred thousand, which is what ``tests/unit/test_traffic.py`` checks.

Two populations share the same integrator.

*Lane-bound* vessels -- cargo, ferries -- carry a fractional progress along a
rail and steer toward it. They are predictable, and that predictability is the
whole reason the fleet can release them with a prediction rather than escort
them (specification section 4).

*Free* vessels -- fishing, small craft -- switch between straight legs and
coordinated turns on a Markov chain. They are not predictable, and the fleet has
to learn that some contacts are simply not worth predicting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from baltic_isr import seeds
from baltic_isr.config import Config
from baltic_isr.world import field
from baltic_isr.world.statics import (
    Statics,
    _gather_along,
    sample_rail,
    sample_rail_tangent,
)

MODE_STRAIGHT = 0
MODE_TURNING = 1

_TURN_RATE_RAD_S = math.radians(3.0) / 60.0
_MAX_TURN_RATE_RAD_S = math.radians(12.0) / 60.0
_P_START_TURN = 0.02
_P_END_TURN = 0.10
_DEVIATION_SPEED_MS = 0.6
# A ship holding station over a point does not stop dead, it works a small
# circle against the current. A dot that stops reads as nothing on a map; a dot
# that circles over a cable reads as loitering, which is what it is.
_LOITER_RADIUS_M = 300.0
_NO_RAIL = -1

# Speed over ground wanders: weather, traffic, schedule. An Ornstein-Uhlenbeck
# process with a two-hour memory and an eight per cent spread, which is what
# stops the rail predictor from being a tautology.
_SPEED_RELAXATION_S = 7200.0
_SPEED_SPREAD = 0.08


@dataclass(frozen=True)
class TrafficState:
    """Every vessel in every world.

    All tensors have leading shape ``(worlds, vessels)``.

    Attributes
    ----------
    xy : Tensor
        Position in local metres, ``(worlds, vessels, 2)``.
    heading : Tensor
        Course over ground in radians, measured clockwise from north.
    speed : Tensor
        Speed through the water in m/s.
    cruise_speed : Tensor
        The speed the vessel returns to when not deviating.
    vessel_class : Tensor
        Index into the configured class list.
    source_level_db : Tensor
        Radiated spectrum level at 1 m.
    rail : Tensor
        Rail index, or ``-1`` for a free vessel.
    progress : Tensor
        Fraction along the rail, wrapping at one.
    lane_offset : Tensor
        Constant cross-track position within the lane, in metres.
    direction : Tensor
        ``+1`` or ``-1`` along the rail. A lane has two ends, and a vessel that
        reaches one turns round rather than reappearing at the other.
    mode : Tensor
        ``MODE_STRAIGHT`` or ``MODE_TURNING``, free vessels only.
    turn_sign : Tensor
        Which way a turning vessel turns.
    deviating : Tensor
        Whether the vessel is inside its deviation window right now.
    deviation_start_s : Tensor
        When the vessel is *armed*. ``inf`` for one that never deviates.
    deviation_end_s : Tensor
        When the window closes, set on the step it actually fires.
    deviation_fired : Tensor
        Whether the window has opened. Arming is scheduled by the clock; firing
        waits for the right place, because a ship stops where it means to and
        not at an hour computed in advance.
    deviation_needs_corridor : Tensor
        Whether this vessel waits for a cable corridor before it fires. Most do.
        The rest stop in open water, so that camping on the cables cannot be the
        whole answer.
    ais_on : Tensor
        Whether the vessel is transmitting its identity.
    """

    xy: torch.Tensor
    heading: torch.Tensor
    speed: torch.Tensor
    cruise_speed: torch.Tensor
    vessel_class: torch.Tensor
    source_level_db: torch.Tensor
    rail: torch.Tensor
    progress: torch.Tensor
    lane_offset: torch.Tensor
    direction: torch.Tensor
    mode: torch.Tensor
    turn_sign: torch.Tensor
    deviating: torch.Tensor
    deviation_start_s: torch.Tensor
    deviation_end_s: torch.Tensor
    deviation_fired: torch.Tensor
    deviation_needs_corridor: torch.Tensor
    ais_on: torch.Tensor

    @property
    def n_worlds(self) -> int:
        """Return how many worlds run in parallel."""
        return int(self.xy.shape[0])

    @property
    def n_vessels(self) -> int:
        """Return how many vessels each world holds."""
        return int(self.xy.shape[1])

    @property
    def velocity(self) -> torch.Tensor:
        """Return the velocity vector of each vessel, ``(worlds, vessels, 2)``."""
        return torch.stack(
            [self.speed * torch.sin(self.heading), self.speed * torch.cos(self.heading)], dim=-1
        )

    @property
    def lane_bound(self) -> torch.Tensor:
        """Return which vessels follow a rail."""
        return self.rail >= 0


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference into ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _lane_position(
    statics: Statics, rail: torch.Tensor, progress: torch.Tensor, offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return where a lane-bound vessel is, and which way it points.

    A ship in a traffic separation scheme holds its side of the lane and
    corrects for the current; it does not get blown across and then chase its
    way back at the same speed, which it could never do. So its position is the
    rail point at its progress pushed sideways by a constant offset, and its
    heading is the rail tangent. That is what makes it predictable, and
    predictability is what the task of specification section 4 rests on.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    rail : Tensor
        Rail index per vessel.
    progress : Tensor
        Fraction along the rail.
    offset : Tensor
        Cross-track offset in metres.

    Returns
    -------
    tuple of Tensor
        Position ``(..., 2)`` and heading ``(...)``.
    """
    anchor = sample_rail(statics, rail, progress)
    tangent = sample_rail_tangent(statics, rail, progress)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
    return anchor + normal * offset.unsqueeze(-1), torch.atan2(tangent[..., 0], tangent[..., 1])


class Traffic:
    """The traffic generator and its step.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry.
    n_worlds : int
        How many worlds to run in parallel.
    """

    def __init__(self, config: Config, statics: Statics, n_worlds: int) -> None:
        self._config = config
        self._statics = statics
        self._worlds = n_worlds
        self._device = statics.device
        self._generator = seeds.torch_generator(config.simulation.seed, "traffic", self._device)

        classes = config.traffic.classes
        rails = {rail.name: index for index, rail in enumerate(config.rails)}
        self._share = torch.tensor(
            [item.share for item in classes], dtype=torch.float32, device=self._device
        )
        self._class_speed = torch.tensor(
            [item.speed_mean_ms for item in classes], dtype=torch.float32, device=self._device
        )
        self._class_sigma = torch.tensor(
            [item.speed_sigma_ms for item in classes], dtype=torch.float32, device=self._device
        )
        self._class_level = torch.tensor(
            [item.source_level_db for item in classes], dtype=torch.float32, device=self._device
        )
        self._class_rail = torch.tensor(
            [rails[item.rail] if item.rail is not None else _NO_RAIL for item in classes],
            dtype=torch.long,
            device=self._device,
        )
        self._class_predictable = torch.tensor(
            [item.predictable for item in classes], dtype=torch.bool, device=self._device
        )

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self._generator, device=self._device)

    def reset(self, seed: int | None = None) -> TrafficState:
        """Populate every world and schedule its deviations.

        Parameters
        ----------
        seed : int, optional
            Overrides the configured seed, so that evaluation can pin a
            scenario the training never saw.

        Returns
        -------
        TrafficState
            A fresh state at ``t = 0``.
        """
        if seed is not None:
            self._generator = seeds.torch_generator(seed, "traffic", self._device)
        worlds, vessels = self._worlds, self._config.traffic.n_vessels
        shape = (worlds, vessels)

        vessel_class = torch.multinomial(
            self._share, worlds * vessels, replacement=True, generator=self._generator
        ).reshape(shape)
        rail = self._class_rail[vessel_class]
        lane_bound = rail >= 0

        progress = self._rand(*shape)
        speed = (
            self._class_speed[vessel_class]
            + self._class_sigma[vessel_class]
            * torch.randn(shape, generator=self._generator, device=self._device)
        ).clamp_min(0.5)

        half_width = self._statics.rail_half_width[rail.clamp_min(0)]
        lane_offset = (2.0 * self._rand(*shape) - 1.0) * half_width
        on_rail, lane_heading = _lane_position(
            self._statics, rail.clamp_min(0), progress, lane_offset
        )
        free = self._free_positions(shape)

        xy = torch.where(lane_bound.unsqueeze(-1), on_rail, free)
        heading = torch.where(lane_bound, lane_heading, self._rand(*shape) * 2.0 * math.pi)

        state = TrafficState(
            xy=xy,
            heading=heading,
            speed=speed,
            cruise_speed=speed,
            vessel_class=vessel_class,
            source_level_db=self._class_level[vessel_class],
            rail=rail,
            progress=progress,
            lane_offset=lane_offset,
            direction=torch.where(self._rand(*shape) < 0.5, -1.0, 1.0),
            mode=torch.full(shape, MODE_STRAIGHT, dtype=torch.long, device=self._device),
            turn_sign=torch.where(self._rand(*shape) < 0.5, -1.0, 1.0),
            deviating=torch.zeros(shape, dtype=torch.bool, device=self._device),
            deviation_start_s=torch.full(shape, math.inf, device=self._device),
            deviation_end_s=torch.full(shape, math.inf, device=self._device),
            deviation_fired=torch.zeros(shape, dtype=torch.bool, device=self._device),
            deviation_needs_corridor=torch.ones(shape, dtype=torch.bool, device=self._device),
            ais_on=torch.ones(shape, dtype=torch.bool, device=self._device),
        )
        return self._schedule_deviations(state)

    def _free_positions(self, shape: tuple[int, int]) -> torch.Tensor:
        """Draw positions uniformly over water, by rejection on a few rounds.

        A fixed number of rounds rather than a loop until success: the shape of
        the computation must not depend on the numbers drawn.
        """
        best = torch.zeros(*shape, 2, device=self._device)
        found = torch.zeros(shape, dtype=torch.bool, device=self._device)
        for _ in range(8):
            candidate = self._rand(*shape, 2) * self._statics.extent
            take = field.is_sea(self._statics, candidate) & ~found
            best = torch.where(take.unsqueeze(-1), candidate, best)
            found = found | take
        centre = self._statics.extent * torch.tensor([0.5, 0.5], device=self._device)
        return torch.where(found.unsqueeze(-1), best, centre)

    def _schedule_deviations(self, state: TrafficState) -> TrafficState:
        """Pick which vessels deviate, when, and over which corridor.

        Only a predictable, lane-bound vessel is worth deviating: the whole
        point is that the fleet had a prediction and the prediction broke. A
        fishing boat behaving oddly is Tuesday.

        The start time is chosen so the vessel reaches a point of its rail that
        lies over a seabed corridor. Loitering in open water is odd; loitering
        over a cable is the scenario.
        """
        worlds, vessels = state.xy.shape[0], state.xy.shape[1]
        count = self._config.fleet.deviations_per_episode
        if count == 0:
            return state

        eligible = self._class_predictable[state.vessel_class] & state.lane_bound
        chosen = torch.topk(
            torch.where(eligible, self._rand(worlds, vessels), torch.zeros(()).to(self._device)),
            k=min(count, vessels),
            dim=1,
        ).indices
        selected = torch.zeros_like(eligible)
        selected.scatter_(1, chosen, True)
        selected = selected & eligible

        over = self._statics.rail_over_corridor[state.rail.clamp_min(0)]
        samples = over.shape[-1]
        pick = torch.where(
            over, self._rand(worlds, vessels, samples), torch.zeros(()).to(self._device)
        )
        target = pick.argmax(dim=-1).to(state.progress.dtype) / (samples - 1)

        # A vessel does not loop its lane, it shuttles along it, so progress is
        # a triangle wave rather than a sawtooth. Unfold it: a vessel running
        # backwards from p is a vessel running forwards from 2 - p, and folding
        # the result reproduces the bounce. A given progress is then reached at
        # u = target or u = 2 - target, modulo the period of two.
        length = self._statics.rail_length[state.rail.clamp_min(0)]
        speed = state.speed.clamp_min(0.5)
        unfolded = torch.where(state.direction > 0.0, state.progress, 2.0 - state.progress)
        forward = (target - unfolded) % 2.0
        backward = ((2.0 - target) - unfolded) % 2.0
        travel = torch.minimum(forward, backward) * length

        # A full there-and-back is two lane lengths. Choosing which pass spreads
        # the deviations over the episode rather than bunching them at the start.
        period = 2.0 * length / speed
        usable = self._config.simulation.episode_s * 0.8
        laps = torch.floor(self._rand(worlds, vessels) * (usable / period).clamp_min(1.0))
        start = (travel / speed + laps * period).clamp(max=usable)

        infinite = torch.full_like(start, math.inf)
        return replace(
            state,
            deviation_start_s=torch.where(selected, start, infinite),
            deviation_needs_corridor=(
                self._rand(worlds, vessels) < self._config.fleet.deviation_corridor_fraction
            ),
        )

    def step(self, state: TrafficState, t: float, dt_s: float) -> TrafficState:
        """Advance every vessel in every world by one tick.

        Parameters
        ----------
        state : TrafficState
            Current state.
        t : float
            Simulation time after the step.
        dt_s : float
            Step length.

        Returns
        -------
        TrafficState
            The advanced state.
        """
        statics = self._statics
        lane_bound = state.lane_bound
        rail = state.rail.clamp_min(0)

        # Armed by the clock, fired by the geography. Computing an arrival time
        # in advance does not survive a speed that wanders: it put only half the
        # deviations over a corridor, which is where they are supposed to be.
        over_corridor = (
            _gather_along(
                statics.rail_over_corridor.unsqueeze(-1).to(state.progress.dtype),
                rail,
                state.progress,
            ).squeeze(-1)
            > 0.5
        )
        armed = (state.deviation_start_s <= t) & ~state.deviation_fired
        wherever = ~state.deviation_needs_corridor
        fires = armed & (over_corridor | wherever) & lane_bound
        fired = state.deviation_fired | fires
        end = torch.where(
            fires,
            torch.full_like(state.deviation_end_s, t + self._config.fleet.deviation_duration_s),
            state.deviation_end_s,
        )
        deviating = fired & (t < end)

        # Speed over ground relaxes toward the vessel's cruise and wanders round
        # it. Without this the rail predictor would be exact, because it would
        # be running the generator backwards.
        decay = math.exp(-dt_s / _SPEED_RELAXATION_S)
        spread = _SPEED_SPREAD * state.cruise_speed * math.sqrt(1.0 - decay**2)
        wandered = (
            state.cruise_speed
            + (state.speed - state.cruise_speed) * decay
            + spread
            * torch.randn(state.speed.shape, generator=self._generator, device=self._device)
        ).clamp_min(0.5)
        speed = torch.where(deviating, torch.full_like(state.speed, _DEVIATION_SPEED_MS), wandered)

        # A lane has two ends. Reaching one means leaving the modelled water, so
        # the vessel turns round instead of teleporting to the far end -- which
        # it used to do, and which made every prediction past one lap nonsense.
        # A deviating vessel makes no way at all: holding station is the point.
        making_way = torch.where(deviating, torch.zeros_like(speed), speed)
        advanced = state.progress + state.direction * making_way * dt_s / statics.rail_length[rail]
        turned = (advanced > 1.0) | (advanced < 0.0)
        direction = torch.where(turned, -state.direction, state.direction)
        progress = torch.where(turned, state.progress, advanced).clamp(0.0, 1.0)
        lane_xy, lane_tangent_heading = _lane_position(statics, rail, progress, state.lane_offset)

        # ... but it does not sit still either. A ship holding position works a
        # small circle against the current, and on a map a dot that circles over
        # a cable reads as loitering where a dot that stops reads as nothing.
        # Phase from the time since it fired. Taken straight off deviation_end_s
        # it is minus infinity on the firing step, because the window closes
        # only later -- and one NaN position is enough to poison the whole track.
        turn_rate = _DEVIATION_SPEED_MS / _LOITER_RADIUS_M
        duration = self._config.fleet.deviation_duration_s
        since_fired = torch.where(deviating, duration - (end - t), torch.zeros_like(speed))
        phase = since_fired * turn_rate
        circle = _LOITER_RADIUS_M * torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
        lane_xy = lane_xy + torch.where(deviating.unsqueeze(-1), circle, torch.zeros_like(circle))
        lane_heading = torch.where(
            direction > 0.0,
            lane_tangent_heading,
            (lane_tangent_heading + math.pi) % (2.0 * math.pi),
        )
        lane_heading = torch.where(
            deviating, (phase + 0.5 * math.pi) % (2.0 * math.pi), lane_heading
        )

        roll = self._rand(*state.mode.shape)
        turning = state.mode == MODE_TURNING
        mode = torch.where(
            turning,
            torch.where(roll < _P_END_TURN, MODE_STRAIGHT, MODE_TURNING),
            torch.where(roll < _P_START_TURN, MODE_TURNING, MODE_STRAIGHT),
        )
        free_heading = state.heading + torch.where(
            mode == MODE_TURNING, state.turn_sign * _TURN_RATE_RAD_S * dt_s, 0.0
        )

        turn = _wrap_angle(free_heading - state.heading).clamp(
            -_MAX_TURN_RATE_RAD_S * dt_s, _MAX_TURN_RATE_RAD_S * dt_s
        )
        free_next = (state.heading + turn) % (2.0 * math.pi)

        # Free vessels are integrated and are pushed about by a current they do
        # not correct for. Lane-bound ones are placed.
        drift = field.current(statics, state.xy, self._config.environment)
        velocity = (
            torch.stack([speed * torch.sin(free_next), speed * torch.cos(free_next)], dim=-1)
            + drift
        )
        free_xy = state.xy + velocity * dt_s

        # Turn back at the shore, and at the box edge too: outside the box the
        # coastline interpolation flattens to its end value, so "at sea" stays
        # true out there and a free vessel would wander off for good.
        outside = (free_xy < 0.0).any(-1) | (free_xy > statics.extent).any(-1)
        blocked = outside | ~field.is_sea(statics, free_xy)
        free_next = torch.where(blocked, (free_next + math.pi) % (2.0 * math.pi), free_next)
        free_xy = torch.where(blocked.unsqueeze(-1), state.xy, free_xy)

        xy = torch.where(lane_bound.unsqueeze(-1), lane_xy, free_xy)
        heading = torch.where(lane_bound, lane_heading, free_next)

        return replace(
            state,
            xy=xy,
            heading=heading,
            speed=speed,
            progress=progress,
            direction=direction,
            mode=torch.where(lane_bound, state.mode, mode),
            deviating=deviating,
            deviation_end_s=end,
            deviation_fired=fired,
            ais_on=~deviating,
        )
