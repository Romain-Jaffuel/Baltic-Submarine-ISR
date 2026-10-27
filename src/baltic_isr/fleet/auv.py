"""The fleet: three regimes, an energy budget, and a base to come home to.

Vectorised over worlds and vehicles on the same terms as the traffic model, and
checked the same way: the operation count of :meth:`Fleet.step` does not grow
with the batch and nothing reads back to the host.

**Vehicles live on a sortie cycle**: out, on station, home, alongside charging,
out again. Nothing is teleported onto station -- every vehicle swims there under
its own power and pays for it.

At ``t = 0`` the fleet is in **sustained rotation**: each vehicle is placed at a
different, evenly spread point of that cycle, so some are working while others
are outbound, coming home, or on charge. That is what a real operation looks
like, and it has a consequence worth stating plainly: **only about half the
fleet is on station at any moment**. Twenty vehicles do not buy twenty pairs of
ears. See ``docs/constants.md`` C-8-02.

The ``cold`` start -- everyone at the quay at ``t = 0`` -- is kept as a contrast.
It costs the first nine hours and the last ten, and it recalls the whole base at
once.

Two things are *not* learned, because they are not tactical choices:

*The forced recall.* Below the energy it takes to reach the nearest base, the
regime and the heading are taken away from the policy. It cannot drown its fleet.

*Surfacing for a fix.* Triggered when dead-reckoning uncertainty reaches half the
staleness gate. Whether the policy should own this instead is open, and batch 7
will tell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from baltic_isr import seeds
from baltic_isr.config import Config
from baltic_isr.fleet import navigation, power
from baltic_isr.world import field
from baltic_isr.world.statics import Statics

MODE_DEPLOYED = 0
MODE_SURFACING = 1
MODE_RETURNING = 2
MODE_CHARGING = 3
MODE_LOST = 4

# How much longer the navigable way home is than the straight line, median
# over the gulf. See _return_margin.
_DETOUR = 1.4

REGIME_DRIFT = 0
REGIME_CRUISE = 1
REGIME_SPRINT = 2

HEADINGS = 8
REGIMES = 3
ACTION_RETURN = HEADINGS * REGIMES
N_ACTIONS = ACTION_RETURN + 1

_MAX_TURN_RATE_RAD_S = math.radians(6.0) / 60.0


@dataclass(frozen=True)
class FleetState:
    """Every vehicle in every world.

    All tensors have leading shape ``(worlds, vehicles)``.

    Attributes
    ----------
    xy : Tensor
        True position in local metres, ``(worlds, vehicles, 2)``.
    estimate : Tensor
        The position the vehicle believes it is at. The gap between this and
        ``xy`` is what manufactures false alarms.
    heading : Tensor
        Course in radians, clockwise from north.
    regime : Tensor
        ``REGIME_DRIFT``, ``REGIME_CRUISE`` or ``REGIME_SPRINT``.
    energy_j : Tensor
        Remaining battery.
    mode : Tensor
        One of the ``MODE_`` constants.
    base : Tensor
        Index of the vehicle's home base.
    heading_bias, scale_error : Tensor
        Fixed instrument errors, drawn once per episode.
    distance_since_fix : Tensor
        Path length since the last GNSS fix.
    timer_s : Tensor
        Countdown for surfacing and for charging.
    station : Tensor
        The point this vehicle transits to, drawn once per episode.
    """

    xy: torch.Tensor
    estimate: torch.Tensor
    heading: torch.Tensor
    regime: torch.Tensor
    energy_j: torch.Tensor
    mode: torch.Tensor
    base: torch.Tensor
    heading_bias: torch.Tensor
    scale_error: torch.Tensor
    distance_since_fix: torch.Tensor
    timer_s: torch.Tensor
    station: torch.Tensor

    @property
    def n_worlds(self) -> int:
        """Return how many worlds run in parallel."""
        return int(self.xy.shape[0])

    @property
    def n_vehicles(self) -> int:
        """Return how many vehicles each world holds."""
        return int(self.xy.shape[1])

    @property
    def at_sea(self) -> torch.Tensor:
        """Return which vehicles are in the water and under way."""
        return (self.mode == MODE_DEPLOYED) | (self.mode == MODE_RETURNING)

    @property
    def listening(self) -> torch.Tensor:
        """Return which vehicles can hear.

        A surfaced vehicle cannot, and that is most of what surfacing costs.
        """
        return self.at_sea

    @property
    def drift_error_m(self) -> torch.Tensor:
        """Return the true distance between belief and position."""
        error: torch.Tensor = (self.estimate - self.xy).norm(dim=-1)
        return error


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference into ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Fleet:
    """The constellation, its energy budget and its state machine.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, which carries the bases.
    n_worlds : int
        How many worlds to run in parallel.
    """

    def __init__(self, config: Config, statics: Statics, n_worlds: int) -> None:
        self._config = config
        self._statics = statics
        self._worlds = n_worlds
        self._device = statics.device
        self._generator = seeds.torch_generator(config.simulation.seed, "fleet", self._device)

        speeds = power.regime_speeds(config.auv)
        self._speed = torch.tensor(
            [speeds["drift"], speeds["cruise"], speeds["sprint"]],
            dtype=torch.float32,
            device=self._device,
        )
        self._power = torch.tensor(
            [power.power_w(value, config.auv) for value in self._speed.tolist()],
            dtype=torch.float32,
            device=self._device,
        )
        self._headings = torch.arange(HEADINGS, device=self._device) * (2.0 * math.pi / HEADINGS)
        self._surface_s = (
            2.0 * config.auv.operating_depth_m / config.auv.vertical_speed_ms
            + config.comms.surface_dwell_s
        )
        self._turnaround_s = config.auv.recharge_s + config.auv.handling_s
        # The limit the configuration states, kept for the places that ask
        # a vehicle whether it may BE somewhere. Getting it there is
        # `field.push_offshore`, which reads a field baked for this number.
        self._min_water_m = config.fleet.min_water_m
        # How long a battery lasts at cruise before the reserve is reached. The
        # whole sortie -- out, on station, home -- has to fit inside it.
        self._usable_s = (
            config.auv.battery_j
            * (1.0 - config.auv.energy_reserve_fraction)
            / power.power_w(config.auv.regimes.cruise_ms, config.auv)
        )

    @property
    def surfacing_s(self) -> float:
        """Return how long one surfacing costs, in seconds."""
        return self._surface_s

    @property
    def usable_s(self) -> float:
        """Return how long a battery lasts at cruise before the reserve."""
        return self._usable_s

    def sortie(self, station: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the timings of one sortie, per vehicle.

        Out at cruise, work, home at cruise, then turn the vehicle round. The
        ratio of ``on_station`` to ``cycle`` is the fraction of the fleet that is
        actually listening at any moment, and it is well under one.

        Parameters
        ----------
        station : Tensor
            Patrol point, shape ``(..., 2)``.
        base : Tensor
            Home base, shape ``(..., 2)``.

        Returns
        -------
        dict
            ``leg``, ``on_station`` and ``cycle``, all in seconds.
        """
        leg = (station - base).norm(dim=-1) / self._config.auv.regimes.cruise_ms
        on_station = (self._usable_s - 2.0 * leg).clamp_min(0.0)
        return {
            "leg": leg,
            "on_station": on_station,
            "cycle": 2.0 * leg + on_station + self._turnaround_s,
        }

    def _berthing(self, base: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Return which vehicles are close enough to a base to berth.

        Two rules, and the configuration states both. At sea a vehicle needs
        water under the keel. Inside the approach to its own base it needs
        only to be off the quay -- "a vehicle may come into shallow water;
        that is what a base is for" -- because the berth at Helsinki has
        twelve metres over it and the fleet asks for fifteen.

        Parameters
        ----------
        base : Tensor
            Index of each vehicle's own base, shape ``(worlds, vehicles)``.
        xy : Tensor
            Where to evaluate, shape ``(worlds, vehicles, 2)``.

        Returns
        -------
        Tensor
            Boolean, shape ``(worlds, vehicles)``.
        """
        home = self._statics.bases[base]
        inside: torch.Tensor = (xy - home).norm(dim=-1) < self._config.fleet.base_approach_m
        return inside

    def _stations(self, shape: tuple[int, int]) -> torch.Tensor:
        """Draw one patrol point per vehicle, uniformly over navigable water.

        Navigable, not merely wet: a station inside the keel clearance is a
        station the vehicle can never reach, and it would spend the episode
        pinned against the coast trying.
        """
        best = torch.zeros(*shape, 2, device=self._device)
        found = torch.zeros(shape, dtype=torch.bool, device=self._device)
        for _ in range(8):
            candidate = self._rand(*shape, 2) * self._statics.extent
            take = field.has_clearance(self._statics, candidate, self._min_water_m) & ~found
            best = torch.where(take.unsqueeze(-1), candidate, best)
            found = found | take
        return torch.where(found.unsqueeze(-1), best, self._statics.extent * 0.5)

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self._generator, device=self._device)

    def reset(self, seed: int | None = None) -> FleetState:
        """Place the fleet, either in sustained rotation or all at the quay.

        In rotation, phases are spread **evenly** rather than drawn at random:
        an even spread is what keeps coverage flat and stops a whole base coming
        home at the same hour, which is the point of rotating at all.

        Parameters
        ----------
        seed : int, optional
            Overrides the configured seed.

        Returns
        -------
        FleetState
            A fresh state at ``t = 0``.
        """
        if seed is not None:
            self._generator = seeds.torch_generator(seed, "fleet", self._device)
        auv = self._config.auv
        worlds, vehicles = self._worlds, self._config.fleet.count
        shape = (worlds, vehicles)

        # Station first, then the base: a vehicle sails from the port nearest
        # its patrol area. Handing bases out round robin instead cost 17 km of
        # transit each way and dropped the duty cycle from 45 % to 28 %.
        station = self._stations(shape)
        base_index = torch.cdist(station, self._statics.bases.expand(worlds, -1, -1)).argmin(-1)
        home = self._statics.bases[base_index]
        bias, scale = navigation.sample_instrument_errors(
            shape, self._config.navigation, self._generator, self._device
        )

        cruise_w = power.power_w(auv.regimes.cruise_ms, auv)
        timings = self.sortie(station, home)
        leg, cycle = timings["leg"], timings["cycle"]
        at_sea_s = self._usable_s

        if self._config.fleet.start == "cold":
            phase = torch.zeros(shape, device=self._device)
        else:
            spread = torch.arange(vehicles, device=self._device).to(torch.float32) / vehicles
            phase = spread.expand(worlds, vehicles) * cycle

        outbound = phase < leg
        returning = (phase >= at_sea_s - leg) & (phase < at_sea_s)
        charging = phase >= at_sea_s

        # Position: along the outbound leg, on station, along the way home, or
        # alongside. Energy follows the same clock, because every leg is cruise.
        out_fraction = (phase / leg.clamp_min(1.0)).clamp(0.0, 1.0).unsqueeze(-1)
        back_fraction = (
            ((phase - (at_sea_s - leg)) / leg.clamp_min(1.0)).clamp(0.0, 1.0).unsqueeze(-1)
        )
        xy = torch.where(
            charging.unsqueeze(-1),
            home,
            torch.where(
                outbound.unsqueeze(-1),
                torch.lerp(home, station, out_fraction),
                torch.where(
                    returning.unsqueeze(-1), torch.lerp(station, home, back_fraction), station
                ),
            ),
        )

        # The transit is a straight line from the base to the station, and a
        # straight line in this gulf can cut a corner of Estonia. Project the
        # placement out to sea so the episode starts legal; from there the step
        # keeps it that way, and the push stays sub-metre.
        xy = field.push_offshore(self._statics, xy, self._berthing(base_index, xy))

        charged = ((phase - at_sea_s) / self._turnaround_s).clamp(0.0, 1.0)
        reserve = auv.energy_reserve_fraction * auv.battery_j
        energy = torch.where(
            charging,
            reserve + (auv.battery_j - reserve) * charged,
            (auv.battery_j - cruise_w * phase).clamp_min(reserve),
        )

        mode = torch.where(
            charging,
            torch.full(shape, MODE_CHARGING, dtype=torch.long, device=self._device),
            torch.where(
                returning,
                torch.full(shape, MODE_RETURNING, dtype=torch.long, device=self._device),
                torch.full(shape, MODE_DEPLOYED, dtype=torch.long, device=self._device),
            ),
        )
        timer = torch.where(
            charging, (1.0 - charged) * self._turnaround_s, torch.zeros(shape, device=self._device)
        )

        # A vehicle already at sea has been dead reckoning for a while, and its
        # belief has drifted accordingly. Starting it with a perfect fix would
        # hand the fleet an accuracy it has not earned.
        travelled = (phase.clamp_max(at_sea_s) * auv.regimes.cruise_ms) % (
            self._config.navigation.surface_trigger_m
            / max(math.sin(math.radians(self._config.navigation.heading_bias_sigma_deg)), 1e-9)
        )
        distance_since_fix = torch.where(
            charging, torch.zeros(shape, device=self._device), travelled
        )

        # Point outbound and charging vehicles at their station, homebound ones
        # at their base, and on-station ones along their current track.
        toward_station = station - xy
        aim = torch.where(returning.unsqueeze(-1), home - xy, toward_station)
        aim = torch.where(
            aim.norm(dim=-1, keepdim=True) > 1.0, aim, self._statics.extent * 0.5 - xy
        )

        # A vehicle already at sea has been dead reckoning for a while and its
        # belief has drifted. Starting it on a perfect fix would hand the fleet
        # an accuracy it has not earned.
        spread = distance_since_fix * math.sin(
            math.radians(self._config.navigation.heading_bias_sigma_deg)
        )
        wander = torch.randn(xy.shape, generator=self._generator, device=self._device)
        estimate = xy + wander * spread.unsqueeze(-1)

        return FleetState(
            xy=xy,
            estimate=estimate,
            heading=torch.atan2(aim[..., 0], aim[..., 1]) % (2.0 * math.pi),
            regime=torch.full(shape, REGIME_CRUISE, dtype=torch.long, device=self._device),
            energy_j=energy,
            mode=mode,
            base=base_index.clone(),
            heading_bias=bias,
            scale_error=scale,
            distance_since_fix=distance_since_fix,
            timer_s=timer,
            station=station,
        )

    def distance_to_base(self, state: FleetState) -> torch.Tensor:
        """Return each vehicle's distance to its own base.

        Parameters
        ----------
        state : FleetState
            Current state.

        Returns
        -------
        Tensor
            Distance in metres, shape ``(worlds, vehicles)``.
        """
        distance: torch.Tensor = (self._statics.bases[state.base] - state.xy).norm(dim=-1)
        return distance

    def step(self, state: FleetState, actions: torch.Tensor, t: float, dt_s: float) -> FleetState:
        """Apply one action per vehicle and advance the fleet.

        Parameters
        ----------
        state : FleetState
            Current state.
        actions : Tensor
            One integer per vehicle in ``[0, N_ACTIONS)``. Eight headings times
            three regimes, plus a voluntary return.
        t : float
            Simulation time after the step.
        dt_s : float
            Step length.

        Returns
        -------
        FleetState
            The advanced state.
        """
        auv = self._config.auv
        bases = self._statics.bases[state.base]
        to_base = bases - state.xy
        range_to_base = to_base.norm(dim=-1)
        base_heading = torch.atan2(to_base[..., 0], to_base[..., 1])

        wants_return = actions == ACTION_RETURN
        commanded_heading = self._headings[(actions % HEADINGS).clamp(0, HEADINGS - 1)]
        commanded_regime = (actions // HEADINGS).clamp(0, REGIMES - 1)

        # The forced recall of specification section 2.5. Not a decision: below
        # the energy it takes to get home at cruise, the policy loses the helm.
        margin = self._return_margin(range_to_base)
        recalled = (state.energy_j < margin) & (state.mode == MODE_DEPLOYED)
        returning = recalled | (wants_return & (state.mode == MODE_DEPLOYED))

        mode = torch.where(returning, torch.full_like(state.mode, MODE_RETURNING), state.mode)

        # Surfacing for a fix, once the belief has drifted half a gate away.
        uncertainty = navigation.uncertainty_m(state.distance_since_fix, self._config.navigation)
        due = (uncertainty > self._config.navigation.surface_trigger_m) & (mode == MODE_DEPLOYED)
        mode = torch.where(due, torch.full_like(mode, MODE_SURFACING), mode)
        timer = torch.where(due, torch.full_like(state.timer_s, self._surface_s), state.timer_s)

        deployed = mode == MODE_DEPLOYED
        heading = torch.where(
            mode == MODE_RETURNING,
            base_heading,
            torch.where(deployed, commanded_heading, state.heading),
        )
        turn = _wrap_angle(heading - state.heading).clamp(
            -_MAX_TURN_RATE_RAD_S * dt_s, _MAX_TURN_RATE_RAD_S * dt_s
        )
        heading = (state.heading + turn) % (2.0 * math.pi)

        regime = torch.where(
            mode == MODE_RETURNING,
            torch.full_like(state.regime, REGIME_CRUISE),
            torch.where(deployed, commanded_regime, state.regime),
        )
        moving = deployed | (mode == MODE_RETURNING)
        speed = torch.where(moving, self._speed[regime], torch.zeros_like(state.energy_j))

        drift = field.current(self._statics, state.xy, self._config.environment)
        through_water = torch.stack([speed * torch.sin(heading), speed * torch.cos(heading)], -1)
        proposed = through_water + drift * moving.unsqueeze(-1)

        # Keep the fleet inside the box. A vehicle that swims out of the area is
        # not lost at sea, it simply stops making progress that way -- and its
        # velocity over ground really is zero, so the dead reckoning has to see
        # zero too. Advancing the estimate here while holding the truth still
        # made the believed position diverge without bound.
        candidate = state.xy + proposed * dt_s
        outside = (candidate < 0.0).any(-1) | (candidate > self._statics.extent).any(-1)
        candidate = torch.where(outside.unsqueeze(-1), state.xy, candidate)

        # And keep it in the water. Without this the fleet simply swam ashore:
        # 24 % of recorded positions were on land and the worst incursion ran
        # 24 km inland, which did not just look wrong, it inflated the coverage,
        # because a vehicle listening from a forest hears nothing.
        #
        # The limit is a DEPTH, not a distance from the waterline: what stops an
        # AUV is water under the keel. Inside the approach to its own base it
        # may come into shallow water -- that is what a base is for -- but not
        # onto the beach, so the clearance relaxes there instead of vanishing.
        xy = field.push_offshore(
            self._statics,
            candidate,
            self._berthing(state.base, candidate),
            came_from=state.xy,
        )

        # The projection is part of the vehicle's motion, so the dead reckoning
        # has to see it. A vehicle held against the shore makes good only the
        # along-shore component, and its DVL measures exactly that.
        over_ground = (xy - state.xy) / dt_s

        estimate = navigation.advance(
            state.estimate, over_ground, state.heading_bias, state.scale_error, dt_s
        )
        distance_since_fix = state.distance_since_fix + over_ground.norm(dim=-1) * dt_s

        draw = torch.where(
            moving, self._power[regime], torch.full_like(state.energy_j, auv.hotel_power_w)
        )
        charging = mode == MODE_CHARGING
        gain = auv.battery_j / auv.recharge_s
        energy = torch.where(
            charging,
            (state.energy_j + gain * dt_s).clamp(max=auv.battery_j),
            (state.energy_j - draw * dt_s).clamp_min(0.0),
        )

        timer = torch.where(charging, timer - dt_s, timer - dt_s).clamp_min(0.0)
        surfaced = (mode == MODE_SURFACING) & (timer <= 0.0)
        fix = navigation.apply_fix(xy, self._config.navigation, self._generator)
        estimate = torch.where(surfaced.unsqueeze(-1), fix, estimate)
        distance_since_fix = torch.where(
            surfaced, torch.zeros_like(distance_since_fix), distance_since_fix
        )
        mode = torch.where(surfaced, torch.full_like(mode, MODE_DEPLOYED), mode)

        home = (mode == MODE_RETURNING) & (range_to_base < self._config.fleet.arrival_radius_m)
        mode = torch.where(home, torch.full_like(mode, MODE_CHARGING), mode)
        timer = torch.where(home, torch.full_like(timer, self._turnaround_s), timer)
        estimate = torch.where(home.unsqueeze(-1), bases, estimate)
        distance_since_fix = torch.where(
            home, torch.zeros_like(distance_since_fix), distance_since_fix
        )

        # Alongside for the handling plus the charge, not merely until the
        # battery reads full: a vehicle does not turn itself round.
        relaunched = charging & (timer <= 0.0)
        mode = torch.where(relaunched, torch.full_like(mode, MODE_DEPLOYED), mode)

        # Dry and away from base. The vehicle is gone for the episode.
        lost = (energy <= 0.0) & (range_to_base >= self._config.fleet.arrival_radius_m)
        mode = torch.where(lost, torch.full_like(mode, MODE_LOST), mode)

        return replace(
            state,
            xy=xy,
            estimate=estimate,
            heading=heading,
            regime=regime,
            energy_j=energy,
            mode=mode,
            distance_since_fix=distance_since_fix,
            timer_s=timer,
        )

    def _return_margin(self, range_to_base: torch.Tensor) -> torch.Tensor:
        """Return the energy a vehicle must keep in hand to reach its base."""
        auv = self._config.auv
        cruise = auv.regimes.cruise_ms
        # Inflated by the measured detour. The range here is a straight line,
        # and in this archipelago the navigable way home is 1.39x that at the
        # median -- so a margin sized on the straight line sends a vehicle
        # back with too little in hand and it dies on the way. Measured by
        # Dijkstra over the navigable grid; the factor is deliberately the
        # median rather than the tail, because sizing for a 4x worst case
        # would keep the whole fleet alongside.
        transit = _DETOUR * power.power_w(cruise, auv) * range_to_base / cruise
        return transit + auv.energy_reserve_fraction * auv.battery_j
