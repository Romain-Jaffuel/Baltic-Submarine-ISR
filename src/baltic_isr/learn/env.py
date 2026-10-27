"""The environment: everything from batches 2 to 5, stepped together.

One decision is fifteen simulation steps. The world runs at 60 s because a cargo
at 6 m/s should not jump a sensor range between two detections; the policy acts
at 900 s because a finer cadence buys nothing and costs discount horizon
(specification section 6.2).

What a vehicle sees is **its own picture**, never the truth: its believed
position, the tracks it holds or has been told about, the water it has swept.
Two vehicles side by side can disagree, and usually do -- that is the whole
point of the task, and it is why the observation is assembled per vehicle rather
than sliced out of a global state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch

from baltic_isr import seeds
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.fleet import navigation
from baltic_isr.fleet.auv import Fleet, FleetState
from baltic_isr.fleet.comms import Comms, Links
from baltic_isr.picture import tracks as tracks_module
from baltic_isr.picture.plausibility import Plausibility
from baltic_isr.picture.presence import Presence, Search
from baltic_isr.picture.tracks import Picture, Tracks
from baltic_isr.sensing import acoustics
from baltic_isr.sensing.detector import Detector
from baltic_isr.world import statics as statics_module
from baltic_isr.world.statics import Statics
from baltic_isr.world.traffic import Traffic, TrafficState


class Observer(Protocol):
    """Something that wants to watch a trace run go by."""

    def __call__(
        self, moment: float, state: EnvState, links: torch.Tensor | None
    ) -> None:  # pragma: no cover - a shape, not an implementation
        """Record one physics tick."""
        ...


PATCH = 7
PATCH_CELLS = PATCH * PATCH
NEAREST_TRACKS = 6
TRACK_FEATURES = 6
SECTORS = 8
# ... plus four for the most neglected stretch of cable: a bearing, how
# stale it is, and how far. Added after a training run paid for cable
# coverage while giving the policy no way to see which stretch needed it.
SELF_FEATURES = 2 + 2 + 1 + 3 + 1 + 3 + 2 + 2 + 2 + 4
OBSERVATION = 3 * PATCH_CELLS + NEAREST_TRACKS * TRACK_FEATURES + SECTORS + SELF_FEATURES

# The oracle of specification section 6.1: the upper bound. It sees the truth
# rather than its own stale picture, and the comms have no effect on it, so the
# gap between it and the decentralised policy IS the cost of decentralisation
# and of the 250-byte budget. Measured, not argued.
#
# Assembled per vehicle rather than once per world, so the same network and the
# same PPO serve both: the only difference between the two policies is then
# what they are told, which is exactly the comparison worth making.
ORACLE_COLUMNS = 8
ORACLE_ROWS = 6
ORACLE_CELLS = ORACLE_COLUMNS * ORACLE_ROWS
ORACLE_CABLE = 24
ORACLE_GLOBAL = 3 * ORACLE_CELLS + ORACLE_CABLE
ORACLE_VEHICLE = 8

SIM_STEP_S = 60.0


@dataclass(frozen=True)
class EnvState:
    """Everything the environment carries between decisions."""

    t: float
    fleet: FleetState
    traffic: TrafficState
    tracks: Tracks
    shore: Tracks
    presence: Presence
    first_seen_s: torch.Tensor
    deviation_seen_s: torch.Tensor
    cable_heard_s: torch.Tensor


@dataclass(frozen=True)
class StepResult:
    """One decision, from every vehicle's point of view."""

    observation: torch.Tensor
    reward: torch.Tensor
    done: bool
    alarms: torch.Tensor
    swept: torch.Tensor
    # Vehicle-seconds spent with something inside detection range, summed over
    # contacts. Specification section 7 asks for it and nothing was producing
    # it: `Scorer.add_contact_time` existed from lot 6 and was never called, so
    # `verification_cost` read 0.0 for every policy ever scored. A metric that
    # silently reports zero is worse than one that is missing.
    contact_s: torch.Tensor


class Environment:
    """The whole simulation, batched over worlds, with one shared policy.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    worlds : int
        How many worlds to run in parallel.
    device : torch.device or str
        Where to run.
    """

    def __init__(
        self,
        config: Config,
        worlds: int,
        device: torch.device | str = "cpu",
        sees: str = "local",
        oracle_fleet_block: bool = False,
    ) -> None:
        if sees not in ("local", "oracle"):
            raise ValueError(f"unknown observation {sees!r}")
        # Specification 6.3 asks the oracle for the state of all N vehicles.
        # Taken literally that is a flat 320-vector whose slot order is
        # arbitrary -- the network has to learn which slot is which vehicle,
        # and the assignment means nothing. The local block already carries
        # the same fact in eight permutation-invariant numbers (the neighbour
        # sectors), so the 320 add dilution, not information. Off by default,
        # kept so the measurement can be repeated.
        self.oracle_fleet_block = oracle_fleet_block
        # "local" is the system: a vehicle's own sensor and its own stale
        # picture. "oracle" is the upper bound of specification section 6.1,
        # which sees the truth. Everything downstream -- the network, PPO, the
        # training loop -- is identical for both, so the measured gap between
        # them is the cost of decentralisation and nothing else.
        self.sees = sees
        self._config = config
        self._worlds = worlds
        self._device = torch.device(device)
        self._statics: Statics = statics_module.build(config, device)
        self._seed = config.simulation.seed

        self._traffic = Traffic(config, self._statics, worlds)
        self._fleet = Fleet(config, self._statics, worlds)
        self._detector = Detector(config, self._statics, self._generator("detector"))
        self._comms = Comms(config, self._statics, self._generator("comms"))
        self._picture = Picture(config, self._statics)
        self._judge = Plausibility(config, self._statics)
        self._search = Search(config, self._statics, worlds, config.fleet.count)

        self._reach = torch.tensor(
            [
                acoustics.detection_range_m(
                    speed,
                    max(item.source_level_db for item in config.traffic.classes),
                    config.acoustics,
                    config.environment,
                )
                for speed in (
                    config.auv.regimes.drift_ms,
                    config.auv.regimes.cruise_ms,
                    config.auv.regimes.sprint_ms,
                )
            ],
            dtype=torch.float32,
            device=self._device,
        )
        rows, columns = self._search.shape
        self._rows, self._columns = rows, columns

    def _generator(self, name: str) -> torch.Generator:
        return seeds.torch_generator(self._seed, name, self._device)

    @property
    def n_vehicles(self) -> int:
        """Return how many vehicles each world holds."""
        return self._config.fleet.count

    @property
    def config(self) -> Config:
        """Return the resolved configuration."""
        return self._config

    @property
    def picture(self) -> Picture:
        """Return the prediction model, for reading ellipses back."""
        return self._picture

    @property
    def search(self) -> Search:
        """Return the search grid, for drawing."""
        return self._search

    @property
    def statics(self) -> Statics:
        """Return the fixed geometry."""
        return self._statics

    def reach_m(self, regime: torch.Tensor) -> torch.Tensor:
        """Return each vehicle's detection range right now.

        Parameters
        ----------
        regime : Tensor
            ``(worlds, vehicles)``.

        Returns
        -------
        Tensor
            Range in metres, same shape.
        """
        return self._reach[regime]

    def reset(self, seed: int | None = None) -> tuple[EnvState, torch.Tensor]:
        """Start an episode.

        Parameters
        ----------
        seed : int, optional
            Overrides the configured seed, for evaluation on a fixed scenario.

        Returns
        -------
        tuple
            The state, and the first observation.
        """
        if seed is not None:
            self._seed = seed
            self._detector = Detector(self._config, self._statics, self._generator("detector"))
            self._comms = Comms(self._config, self._statics, self._generator("comms"))
        worlds, vessels = self._worlds, self._config.traffic.n_vessels
        state = EnvState(
            t=0.0,
            fleet=self._fleet.reset(seed),
            traffic=self._traffic.reset(seed),
            tracks=tracks_module.empty(worlds, self.n_vehicles, vessels, self._device),
            shore=self._comms.shore(worlds, vessels),
            presence=self._search.empty(),
            first_seen_s=torch.full((worlds, vessels), torch.inf, device=self._device),
            deviation_seen_s=torch.full((worlds, vessels), torch.inf, device=self._device),
            cable_heard_s=torch.full(
                (worlds, self._statics.cable_xy.shape[0]),
                -self._config.reward.cable_horizon_s,
                device=self._device,
            ),
        )
        return state, self.observe(state)

    def step(
        self,
        state: EnvState,
        actions: torch.Tensor,
        observer: Observer | None = None,
    ) -> tuple[EnvState, StepResult]:
        """Advance one decision: fifteen simulation steps, then the reward.

        Parameters
        ----------
        state : EnvState
            Current state.
        actions : Tensor
            ``(worlds, vehicles)`` in ``[0, N_ACTIONS)``.
        observer : Observer, optional
            Called once per physics tick, for recording a trace. **Never passed
            during training**: it reads tensors back to the host, which is the
            one thing the step is otherwise forbidden to do. When it is None the
            cost is a single comparison against None per tick.

        Returns
        -------
        tuple
            The new state, and what the decision produced.
        """
        config = self._config
        ticks = max(1, round(config.simulation.decision_interval_s / SIM_STEP_S))
        every = max(1, round(config.simulation.perception_interval_s / SIM_STEP_S))
        gate = self._picture.gate_m(state.tracks)
        before = self._picture.uncertainty_m(state.tracks, state.t).clamp(max=gate)
        energy_before = state.fleet.energy_j
        potential_before = self._cable_potential(state)

        fleet, traffic = state.fleet, state.traffic
        tracks, shore, presence = state.tracks, state.shore, state.presence
        first_seen, deviation_seen = state.first_seen_s, state.deviation_seen_s
        cable_heard = state.cable_heard_s
        alarms = torch.zeros_like(fleet.energy_j)
        swept = torch.zeros_like(fleet.energy_j)
        relayed = torch.zeros_like(fleet.energy_j)
        guarded = torch.zeros_like(fleet.energy_j)
        contact_s = torch.zeros_like(fleet.energy_j)
        moment = state.t

        for tick in range(ticks):
            moment += SIM_STEP_S
            fleet = self._fleet.step(fleet, actions, moment, SIM_STEP_S)
            traffic = self._traffic.step(traffic, moment, SIM_STEP_S)

            # Hulls move every tick; ears, tracks and radios every fifth. The
            # observer still fires every tick, because motion is what wants the
            # frame rate -- 2880 frames over 48 h is 48 a second of film, where
            # perception ticks alone would give ten.
            perceive = (tick + 1) % every == 0 or tick + 1 == ticks
            if not perceive:
                if observer is not None:
                    observer(
                        moment,
                        self._snapshot(
                            moment,
                            fleet,
                            traffic,
                            tracks,
                            shore,
                            presence,
                            first_seen,
                            deviation_seen,
                            cable_heard,
                        ),
                        None,
                    )
                continue

            contacts = self._detector.listen(
                fleet.xy,
                fleet.estimate,
                fleet.regime,
                fleet.listening,
                traffic.xy,
                traffic.source_level_db,
                traffic.vessel_class,
                torch.ones_like(traffic.deviating),
                trials=every,
            )
            verdict = self._judge.assess(
                contacts.position,
                traffic.velocity.unsqueeze(1).expand(-1, self.n_vehicles, -1, -1),
                contacts.vessel_class,
            )
            was_mine = tracks.issuer == torch.arange(
                self.n_vehicles, device=self._device, dtype=torch.int16
            ).reshape(1, -1, 1)
            tracks, deviated = self._picture.update(
                tracks,
                moment,
                contacts.detected,
                contacts.position,
                fleet.estimate,
                verdict.plausible,
                self._growth(traffic, contacts.vessel_class),
                traffic.rail.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.progress.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.direction.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.velocity.unsqueeze(1).expand(-1, self.n_vehicles, -1, -1),
                navigation.uncertainty_m(fleet.distance_since_fix, config.navigation),
            )
            alarms = alarms + (deviated & traffic.deviating.unsqueeze(1)).float().sum(-1)
            relayed = relayed + (contacts.detected & ~was_mine & tracks.known).float().sum(-1)
            # Counted per (vehicle, vessel) pair, so sitting beside one cargo
            # for an hour costs what it should: this is the term that punishes
            # escorting rather than releasing with a prediction.
            contact_s = contact_s + contacts.detected.float().sum(-1) * (every * SIM_STEP_S)

            heard = contacts.detected.any(dim=1)
            first_seen = torch.where(
                heard & ~torch.isfinite(first_seen), torch.full_like(first_seen, moment), first_seen
            )
            caught = (deviated & traffic.deviating.unsqueeze(1)).any(dim=1)
            deviation_seen = torch.where(
                caught & ~torch.isfinite(deviation_seen),
                torch.full_like(deviation_seen, moment),
                deviation_seen,
            )

            links = self._comms.links(fleet.xy, self._velocity(fleet), fleet.listening)
            tracks, _ = self._comms.exchange(tracks, links)
            surfaced = fleet.mode == auv_module.MODE_SURFACING
            tracks, shore, _ = self._comms.surface_exchange(tracks, shore, surfaced, traffic.ais_on)

            presence, removed = self._search.step(
                presence,
                fleet.estimate,
                fleet.listening,
                self.reach_m(fleet.regime),
                every * SIM_STEP_S,
            )
            swept = swept + removed
            cable_heard, refreshed = self._watch_cables(
                cable_heard, fleet, moment, every * SIM_STEP_S
            )
            guarded = guarded + refreshed
            if observer is not None:
                observer(
                    moment,
                    self._snapshot(
                        moment,
                        fleet,
                        traffic,
                        tracks,
                        shore,
                        presence,
                        first_seen,
                        deviation_seen,
                        cable_heard,
                    ),
                    links.reachable,
                )

        after = self._picture.uncertainty_m(tracks, moment).clamp(max=gate)
        new = EnvState(
            t=moment,
            fleet=fleet,
            traffic=traffic,
            tracks=tracks,
            shore=shore,
            presence=presence,
            first_seen_s=first_seen,
            deviation_seen_s=deviation_seen,
            cable_heard_s=cable_heard,
        )
        # Potential-based shaping, which needs the state on both sides of the
        # decision, so it is built here rather than inside `_reward`.
        settings = self._config.reward
        shaping = settings.cable_shaping * (
            settings.shaping_discount * self._cable_potential(new) - potential_before
        )
        reward = (
            self._reward(
                before=before,
                after=after,
                gate=gate,
                tracks=tracks,
                alarms=alarms,
                relayed=relayed,
                swept=swept,
                guarded=guarded,
                energy_spent=(energy_before - fleet.energy_j).clamp_min(0.0),
                lost=fleet.mode == auv_module.MODE_LOST,
                expired=self._picture.stale(tracks, moment),
            )
            + shaping
        )
        return new, StepResult(
            observation=self.observe(new),
            reward=reward,
            done=moment >= self._config.simulation.episode_s,
            alarms=alarms,
            swept=swept,
            contact_s=contact_s,
        )

    def _snapshot(
        self,
        moment: float,
        fleet: FleetState,
        traffic: TrafficState,
        tracks: Tracks,
        shore: Tracks,
        presence: Presence,
        first_seen: torch.Tensor,
        deviation_seen: torch.Tensor,
        cable_heard: torch.Tensor,
    ) -> EnvState:
        """Bundle the mid-decision state for an observer."""
        return EnvState(
            t=moment,
            fleet=fleet,
            traffic=traffic,
            tracks=tracks,
            shore=shore,
            presence=presence,
            first_seen_s=first_seen,
            deviation_seen_s=deviation_seen,
            cable_heard_s=cable_heard,
        )

    def _velocity(self, fleet: FleetState) -> torch.Tensor:
        """Return each vehicle's velocity vector, for the Doppler."""
        speed = self._fleet_speed(fleet)
        return torch.stack(
            [speed * torch.sin(fleet.heading), speed * torch.cos(fleet.heading)], dim=-1
        )

    def _fleet_speed(self, fleet: FleetState) -> torch.Tensor:
        regimes = self._config.auv.regimes
        table = torch.tensor(
            [regimes.drift_ms, regimes.cruise_ms, regimes.sprint_ms], device=self._device
        )
        return torch.where(fleet.at_sea, table[fleet.regime], torch.zeros_like(fleet.energy_j))

    def _growth(self, traffic: TrafficState, reported: torch.Tensor) -> torch.Tensor:
        """Return the ellipse growth implied by the class the observer heard."""
        table = torch.tensor(
            [item.growth_m_per_s for item in self._config.traffic.classes], device=self._device
        )
        del traffic
        return table[reported]

    def _watch_cables(
        self, heard_s: torch.Tensor, fleet: FleetState, moment: float, dt_s: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the cable clock, and credit whoever refreshed it.

        The reward used to say nothing about the cables at all. A vehicle
        could see where they were -- the corridor distance is in its
        observation -- and was never paid a thing for watching one, while the
        map the project is judged on shades exactly that. A policy trained on
        the old reward would have optimised one quantity while the film
        measured another, and nobody could have said whether it had learned.

        The credit is **per vehicle and proportional to the staleness it
        clears**: refreshing a stretch heard a minute ago is worth nothing,
        refreshing one last heard six hours back is worth full marks. That
        makes the signal dense and locally attributable, which is what PPO
        needs -- a shared penalty on total exposure would tell forty vehicles
        the same thing whatever each of them did.

        Parameters
        ----------
        heard_s : Tensor
            When each cable sample was last within earshot, ``(worlds, samples)``.
        fleet : FleetState
            Current fleet.
        moment : float
            Now.
        dt_s : float
            Seconds this perception tick stands for.

        Returns
        -------
        tuple
            The updated clock, and the credit per vehicle ``(worlds, vehicles)``.
        """
        samples = self._statics.cable_xy
        if samples.shape[0] == 0:
            return heard_s, torch.zeros_like(fleet.energy_j)

        gap = (fleet.xy.unsqueeze(2) - samples.reshape(1, 1, -1, 2)).norm(dim=-1)
        within = (gap <= self.reach_m(fleet.regime).unsqueeze(-1)) & fleet.listening.unsqueeze(-1)
        # NO UPPER CLAMP, and that is the whole lesson of the first training
        # run. Clamped at one, a stretch unheard for fifty-four hours was worth
        # exactly what one unheard for six was worth -- so the cheapest way to
        # earn the term was to work a nearby subset over and over and never
        # travel to the rest. The policy duly did: it refreshed the median
        # sample MORE often than the lawnmower (54 visits against 36) while
        # abandoning **a quarter of the network entirely**, and scored 0.626 on
        # a term meant to measure coverage while posting the worst coverage of
        # the three. Clearing fifty-four hours really is nine times clearing
        # six, and the arithmetic has to say so or the incentive points at the
        # nearest stale sample forever. The episode bounds this at about nine.
        #
        # Credit in stale-sample-units, not as a fraction of the whole cable
        # network. Dividing by all 226 samples made a vehicle sitting on a
        # forgotten cable worth 0.4 % of a decision's reward -- an incentive
        # nothing can feel. One unit is one sample refreshed from fully stale.
        staleness = ((moment - heard_s) / self._config.reward.cable_horizon_s).clamp_min(0.0)
        earned = within.float() * staleness.unsqueeze(1)
        if self._config.reward.cable_shared:
            # One refresh pays one refresh. The clock below resets once however
            # many vehicles are in range, so paying each of them the full
            # staleness made the fleet pay three times for one piece of work --
            # 66 % of the credit was going out twice or more.
            earned = earned / within.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        credit = earned.sum(dim=-1)

        covered = within.any(dim=1)
        updated = torch.where(covered, torch.full_like(heard_s, moment), heard_s)
        del dt_s
        return updated, credit

    def _reward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        gate: torch.Tensor,
        tracks: Tracks,
        alarms: torch.Tensor,
        relayed: torch.Tensor,
        swept: torch.Tensor,
        guarded: torch.Tensor,
        energy_spent: torch.Tensor,
        lost: torch.Tensor,
        expired: torch.Tensor,
    ) -> torch.Tensor:
        """Build the per-vehicle reward of specification section 6.4."""
        settings = self._config.reward
        gained = ((before - after) / gate).clamp_min(0.0)
        mine = tracks.issuer == torch.arange(
            self.n_vehicles, device=self._device, dtype=torch.int16
        ).reshape(1, -1, 1)
        weighted = torch.where(mine, gained, gained * settings.relay_multiplier)
        if settings.verification_shared:
            # Divide the credit for one contact among everyone who just earned
            # it. Before this, "coordination" was operationalised as the relay
            # bonus -- 1.6x for verifying a track that came from somebody else
            # -- and a track only arrives from somebody else if you are inside
            # their 4 km modem range. So the reward paid vehicles to huddle,
            # and 63 % of them sat inside a neighbour's modem range. Sharing
            # turns the incentive into "cover what the others do not".
            weighted = weighted / (gained > 0.0).float().sum(dim=1, keepdim=True).clamp_min(1.0)
        del relayed

        verification = settings.verification * weighted.sum(dim=-1)
        alarm = settings.alarm * alarms
        # A FRACTION of what the vehicle holds, not a tally of it. Counting
        # expiries made the penalty scale with how much the fleet knew: 180
        # vessels, most tracks stale, 0.1 each, and the term reached -1800 over
        # an episode -- a thousand times the verification signal it was meant to
        # nudge. Specification 6.4 asks for small and shared.
        holding = tracks.known.float().sum(dim=-1).clamp_min(1.0)
        share = expired.float().sum(dim=-1) / holding
        staleness = settings.staleness * share.mean(dim=-1, keepdim=True)
        exploration = settings.exploration * swept
        # What the film is judged on, finally in the thing being optimised.
        cable = settings.cable * guarded
        energy = settings.energy * energy_spent / self._config.auv.battery_j
        penalty = settings.loss * lost.float()
        return verification + alarm + exploration + cable - staleness - energy - penalty

    # --- observation ---------------------------------------------------------

    @property
    def observation_width(self) -> int:
        """Return how wide this environment's observation is."""
        if self.sees == "oracle":
            fleet = self._config.fleet.count if self.oracle_fleet_block else 0
            truth = NEAREST_TRACKS * TRACK_FEATURES
            return OBSERVATION + truth + ORACLE_GLOBAL + fleet * ORACLE_VEHICLE
        return OBSERVATION

    def observe(self, state: EnvState) -> torch.Tensor:
        """Assemble what every vehicle can see.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, observation_width)``.
        """
        if self.sees == "oracle":
            return self.observe_oracle(state)
        return self._observe_local(state)

    def _observe_local(self, state: EnvState) -> torch.Tensor:
        """Assemble the decentralised observation: sensor, messages, own picture."""
        fleet = state.fleet
        worlds, vehicles = fleet.xy.shape[0], fleet.xy.shape[1]
        extent = self._statics.extent

        patches = self._patches(state)
        nearest = self._nearest_tracks(state)
        sectors = self._neighbour_sectors(fleet)

        believed = fleet.estimate / extent * 2.0 - 1.0
        heading = torch.stack([torch.sin(fleet.heading), torch.cos(fleet.heading)], dim=-1)
        energy = (fleet.energy_j / self._config.auv.battery_j).unsqueeze(-1)
        regime = torch.nn.functional.one_hot(fleet.regime, auv_module.REGIMES).to(believed.dtype)

        to_base = self._statics.bases[fleet.base] - fleet.estimate
        base_range = to_base.norm(dim=-1, keepdim=True)
        base = torch.cat([to_base / base_range.clamp_min(1.0), base_range / extent.max()], dim=-1)
        margin = (
            self._fleet.distance_to_base(fleet)
            * self._config.auv.hotel_power_w
            / self._config.auv.battery_j
        ).unsqueeze(-1)

        drift = navigation.uncertainty_m(fleet.distance_since_fix, self._config.navigation)
        nav = torch.stack(
            [fleet.distance_since_fix / 100_000.0, drift / self._config.fleet.gate_m], dim=-1
        )

        corridor = statics_module.sample_raster(
            self._statics, self._statics.corridor_distance_m, fleet.estimate
        )
        rail = statics_module.sample_raster(
            self._statics, self._statics.rail_distance_m, fleet.estimate
        )
        geography = torch.stack(
            [corridor / 50_000.0, rail / 50_000.0, corridor / 50_000.0, rail / 50_000.0], dim=-1
        )

        parts = [
            patches,
            nearest,
            sectors,
            believed,
            heading,
            energy,
            regime,
            margin,
            base,
            nav,
            geography,
            self._forgotten_cable(state),
        ]
        observation = torch.cat(parts, dim=-1)
        assert observation.shape == (worlds, vehicles, OBSERVATION), observation.shape
        return observation

    def observe_oracle(self, state: EnvState) -> torch.Tensor:
        """Assemble what a commander who could see everything would see.

        **The whole local observation, with every belief replaced by truth**,
        and the global blocks on top. That construction is the point: an upper
        bound has to *dominate* the thing it bounds, and the first version of
        this did not. It gave the truth on an eight-by-six raster -- cells of
        17 by 18 km, three times coarser than the detection range -- while the
        decentralised policy got six-kilometre cells, its six nearest contacts
        with their uncertainty, and a bearing to the most neglected cable. It
        knew more and could aim less, and it lost on every metric to the
        policy it was supposed to bound. A bound that loses is not a bound.

        So the oracle now gets the same blocks from the same code, fed a
        perfect picture: exact navigation, every vessel known, no staleness,
        no covariance, no messages. Whatever it cannot do is then a limit of
        the fleet rather than of what it was told, and the gap to the
        decentralised policy is the price of a sonar and 250 bytes.

        Parameters
        ----------
        state : EnvState
            Current state, truth included.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, observation_width)``.
        """
        fleet, traffic = state.fleet, state.traffic
        worlds, vehicles = fleet.xy.shape[0], fleet.xy.shape[1]
        extent = self._statics.extent

        # The REAL local observation, not one built from a perfect picture.
        # Building it from truth was the second mistake: verification is 75 %
        # of the reward and pays for reducing sigma on the **fleet's own**
        # table, so a perfect picture made three of the six track features
        # dead constants -- uncertainty 0.0008 of spread, plausibility and age
        # flat -- while the real sigma on held tracks sat at a median of 1542
        # m. The oracle was being graded on an exam it was not shown.
        #
        # It must SEE the picture because the picture's staleness is what it
        # is paid to fix; it simply is not LIMITED by it, because the truth
        # arrives beside it. Those are different things and I had merged them.
        local = self._observe_local(state)
        truth = self._nearest_threats(state)

        traffic_grid = self._raster(traffic.xy, extent, torch.ones_like(traffic.xy[..., 0]))
        deviating_grid = self._raster(traffic.xy, extent, traffic.deviating.float())
        fleet_grid = self._raster(fleet.xy, extent, torch.ones_like(fleet.xy[..., 0]))

        stride = max(self._statics.cable_xy.shape[0] // ORACLE_CABLE, 1)
        clock = state.cable_heard_s[:, ::stride][:, :ORACLE_CABLE]
        cable = ((state.t - clock) / self._config.reward.cable_horizon_s / 8.0).clamp(0.0, 1.0)
        if cable.shape[1] < ORACLE_CABLE:
            cable = torch.nn.functional.pad(cable, (0, ORACLE_CABLE - cable.shape[1]))

        shared = torch.cat([traffic_grid, deviating_grid, fleet_grid, cable], dim=-1)
        parts = [local, truth, shared.unsqueeze(1).expand(-1, vehicles, -1)]
        if self.oracle_fleet_block:
            own = self._vehicle_block(fleet, extent)
            parts.append(own.reshape(worlds, 1, vehicles * ORACLE_VEHICLE).expand(-1, vehicles, -1))
        return torch.cat(parts, dim=-1)

    def _cable_potential(self, state: EnvState) -> torch.Tensor:
        """Return each vehicle's closeness to the cable most worth going to.

        The potential of a potential-based shaping term. Shaping of the form
        ``gamma * phi(s') - phi(s)`` leaves the optimal policy unchanged
        whatever ``phi`` is, which is why it is the safe way to make a distant
        goal visible: raising the discount instead would change what the
        policy is optimising, not just what it can see.

        And something has to. The discount horizon is 22 km of lookahead,
        while the far end of EstLink 2 sits 79 km from the nearest base -- the
        trip is invisible to the value function, and 45 % of that cable was
        never approached against 16-24 % for the others.

        Parameters
        ----------
        state : EnvState
            Current state.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)``, higher when closer to neglected cable.
        """
        samples = self._statics.cable_xy
        if samples.shape[0] == 0:
            return torch.zeros_like(state.fleet.energy_j)
        horizon = self._config.reward.cable_horizon_s
        staleness = ((state.t - state.cable_heard_s) / horizon).clamp_min(0.0)
        gap = (samples.reshape(1, 1, -1, 2) - state.fleet.xy.unsqueeze(2)).norm(dim=-1)
        worth = staleness.unsqueeze(1) / (1.0 + gap / self._config.fleet.gate_m)
        chosen = worth.argmax(dim=-1, keepdim=True)
        distance: torch.Tensor = gap.gather(2, chosen).squeeze(-1)
        return -distance / float(self._statics.extent.max())

    def _nearest_threats(self, state: EnvState) -> torch.Tensor:
        """Return the ground truth about the vessels that matter, per vehicle.

        The oracle's reason to exist. Feeding it ``_nearest_tracks`` on a
        perfect picture was tried and is almost useless: that block ranks by
        **urgency**, and on a picture with no uncertainty every urgency is
        zero, so it returned six vessels chosen by a tie-break and carried no
        deviation flag at all. Truth has to be asked a different question.

        Deviating vessels first, nearest first within that, and each one
        carries what makes it a target: where it is, whether it is deviating,
        and whether it is over a cable. That is the ground truth of
        specification section 6.3, in the form the eight-heading action space
        can act on.

        Parameters
        ----------
        state : EnvState
            Current state, truth included.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, NEAREST_TRACKS * TRACK_FEATURES)``.
        """
        traffic = state.traffic
        offset = traffic.xy.unsqueeze(1) - state.fleet.xy.unsqueeze(2)
        distance = offset.norm(dim=-1)

        over_cable = (
            statics_module.sample_raster(
                self._statics, self._statics.corridor_distance_m, traffic.xy
            )
            <= self._config.corridors[0].half_width_m
        )
        deviating = traffic.deviating.unsqueeze(1).expand_as(distance)
        threatening = (traffic.deviating & over_cable).unsqueeze(1).expand_as(distance)

        # Over a cable and misbehaving outranks merely misbehaving, which
        # outranks everything else; distance breaks the tie inside each class.
        rank = threatening.float() * 2.0 + deviating.float() - distance / 1e6
        order = rank.topk(NEAREST_TRACKS, dim=-1).indices

        picked = offset.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
        span = picked.norm(dim=-1).clamp_min(1.0)
        speed = traffic.speed.unsqueeze(1).expand_as(distance)
        features = torch.stack(
            [
                picked[..., 0] / span,
                picked[..., 1] / span,
                (span / 50_000.0).clamp(0.0, 2.0),
                deviating.gather(2, order).to(span.dtype),
                threatening.gather(2, order).to(span.dtype),
                (speed.gather(2, order) / 10.0).clamp(0.0, 2.0),
            ],
            dim=-1,
        )
        return features.reshape(*features.shape[:2], NEAREST_TRACKS * TRACK_FEATURES)

    def _perfect(self, state: EnvState) -> Tracks:
        """Return the track table a fleet with no sensor limits would hold.

        Every vessel, at its true position and velocity, seen this instant,
        with a covariance small enough to be noise. The plausibility flag
        carries the **ground truth** of deviation rather than a model's guess,
        which is what specification section 6.3 asks the oracle to be given.
        """
        traffic = state.traffic
        tracks = state.tracks
        worlds, vehicles, vessels = tracks.last_seen_s.shape

        def spread(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(1).expand(-1, vehicles, *value.shape[1:])

        tiny = torch.full((worlds, vehicles, vessels), 1.0, device=self._device)
        zero = torch.zeros_like(tiny)
        truth = spread(traffic.deviating)
        me = torch.arange(vehicles, device=self._device, dtype=torch.int16)
        # Every field from the truth, not one carried over from the live
        # table: rail and progress feed the lane predictor, and leaving them
        # as the fleet believed them would let the picture -- and therefore
        # the comms -- back into an observation that must not depend on them.
        return replace(
            tracks,
            last_seen_s=torch.full_like(tracks.last_seen_s, state.t),
            xy=spread(traffic.xy),
            velocity=spread(traffic.velocity),
            cov=torch.stack([tiny, zero, tiny], dim=-1),
            rail=spread(traffic.rail).to(tracks.rail.dtype),
            progress=spread(traffic.progress),
            direction=spread(traffic.direction),
            growth_m_per_s=self._growth(traffic, spread(traffic.vessel_class)),
            issuer=me.reshape(1, vehicles, 1).expand_as(tracks.issuer),
            plausible=~truth,
            alarmed=truth,
        )

    def _vehicle_block(self, fleet: FleetState, extent: torch.Tensor) -> torch.Tensor:
        """Return the ``N x 8`` fleet state of specification section 6.3."""
        return torch.cat(
            [
                fleet.xy / extent * 2.0 - 1.0,
                torch.stack([torch.sin(fleet.heading), torch.cos(fleet.heading)], dim=-1),
                (fleet.energy_j / self._config.auv.battery_j).unsqueeze(-1),
                torch.nn.functional.one_hot(fleet.regime, auv_module.REGIMES).to(fleet.xy.dtype),
            ],
            dim=-1,
        )

    def _raster(self, xy: torch.Tensor, extent: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Bin points onto the coarse oracle grid, ``(worlds, cells)``."""
        cell = (xy / extent * torch.tensor([ORACLE_COLUMNS, ORACLE_ROWS], device=self._device)).to(
            torch.long
        )
        column = cell[..., 0].clamp(0, ORACLE_COLUMNS - 1)
        row = cell[..., 1].clamp(0, ORACLE_ROWS - 1)
        flat = row * ORACLE_COLUMNS + column
        grid = torch.zeros(xy.shape[0], ORACLE_CELLS, device=self._device)
        grid.scatter_add_(1, flat, weight)
        return grid / max(float(xy.shape[1]) / ORACLE_CELLS, 1.0)

    def _forgotten_cable(self, state: EnvState) -> torch.Tensor:
        """Return where the most neglected stretch of cable is, per vehicle.

        The first training run paid for keeping cables heard and gave the
        policy no way to tell which one needed hearing: the observation
        carried the *distance* to a corridor and never its clock. A policy
        that cannot see which stretch is stale can only learn a fixed spatial
        habit, not a patrol -- and that is exactly what came out, a fleet that
        worked the convenient half of the network and abandoned a quarter of
        it. An incentive you cannot aim at is a lottery.

        Four numbers: the way to the stalest stretch, how stale it is, and how
        far. The direction is a unit vector so the policy reads a bearing
        rather than a map position, which is the form the eight-heading action
        space can act on directly.

        Parameters
        ----------
        state : EnvState
            Current state, for the cable clock.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, 4)``.
        """
        fleet = state.fleet
        samples = self._statics.cable_xy
        if samples.shape[0] == 0:
            return torch.zeros(*fleet.xy.shape[:2], 4, device=self._device)

        horizon = self._config.reward.cable_horizon_s
        staleness = ((state.t - state.cable_heard_s) / horizon).clamp_min(0.0)
        offset = samples.reshape(1, 1, -1, 2) - fleet.estimate.unsqueeze(2)
        gap = offset.norm(dim=-1)

        # Worth going for = stale, and not on the other side of the gulf. The
        # ratio is the quantity the policy is actually being paid on, per
        # kilometre it would have to travel to collect it.
        spread = staleness.unsqueeze(1).expand(-1, gap.shape[1], -1)
        worth = spread / (1.0 + gap / self._config.fleet.gate_m)
        chosen = worth.argmax(dim=-1, keepdim=True)
        bearing = offset.gather(2, chosen.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)
        distance = gap.gather(2, chosen).squeeze(-1)
        # Staleness is divided by eight because an episode is 48 h against a
        # 6 h horizon: that is the most a stretch can be neglected by, so the
        # feature lands in [0, 1] without a clamp that would hide the tail.
        return torch.cat(
            [
                bearing / distance.clamp_min(1.0).unsqueeze(-1),
                (spread.gather(2, chosen).squeeze(-1) / 8.0).unsqueeze(-1),
                (distance / float(self._statics.extent.max())).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _patches(self, state: EnvState) -> torch.Tensor:
        """Return the three 7x7 maps centred on each vehicle."""
        from baltic_isr.picture.presence import PRESENCE_CELL_M

        rows, columns = self._rows, self._columns
        cell = (state.fleet.estimate / PRESENCE_CELL_M).floor().to(torch.long)
        half = PATCH // 2
        offsets = torch.arange(-half, half + 1, device=self._device)
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        column = (cell[..., 0].unsqueeze(-1).unsqueeze(-1) + grid_x).clamp(0, columns - 1)
        row = (cell[..., 1].unsqueeze(-1).unsqueeze(-1) + grid_y).clamp(0, rows - 1)
        flat = (row * columns + column).reshape(*cell.shape[:2], PATCH_CELLS)

        probability = state.presence.probability.gather(2, flat)
        age = (state.presence.age_s.gather(2, flat) / 21_600.0).clamp(0.0, 2.0)
        staleness = self._staleness_map(state).gather(2, flat)
        return torch.cat([probability, age, staleness], dim=-1)

    def _staleness_map(self, state: EnvState) -> torch.Tensor:
        """Return, per cell, how close the tracks predicted into it are to expiry.

        The map of *where my picture is about to lie*, which specification
        section 6.3 calls the heart of the observation. Built by scattering each
        held track into the cell it is predicted to be in.
        """
        from baltic_isr.picture.presence import PRESENCE_CELL_M

        position, _ = self._picture.predict(state.tracks, state.t)
        gate = self._picture.gate_m(state.tracks)
        ratio = (self._picture.uncertainty_m(state.tracks, state.t) / gate).clamp(0.0, 1.0)
        ratio = torch.where(state.tracks.known, ratio, torch.zeros_like(ratio))

        cell = (position / PRESENCE_CELL_M).floor().to(torch.long)
        column = cell[..., 0].clamp(0, self._columns - 1)
        row = cell[..., 1].clamp(0, self._rows - 1)
        index = row * self._columns + column

        shape = (*state.tracks.last_seen_s.shape[:2], self._rows * self._columns)
        out = torch.zeros(shape, device=self._device)
        return out.scatter_reduce(2, index, ratio, reduce="amax", include_self=True)

    def _nearest_tracks(self, state: EnvState) -> torch.Tensor:
        """Return the handful of held contacts each vehicle should care about."""
        tracks = state.tracks
        position, _ = self._picture.predict(tracks, state.t)
        gate = self._picture.gate_m(tracks)
        uncertainty = self._picture.uncertainty_m(tracks, state.t)

        offset = position - state.fleet.estimate.unsqueeze(2)
        distance = offset.norm(dim=-1)
        # Rank by how close to expiry, not by how near: the point of the task is
        # to go where the picture is about to fail, not where it is comfortable.
        urgency = torch.where(
            tracks.known, (uncertainty / gate).clamp(0.0, 2.0), torch.full_like(distance, -1.0)
        )
        order = urgency.topk(NEAREST_TRACKS, dim=-1).indices

        picked = offset.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
        span = picked.norm(dim=-1).clamp_min(1.0)
        mine = tracks.issuer == torch.arange(
            self.n_vehicles, device=self._device, dtype=torch.int16
        ).reshape(1, -1, 1)
        features = torch.stack(
            [
                picked[..., 0] / span,
                picked[..., 1] / span,
                (span / 50_000.0).clamp(0.0, 2.0),
                urgency.gather(2, order),
                tracks.known.gather(2, order).to(span.dtype),
                (~mine).gather(2, order).to(span.dtype),
            ],
            dim=-1,
        )
        return features.reshape(*features.shape[:2], NEAREST_TRACKS * TRACK_FEATURES)

    def _neighbour_sectors(self, fleet: FleetState) -> torch.Tensor:
        """Return how many vehicles sit in each of eight bearings, within modem range."""
        offset = fleet.estimate.unsqueeze(2) - fleet.estimate.unsqueeze(1)
        distance = offset.norm(dim=-1)
        near = (distance > 0.0) & (distance < self._config.comms.range_m)
        angle = torch.atan2(offset[..., 0], offset[..., 1]) % (2.0 * torch.pi)
        sector = (angle / (2.0 * torch.pi / SECTORS)).floor().to(torch.long).clamp(0, SECTORS - 1)
        counts = torch.zeros((*fleet.estimate.shape[:2], SECTORS), device=self._device).scatter_add(
            2, sector, near.to(torch.float32)
        )
        return (counts / 4.0).clamp(0.0, 1.0)

    def links(self, state: EnvState) -> Links:
        """Return the current acoustic graph, for drawing."""
        return self._comms.links(state.fleet.xy, self._velocity(state.fleet), state.fleet.listening)
