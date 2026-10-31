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

import math
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

# The suspect block: where the one contact this vehicle believes is misbehaving
# lies, and whether the fleet is already on it.
#
# `_nearest_tracks` ranks by how close a prediction is to expiring, which is
# the right order for keeping a picture and the wrong one for running down a
# suspect -- a vessel sitting still over a cable has the freshest, least urgent
# track on the board, so it sorted last and the policy never saw it. Six
# numbers: the bearing, the range, THE RANGE AGAIN AT ESCORT SCALE, whether
# there is a suspect at all, how many linked neighbours are already shadowing
# it, whether the report came over the modem rather than from this vehicle's
# own ears, and how stale the verdict is.
#
# The second range is the one that matters and it was missing. Both were
# `distance / 50 000 m`, a scale chosen for keeping a picture across the gulf,
# which puts the ENTIRE paid escort radius inside six hundredths of one unit:
# 0.006 alongside, 0.060 at the rim. Among a hundred and seventy other inputs
# a network cannot resolve that, so the policy could see that it had a suspect
# and roughly which way, and could not see whether it was on top of it.
# Measured consequence: vehicles held station a median 68 min and left 70 % of
# red vessels while they were still red, abandoning 2.0 a decision for 0.5,
# with 80 % of their battery left. The reward said stay at every duration
# tested; the observation could not say where it was.
SUSPECT_FEATURES = 8

OBSERVATION = (
    3 * PATCH_CELLS + NEAREST_TRACKS * TRACK_FEATURES + SECTORS + SUSPECT_FEATURES + SELF_FEATURES
)

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

# Stands in for the range to a contact that is not flagged, so the argmin that
# picks the nearest suspect can never choose one on merit. Larger than the box
# diagonal by a wide margin.
_NOWHERE_M = 1.0e9


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
    # Which vessel each vehicle is the escort of, or -1. `(worlds, vehicles)`.
    #
    # A cap that is recomputed every look is not a cap, it is a lottery. The
    # credit used to pay the nearest `shadow_team`, re-ranked three times a
    # decision; with ten drones milling at similar ranges every one of them
    # spent part of the time in the first five, so the surplus collected 1.335
    # a decision of a term that was supposed to pay it nothing, and staying was
    # worth 1.087 against 0.968 for going back on patrol. The crowd was the
    # correct response to that reward. Rank churn turns exclusion into
    # rationing, and rationing a big enough pot still draws a mob.
    #
    # A claim is held instead of recomputed: once five vehicles are on a
    # vessel, a sixth cannot be paid for it however close it gets, and the
    # answer does not change between looks or between decisions. It is also
    # what "unless they are busy" means operationally -- a vehicle with a claim
    # is not available for the next contact.
    escort_claim: torch.Tensor


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
        # How far inside its hearing a vehicle has to be before a cable counts
        # as watched: the widest corridor half-width, so the whole sensitive
        # zone is covered and not just the line through it. See _cable_credit.
        self._corridor_margin_m = max(
            (corridor.half_width_m for corridor in config.corridors), default=0.0
        )

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
            escort_claim=torch.full(
                (worlds, self.n_vehicles), -1, dtype=torch.int16, device=self._device
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
        suspect_before = self._suspect_potential(state)

        fleet, traffic = state.fleet, state.traffic
        tracks, shore, presence = state.tracks, state.shore, state.presence
        first_seen, deviation_seen = state.first_seen_s, state.deviation_seen_s
        cable_heard = state.cable_heard_s
        alarms = torch.zeros_like(fleet.energy_j)
        swept = torch.zeros_like(fleet.energy_j)
        relayed = torch.zeros_like(fleet.energy_j)
        guarded = torch.zeros_like(fleet.energy_j)
        contact_s = torch.zeros_like(fleet.energy_j)
        shadowed = torch.zeros_like(fleet.energy_j)
        crowded = torch.zeros_like(fleet.energy_j)
        packed = torch.zeros_like(fleet.energy_j)
        looks = 0
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
                            state.escort_claim,
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
            # 0.5 is the loiter sigmoid's own midpoint, not a new constant:
            # above it the contact is running slower than its class usually
            # does. Paired with the corridor test it is the signature the
            # project is named after, and it is the only thing the escort is
            # paid to sit on.
            called = (verdict.loitering > 0.5) & verdict.over_corridor
            tracks, deviated = self._picture.update(
                tracks,
                moment,
                contacts.detected,
                contacts.position,
                fleet.estimate,
                verdict.plausible,
                called,
                self._growth(traffic, contacts.vessel_class),
                traffic.rail.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.progress.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.direction.unsqueeze(1).expand(-1, self.n_vehicles, -1),
                traffic.velocity.unsqueeze(1).expand(-1, self.n_vehicles, -1, -1),
                navigation.uncertainty_m(fleet.distance_since_fix, config.navigation),
            )
            # A deviation is noticed when the fleet CONCLUDES one, not only when
            # a track contradicts its prediction by three sigma. `deviated` is a
            # tracking event and it is structurally blind to the thing this
            # project is named after: a vessel that stops makes its prediction
            # run away, the covariance grows, the track EXPIRES, and `held` goes
            # false before the residual ever trips the gate -- so a long loiter
            # raises no alarm at all, it just restarts the track. Measured, the
            # tracking test notices 33-48 % of scheduled deviations while the
            # plausibility judgement notices 94-100 % of them.
            #
            # Gated on the believed speed, because the raw judgement is not a
            # detector: ungated it also flags 64 % of innocent vessels over an
            # episode. Under 1.0 m/s that falls to 0.4 % for 71-73 % recall, and
            # whatever it does flag is counted against the false alarm rate in
            # metrics.outcome, so the trade is visible rather than bought.
            slow = tracks.velocity.norm(dim=-1) < config.reward.shadow_slow_ms
            noticed = deviated | (called & contacts.detected & slow)
            # `noticed` scores the run; `deviated` pays it, and the split is
            # measured rather than tidy. Paying the alarm on the judgement was
            # tried and it makes the fleet worse at the thing it was meant to
            # fix: on the film seed it took six red points reached down to
            # four, with two never heard at all at 19.7 and 21.1 km, while it
            # piled six to nine vehicles onto the two it had already found.
            # Every version of this that raised the payoff for attending to a
            # found suspect bought it out of the search that finds the next
            # one. The metric should say what the fleet knew; what the fleet
            # is paid for is a separate question with a worse answer.
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
            caught = (noticed & traffic.deviating.unsqueeze(1)).any(dim=1)
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
            # Averaged over the decision rather than summed, so the term does
            # not change scale if the perception interval does.
            suspected = self._suspected(moment, tracks)
            shadowed = shadowed + self._shadow(suspected, fleet, traffic, state.escort_claim)
            crowded = crowded + self._crowding(suspected, fleet, traffic, state.escort_claim)
            packed = packed + self._packing(fleet, state.escort_claim)
            looks += 1
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
                        state.escort_claim,
                    ),
                    links.reachable,
                )

        after = self._picture.uncertainty_m(tracks, moment).clamp(max=gate)
        claim = self._claims(state.escort_claim, moment, fleet, traffic, tracks)
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
            escort_claim=claim,
        )
        # Potential-based shaping, which needs the state on both sides of the
        # decision, so it is built here rather than inside `_reward`.
        settings = self._config.reward
        shaping = settings.cable_shaping * (
            settings.shaping_discount * self._cable_potential(new) - potential_before
        ) + settings.shadow_shaping * (
            settings.shaping_discount * self._suspect_potential(new) - suspect_before
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
                shadowed=shadowed / max(looks, 1),
                crowded=crowded / max(looks, 1),
                packed=packed / max(looks, 1),
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
        claim: torch.Tensor,
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
            escort_claim=claim,
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
        # The detection range LESS the corridor half-width, which is the
        # difference between watching a line and watching the zone that matters.
        # A cable counted as watched at the full 5.95 km of cruise hearing only
        # guarantees that a vessel sitting exactly on the line would be heard;
        # a vessel 3 km off it on the far side is 8.95 km from the vehicle and
        # is not. Measured, deviations happen 0.55 to 3.15 km from the nearest
        # cable sample -- inside the corridor, which is what `over_corridor`
        # fires on -- and the one the fleet missed sat 2.98 km off a stretch
        # that was being "heard" 44 % of its window, with the nearest vehicle
        # 6.8 km away the whole time. Eight hundred and fifty metres short, and
        # the reward had already paid for that cable.
        #
        # Requiring the vehicle to cover the whole corridor makes a patrol mean
        # being on the cable rather than within earshot of it, which is both
        # the honest reading of the claim the film makes and the thing that
        # sweeps the water the deviations are actually in.
        reach = (self.reach_m(fleet.regime) - self._corridor_margin_m).clamp_min(0.0)
        within = (gap <= reach.unsqueeze(-1)) & fleet.listening.unsqueeze(-1)
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
        # Squaring this was tried, to pay for the trip to the ends of the
        # network, and it fails twice over. It does not work: the eastern sixth
        # of the cables still saw 39 visits against 1 080 for the middle and 21
        # samples were never visited at all. And it wrecks the learning -- the
        # trained policy scored 0.398 against 0.927 for the scripted cable
        # patrol and 0.667 for the lawnmower, BELOW BOTH FLOORS on its own
        # objective, because a squared term makes the payout rare and enormous
        # and PPO cannot estimate an advantage through that variance.
        #
        # The reason no exponent works is the discount, not the weight. At
        # gamma 0.95 over 900 s decisions the effective horizon is about twenty
        # decisions, five hours. The far end of EstLink 2 is 79 km from the
        # nearest base, sixteen to eighteen hours of cruise, and 0.95^144 is
        # 0.0006: the reward waiting there is multiplied by six ten-thousandths
        # before the value function ever sees it. No price fixes a horizon. A
        # third base in the east would, and so would a longer discount; both
        # are larger changes than a reward term and neither is made here.
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
        shadowed: torch.Tensor,
        crowded: torch.Tensor,
        packed: torch.Tensor,
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
        # `crowded` is the escort's other half: `shadowed` says who is paid to
        # be alongside, and this says that standing in the same water with no
        # claim is worse than going back to work. The cap alone left a queue.
        return (
            verification
            + alarm
            + exploration
            + cable
            + shadowed
            - crowded
            - packed
            - staleness
            - energy
            - penalty
        )

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
        suspect = self._nearest_suspect(state)

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
            suspect,
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

    def _suspect_potential(self, state: EnvState) -> torch.Tensor:
        """Return each vehicle's closeness to the suspect it ought to be closing on.

        The potential of a second potential-based shaping term, for the same
        reason the cables needed the first: ``gamma * phi(s') - phi(s)`` leaves
        the optimal policy untouched whatever ``phi`` is, so it is the safe way
        to make a goal visible that the value function cannot otherwise see.

        Here what it cannot see is a deadline. The shadow credit pays nothing
        until the vehicle is inside the detection range, a deviation lasts
        three hours, and cruise covers 13 km in three hours -- so to any
        vehicle further out than that the credit is a reward it cannot reach,
        and PPO has no gradient to follow towards it. Measured on the first
        trained policy: the fleet held a flag on a live suspect in 118 of 137
        decisions and had nobody within range 97.6 % of the time.

        Zero where nothing is flagged, rather than some large negative number.
        Shaping stays policy-invariant under any potential, but one that
        lurches when a suspect appears pays out a windfall nobody's action
        earned, and that is noise in the advantage.

        **Exponential, and that is the load-bearing choice.** A linear ramp was
        tried first and has a dead zone in it. The shaping pays
        ``k * (gamma * phi(s') - phi(s))``, which is positive only while
        ``gamma * dphi > (1 - gamma) * phi``; with a linear ramp ``dphi`` is
        constant while ``phi`` keeps growing, so closing stops paying at
        ``reach * (1 - gamma * dphi / (1 - gamma))`` -- measured, 9.5 km, and
        from there to the 6 km radius every decision spent closing was a
        LOSS. The fleet would have been nudged to a ring around the suspect and
        held outside the range it was being paid to enter.

        With ``phi = exp(-d/L)`` the same condition reduces to
        ``L < step / ln(1 / gamma)``, which carries no ``d`` at all: one
        constraint on the length scale buys a closing gradient that is positive
        at every range. At 900 s of cruise and a 0.95 discount the bound is
        21 km, and ``shadow_reach_m`` is the e-folding length held well inside
        it.

        Parameters
        ----------
        state : EnvState
            Current state.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)`` in ``[0, 1]``, higher when closer.
        """
        settings = self._config.reward
        if settings.shadow_shaping <= 0.0:
            return torch.zeros_like(state.fleet.energy_j)
        flagged = self._suspected(state.t, state.tracks)
        gap = (state.traffic.xy.unsqueeze(1) - state.fleet.xy.unsqueeze(2)).norm(dim=-1)
        # A vessel pulls only its own escort: the vehicles that hold a claim on
        # it, and nobody else. The claim is the same object the credit is paid
        # on, so the shaping and the reward cannot disagree about who is
        # supposed to be going.
        #
        # This was `inside | (occupancy < team)`, and that disjunction defeated
        # the cap for everybody it mattered for: a vehicle already inside the
        # radius stayed eligible whatever the occupancy, so the sixth to arrive
        # kept a potential near 1.0 and leaving cost it the shaping. Replacing
        # it with a fresh rank was no better and for a deeper reason --
        # potential-based shaping provably leaves the optimal policy alone, so
        # no shaping change could ever have thinned a crowd. Only the credit
        # can, and the credit now follows the claim.
        vessel_index = torch.arange(gap.shape[-1], device=self._device).view(1, 1, -1)
        eligible = flagged & (state.escort_claim.unsqueeze(-1).long() == vessel_index)
        # The nearest eligible one, so a vehicle already beside a suspect is not
        # dragged off it by a second one further away: that is the "unless they
        # are busy" clause, and it needs no flag of its own.
        nearest = torch.where(eligible, gap, torch.full_like(gap, _NOWHERE_M)).min(dim=-1).values
        closeness = torch.exp(-nearest / settings.shadow_reach_m)
        return torch.where(eligible.any(dim=-1), closeness, torch.zeros_like(closeness))

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
            deviant=truth,
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

    def _suspected(self, t: float, tracks: Tracks) -> torch.Tensor:
        """Return which contacts each vehicle currently believes are misbehaving.

        The plausibility model's own verdict, latched: once this vehicle --
        or any vehicle that has since reached it over the modem -- has watched
        the contact loiter over a cable, the contact stays a target for the
        rest of the episode. This one predicate is what the suspect
        observation points at, what the shadow credit is paid on, and what the
        shaping pulls towards: if they disagreed, the policy would be paid for
        something it could not see.

        **It used to expire after ``shadow_memory_s``, and that single line was
        why no fleet ever converged on anything.** Measured over 22 real
        deviations: they last 3.0 h, the nearest deployed vehicle is 22.8 km
        away -- 5.3 h at cruise -- and 900 s of verdict buys 1.08 km of
        travel. So the flag died after one decision, the shaping potential
        went with it, and the only vehicles that could ever collect were the
        ones already on station. The policy's answer was the correct one for
        that reward: sit where suspects usually appear. Permanence costs
        nothing on the wire -- the bit was already in the packet -- and it is
        what makes a 4 km modem sufficient, since the finder now keeps the
        verdict indefinitely and hands it to the first vehicle it meets.
        ``shadow_memory_s`` survives only as the scale the observation's
        staleness feature is measured in.

        ``alarmed`` was used here first and is the wrong flag. It records that
        the vehicle once watched the contact contradict its prediction, which
        ordinary manoeuvring, navigation error and class confusion all do --
        and it is never cleared, so it accumulated to 39 vessels per vehicle
        over an episode. Measured against ground truth, a fresh ``alarmed``
        flag indicated a vessel actually deviating 0.5 % of the time, against
        8.3 % for a fresh implausible verdict: seventeen times the precision
        and twenty-five times the catch. With ``alarmed`` the suspect block
        pointed at a contact last heard eight hours ago and 23 km away, and
        never once at a live deviation.

        Returns
        -------
        Tensor
            Boolean, ``(worlds, vehicles, vessels)``.
        """
        # The verdict is permanent; acting on it needs a position. A vehicle
        # whose track has grown past the staleness gate has not forgotten that
        # the vessel is dangerous -- it has forgotten where the vessel is, and
        # there is nothing to converge on. It goes back to patrol, and when
        # anybody re-acquires the contact the latched verdict makes it a target
        # again at once, with nobody having to witness the loitering a second
        # time. That is where the permanence pays.
        #
        # Without this gate the flag outlives the fix by hours: measured, the
        # informed vehicle's belief about a flagged vessel sits at a median
        # 10.3 km but a p90 of 57.9 km, so a tenth of the suspicions point at
        # open water the vessel left long ago.
        # And slow enough to hold station on, by this vehicle's own believed
        # velocity. The verdict being permanent is what makes this necessary:
        # a watch list is not a target list, and a vessel that has resumed
        # passage at six metres a second is not something a 1.2 m/s vehicle
        # escorts. It is the condition that keeps `deviant` from decaying into
        # the thing `alarmed` was rejected for being.
        slow = tracks.velocity.norm(dim=-1) < self._config.reward.shadow_slow_ms
        suspected: torch.Tensor = (
            tracks.deviant & tracks.known & slow & ~self._picture.stale(tracks, t)
        )
        return suspected

    def _nearest_suspect(self, state: EnvState) -> torch.Tensor:
        """Return where this vehicle's suspect lies, and who else is on it.

        The suspect is the nearest contact the vehicle holds and has flagged --
        ``alarmed``, meaning it has watched the vessel contradict its own
        prediction. The flag travels over the modem as part of the track, so a
        vehicle that has heard nothing itself can still be looking at a suspect
        somebody three kilometres away found, which is the whole point of
        carrying the link.

        Returns
        -------
        Tensor
            ``(worlds, vehicles, SUSPECT_FEATURES)``.
        """
        tracks = state.tracks
        fleet = state.fleet
        vehicles = fleet.estimate.shape[1]
        position, _ = self._picture.predict(tracks, state.t)
        offset = position - fleet.estimate.unsqueeze(2)
        gap = offset.norm(dim=-1)

        flagged = self._suspected(state.t, tracks)
        ranked = torch.where(flagged, gap, torch.full_like(gap, _NOWHERE_M))
        pick = ranked.argmin(dim=-1, keepdim=True)
        found = flagged.any(dim=-1, keepdim=True).to(gap.dtype)

        chosen = offset.gather(2, pick.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)
        span = chosen.norm(dim=-1, keepdim=True).clamp_min(1.0)
        mine = tracks.issuer == torch.arange(
            self.n_vehicles, device=self._device, dtype=torch.int16
        ).reshape(1, -1, 1)
        told = (~mine).gather(2, pick).to(gap.dtype)

        # How many linked neighbours are already shadowing the same vessel.
        #
        # Within modem range, which is the graph the film draws -- so the
        # number the policy reads is the number a viewer can count. Without
        # it the reward's team cap is invisible: a vehicle cannot tell whether
        # it would be the third on station or the thirtieth, and the fleet
        # either all comes or none does.
        on_it = (gap < self._config.reward.shadow_radius_m) & flagged
        # on_it is indexed by the vessel each OTHER vehicle is near; this
        # vehicle wants it indexed by its OWN pick, hence the transpose.
        theirs = on_it.permute(0, 2, 1).gather(1, pick.expand(-1, -1, vehicles))
        spacing = (fleet.estimate.unsqueeze(2) - fleet.estimate.unsqueeze(1)).norm(dim=-1)
        linked = (spacing > 0.0) & (spacing < self._config.comms.range_m)
        crowd = (linked & theirs).to(gap.dtype).sum(dim=-1, keepdim=True)

        # How stale the verdict is, as a fraction of how long one is honoured.
        # Without it the policy cannot tell a contact judged thirty seconds ago
        # from one judged fourteen minutes ago, and only the first is worth
        # setting off for.
        age = (state.t - tracks.last_seen_s).gather(2, pick)
        staleness = (age / self._config.reward.shadow_memory_s).clamp(0.0, 1.0)

        features = torch.cat(
            [
                chosen / span,
                (span / 50_000.0).clamp(0.0, 2.0),
                # The same range over the radius the escort is actually paid
                # for, so one unit is the rim and the whole station-keeping
                # decision is resolved rather than crushed into a rounding
                # error. See SUSPECT_FEATURES.
                (span / self._config.reward.shadow_radius_m).clamp(0.0, 2.0),
                found,
                # Clamped at THREE times the cap, not at one. Clamped at one a
                # vehicle could not tell five neighbours on the vessel from
                # fifteen -- the feature saturated exactly where the decision
                # it informs begins, which is "is this escort already full".
                # Measured on the film that produced: mobs of 12, 13 and 15 on
                # one vessel while other red vessels in the same frame had
                # nobody. The cap in the reward cannot be obeyed by a policy
                # that cannot see it being exceeded.
                (crowd / float(self._config.reward.shadow_team)).clamp(0.0, 3.0),
                told,
                staleness,
            ],
            dim=-1,
        )
        # Nothing flagged means nothing to say. Left unmasked, the block feeds
        # the bearing to whichever vessel happened to sort first, which is a
        # direction with no meaning behind it.
        return features * found

    def _claims(
        self,
        held: torch.Tensor,
        moment: float,
        fleet: FleetState,
        traffic: TrafficState,
        tracks: Tracks,
    ) -> torch.Tensor:
        """Return who is the escort of what, carrying claims forward.

        Once per decision, not once per look, and that is the entire point: a
        cap re-ranked three times a decision pays everybody a share instead of
        paying five and refusing the rest. See `EnvState.escort_claim`.

        Two steps, in this order, because keeping beats taking:

        **Release.** A claim survives while the vehicle is deployed, still
        believes the vessel is a suspect, and is still within `shadow_reach_m`.
        Any of those failing frees the slot, which is what stops a vehicle
        holding one from the far side of the gulf.

        Released on the REACH and not on the paid radius, and the difference is
        not a detail. Slots are granted out to 12 km so a vessel can call in a
        vehicle that has not arrived yet; releasing at the 3 km payment radius
        took the slot straight back on the next decision, because a vehicle
        called from 10 km is 10 km away when it is called. So every claim
        outside 3 km churned once a decision and was refilled by whoever was
        nearest -- the fresh ranking this method exists to replace, surviving
        in the one place the test did not look. Measured consequences: the
        attraction flickered for everyone inbound, so more vehicles were drawn
        than there were slots, and the five places went to the five that
        arrived first rather than the five that were tasked. That is where the
        queue came from.

        **Fill.** Free slots go to the nearest unclaimed vehicles that hold the
        verdict themselves, out to the shaping's reach -- so a vessel can call
        in a vehicle that has not arrived yet, which is the difference between
        an escort and a coincidence. Ranking only decides who fills an empty
        slot; it never evicts an incumbent, so the answer stops churning.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)`` of vessel indices, ``-1`` for unclaimed.
        """
        settings = self._config.reward
        team = int(settings.shadow_team)
        deployed = fleet.mode == auv_module.MODE_DEPLOYED
        flagged = self._suspected(moment, tracks)
        gap = (traffic.xy.unsqueeze(1) - fleet.xy.unsqueeze(2)).norm(dim=-1)

        worlds, vehicles, vessels = flagged.shape
        rows = torch.arange(vehicles, device=self._device)

        # --- release ---------------------------------------------------------
        claim = held.long()
        has = claim >= 0
        safe = claim.clamp_min(0)
        still = (
            deployed
            & flagged.gather(2, safe.unsqueeze(-1)).squeeze(-1)
            & (gap.gather(2, safe.unsqueeze(-1)).squeeze(-1) < settings.shadow_reach_m)
        )
        claim = torch.where(has & still, claim, torch.full_like(claim, -1))

        # --- fill ------------------------------------------------------------
        taken = torch.zeros((worlds, vessels), dtype=torch.long, device=self._device)
        taken.scatter_add_(1, claim.clamp_min(0), (claim >= 0).long())
        free = team - taken

        wants = (
            flagged
            & deployed.unsqueeze(-1)
            & (gap < settings.shadow_reach_m)
            & (claim < 0).unsqueeze(-1)
            & (free > 0).unsqueeze(1)
        )
        # Each unclaimed vehicle applies to ONE vessel, the nearest it could
        # serve. Letting a vehicle be a candidate for several at once is what
        # makes the assignment need a loop; one application each makes it two
        # vectorised steps and costs nothing real, since a vehicle can only
        # escort one ship anyway.
        ranked = torch.where(wants, gap, torch.full_like(gap, _NOWHERE_M))
        applied, choice = ranked.min(dim=-1)
        applying = applied < _NOWHERE_M

        # Then each vessel takes its nearest applicants. A vehicle's place in
        # the queue is how many applicants to the same vessel are closer than
        # it, which needs no sort.
        same = (choice.unsqueeze(2) == choice.unsqueeze(1)) & applying.unsqueeze(2)
        same = same & applying.unsqueeze(1)
        closer = same & (applied.unsqueeze(1) < applied.unsqueeze(2))
        queue = closer.sum(dim=2)

        room = free.gather(1, choice.clamp_min(0))
        granted = applying & (queue < room)
        del rows
        return torch.where(granted, choice, claim).to(torch.int16)

    def _shadow(
        self,
        suspected: torch.Tensor,
        fleet: FleetState,
        traffic: TrafficState,
        claim: torch.Tensor,
    ) -> torch.Tensor:
        """Return the credit for holding contact on a vessel believed to be misbehaving.

        Paid on the TRUE geometry against the vehicle's OWN verdict: you are
        paid for actually being beside a vessel your plausibility model has
        called odd. Believing it from across the gulf earns nothing.

        On the belief and not on the truth, which is the one thing here that
        took two trainings to get right. Paid on the truth, the expected value
        of setting off is the payoff times the 8.3 % chance the contact really
        is deviating -- 0.08 a decision against 1.37 for staying on patrol --
        so the fleet correctly ignored it. A reward has to be reachable from
        what the agent knows.

        Paid for ONE vessel -- the best of them -- and only to the nearest
        ``shadow_team`` of the vehicles on it. Both halves were learned the
        hard way: summed over vessels the term paid for standing in traffic,
        and with a tapering split the sixth arrival still earned enough to
        come, so seventeen vehicles converged on one contact.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)``.
        """
        settings = self._config.reward
        if settings.shadow <= 0.0:
            return torch.zeros_like(fleet.energy_j)
        radius = settings.shadow_radius_m
        # The team size is not read here any more: who is on the team is
        # `_claims`' answer, carried in `claim`, and re-deriving it from the
        # size was the re-ranking this method had to stop doing.
        gap = (traffic.xy.unsqueeze(1) - fleet.xy.unsqueeze(2)).norm(dim=-1)
        on_it = suspected & (gap < radius) & (fleet.mode == auv_module.MODE_DEPLOYED).unsqueeze(-1)
        # An exponential core, shifted so it still reaches zero at the radius.
        #
        # This was a squared ramp, and squaring is the reason the escort used to
        # stop short. `(1 - g/R)^2` has its steepest gradient at the vessel and
        # ZERO gradient at the rim, which is where a vehicle arrives: the
        # moment it crossed into the radius the term went quiet, so it parked.
        # Measured on that term, median closest approach was 1.47 km of a 3 km
        # radius and the fleet drifted 67 % of the time it was on station,
        # because drifting costs no fuel and the rim paid nearly as well as
        # being alongside.
        #
        # The shift matters as much as the exponential. A bare `exp(-g/L)` pays
        # for hanging about at the edge, which is what the ramp was protecting
        # against; subtracting the rim value and renormalising keeps the ramp's
        # one virtue -- the edge is worth nothing -- and adds a gradient that is
        # steep at every range inside it. At 700 m and `shadow` 16: 0 at 3 km,
        # 0.71 at 2 km, 3.66 at 1 km, 10.35 at 300 m.
        #
        # 0.71 at the rim is chosen, not incidental: cable patrol pays 0.88 a
        # decision, so a vehicle can no longer do better than its day job by
        # loitering in a wide circle. That is the old failure made arithmetically
        # impossible rather than merely discouraged.
        floor = math.exp(-radius / settings.shadow_hug_m)
        core = (torch.exp(-gap / settings.shadow_hug_m) - floor) / (1.0 - floor)
        closeness = core.clamp(0.0, 1.0) * on_it.to(gap.dtype)
        # Only the claimed escort is paid for a given vessel, and the rest are
        # paid nothing at all. The split used to taper -- each vehicle's share
        # flattened past the cap instead of stopping -- so the sixth to arrive
        # still earned about what patrolling earned and seventeen piled onto one
        # contact. A cliff is what makes a team a team.
        #
        # The claim, not a fresh ranking. Re-ranking every look paid the
        # surplus 1.335 a decision of a term meant to pay it nothing, because
        # with ten drones at similar ranges each spent part of its time in the
        # first five. See `EnvState.escort_claim`.
        vessel_index = torch.arange(gap.shape[-1], device=self._device).view(1, 1, -1)
        engaged = on_it & (claim.unsqueeze(-1).long() == vessel_index)
        # FULL credit to each of the five, not a share. The 1/n split was put
        # in to stop a pile forming and it worked by making the fifth arrival
        # earn a fifth of the first, which is a disincentive to be the fifth --
        # the escort settled at one to three and the ask is five. What stops
        # the pile now is the predicate, not the arithmetic: the slow gate
        # admits 0.4 % of the (vehicle, vessel) pairs it used to, so the credit
        # cannot be collected by standing in traffic and a crowd can only form
        # on a vessel that really is sitting still over a cable. On that vessel
        # a crowd of five is the thing being asked for.
        paid = closeness * engaged.to(gap.dtype)
        # One vessel, the best one, and not the sum over all of them. Summed,
        # the term paid for *standing where contacts are dense* rather than for
        # escorting any particular ship: a deployed vehicle had a mean of 6.04
        # vessels inside this radius, so parking in the lane collected over and
        # over while following one vessel collected once. That is the whole of
        # why the trained fleet balled up in the middle of the gulf.
        credit: torch.Tensor = settings.shadow * paid.max(dim=-1).values
        return credit

    def _packing(self, fleet: FleetState, claim: torch.Tensor) -> torch.Tensor:
        """Charge each vehicle for the deployed hulls packed in around it.

        Measured on a recorded episode, the median vehicle has no neighbour
        within 3 km at all -- but 8.1 % of deployed vehicle-instants sit in a
        knot of six or more, and the worst holds fourteen hulls inside two
        kilometres. Those piles are what a viewer sees, and nothing in the
        reward spoke against them: the escort cap limits who is PAID beside a
        suspect, and `_crowding` charges the unpaid queue beside one, but
        neither has anything to say about a clump on patrol.

        **A vehicle holding an escort claim pays nothing.** The first version
        charged everyone in the knot and that was a real mistake, visible in
        one measurement: escort effort spent on a vessel that was actually
        deviating fell from 83.3 % to 54.1 %. Charging the legitimate team
        made a crowded real rover expensive and a lonely false positive free,
        so the fleet drifted towards isolated suspects -- and a false positive
        is isolated by construction. The reward was paying vehicles to avoid
        the thing that mattered.

        Exempting claim-holders is not a return to ranking. A claim is a held
        lease, granted once and kept across decisions; the failure it replaced
        was a fresh rank re-drawn every decision, which taught the policy
        nothing because it carried no information from one step to the next.

        What is left is charged symmetrically: among the vehicles that are not
        escorting anything, everyone in the pile pays the same and nobody is
        ranked. The allowance is the paid escort team, so a full escort plus
        its own water is free and the next unengaged hull to arrive is not.
        """
        settings = self._config.reward
        if settings.packing_weight <= 0.0:
            return torch.zeros_like(fleet.energy_j)
        deployed = fleet.mode == auv_module.MODE_DEPLOYED
        gap = (fleet.xy.unsqueeze(2) - fleet.xy.unsqueeze(1)).norm(dim=-1)
        near = (gap < settings.packing_radius_m) & deployed.unsqueeze(1) & deployed.unsqueeze(2)
        # Every deployed vehicle counts itself, and nobody else does. The count
        # still includes the escorts -- the water really is that full -- but
        # they are not the ones asked to leave it.
        neighbours = near.sum(dim=-1) - deployed.long()
        surplus = (neighbours - settings.packing_allowance).clamp_min(0)
        unengaged = (claim.long() < 0) & deployed
        return settings.packing_weight * surplus.to(gap.dtype) * unengaged.to(gap.dtype)

    def _crowding(
        self,
        suspected: torch.Tensor,
        fleet: FleetState,
        traffic: TrafficState,
        claim: torch.Tensor,
    ) -> torch.Tensor:
        """Return what a vehicle is charged for standing on a suspect it does not escort.

        The team cap governs who is PAID, and that turned out not to govern who
        comes. Claims are exact -- five a vessel, never six, verified over a
        run -- and the crowd inside a radius was still eleven at the ninetieth
        percentile, of which 4.8 escorted this vessel, 3.3 escorted a different
        red vessel whose circle overlapped, and 4.25 escorted nothing at all.

        The 4.25 are a queue, not transients: measured, an unclaimed vehicle
        stays 60 minutes in median, only 22 % of spells are a single decision
        passing through, and 31 % end by winning a vacated slot. A slot pays
        4.29 a decision against 0.84 on patrol and is held for 135 minutes, so
        waiting is worth +1.66 discounted against going back to work even
        though the wait itself pays nothing. Exclusion cannot fix that, because
        what draws the queue is the vacancy rather than the payment.

        So the wait is charged. Flat, once, however many circles a vehicle is
        standing in -- the cost is having no task, not how much water it
        crowds. A vehicle holding a claim pays nothing wherever it is, which is
        what keeps this away from the escort: it falls only on `claim < 0`.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)``, zero or the charge.
        """
        settings = self._config.reward
        if settings.shadow_crowding <= 0.0:
            return torch.zeros_like(fleet.energy_j)
        gap = (traffic.xy.unsqueeze(1) - fleet.xy.unsqueeze(2)).norm(dim=-1)
        deployed = (fleet.mode == auv_module.MODE_DEPLOYED).unsqueeze(-1)
        on_it = suspected & (gap < settings.shadow_radius_m) & deployed
        loitering = on_it.any(dim=-1) & (claim.long() < 0)
        return settings.shadow_crowding * loitering.to(gap.dtype)

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
