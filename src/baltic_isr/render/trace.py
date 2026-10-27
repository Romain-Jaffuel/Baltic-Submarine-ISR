"""One world, recorded frame by frame, so the renderer has something to draw.

This is the observer specification section 6.5 describes: it does not exist
during training. It is a callback the environment invokes once per *physics*
tick, which is where the frame rate comes from -- 48 h at 60 s is 2880 frames,
and a sixty-second film wants about fifty a second. Recording at the decision
cadence instead would give 192 frames and a slideshow.

It reads tensors back to the host every tick, which is exactly the thing the
training path is forbidden to do. That is fine here and nowhere else: a trace
run is one world, played once, to make a picture.

What is kept is what section 9.4 says the map shows, and no more. In particular
the per-vehicle pictures are collapsed to the **fleet's best**: forty vehicles
times a hundred and eighty vessels of ellipse, every frame, is a gigabyte of
nothing anybody can look at. The renderer wants to show what the fleet knows,
and the fleet knows whatever its freshest holder knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from baltic_isr.config import Config
from baltic_isr.learn.env import Environment, EnvState
from baltic_isr.picture.tracks import Picture
from baltic_isr.sensing import acoustics

FOG_EVERY = 5


@dataclass
class Trace:
    """A recorded episode, ready to draw.

    Every array leads with the frame axis. Positions are local metres, so they
    plot directly against the box.
    """

    times_s: list[float] = field(default_factory=list)
    vehicle_xy: list[np.ndarray] = field(default_factory=list)
    vehicle_estimate: list[np.ndarray] = field(default_factory=list)
    vehicle_regime: list[np.ndarray] = field(default_factory=list)
    vehicle_energy: list[np.ndarray] = field(default_factory=list)
    vehicle_mode: list[np.ndarray] = field(default_factory=list)
    vehicle_reach_m: list[np.ndarray] = field(default_factory=list)
    vessel_xy: list[np.ndarray] = field(default_factory=list)
    vessel_deviating: list[np.ndarray] = field(default_factory=list)
    track_xy: list[np.ndarray] = field(default_factory=list)
    track_cov: list[np.ndarray] = field(default_factory=list)
    track_known: list[np.ndarray] = field(default_factory=list)
    track_alarmed: list[np.ndarray] = field(default_factory=list)
    track_age_s: list[np.ndarray] = field(default_factory=list)
    links: list[np.ndarray] = field(default_factory=list)
    fog_frames: list[int] = field(default_factory=list)
    fog_age_s: list[np.ndarray] = field(default_factory=list)
    vessel_class: np.ndarray | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        """Return how many frames were recorded."""
        return len(self.times_s)

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the trace as stacked arrays, ready for ``savez``."""
        out: dict[str, np.ndarray] = {
            "times_s": np.asarray(self.times_s, dtype=np.float32),
            "fog_frames": np.asarray(self.fog_frames, dtype=np.int32),
        }
        for name in (
            "vehicle_xy",
            "vehicle_estimate",
            "vehicle_regime",
            "vehicle_energy",
            "vehicle_mode",
            "vehicle_reach_m",
            "vessel_xy",
            "vessel_deviating",
            "track_xy",
            "track_cov",
            "track_known",
            "track_alarmed",
            "track_age_s",
            "fog_age_s",
        ):
            out[name] = np.stack(getattr(self, name))
        if self.vessel_class is not None:
            out["vessel_class"] = self.vessel_class
        # Links are a handful of pairs per frame, so they are stored flat with
        # an offset table rather than padded to a worst case nobody reaches.
        counts = np.asarray([pair.shape[0] for pair in self.links], dtype=np.int32)
        out["link_counts"] = counts
        out["link_pairs"] = (
            np.concatenate(self.links) if self.links else np.zeros((0, 2), dtype=np.int16)
        )
        return out

    def save(self, path: Path) -> Path:
        """Write the trace to a compressed archive.

        Parameters
        ----------
        path : Path
            Destination.

        Returns
        -------
        Path
            Where it was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # The numpy stub binds **kwargs to allow_pickle rather than to the
        # arrays, so the call type-checks wrong while being correct.
        np.savez_compressed(str(path), **self.arrays())  # type: ignore[arg-type]
        return path


class Recorder:
    """Pulls one world out of a running environment, once per physics tick.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    picture : Picture
        The prediction model, for reading ellipses back.
    world : int
        Which world to follow.
    """

    def __init__(self, config: Config, picture: Picture, world: int = 0) -> None:
        self._config = config
        self._picture = picture
        self._world = world
        self.trace = Trace()
        self._tick = 0
        # The hearing radius is drawn as a faint disc that collapses when a
        # vehicle sprints: specification 9.4 calls it the project's central
        # trade-off, on screen, without a caption. So it has to be the real one.
        loudest = max(item.source_level_db for item in config.traffic.classes)
        self._reach = torch.tensor(
            [
                acoustics.detection_range_m(speed, loudest, config.acoustics, config.environment)
                for speed in (
                    config.auv.regimes.drift_ms,
                    config.auv.regimes.cruise_ms,
                    config.auv.regimes.sprint_ms,
                )
            ],
            dtype=torch.float32,
        )

    def __call__(self, moment: float, state: EnvState, links: torch.Tensor | None) -> None:
        """Record one frame.

        Parameters
        ----------
        moment : float
            Simulation time.
        state : EnvState
            The state at that tick.
        links : Tensor, optional
            The acoustic adjacency, when this tick computed one.
        """
        index = self._world
        trace = self.trace
        fleet, traffic, tracks = state.fleet, state.traffic, state.tracks

        trace.times_s.append(moment)
        trace.vehicle_xy.append(_numpy(fleet.xy[index]))
        trace.vehicle_estimate.append(_numpy(fleet.estimate[index]))
        trace.vehicle_regime.append(_numpy(fleet.regime[index], np.int8))
        trace.vehicle_energy.append(_numpy(fleet.energy_j[index] / self._config.auv.battery_j))
        trace.vehicle_mode.append(_numpy(fleet.mode[index], np.int8))
        trace.vehicle_reach_m.append(
            self._reach[fleet.regime[index].cpu()].numpy().astype(np.float32)
        )
        trace.vessel_xy.append(_numpy(traffic.xy[index]))
        trace.vessel_deviating.append(_numpy(traffic.deviating[index], np.bool_))
        if trace.vessel_class is None:
            trace.vessel_class = _numpy(traffic.vessel_class[index], np.int8)

        # The fleet's picture is its freshest holder's, per vessel.
        predicted, grown = self._picture.predict(tracks, moment)
        freshest = tracks.last_seen_s[index].argmax(dim=0)
        vessels = torch.arange(freshest.shape[0], device=freshest.device)
        trace.track_xy.append(_numpy(predicted[index, freshest, vessels]))
        trace.track_cov.append(_numpy(grown[index, freshest, vessels]))
        trace.track_known.append(_numpy(tracks.known[index].any(dim=0), np.bool_))
        # The freshest holder's verdict, not an OR over forty vehicles. OR-ing
        # paints a vessel red because somebody has a stale alarm on it, and with
        # forty holders it saturates: 118 of 180 were red at the end of a run.
        trace.track_alarmed.append(_numpy(tracks.alarmed[index, freshest, vessels], np.bool_))
        age = moment - tracks.last_seen_s[index, freshest, vessels]
        trace.track_age_s.append(
            _numpy(torch.where(torch.isfinite(age), age, torch.zeros_like(age)))
        )

        if links is None:
            # Perception ran four ticks ago; the graph it found still stands.
            trace.links.append(trace.links[-1] if trace.links else np.zeros((0, 2), dtype=np.int16))
        else:
            upper = torch.triu(links[index], diagonal=1)
            trace.links.append(upper.nonzero().to(torch.int16).cpu().numpy())

        if self._tick % FOG_EVERY == 0:
            trace.fog_frames.append(len(trace.times_s) - 1)
            trace.fog_age_s.append(_numpy(state.presence.age_s[index].min(dim=0).values))
        self._tick += 1


def _numpy(tensor: torch.Tensor, dtype: type = np.float32) -> np.ndarray:
    """Pull a tensor to the host. Only ever called from a trace run."""
    return tensor.detach().cpu().numpy().astype(dtype)


def record(
    env: Environment,
    policy: object,
    seed: int = 999_999,
    world: int = 0,
) -> Trace:
    """Play one episode and keep every frame.

    Parameters
    ----------
    env : Environment
        The environment, which must be built with at least ``world + 1`` worlds.
    policy : object
        Anything with ``act(fleet_state, t) -> actions``. A policy that needs
        more than the fleet -- a learned one reads the whole observation --
        sets ``wants_state = True`` and is handed the ``EnvState`` instead.
        An explicit flag rather than signature sniffing: the scripted policies
        and the learned one genuinely want different things, and hiding that
        behind a try/except would turn a typo into a silent fallback.
    seed : int
        Scenario seed.
    world : int
        Which world to follow.

    Returns
    -------
    Trace
        The recording.
    """
    recorder = Recorder(env.config, env.picture, world)
    state, _ = env.reset(seed=seed)
    decisions = int(env.config.simulation.episode_s / env.config.simulation.decision_interval_s)
    for _ in range(decisions):
        subject = state if getattr(policy, "wants_state", False) else state.fleet
        action = policy.act(subject, state.t)  # type: ignore[attr-defined]
        state, _ = env.step(state, action, observer=recorder)
    recorder.trace.meta = {
        "scenario": env.config.scenario,
        "fingerprint": env.config.fingerprint,
        "seed": seed,
        "extent_m": _numpy(env.statics.extent),
        "vehicles": env.n_vehicles,
    }
    return recorder.trace
