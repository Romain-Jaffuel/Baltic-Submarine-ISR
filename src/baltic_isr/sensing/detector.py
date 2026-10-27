"""Who hears what, and how badly they place it.

Detection range is not a constant here. It is solved per pair, from the source
level of that vessel against the noise floor at that listener -- which depends on
the listener's own regime and on how much traffic is grinding away around it. A
vehicle sprinting inside a busy lane hears almost nothing; the same vehicle
drifting in a quiet corner hears a long way.

Inverting the transmission loss per pair, per step, is not something to do with a
bisection. So the loss curve is tabulated once at setup and inverted by
interpolation, which is a gather and some arithmetic.

**A measurement carries the listener's own navigation error.** A passive array
bears well and ranges badly, and the vehicle adds whatever its dead reckoning is
wrong by. That is the mechanism by which a mis-navigated vehicle issues
predictions nobody can verify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from baltic_isr.config import Config
from baltic_isr.sensing import acoustics
from baltic_isr.world import field
from baltic_isr.world.statics import Statics

_TABLE_SIZE = 512
_MIN_BUDGET_DB = 0.0
_MAX_BUDGET_DB = 200.0


@dataclass(frozen=True)
class Contacts:
    """What one step of listening produced.

    Attributes
    ----------
    detected : Tensor
        ``(worlds, vehicles, vessels)``, whether this vehicle heard this vessel.
    position : Tensor
        ``(worlds, vehicles, vessels, 2)``, where it thinks the vessel is, in
        the listener's own distorted frame.
    vessel_class : Tensor
        ``(worlds, vehicles, vessels)``, the class it thinks it heard.
    range_m : Tensor
        ``(worlds, vehicles, vessels)``, true slant range, for diagnostics.
    """

    detected: torch.Tensor
    position: torch.Tensor
    vessel_class: torch.Tensor
    range_m: torch.Tensor


class Detector:
    """Passive detection, batched over worlds, vehicles and vessels.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, for the device and the environment.
    generator : torch.Generator
        Seeded stream for detection draws and measurement noise.
    """

    def __init__(self, config: Config, statics: Statics, generator: torch.Generator) -> None:
        self._config = config
        self._statics = statics
        self._generator = generator
        self._device = statics.device

        sound, environment = config.acoustics, config.environment
        # Tabulate the range at which a given level budget runs out, so that the
        # per-pair inversion is an interpolation rather than a search.
        budgets = torch.linspace(_MIN_BUDGET_DB, _MAX_BUDGET_DB, _TABLE_SIZE)

        def reach(budget: float) -> float:
            def excess(range_m: float) -> float:
                return budget - acoustics.transmission_loss_db(range_m, environment)

            return acoustics._solve_range(excess, 1.0, 200_000.0)

        ranges = [reach(float(budget)) for budget in budgets]
        self._budget_axis = budgets.to(self._device)
        self._range_table = torch.tensor(ranges, dtype=torch.float32, device=self._device)
        self._array_gain = acoustics.array_gain_db(sound)
        self._threshold = acoustics.detection_threshold_db(sound)
        self._self_noise = torch.tensor(
            [
                acoustics.noise_after_array_db(speed, sound.band_centre_hz, sound, environment)
                for speed in (
                    config.auv.regimes.drift_ms,
                    config.auv.regimes.cruise_ms,
                    config.auv.regimes.sprint_ms,
                )
            ],
            dtype=torch.float32,
            device=self._device,
        )
        self._quiet_range = acoustics.detection_range_m(
            config.auv.regimes.drift_ms,
            max(item.source_level_db for item in config.traffic.classes),
            sound,
            environment,
        )
        self._n_classes = len(config.traffic.classes)

    def range_for_budget(self, budget_db: torch.Tensor) -> torch.Tensor:
        """Return the range at which a level budget runs out.

        Parameters
        ----------
        budget_db : Tensor
            ``SL - NL + AG - DT``, any shape.

        Returns
        -------
        Tensor
            Range in metres, same shape.
        """
        position = (
            (budget_db - _MIN_BUDGET_DB) / (_MAX_BUDGET_DB - _MIN_BUDGET_DB) * (_TABLE_SIZE - 1)
        ).clamp(0.0, _TABLE_SIZE - 1)
        low = position.floor().to(torch.long)
        high = (low + 1).clamp(max=_TABLE_SIZE - 1)
        weight = position - low.to(position.dtype)
        return torch.lerp(self._range_table[low], self._range_table[high], weight)

    def noise_floor_db(
        self,
        listeners: torch.Tensor,
        regime: torch.Tensor,
        sources: torch.Tensor,
        source_level_db: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Return the noise level at each listener's beamformer output.

        Its own flow and machinery noise, plus every vessel too far to resolve.

        Parameters
        ----------
        listeners : Tensor
            ``(worlds, vehicles, 2)``.
        regime : Tensor
            ``(worlds, vehicles)``, which regime each vehicle is in.
        sources : Tensor
            ``(worlds, vessels, 2)``.
        source_level_db : Tensor
            ``(worlds, vessels)``.
        active : Tensor
            ``(worlds, vessels)``.

        Returns
        -------
        Tensor
            ``(worlds, vehicles)`` in dB.
        """
        traffic = field.ambient_noise_db(
            self._statics,
            listeners,
            sources,
            source_level_db,
            active,
            self._config.environment,
            self._quiet_range,
        )
        own = self._self_noise[regime]
        excess = traffic - self._config.environment.ambient_noise_db
        return 10.0 * torch.log10(
            torch.pow(10.0, own / 10.0)
            + torch.pow(10.0, (own + excess) / 10.0)
            - torch.pow(10.0, own / 10.0)
        )

    def listen(
        self,
        listeners: torch.Tensor,
        believed: torch.Tensor,
        regime: torch.Tensor,
        listening: torch.Tensor,
        sources: torch.Tensor,
        source_level_db: torch.Tensor,
        vessel_class: torch.Tensor,
        active: torch.Tensor,
        trials: int = 1,
    ) -> Contacts:
        """Run one step of listening.

        Parameters
        ----------
        listeners : Tensor
            True vehicle positions, ``(worlds, vehicles, 2)``.
        believed : Tensor
            What each vehicle thinks its position is, same shape. Every
            measurement is expressed against this, so navigation error leaks
            straight into the picture.
        regime : Tensor
            ``(worlds, vehicles)``.
        listening : Tensor
            ``(worlds, vehicles)``, false for a surfaced or charging vehicle.
        sources : Tensor
            True vessel positions, ``(worlds, vessels, 2)``.
        source_level_db : Tensor
            ``(worlds, vessels)``.
        vessel_class : Tensor
            ``(worlds, vessels)``.
        active : Tensor
            ``(worlds, vessels)``.
        trials : int
            How many 60 s opportunities this call stands for. Detection is a
            Bernoulli trial per opportunity, so the probability compounds:
            calling once for five ticks is the same process, sampled coarsely.

        Returns
        -------
        Contacts
            Detections and their measurements.
        """
        sound = self._config.acoustics
        offset = sources.unsqueeze(1) - listeners.unsqueeze(2)
        distance = offset.norm(dim=-1).clamp_min(1.0)

        noise = self.noise_floor_db(listeners, regime, sources, source_level_db, active)
        budget = source_level_db.unsqueeze(1) - noise.unsqueeze(2) + self._array_gain
        budget = budget - self._threshold
        reach = self.range_for_budget(budget).clamp_min(1.0)

        single = sound.max_probability * torch.exp(-((distance / reach) ** 2))
        probability = 1.0 - (1.0 - single) ** trials
        draw = torch.rand(probability.shape, generator=self._generator, device=self._device)
        detected = (
            (draw < probability) & listening.unsqueeze(2) & active.unsqueeze(1) & (distance < reach)
        )

        # Bearing error is across the line of sight, range error along it. The
        # second is fifteen times the first, which is why a passive contact is a
        # smear pointing at the listener.
        along = offset / distance.unsqueeze(-1)
        across = torch.stack([along[..., 1], -along[..., 0]], dim=-1)
        shape = distance.shape
        down = torch.randn(shape, generator=self._generator, device=self._device)
        cross = torch.randn(shape, generator=self._generator, device=self._device)
        error = along * (distance * sound.range_error_fraction * down).unsqueeze(-1) + across * (
            distance * math.radians(sound.bearing_sigma_deg) * cross
        ).unsqueeze(-1)

        # The listener places the contact relative to where it believes it is.
        position = believed.unsqueeze(2) + offset + error

        confused = (
            torch.rand(shape, generator=self._generator, device=self._device)
            < sound.class_confusion
        )
        wrong = torch.randint(
            self._n_classes, shape, generator=self._generator, device=self._device
        )
        reported = torch.where(confused, wrong, vessel_class.unsqueeze(1).expand(shape))

        return Contacts(
            detected=detected, position=position, vessel_class=reported, range_m=distance
        )
