"""What a run achieved, in the terms of specification section 7.

None of these is the reward. The reward is a training signal shaped to be dense
and learnable; these are the claims the project is willing to make, and they are
measured on fixed seeds the training never saw.

**None of them measures time spent in sensor range.** That was the v1 metric and
it rewarded escorting a cargo that was behaving itself. What is measured instead
is whether the fleet's picture was *right*, how fast it noticed when it stopped
being right, and what the picture cost to maintain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from baltic_isr.config import Config
from baltic_isr.learn.env import EnvState
from baltic_isr.picture.tracks import Picture


@dataclass(frozen=True)
class Outcome:
    """One run's score.

    Attributes
    ----------
    alarm_delay_median_s, alarm_delay_p90_s : float
        Time between a vessel beginning to deviate and the fleet noticing. The
        P90 is the honest number; the median flatters a fleet that catches the
        easy ones.
    deviations_missed : float
        Fraction never noticed at all.
    picture_accuracy : float
        Fraction of vessel-time where the fleet's best prediction is within two
        kilometres of the truth. Satisfiable **by prediction**: a cargo
        correctly predicted and never revisited counts as known, which is the
        behaviour the task wants.
    expired_fraction : float
        Fraction of held tracks that went stale without anyone checking.
    verification_cost : float
        Vehicle-seconds of sensor contact per vessel-hour of picture held. Lower
        is better; this is where escorting a cargo is punished.
    handoff_rate : float
        Fraction of held tracks whose prediction came from another vehicle. The
        only number that depends on the modem.
    false_alarm_rate : float
        Alarms per vessel-day raised on vessels that were behaving.
    never_detected : float
        Fraction of vessels the fleet never heard at all.
    """

    alarm_delay_median_s: float
    alarm_delay_p90_s: float
    deviations_missed: float
    picture_accuracy: float
    expired_fraction: float
    verification_cost: float
    handoff_rate: float
    false_alarm_rate: float
    never_detected: float


class Scorer:
    """Accumulates a run and scores it.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    picture : Picture
        The prediction model, for reading uncertainties back.
    """

    ACCURATE_M = 2000.0

    def __init__(self, config: Config, picture: Picture) -> None:
        self._config = config
        self._picture = picture
        self._accurate = 0.0
        self._samples = 0.0
        self._contact_s = 0.0
        self._held_s = 0.0
        self._expired = 0.0
        self._held = 0.0
        self._third_party = 0.0
        self._false = 0.0
        self._was_alarmed: torch.Tensor | None = None

    def observe(self, state: EnvState, dt_s: float) -> None:
        """Fold one decision into the running score.

        Parameters
        ----------
        state : EnvState
            The state after that decision.
        dt_s : float
            How long it covered.
        """
        tracks = state.tracks
        known = tracks.known
        predicted, _ = self._picture.predict(tracks, state.t)
        truth = state.traffic.xy.unsqueeze(1)

        # The fleet's best guess per vessel is its freshest holder's, which is
        # what a commander ashore would be looking at.
        freshest = tracks.last_seen_s.argmax(dim=1, keepdim=True)
        best = predicted.gather(1, freshest.unsqueeze(-1).expand(-1, -1, -1, 2))
        held_anywhere = known.any(dim=1)
        error = (best.squeeze(1) - truth.squeeze(1)).norm(dim=-1)

        self._accurate += float((held_anywhere & (error < self.ACCURATE_M)).sum()) * dt_s
        self._samples += float(state.traffic.xy.shape[0] * state.traffic.xy.shape[1]) * dt_s

        self._held += float(known.sum())
        self._held_s += float(held_anywhere.sum()) * dt_s
        self._expired += float(self._picture.stale(tracks, state.t).sum())

        mine = tracks.issuer == torch.arange(
            known.shape[1], device=known.device, dtype=torch.int16
        ).reshape(1, -1, 1)
        self._third_party += float((known & ~mine).sum())

        # An alarm is an event. Counting the standing state once per decision
        # turned six false alarms into eighty per vessel-day, because an alarm
        # that is never cleared was counted a hundred and ninety-two times.
        innocent = (known & tracks.alarmed & ~state.traffic.deviating.unsqueeze(1)).any(dim=1)
        if self._was_alarmed is None:
            self._was_alarmed = torch.zeros_like(innocent)
        self._false += float((innocent & ~self._was_alarmed).sum())
        self._was_alarmed = innocent

    def add_contact_time(self, seconds: float) -> None:
        """Record vehicle-seconds spent inside detection range of something."""
        self._contact_s += seconds

    def score(self, state: EnvState) -> Outcome:
        """Return the run's score.

        Parameters
        ----------
        state : EnvState
            The final state, which carries when each deviation was caught.

        Returns
        -------
        Outcome
            The numbers of specification section 7.
        """
        started = state.traffic.deviation_start_s
        caught = state.deviation_seen_s
        scheduled = torch.isfinite(started)
        noticed = scheduled & torch.isfinite(caught)
        delay = (caught - started)[noticed]

        episode_days = self._config.simulation.episode_s / 86_400.0
        vessels = float(state.traffic.xy.shape[0] * state.traffic.xy.shape[1])
        held_hours = max(self._held_s / 3600.0, 1.0)

        return Outcome(
            alarm_delay_median_s=float(delay.median()) if delay.numel() else float("inf"),
            alarm_delay_p90_s=float(delay.quantile(0.9)) if delay.numel() else float("inf"),
            deviations_missed=1.0 - float(noticed.sum()) / max(float(scheduled.sum()), 1.0),
            picture_accuracy=self._accurate / max(self._samples, 1.0),
            expired_fraction=self._expired / max(self._held, 1.0),
            verification_cost=self._contact_s / held_hours,
            handoff_rate=self._third_party / max(self._held, 1.0),
            false_alarm_rate=self._false / max(vessels * episode_days, 1.0),
            never_detected=1.0
            - float(torch.isfinite(state.first_seen_s).sum()) / max(vessels, 1.0),
        )


# Every chart in this repo reads the same way up, on one scale: **three
# percentages, and more is better in all of them.** The measurements
# underneath do not agree on either count -- two of the three are failures
# counted, where less is more, and one is a duration in hours -- so an early
# version drew five panels captioned "lower is better" or "higher is better".
# That is five charts to read and a direction to hold in your head for each,
# which is not a comparison.
#
# These are the same measurements, restated as successes out of a hundred:
#
#   manoeuvres noticed   1 - missed                  an exact complement
#   cable kept fresh     1 - mean gap / 24 h         the gap against a day
#   picture accuracy     unchanged                   already a success
#
# All three transforms are monotone, so no ranking moves; only the arithmetic
# the reader would otherwise have to do, which is now none.
UPRIGHT = (
    ("manoeuvres_noticed", "Manoeuvres detected"),
    ("cable_kept_fresh", "Cable coverage"),
    ("picture_accuracy", "Track accuracy"),
)

# The window the cable gap is measured against. A day, because a cargo takes
# six hours to cross the box and the worst policy here leaves a stretch unheard
# for twenty-one, so a day is the scale the whole range fits on.
FRESHNESS_WINDOW_H = 24.0


def upright(row: Mapping[str, float]) -> dict[str, float]:
    """Restate one scored run as three percentages, more being better in each.

    Parameters
    ----------
    row : mapping
        Must carry ``deviations_missed``, ``cable_exposure_h`` and
        ``picture_accuracy``.

    Returns
    -------
    dict
        The keys named in `UPRIGHT`, each in percent.
    """
    gap = float(row["cable_exposure_h"])
    fresh = 100.0 * (1.0 - gap / FRESHNESS_WINDOW_H)
    return {
        "manoeuvres_noticed": 100.0 * (1.0 - float(row["deviations_missed"])),
        "cable_kept_fresh": min(max(fresh, 0.0), 100.0),
        "picture_accuracy": 100.0 * float(row["picture_accuracy"]),
    }
