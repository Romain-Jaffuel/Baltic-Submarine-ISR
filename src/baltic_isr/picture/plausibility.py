"""Is this contact behaving? A function, deliberately, and not a network.

Specification section 12.5 requires that "why was this vessel released?" be
answerable by reading the code. So this is arithmetic over four inspectable
quantities, and it returns them alongside its verdict rather than only a score.

**Surprise is class-conditional, and that is the whole idea.** A cargo at half a
knot is a problem; a fishing boat at half a knot is fishing. The model therefore
measures speed in standard deviations of *that class's own* distribution rather
than against an absolute threshold, which is what stops the fleet crying wolf
every time a trawler slows down.

**It uses only what a submerged vehicle can actually sense.** Two of the four
signatures of specification section 4.3 -- an AIS gap, and a mismatch between the
declared and the acoustic position -- need AIS, and AIS is VHF: it does not reach
sixty metres down. A vehicle can only learn those from the base, over the surface
burst, which is batch 5. What is left here is acoustic and geographic: how fast,
what class, how near a lane, how near a cable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baltic_isr.config import Config
from baltic_isr.world.statics import Statics, sample_raster


@dataclass(frozen=True)
class Assessment:
    """A verdict, and the reasons for it.

    Attributes
    ----------
    plausible : Tensor
        Whether this contact can be released with a prediction.
    score : Tensor
        The verdict as a number in ``[0, 1]``.
    loitering : Tensor
        How far below its class's usual speed it is running, as a graded value
        in ``[0, 1]`` rather than a threshold -- a cargo at four knots is odd,
        at one knot it is a problem, and a step function would call both the
        same.
    over_corridor : Tensor
        Whether it sits above a cable or a pipe.
    off_lane : Tensor
        Whether a lane-bound class is far from any lane.
    """

    plausible: torch.Tensor
    score: torch.Tensor
    loitering: torch.Tensor
    over_corridor: torch.Tensor
    off_lane: torch.Tensor


class Plausibility:
    """The released-or-not decision, batched.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry, for the lane and corridor distance rasters.
    """

    def __init__(self, config: Config, statics: Statics) -> None:
        self._config = config
        self._statics = statics
        device = statics.device
        classes = config.traffic.classes
        self._mean = torch.tensor(
            [item.speed_mean_ms for item in classes], dtype=torch.float32, device=device
        )
        # A class with no spread would divide by zero and call every speed
        # infinitely surprising.
        self._sigma = torch.tensor(
            [max(item.speed_sigma_ms, 0.2) for item in classes], dtype=torch.float32, device=device
        )
        self._lane_bound = torch.tensor(
            [item.rail is not None for item in classes], dtype=torch.bool, device=device
        )
        self._half_width = max(rail.width_m for rail in config.rails) / 2.0
        self._corridor_half_width = max(item.half_width_m for item in config.corridors)

    def assess(
        self, position: torch.Tensor, velocity: torch.Tensor, vessel_class: torch.Tensor
    ) -> Assessment:
        """Judge a batch of contacts.

        Parameters
        ----------
        position : Tensor
            Where the observer places the contact, ``(..., 2)`` in local metres.
        velocity : Tensor
            Observed velocity, ``(..., 2)``.
        vessel_class : Tensor
            The class the observer thinks it heard. Wrong sometimes -- the
            detector confuses classes, and a misread class is a false alarm in
            its own right.

        Returns
        -------
        Assessment
            Verdict and reasons.
        """
        settings = self._config.plausibility
        speed = velocity.norm(dim=-1)
        surprise = (self._mean[vessel_class] - speed) / self._sigma[vessel_class]
        loitering = torch.sigmoid(settings.loiter_sharpness * (surprise - settings.loiter_sigmas))

        to_corridor = sample_raster(self._statics, self._statics.corridor_distance_m, position)
        over_corridor = to_corridor < self._corridor_half_width

        to_lane = sample_raster(self._statics, self._statics.rail_distance_m, position)
        off_lane = self._lane_bound[vessel_class] & (
            to_lane > settings.off_lane_multiple * self._half_width
        )

        penalty = (
            loitering * (settings.loiter_penalty + settings.corridor_penalty * over_corridor)
            + settings.off_lane_penalty * off_lane
        )
        score = (1.0 - penalty).clamp(0.0, 1.0)
        return Assessment(
            plausible=score >= settings.threshold,
            score=score,
            loitering=loitering,
            over_corridor=over_corridor,
            off_lane=off_lane,
        )

    def explain(self, assessment: Assessment, index: tuple[int, ...]) -> str:
        """Return why one contact was judged as it was, in words.

        Only for reading a run back; nothing in the training path calls it.

        Parameters
        ----------
        assessment : Assessment
            A batch of verdicts.
        index : tuple of int
            Which one to explain.

        Returns
        -------
        str
            One line.
        """
        verdict = "released" if bool(assessment.plausible[index]) else "held"
        reasons = []
        if float(assessment.loitering[index]) > 0.5:
            reasons.append(f"loitering {float(assessment.loitering[index]):.2f}")
        if bool(assessment.over_corridor[index]):
            reasons.append("over a corridor")
        if bool(assessment.off_lane[index]):
            reasons.append("off its lane")
        detail = ", ".join(reasons) if reasons else "nothing out of the ordinary"
        return f"{verdict} (score {float(assessment.score[index]):.2f}): {detail}"
