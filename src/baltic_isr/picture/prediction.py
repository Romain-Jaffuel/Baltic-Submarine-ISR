"""How far ahead a contact can be predicted, and by which rule.

Releasing a cargo instead of escorting it (specification section 4) only works
if the prediction issued in its place is worth something. How long it is worth
something is not a matter of opinion: it is measured, by running the traffic
model and watching two predictors fail.

*Constant velocity* takes the last observed state and extrapolates. It is all
one can do for a fishing boat.

*Rail following* advances the contact along the lane it was found in, at the
speed it was making. The fleet is entitled to know where the shipping lanes are
-- they are published -- and that knowledge is most of what makes a cargo
predictable.

The error of each grows with elapsed time; where it crosses the staleness gate
is the useful horizon. :func:`measure_horizons` produces those numbers and
``baltic-isr predictability`` prints them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baltic_isr.config import Config
from baltic_isr.world.statics import Statics, sample_rail, sample_rail_tangent


@dataclass(frozen=True)
class Horizon:
    """What one predictor achieves on one class of vessel.

    Attributes
    ----------
    name : str
        Vessel class.
    rule : str
        ``"constant_velocity"`` or ``"rail"``.
    growth_m_per_s : float
        Slope of position error against elapsed time, as a diagnostic.
    horizon_s : float
        When the median error first reaches the staleness gate. Read off the
        curve rather than off the fitted slope, because the error is not linear
        -- a lane-bound vessel is predictable for hours and then turns.
    error_at_hour_m : float
        Median error one hour ahead, which is the scale the decision cadence
        actually cares about.
    """

    name: str
    rule: str
    growth_m_per_s: float
    horizon_s: float
    error_at_hour_m: float


def constant_velocity(
    position: torch.Tensor, velocity: torch.Tensor, elapsed_s: torch.Tensor
) -> torch.Tensor:
    """Extrapolate a contact in a straight line.

    Parameters
    ----------
    position : Tensor
        State at the last observation, ``(..., 2)``.
    velocity : Tensor
        Velocity at the last observation, ``(..., 2)``.
    elapsed_s : Tensor
        Seconds since, broadcastable to ``position``'s leading shape.

    Returns
    -------
    Tensor
        Predicted position, ``(..., 2)``.
    """
    return position + velocity * elapsed_s.unsqueeze(-1)


def along_rail(
    statics: Statics,
    rail: torch.Tensor,
    progress: torch.Tensor,
    speed_ms: torch.Tensor,
    offset_m: torch.Tensor,
    direction: torch.Tensor,
    elapsed_s: torch.Tensor,
) -> torch.Tensor:
    """Advance a contact along the lane it was found in.

    Parameters
    ----------
    statics : Statics
        Device geometry.
    rail : Tensor
        Rail index per contact.
    progress : Tensor
        Fraction along the rail at the last observation.
    speed_ms : Tensor
        Speed made good at the last observation.
    offset_m : Tensor
        Observed cross-track position within the lane.
    direction : Tensor
        Which way along the rail the contact was heading, ``+1`` or ``-1``.
    elapsed_s : Tensor
        Seconds since.

    Returns
    -------
    Tensor
        Predicted position, ``(..., 2)``.
    """
    index = rail.clamp_min(0)
    # Clamped rather than wrapped: the predictor cannot see past the end of the
    # lane either, and pretending otherwise would flatter it.
    advanced = (progress + direction * speed_ms * elapsed_s / statics.rail_length[index]).clamp(
        0.0, 1.0
    )
    anchor = sample_rail(statics, index, advanced)
    tangent = sample_rail_tangent(statics, index, advanced)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
    return anchor + normal * offset_m.unsqueeze(-1)


def fit_growth(elapsed_s: torch.Tensor, error_m: torch.Tensor) -> float:
    """Fit ``error = growth * elapsed`` through the origin.

    A line through the origin rather than a free intercept: at zero elapsed time
    the prediction *is* the observation, so an intercept would only absorb
    measurement noise that belongs elsewhere in the budget.

    Parameters
    ----------
    elapsed_s : Tensor
        Elapsed times.
    error_m : Tensor
        Position errors at those times.

    Returns
    -------
    float
        Metres of error per second.
    """
    numerator = float((elapsed_s * error_m).sum())
    denominator = float((elapsed_s * elapsed_s).sum())
    return numerator / denominator if denominator > 0.0 else 0.0


def measure_horizons(
    config: Config,
    statics: Statics,
    worlds: int = 16,
    horizon_s: float = 21_600.0,
    samples: int = 24,
) -> list[Horizon]:
    """Run the traffic model and measure how fast each predictor goes wrong.

    This is the batch 4 acceptance criterion: the useful prediction horizon of
    each vessel class comes out of the simulator, not out of a table.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry.
    worlds : int
        How many worlds to average over.
    horizon_s : float
        How far ahead to push each prediction.
    samples : int
        How many points to sample along that span.

    Returns
    -------
    list of Horizon
        One entry per class per rule.
    """
    from baltic_isr.world.traffic import Traffic

    generator = Traffic(config, statics, worlds)
    state = generator.reset()
    step_s = horizon_s / samples

    start_xy = state.xy.clone()
    start_velocity = state.velocity.clone()
    start_rail = state.rail.clone()
    start_progress = state.progress.clone()
    start_speed = state.speed.clone()
    start_offset = state.lane_offset.clone()
    start_direction = state.direction.clone()
    vessel_class = state.vessel_class.clone()

    elapsed: list[float] = []
    errors: dict[str, list[torch.Tensor]] = {"constant_velocity": [], "rail": []}
    for index in range(1, samples + 1):
        moment = index * step_s
        state = generator.step(state, moment, step_s)
        span = torch.full_like(start_speed, moment)
        elapsed.append(moment)
        errors["constant_velocity"].append(
            (constant_velocity(start_xy, start_velocity, span) - state.xy).norm(dim=-1)
        )
        errors["rail"].append(
            (
                along_rail(
                    statics,
                    start_rail,
                    start_progress,
                    start_speed,
                    start_offset,
                    start_direction,
                    span,
                )
                - state.xy
            ).norm(dim=-1)
        )

    gate = config.fleet.gate_m
    span_tensor = torch.tensor(elapsed, device=statics.device)
    results: list[Horizon] = []
    for class_index, item in enumerate(config.traffic.classes):
        member = vessel_class == class_index
        if not bool(member.any()):
            continue
        for rule, series in errors.items():
            if rule == "rail" and item.rail is None:
                continue
            medians = torch.stack([value[member].median() for value in series])
            crossed = (medians >= gate).nonzero()
            horizon = float(span_tensor[crossed[0, 0]]) if crossed.numel() else float("inf")
            hour = min(range(len(elapsed)), key=lambda index: abs(elapsed[index] - 3600.0))
            results.append(
                Horizon(
                    name=item.name,
                    rule=rule,
                    growth_m_per_s=fit_growth(span_tensor, medians),
                    horizon_s=horizon,
                    error_at_hour_m=float(medians[hour]),
                )
            )
    return results
