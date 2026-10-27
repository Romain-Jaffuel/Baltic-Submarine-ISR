"""Dead reckoning, and why the estimate walks away from the truth.

Specification v4 section 5.2 said the dominant error is the current the vehicle
cannot estimate. **That is wrong for this vehicle in this basin**, and batch 3
is where it showed. A Doppler velocity log bottom-locks comfortably in fifty
metres of water, and a bottom-locked DVL measures velocity over the *ground* --
the current is in the measurement, not missing from it.

What is left is the instrument itself, and one term dominates the others by an
order of magnitude: a compass bias rotates every increment by the same small
angle, so the error grows as ``distance x sin(bias)`` and never averages out. At
one degree over a hundred kilometres that is 1.7 km, which is most of the three
kilometre staleness gate. Scale-factor error contributes a tenth of that, and
the random walk a hundredth.

This matters beyond navigation: a vehicle that believes it is three kilometres
from where it is issues predictions that are wrong by three kilometres, and the
vehicle that goes to verify one finds nothing. That is a **false alarm**
manufactured by the navigation system, and it is the mechanism specification
section 5.2 cares about.
"""

from __future__ import annotations

import math

import torch

from baltic_isr.config import Navigation


def sample_instrument_errors(
    shape: tuple[int, ...],
    navigation: Navigation,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the fixed instrument errors of each vehicle.

    Drawn once per vehicle per episode, not per step: a compass bias is a
    property of the compass. Redrawing it every step would let it average away,
    which is precisely what a bias does not do.

    Parameters
    ----------
    shape : tuple of int
        Leading shape, usually ``(worlds, vehicles)``.
    navigation : Navigation
        Error magnitudes.
    generator : torch.Generator
        Seeded stream.
    device : torch.device or str
        Where to place the tensors.

    Returns
    -------
    tuple of Tensor
        Heading bias in radians, and DVL scale error as a fraction.
    """
    bias = torch.randn(shape, generator=generator, device=device) * math.radians(
        navigation.heading_bias_sigma_deg
    )
    scale = torch.randn(shape, generator=generator, device=device) * navigation.dvl_scale_error
    return bias, scale


def advance(
    estimate: torch.Tensor,
    ground_velocity: torch.Tensor,
    heading_bias: torch.Tensor,
    scale_error: torch.Tensor,
    dt_s: float,
) -> torch.Tensor:
    """Integrate one step of dead reckoning.

    The vehicle measures its true velocity over ground, then rotates it by its
    compass bias and scales it by its log error before integrating. Both errors
    are systematic, so the result diverges steadily rather than wandering.

    Parameters
    ----------
    estimate : Tensor
        Believed position, shape ``(..., 2)``.
    ground_velocity : Tensor
        True velocity over ground, shape ``(..., 2)``.
    heading_bias : Tensor
        Compass bias in radians, shape ``(...)``.
    scale_error : Tensor
        Log scale error as a fraction, shape ``(...)``.
    dt_s : float
        Step length.

    Returns
    -------
    Tensor
        The updated estimate.
    """
    cos, sin = torch.cos(heading_bias), torch.sin(heading_bias)
    east, north = ground_velocity[..., 0], ground_velocity[..., 1]
    rotated = torch.stack([cos * east - sin * north, sin * east + cos * north], dim=-1)
    return estimate + rotated * (1.0 + scale_error).unsqueeze(-1) * dt_s


def uncertainty_m(distance_since_fix: torch.Tensor, navigation: Navigation) -> torch.Tensor:
    """Return what the vehicle believes its position error to be.

    Not the true error -- the vehicle cannot know that -- but the one-sigma it
    can compute from how far it has run since its last fix. It is what the
    surfacing decision and the prediction covariance are both built on.

    Parameters
    ----------
    distance_since_fix : Tensor
        Path length since the last GNSS fix, in metres.
    navigation : Navigation
        Error magnitudes.

    Returns
    -------
    Tensor
        One-sigma position uncertainty in metres.
    """
    bias = distance_since_fix * math.sin(math.radians(navigation.heading_bias_sigma_deg))
    scale = distance_since_fix * navigation.dvl_scale_error
    walk = navigation.random_walk_m_per_sqrt_m * torch.sqrt(distance_since_fix.clamp_min(0.0))
    return torch.sqrt(bias**2 + scale**2 + walk**2 + navigation.gnss_sigma_m**2)


def apply_fix(
    truth: torch.Tensor,
    navigation: Navigation,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return the position a surfaced vehicle adopts from GNSS.

    Parameters
    ----------
    truth : Tensor
        True position, shape ``(..., 2)``.
    navigation : Navigation
        Fix accuracy.
    generator : torch.Generator
        Seeded stream.

    Returns
    -------
    Tensor
        The corrected estimate.
    """
    noise = torch.randn(truth.shape, generator=generator, device=truth.device)
    return truth + noise * navigation.gnss_sigma_m
