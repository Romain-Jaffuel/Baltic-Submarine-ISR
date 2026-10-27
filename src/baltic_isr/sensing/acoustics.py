"""Passive sonar equation, and the ranges it produces.

Detection range is not a configured number. It is solved from a level budget,
because the whole project rests on how fast a vehicle can go before it stops
hearing, and a postulated range would beg that question. See docs/constants.md.

Everything here is a setup-time scalar computation. The training path reads the
ranges this module returns; it never calls it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from baltic_isr.config import Acoustics, Comms, Environment

_BISECTION_STEPS = 60
_MAX_RANGE_M = 200_000.0
_MIN_RANGE_M = 1.0


def array_gain_db(acoustics: Acoustics) -> float:
    """Return the directivity index of the flank array at the band centre.

    A line array of aperture ``L`` long compared with the wavelength gives
    ``10 log(2L/lambda)``. On a 3.5 m vehicle the aperture is small, which is
    why the design band sits above a kilohertz: below it the array has no
    directivity worth the name.

    Parameters
    ----------
    acoustics : Acoustics
        Band and array parameters.

    Returns
    -------
    float
        Directivity index in dB, floored at zero.
    """
    wavelength = acoustics.sound_speed_ms / acoustics.band_centre_hz
    return max(0.0, 10.0 * math.log10(2.0 * acoustics.array_aperture_m / wavelength))


def transmission_loss_db(range_m: float, environment: Environment) -> float:
    """Return one-way transmission loss over a shallow-water path.

    Spherical spreading out to one water depth, cylindrical beyond, plus a
    linear term. The linear term lumps molecular absorption with bottom
    interaction; in the Gulf of Finland the second dominates the first sixty to
    one, because brackish water barely absorbs and soft clay absorbs a great
    deal.

    Parameters
    ----------
    range_m : float
        Slant range in metres.
    environment : Environment
        Water depth and the two linear loss terms.

    Returns
    -------
    float
        Transmission loss in dB.
    """
    depth = environment.water_depth_m
    linear = environment.absorption_db_per_km + environment.bottom_loss_db_per_km
    if range_m > depth:
        spread = 20.0 * math.log10(depth) + 10.0 * math.log10(range_m / depth)
    else:
        spread = 20.0 * math.log10(max(range_m, _MIN_RANGE_M))
    return spread + linear * range_m / 1000.0


def self_noise_db(
    speed_ms: float, frequency_hz: float, acoustics: Acoustics, environment: Environment
) -> float:
    """Return the vehicle's own noise at the array, as a spectrum level.

    Two sources with very different spectra, summed in power.

    Turbulent flow over the hull rises steeply with speed and falls steeply with
    frequency: it is a low-frequency phenomenon. Referencing it to the crossover
    speed -- where it equals ambient -- makes the assumption inspectable, which
    matters because it is the single most load-bearing number in the model.

    Thruster cavitation appears above an onset speed and is broadband, so it
    rolls off far more gently. That difference is why the sonar band and the
    modem band are affected so unequally.

    Parameters
    ----------
    speed_ms : float
        Speed through the water.
    frequency_hz : float
        Frequency at which the level is wanted.
    acoustics : Acoustics
        Crossover speed, slopes and rolloffs.
    environment : Environment
        Ambient level, which anchors the crossover.

    Returns
    -------
    float
        Self-noise spectrum level in dB re 1 uPa^2/Hz, or ``-inf`` at rest.
    """
    decades = math.log10(frequency_hz / acoustics.band_centre_hz)
    power = 0.0
    if speed_ms > 0.0:
        flow = (
            environment.ambient_noise_db
            + acoustics.flow_noise_slope * math.log10(speed_ms / acoustics.flow_noise_crossover_ms)
            - acoustics.flow_noise_rolloff_db_per_decade * decades
        )
        power += 10.0 ** (flow / 10.0)
    if speed_ms >= acoustics.machinery_onset_ms:
        cavitation = (
            environment.ambient_noise_db
            + acoustics.machinery_penalty_db
            - acoustics.machinery_rolloff_db_per_decade * decades
        )
        power += 10.0 ** (cavitation / 10.0)
    return 10.0 * math.log10(power) if power > 0.0 else -math.inf


def noise_after_array_db(
    speed_ms: float, frequency_hz: float, acoustics: Acoustics, environment: Environment
) -> float:
    """Return the noise level at the beamformer output.

    Ambient noise is close to isotropic and the array works against it at full
    directivity. Flow noise is not: the pressure field of a turbulent boundary
    layer is correlated across the array, so beamforming buys much less. The two
    therefore get different gains before they are summed in power.

    Parameters
    ----------
    speed_ms : float
        Speed through the water.
    frequency_hz : float
        Frequency at which the level is wanted.
    acoustics : Acoustics
        Array and self-noise parameters.
    environment : Environment
        Ambient level.

    Returns
    -------
    float
        Noise spectrum level in dB after array gain.
    """
    ambient = environment.ambient_noise_db - array_gain_db(acoustics)
    own = self_noise_db(speed_ms, frequency_hz, acoustics, environment)
    own_after = own - acoustics.self_noise_array_gain_db
    return 10.0 * math.log10(10.0 ** (ambient / 10.0) + 10.0 ** (own_after / 10.0))


def detection_threshold_db(acoustics: Acoustics) -> float:
    """Return the threshold of a broadband energy detector.

    ``5 log(d B / T)`` for detection index ``d``, bandwidth ``B`` and
    integration time ``T``. Integrating longer buys sensitivity at the price of
    smearing a moving target, which is why ``T`` is ten seconds and not a
    minute.

    Parameters
    ----------
    acoustics : Acoustics
        Band width, integration time and detection index.

    Returns
    -------
    float
        Detection threshold in dB.
    """
    return 5.0 * math.log10(
        acoustics.detection_index * acoustics.band_width_hz / acoustics.integration_s
    )


def signal_excess_db(
    range_m: float,
    speed_ms: float,
    source_level_db: float,
    acoustics: Acoustics,
    environment: Environment,
) -> float:
    """Return signal excess at a range: positive means detected.

    Parameters
    ----------
    range_m : float
        Range to the source.
    speed_ms : float
        Own speed, which sets self-noise.
    source_level_db : float
        Source spectrum level at 1 m, at the band centre.
    acoustics : Acoustics
        Band, array and detector parameters.
    environment : Environment
        Propagation and ambient noise.

    Returns
    -------
    float
        Signal excess in dB.
    """
    return (
        source_level_db
        - transmission_loss_db(range_m, environment)
        - noise_after_array_db(speed_ms, acoustics.band_centre_hz, acoustics, environment)
        - detection_threshold_db(acoustics)
    )


def _solve_range(excess: Callable[[float], float], lower: float, upper: float) -> float:
    """Bisect a monotonically decreasing excess function for its zero."""
    if excess(lower) < 0.0:
        return 0.0
    if excess(upper) > 0.0:
        return upper
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        if excess(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def detection_range_m(
    speed_ms: float,
    source_level_db: float,
    acoustics: Acoustics,
    environment: Environment,
) -> float:
    """Return the range at which signal excess reaches zero.

    Parameters
    ----------
    speed_ms : float
        Own speed. This is the whole point: the answer depends on it.
    source_level_db : float
        Source spectrum level at 1 m.
    acoustics : Acoustics
        Band, array and detector parameters.
    environment : Environment
        Propagation and ambient noise.

    Returns
    -------
    float
        Detection range in metres.
    """

    def excess(range_m: float) -> float:
        return signal_excess_db(range_m, speed_ms, source_level_db, acoustics, environment)

    return _solve_range(excess, _MIN_RANGE_M, _MAX_RANGE_M)


def level_limited_modem_range_m(
    speed_ms: float, acoustics: Acoustics, comms: Comms, environment: Environment
) -> float:
    """Return how far the modem would reach if only levels mattered.

    It reaches much further than the modem is actually usable, which is the
    point of computing it: the binding constraint on the link is not the level
    budget. See :func:`modem_range_m`.

    Parameters
    ----------
    speed_ms : float
        Own speed.
    acoustics : Acoustics
        Self-noise parameters.
    comms : Comms
        Modem parameters.
    environment : Environment
        Propagation and ambient noise.

    Returns
    -------
    float
        Range in metres.
    """
    noise = noise_after_array_db(speed_ms, comms.frequency_hz, acoustics, environment)
    budget = comms.source_level_db - noise - comms.required_snr_db

    def excess(range_m: float) -> float:
        return budget - transmission_loss_db(range_m, environment)

    return _solve_range(excess, _MIN_RANGE_M, _MAX_RANGE_M)


def modem_range_m(comms: Comms) -> float:
    """Return the usable modem range, which multipath sets.

    Independent of speed, and that is a finding rather than a simplification.
    Flow noise is a low-frequency phenomenon: by 25 kHz it has rolled off far
    below ambient, so a moving vehicle is no deafer to its modem than a
    stationary one. What speed does cost the link is Doppler, and that is
    :func:`doppler_success` -- a loss of packets, not of range.

    Parameters
    ----------
    comms : Comms
        Modem parameters, carrying the multipath-limited range.

    Returns
    -------
    float
        Range in metres.
    """
    return comms.range_m


def doppler_success(relative_speed_ms: float, comms: Comms) -> float:
    """Return the probability a packet survives the Doppler shift.

    Motion compresses the acoustic time scale by ``v/c``. At 25 kHz and 2.5 m/s
    that is 43 Hz of shift, but the damage is done by the compression of the
    symbol interval across a long packet, not by the shift itself. A modem with
    modest Doppler tracking holds together up to about a metre and a half per
    second and degrades sharply beyond.

    This is the real cost of speed on communications, and it replaces the
    speed-dependent modem range of specification v2, which attributed the
    degradation to self-noise and was wrong about the frequency.

    Parameters
    ----------
    relative_speed_ms : float
        Closing or opening speed between the two vehicles.
    comms : Comms
        Modem parameters, carrying the Doppler tolerance.

    Returns
    -------
    float
        Probability in ``[0, 1]``.
    """
    ratio = abs(relative_speed_ms) / comms.doppler_tolerance_ms
    return math.exp(-(ratio**2))
