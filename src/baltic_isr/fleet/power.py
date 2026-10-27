"""The speed-versus-endurance law, and what it costs to come home.

Pure functions over configuration. No vehicle state lives here -- batch 3 puts
the state machine on top. Keeping the law separate is what lets the test suite
check it against the published endurance figure without building a fleet.
"""

from __future__ import annotations

import math

from baltic_isr.config import Auv


def power_w(speed_ms: float, auv: Auv) -> float:
    """Return total electrical draw at a given speed.

    ``P = P_hotel + k v^3``. The cube is why the three regimes are not three
    points on a line: doubling speed multiplies propulsion by eight.

    Parameters
    ----------
    speed_ms : float
        Speed through the water.
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Power in watts.
    """
    return auv.hotel_power_w + auv.propulsion_coefficient * speed_ms**3


def endurance_s(speed_ms: float, auv: Auv) -> float:
    """Return how long a full battery lasts at a given speed.

    Parameters
    ----------
    speed_ms : float
        Speed through the water.
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Endurance in seconds.
    """
    return auv.battery_j / power_w(speed_ms, auv)


def range_m(speed_ms: float, auv: Auv) -> float:
    """Return the distance a full battery buys at a given speed.

    Not monotonic in the way intuition expects: it peaks well below maximum
    speed, because the cube in the denominator outruns the linear numerator.

    Parameters
    ----------
    speed_ms : float
        Speed through the water.
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Range in metres.
    """
    return speed_ms * endurance_s(speed_ms, auv)


def most_efficient_speed_ms(auv: Auv) -> float:
    """Return the speed that maximises distance per battery.

    Differentiating ``v E / (P_h + k v^3)`` gives ``v = (P_h / 2k)^(1/3)``. It
    is a strikingly slow speed, and it is why a transit regime and a search
    regime are not the same thing.

    Parameters
    ----------
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Speed in m/s.
    """
    return float((auv.hotel_power_w / (2.0 * auv.propulsion_coefficient)) ** (1.0 / 3.0))


def energy_to_travel_j(distance_m: float, speed_ms: float, auv: Auv) -> float:
    """Return the energy needed to cover a distance at a speed.

    Parameters
    ----------
    distance_m : float
        Distance over the ground.
    speed_ms : float
        Speed through the water.
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Energy in joules.
    """
    if speed_ms <= 0.0:
        return math.inf
    return power_w(speed_ms, auv) * distance_m / speed_ms


def return_margin_j(distance_to_base_m: float, auv: Auv) -> float:
    """Return the energy a vehicle must keep in hand to get home.

    The forced recall of specification section 2.5 compares this against what
    is left. It assumes the cruise regime, because coming home at a sprint
    costs more than it saves and drifting home is not coming home.

    Parameters
    ----------
    distance_to_base_m : float
        Distance to the nearest base.
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    float
        Energy in joules, including the untouchable reserve.
    """
    transit = energy_to_travel_j(distance_to_base_m, auv.regimes.cruise_ms, auv)
    return transit + auv.energy_reserve_fraction * auv.battery_j


def regime_speeds(auv: Auv) -> dict[str, float]:
    """Return the three regime speeds by name.

    Parameters
    ----------
    auv : Auv
        Vehicle parameters.

    Returns
    -------
    dict
        Mapping of regime name to speed in m/s.
    """
    return {
        "drift": auv.regimes.drift_ms,
        "cruise": auv.regimes.cruise_ms,
        "sprint": auv.regimes.sprint_ms,
    }
