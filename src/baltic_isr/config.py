"""Configuration schema, loading and fingerprinting.

Two base files describe the vehicle and the world; a scenario file overlays
them. Nothing carries a default that is not also written in ``config/``, so the
YAML is the single place a number lives and docs/constants.md can point at it.

Every model is frozen and forbids unknown keys: a typo in a scenario file is an
error, not a silently ignored setting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = _PACKAGE_ROOT / "config"

LonLat = tuple[float, float]


class Frozen(BaseModel):
    """Base for every configuration section."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Simulation(Frozen):
    """Cadence and reproducibility."""

    decision_interval_s: float = Field(gt=0.0)
    perception_interval_s: float = Field(gt=0.0)
    episode_s: float = Field(gt=0.0)
    seed: int


class Geo(Frozen):
    """The working box and its projection."""

    projected_crs: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    sea_area_m2: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lon_min >= self.lon_max or self.lat_min >= self.lat_max:
            raise ValueError("box bounds must be ordered min < max")
        return self


class Environment(Frozen):
    """Water column, bottom and the noise floor they produce."""

    water_depth_m: float = Field(gt=0.0)
    ambient_noise_db: float
    absorption_db_per_km: float = Field(ge=0.0)
    bottom_loss_db_per_km: float = Field(ge=0.0)
    current_speed_ms: float = Field(ge=0.0)
    current_direction_deg: float


class Regimes(Frozen):
    """The three operating points. This is the variable the project studies."""

    drift_ms: float = Field(ge=0.0)
    cruise_ms: float = Field(gt=0.0)
    sprint_ms: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.drift_ms < self.cruise_ms < self.sprint_ms:
            raise ValueError("regimes must be ordered drift < cruise < sprint")
        return self


class Auv(Frozen):
    """The reference vehicle. See docs/constants.md for every number here."""

    length_m: float = Field(gt=0.0)
    diameter_m: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    battery_j: float = Field(gt=0.0)
    hotel_power_w: float = Field(gt=0.0)
    propulsion_coefficient: float = Field(gt=0.0)
    max_speed_ms: float = Field(gt=0.0)
    operating_depth_m: float = Field(gt=0.0)
    vertical_speed_ms: float = Field(gt=0.0)
    energy_reserve_fraction: float = Field(ge=0.0, lt=1.0)
    recharge_s: float = Field(gt=0.0)
    handling_s: float = Field(ge=0.0)
    regimes: Regimes

    @model_validator(mode="after")
    def _sprint_within_envelope(self) -> Self:
        if self.regimes.sprint_ms > self.max_speed_ms:
            raise ValueError("sprint regime exceeds the vehicle envelope")
        return self


class Acoustics(Frozen):
    """Band, array and detector. Detection range is solved from these."""

    band_centre_hz: float = Field(gt=0.0)
    band_width_hz: float = Field(gt=0.0)
    sound_speed_ms: float = Field(gt=0.0)
    array_aperture_m: float = Field(gt=0.0)
    integration_s: float = Field(gt=0.0)
    detection_index: float = Field(gt=0.0)
    flow_noise_crossover_ms: float = Field(gt=0.0)
    flow_noise_slope: float = Field(gt=0.0)
    flow_noise_rolloff_db_per_decade: float = Field(ge=0.0)
    machinery_onset_ms: float = Field(gt=0.0)
    machinery_penalty_db: float = Field(ge=0.0)
    machinery_rolloff_db_per_decade: float = Field(ge=0.0)
    self_noise_array_gain_db: float = Field(ge=0.0)
    max_probability: float = Field(gt=0.0, le=1.0)
    bearing_sigma_deg: float = Field(ge=0.0)
    range_error_fraction: float = Field(ge=0.0)
    false_alarm_rate_hz: float = Field(ge=0.0)
    class_confusion: float = Field(ge=0.0, le=1.0)


class Comms(Frozen):
    """Acoustic modem and the cost of surfacing for radio."""

    frequency_hz: float = Field(gt=0.0)
    range_m: float = Field(gt=0.0)
    bitrate_bps: float = Field(gt=0.0)
    slot_s: float = Field(gt=0.0)
    protocol_latency_s: float = Field(ge=0.0)
    source_level_db: float
    required_snr_db: float
    doppler_tolerance_ms: float = Field(gt=0.0)
    surface_dwell_s: float = Field(ge=0.0)
    surface_energy_j: float = Field(ge=0.0)

    @property
    def payload_bytes(self) -> int:
        """Return what one exchange carries, which is the design constraint."""
        return int(self.bitrate_bps * self.slot_s / 8.0)


class VesselClass(Frozen):
    """One traffic population."""

    name: str
    share: float = Field(gt=0.0, le=1.0)
    speed_mean_ms: float = Field(gt=0.0)
    speed_sigma_ms: float = Field(ge=0.0)
    source_level_db: float
    rail: str | None = None
    predictable: bool = True
    growth_m_per_s: float = Field(gt=0.0)


class Rail(Frozen):
    """A navigation lane. Prediction is only possible because these exist."""

    name: str
    width_m: float = Field(gt=0.0)
    vertices: list[LonLat] = Field(min_length=2)


class Corridor(Frozen):
    """A seabed asset. Loitering over one is what makes behaviour suspicious."""

    name: str
    kind: Literal["hvdc", "gas", "fibre"]
    half_width_m: float = Field(gt=0.0)
    vertices: list[LonLat] = Field(min_length=2)


class Base(Frozen):
    """A shore charging point."""

    name: str
    lon: float
    lat: float


class Traffic(Frozen):
    """The surface picture the fleet has to keep."""

    n_vessels: int = Field(gt=0)
    classes: list[VesselClass] = Field(min_length=1)

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> Self:
        total = sum(item.share for item in self.classes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"vessel class shares must sum to 1, got {total}")
        return self


class Navigation(Frozen):
    """Dead reckoning, and why the estimate drifts away from the truth."""

    heading_bias_sigma_deg: float = Field(ge=0.0)
    dvl_scale_error: float = Field(ge=0.0)
    random_walk_m_per_sqrt_m: float = Field(ge=0.0)
    gnss_sigma_m: float = Field(ge=0.0)
    surface_trigger_m: float = Field(gt=0.0)


class Prediction(Frozen):
    """How fast a held prediction goes wrong, and when it is contradicted."""

    rail_growth_m_per_s: float = Field(gt=0.0)
    # How a contact's ellipse grows once it is off its lane. "class" uses the
    # transit speed of the class that was heard -- right for a vessel in
    # transit, and 22x too pessimistic for a loitering one, which is the only
    # kind that matters. See Picture._growth_for and docs/constants.md C-10-04.
    growth_model: Literal["class", "observed", "blended", "drift"] = "class"
    growth_floor_ms: float = Field(gt=0.0, default=0.35)
    growth_manoeuvre: float = Field(gt=0.0, default=1.0)
    # "drift" only: the shortest interval it will divide by, and how much of
    # the new measurement it folds into the stored rate.
    growth_floor_s: float = Field(gt=0.0, default=300.0)
    growth_memory: float = Field(gt=0.0, le=1.0, default=0.4)
    gate_mahalanobis: float = Field(gt=0.0)
    alarm_gate_m: float = Field(gt=0.0)
    presence_recovery_s: float = Field(gt=0.0)
    capacity: int = Field(gt=0)
    # Where the search grid expects to find something worth finding. The lane
    # term is traffic; the cable term is what the traffic might be there to
    # damage. Both zero gives a flat prior over the water.
    search_lane_weight: float = Field(ge=0.0, default=0.75)
    search_cable_weight: float = Field(ge=0.0, default=0.0)
    search_floor: float = Field(gt=0.0, default=0.25)


class PlausibilityConfig(Frozen):
    """The released-or-not rule. Explicit by requirement, see spec section 12.5."""

    loiter_sigmas: float = Field(ge=0.0)
    loiter_sharpness: float = Field(gt=0.0)
    loiter_penalty: float = Field(ge=0.0, le=1.0)
    corridor_penalty: float = Field(ge=0.0, le=1.0)
    off_lane_penalty: float = Field(ge=0.0, le=1.0)
    off_lane_multiple: float = Field(gt=0.0)
    threshold: float = Field(gt=0.0, lt=1.0)


class Reward(Frozen):
    """What the policy is paid on. Specification section 6.4."""

    verification: float = Field(ge=0.0)
    relay_multiplier: float = Field(ge=1.0)
    alarm: float = Field(ge=0.0)
    staleness: float = Field(ge=0.0)
    exploration: float = Field(ge=0.0)
    energy: float = Field(ge=0.0)
    loss: float = Field(ge=0.0)
    # Keeping the cables listened to. Zero reproduces the reward as it was
    # before the cables entered it, which is how the two are told apart.
    cable: float = Field(ge=0.0, default=0.0)
    cable_horizon_s: float = Field(gt=0.0, default=21600.0)
    # Pay once for work done once. Without these, every vehicle in range
    # collects the full credit while the job is done a single time: measured,
    # 66 % of the cable credit was paid twice or more, and 3.36 vehicles were
    # paid for hearing the same vessel. Stacking was subsidised, and the fleet
    # duly stacked -- 78 % of vehicles sat within one detection range of a
    # neighbour. See docs/constants.md C-12-07.
    cable_shared: bool = True
    verification_shared: bool = True
    # Potential-based shaping toward the most neglected cable. The discount
    # horizon is 22 km of lookahead and the far end of EstLink 2 is 79 km from
    # the nearest base, so the trip is simply invisible to the value function.
    # Shaping makes a distant goal visible without moving the optimum, which
    # is what raising gamma would risk.
    cable_shaping: float = Field(ge=0.0, default=0.0)
    # Must match the PPO discount or the shaping stops being policy-invariant.
    shaping_discount: float = Field(gt=0.0, lt=1.0, default=0.95)


class Fleet(Frozen):
    """How many vehicles, where they launch from, and how deviations are injected."""

    count: int = Field(gt=0)
    start: Literal["rotation", "cold"] = "rotation"
    arrival_radius_m: float = Field(gt=0.0)
    # The shore, as the depth of water the vehicle needs rather than a distance
    # from the waterline: a keel clearance is physical, a setback is not. Must
    # stay inside the shelf of world.field, which converts it.
    min_water_m: float = Field(gt=0.0, lt=70.0)
    berth_clearance_m: float = Field(gt=0.0)
    base_approach_m: float = Field(gt=0.0)
    deviations_per_episode: int = Field(ge=0)
    deviation_duration_s: float = Field(gt=0.0)
    deviation_corridor_fraction: float = Field(ge=0.0, le=1.0)
    gate_m: float = Field(gt=0.0)
    track_capacity: int = Field(gt=0)


class Config(Frozen):
    """A fully resolved configuration."""

    schema_version: int
    scenario: str
    simulation: Simulation
    geo: Geo
    environment: Environment
    auv: Auv
    acoustics: Acoustics
    comms: Comms
    traffic: Traffic
    fleet: Fleet
    navigation: Navigation
    prediction: Prediction
    plausibility: PlausibilityConfig
    reward: Reward
    rails: list[Rail]
    corridors: list[Corridor]
    bases: list[Base] = Field(min_length=1)

    @model_validator(mode="after")
    def _known_schema(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        return self

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the whole configuration.

        Written into every run manifest beside the seed. The two together are
        what makes a result reproducible bit for bit.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge an overlay into a base mapping, returning a new mapping."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    """Read one YAML file into a mapping."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def load(scenario: str = "baseline", directory: Path | None = None) -> Config:
    """Load the base configuration and overlay one scenario.

    Parameters
    ----------
    scenario : str
        Name of a file in ``config/scenarios/``, without the extension.
    directory : Path, optional
        Configuration root. Defaults to the repository's ``config/``.

    Returns
    -------
    Config
        A validated, frozen configuration.
    """
    root = directory or CONFIG_DIR
    merged: dict[str, Any] = {}
    for name in ("auv.yaml", "world.yaml"):
        merged = _merge(merged, _read(root / name))
    scenario_path = root / "scenarios" / f"{scenario}.yaml"
    if not scenario_path.is_file():
        available = sorted(path.stem for path in (root / "scenarios").glob("*.yaml"))
        raise FileNotFoundError(f"unknown scenario {scenario!r}; available: {available}")
    merged = _merge(merged, _read(scenario_path))
    merged.setdefault("scenario", scenario)
    return Config.model_validate(merged)
