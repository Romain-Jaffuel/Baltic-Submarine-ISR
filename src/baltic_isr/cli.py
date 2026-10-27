"""Command line entry points.

``baltic-isr budget`` recomputes every derived number in docs/constants.md from
``config/``. The document quotes that output rather than carrying hand-copied
figures, so the two cannot drift apart silently.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import typer

from baltic_isr import config, geo
from baltic_isr.fleet import auv, power
from baltic_isr.picture import prediction
from baltic_isr.sensing import acoustics
from baltic_isr.world import field, statics, traffic

app = typer.Typer(add_completion=False, help="Speed versus endurance in the Gulf of Finland.")

_PUBLISHED_ENDURANCE_H = 24.0
_PUBLISHED_SPEED_MS = 1.543


@app.command()
def budget(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    target: str = typer.Option("cargo", help="Vessel class the detection range is quoted for."),
) -> None:
    """Print the level budget and everything derived from it."""
    resolved = config.load(scenario)
    vessel = next((item for item in resolved.traffic.classes if item.name == target), None)
    if vessel is None:
        available = [item.name for item in resolved.traffic.classes]
        raise typer.BadParameter(f"unknown class {target!r}; available: {available}")

    sound, environment, comms = resolved.acoustics, resolved.environment, resolved.comms
    geometry = geo.build(resolved)

    typer.echo(f"scenario {resolved.scenario}  fingerprint {resolved.fingerprint}")
    typer.echo(f"seed {resolved.simulation.seed}  payload {comms.payload_bytes} bytes/exchange")

    typer.echo("\npower law")
    for speed in (0.0, power.most_efficient_speed_ms(resolved.auv), 1.2, _PUBLISHED_SPEED_MS, 2.5):
        typer.echo(
            f"  {speed:5.3f} m/s  {power.power_w(speed, resolved.auv):7.1f} W"
            f"  {power.endurance_s(speed, resolved.auv) / 3600:7.2f} h"
            f"  {power.range_m(speed, resolved.auv) / 1000:7.1f} km"
        )
    hours = power.endurance_s(_PUBLISHED_SPEED_MS, resolved.auv) / 3600.0
    error = abs(hours - _PUBLISHED_ENDURANCE_H) / _PUBLISHED_ENDURANCE_H
    typer.echo(
        f"  calibration: {hours:.2f} h at 3 kt vs {_PUBLISHED_ENDURANCE_H} published"
        f"  ({error * 100:.2f} % error)"
    )

    typer.echo(
        f"\nlevel budget   SL {vessel.source_level_db:.0f}"
        f"  AG {acoustics.array_gain_db(sound):.1f}"
        f"  DT {acoustics.detection_threshold_db(sound):.1f}"
        f"  ambient {environment.ambient_noise_db:.0f}"
    )
    ranges: dict[str, float] = {}
    for name, speed in power.regime_speeds(resolved.auv).items():
        own = acoustics.self_noise_db(speed, sound.band_centre_hz, sound, environment)
        noise = acoustics.noise_after_array_db(speed, sound.band_centre_hz, sound, environment)
        detection = acoustics.detection_range_m(speed, vessel.source_level_db, sound, environment)
        own_modem = acoustics.self_noise_db(speed, comms.frequency_hz, sound, environment)
        ranges[name] = detection
        typer.echo(
            f"  {name:7s} {speed:4.1f} m/s   self {own:6.1f}  NL {noise:5.1f}"
            f"  R_det {detection / 1000:5.2f} km"
            f"   | 25 kHz self {own_modem:6.1f}"
            f"  p_doppler {acoustics.doppler_success(speed, comms):5.3f}"
        )
    typer.echo(
        f"  modem {acoustics.modem_range_m(comms) / 1000:.1f} km at every regime"
        f"  (level budget alone would allow"
        f" {acoustics.level_limited_modem_range_m(0.0, sound, comms, environment) / 1000:.1f})"
    )
    typer.echo(
        f"  drift/cruise {ranges['drift'] / ranges['cruise']:.2f}x"
        f"   cruise/sprint {ranges['cruise'] / ranges['sprint']:.2f}x"
    )

    centre = geometry.frame.to_xy(
        0.5 * (resolved.geo.lon_min + resolved.geo.lon_max),
        0.5 * (resolved.geo.lat_min + resolved.geo.lat_max),
    ).reshape(1, 2)
    to_base = float(geometry.distance_to_base(centre)[0])
    typer.echo(f"\ntransit   base to area centre {to_base / 1000:.1f} km")
    for name, speed in power.regime_speeds(resolved.auv).items():
        if speed <= 0.0:
            continue
        energy = power.energy_to_travel_j(2.0 * to_base, speed, resolved.auv)
        share = energy / resolved.auv.battery_j
        verdict = "  IMPOSSIBLE" if share > 1.0 else ""
        typer.echo(
            f"  {name:7s} one way {to_base / speed / 3600:5.2f} h"
            f"   round trip {share * 100:6.1f} % of battery{verdict}"
        )

    typer.echo(f"\nfleet   sea area {resolved.geo.sea_area_m2 / 1e6:.0f} km2")
    for count in (10, 15, 20, 25, 30, 44):
        marker = "  <- configured" if count == resolved.fleet.count else ""
        cover = count * math.pi * ranges["cruise"] ** 2 / resolved.geo.sea_area_m2
        typer.echo(f"  N={count:3d}  instantaneous cruise coverage {cover * 100:5.1f} %{marker}")


@app.command()
def scenarios() -> None:
    """List the scenarios and the detection ranges each one produces."""
    for path in sorted((config.CONFIG_DIR / "scenarios").glob("*.yaml")):
        resolved = config.load(path.stem)
        vessel = resolved.traffic.classes[0]
        derived = [
            acoustics.detection_range_m(
                speed, vessel.source_level_db, resolved.acoustics, resolved.environment
            )
            / 1000.0
            for speed in power.regime_speeds(resolved.auv).values()
        ]
        typer.echo(
            f"  {path.stem:10s} {resolved.fingerprint}"
            f"  R_det {derived[0]:5.2f} / {derived[1]:5.2f} / {derived[2]:5.2f} km"
        )


@app.command()
def world(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    worlds: int = typer.Option(64, help="How many worlds to run in parallel."),
    hours: float = typer.Option(48.0, help="Simulated hours."),
) -> None:
    """Run the traffic model and report what it produced."""
    resolved = config.load(scenario)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = statics.build(resolved, device)
    generator = traffic.Traffic(resolved, fixed, worlds)
    state = generator.reset()

    width, height = float(fixed.extent[0]), float(fixed.extent[1])
    water = float(field.sea_area_m2(fixed))
    typer.echo(f"scenario {resolved.scenario}  device {device}  fingerprint {resolved.fingerprint}")
    typer.echo(
        f"box {width / 1000:.1f} x {height / 1000:.1f} km"
        f"   water {water / 1e6:.0f} km2 ({water / (width * height) * 100:.0f} %)"
    )
    for index, rail in enumerate(resolved.rails):
        over = float(fixed.rail_over_corridor[index].float().mean())
        typer.echo(
            f"  rail {rail.name:18s} {float(fixed.rail_length[index]) / 1000:6.1f} km"
            f"   {over * 100:4.0f} % over a corridor"
        )

    scheduled = torch.isfinite(state.deviation_start_s)
    typer.echo(
        f"\n{worlds} worlds x {state.n_vessels} vessels"
        f"   {float(state.lane_bound.float().mean()) * 100:.0f} % lane-bound"
        f"   {float(scheduled.sum(dim=1).float().mean()):.1f} deviations per world"
    )

    step_s = 60.0
    ticks = int(hours * 3600.0 / step_s)
    started = time.perf_counter()
    for tick in range(ticks):
        state = generator.step(state, (tick + 1) * step_s, step_s)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    anchor = statics.sample_rail(fixed, state.rail.clamp_min(0), state.progress)
    offset = (anchor - state.xy).norm(dim=-1)[state.lane_bound]
    typer.echo(
        f"{hours:.0f} h simulated in {elapsed:.1f} s"
        f"   ({worlds * ticks / elapsed / 1000:.0f} k world-steps/s)"
    )
    typer.echo(
        f"  at sea {float(field.is_sea(fixed, state.xy).float().mean()) * 100:.1f} %"
        f"   lane error median {float(offset.median()):.0f} m"
        f"   p99 {float(offset.quantile(0.99)):.0f} m"
    )


@app.command()
def fleet(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    worlds: int = typer.Option(32, help="How many worlds to run in parallel."),
    start: str = typer.Option("", help="Override the start mode: rotation or cold."),
) -> None:
    """Run the fleet through an episode and report what the rotation delivers."""
    resolved = config.load(scenario)
    if start:
        resolved = resolved.model_copy(
            update={"fleet": resolved.fleet.model_copy(update={"start": start})}
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = statics.build(resolved, device)
    units = auv.Fleet(resolved, fixed, worlds)
    state = units.reset()
    count = state.n_vehicles

    timings = units.sortie(state.station, fixed.bases[state.base])
    leg = float(timings["leg"].mean())
    typer.echo(f"scenario {resolved.scenario}  device {device}  start {resolved.fleet.start}")
    typer.echo(
        f"{count} vehicles   usable battery {units.usable_s / 3600:.1f} h at cruise"
        f"   operational radius {units.usable_s / 2 * resolved.auv.regimes.cruise_ms / 1000:.1f} km"
    )
    typer.echo(
        f"sortie   transit {leg * resolved.auv.regimes.cruise_ms / 1000:.1f} km"
        f" = {leg / 3600:.1f} h each way"
        f"   on station {float(timings['on_station'].mean()) / 3600:.1f} h"
        f"   cycle {float(timings['cycle'].mean()) / 3600:.1f} h"
    )
    typer.echo(
        f"  duty cycle {float((timings['on_station'] / timings['cycle']).mean()) * 100:.1f} %"
        f"  -- only that fraction of the fleet is working at any moment"
    )

    step_s = 300.0
    ticks = int(resolved.simulation.episode_s / step_s)
    working, deployed = [], []
    for tick in range(ticks):
        action = auv.REGIME_CRUISE * auv.HEADINGS + _bearing_to(state.station, state.xy)
        state = units.step(state, action, (tick + 1) * step_s, step_s)
        deployed.append(float((state.mode == auv.MODE_DEPLOYED).float().mean()))
        on_station = state.listening & (units.distance_to_base(state) > 15_000.0)
        working.append(float(on_station.float().mean()))

    mean_working = sum(working) / len(working)
    spread = max(deployed) - min(deployed)
    typer.echo(
        f"\nover {resolved.simulation.episode_s / 3600:.0f} h:"
        f" {mean_working * 100:.1f} % of the fleet on station on average"
        f" = {mean_working * count:.1f} vehicles of {count}"
    )
    typer.echo(f"  deployed fraction varies by {spread * 100:.1f} points across the episode")
    if spread < 0.1:
        typer.echo("  flat: no synchronised recall")
    else:
        typer.echo("  swings: the whole base comes and goes together")
    coverage = mean_working * count * math.pi * 5910.0**2 / resolved.geo.sea_area_m2
    typer.echo(f"  instantaneous coverage at cruise {coverage * 100:.1f} % of the water")
    typer.echo(
        f"  drift {float(state.drift_error_m.mean()):.0f} m mean"
        f"   lost {int((state.mode == auv.MODE_LOST).sum())} of {state.mode.numel()}"
    )


def _bearing_to(target: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Return the eight-point heading index from a position toward a target."""
    direction = target - position
    angle = torch.atan2(direction[..., 0], direction[..., 1]) % (2.0 * math.pi)
    return torch.round(angle / (2.0 * math.pi / auv.HEADINGS)).long() % auv.HEADINGS


@app.command()
def predictability(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    worlds: int = typer.Option(16, help="How many worlds to average over."),
    hours: float = typer.Option(6.0, help="How far ahead to push each prediction."),
    figure: bool = typer.Option(True, help="Also write out/predictability.png."),
) -> None:
    """Measure how long a prediction is worth holding, per vessel class."""
    resolved = config.load(scenario)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = statics.build(resolved, device)

    typer.echo(f"scenario {resolved.scenario}  device {device}  gate {resolved.fleet.gate_m:.0f} m")
    typer.echo(f"{'class':10s} {'rule':18s} {'error at 1 h':>13s} {'useful horizon':>15s}")
    horizons = prediction.measure_horizons(
        resolved, fixed, worlds=worlds, horizon_s=hours * 3600.0, samples=36
    )
    for item in horizons:
        span = (
            f"> {hours:.0f} h"
            if item.horizon_s == float("inf")
            else f"{item.horizon_s / 3600:.2f} h"
        )
        typer.echo(f"{item.name:10s} {item.rule:18s} {item.error_at_hour_m:11.0f} m {span:>15s}")

    lane = {item.name: item for item in horizons if item.rule == "rail"}
    straight = {item.name: item for item in horizons if item.rule == "constant_velocity"}
    for name, rail_case in lane.items():
        gain = straight[name].error_at_hour_m / max(rail_case.error_at_hour_m, 1.0)
        typer.echo(f"\n  knowing the lane is worth {gain:.0f}x on a {name} at one hour")

    if figure:
        from baltic_isr.render import figures

        typer.echo(f"  wrote {figures.predictability(resolved, fixed, worlds=worlds)}")


@app.command()
def replay(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    trace_path: str = typer.Option("out/lawnmower.npz", help="Trace to read."),
    seconds: float = typer.Option(60.0, help="Length of the film."),
    record_first: bool = typer.Option(False, help="Record a fresh trace before drawing."),
) -> None:
    """Draw a recorded episode as a map, and write it to out/."""
    import numpy as np

    from baltic_isr.baselines.lawnmower import Lawnmower
    from baltic_isr.learn.env import Environment
    from baltic_isr.render import trace as trace_module
    from baltic_isr.render.replay import Replay

    resolved = config.load(scenario)
    destination = Path(trace_path)
    if record_first or not destination.is_file():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        environment = Environment(resolved, 1, device)
        recorded = trace_module.record(environment, Lawnmower(resolved, environment.statics))
        recorded.save(destination)
        typer.echo(f"  recorded {recorded.frames} frames to {destination}")

    arrays = dict(np.load(destination))
    drawing = Replay(resolved, arrays)
    every = max(1, round(drawing.frames / (seconds * 24.0)))
    typer.echo(f"  {drawing.frames} frames, keeping one in {every}, at 24 fps")
    typer.echo(f"  wrote {drawing.animate(every=every)}")


@app.command()
def train(
    scenario: str = typer.Option("baseline", help="Scenario to resolve."),
    worlds: int = typer.Option(128, help="Parallel worlds."),
    updates: int = typer.Option(150, help="Rollout-and-update cycles."),
    rollout: int = typer.Option(32, help="Decisions collected per update."),
    compare_only: bool = typer.Option(False, help="Score a saved policy without training."),
) -> None:
    """Train the decentralised policy, then score it against both floors."""
    from baltic_isr.learn import ppo
    from baltic_isr.learn import train as training

    resolved = config.load(scenario)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = training.OUT / "decentralised.pt"

    if compare_only:
        if not saved.is_file():
            raise typer.BadParameter(f"nothing saved at {saved}")
        policy, _ = training.load(saved, device)
    else:
        policy, log = training.train(
            resolved,
            worlds=worlds,
            updates=updates,
            device=device,
            settings=ppo.Settings(rollout=rollout),
        )
        typer.echo(f"  {log[-1]['transitions'] / 1e6:.2f} M transitions")
        typer.echo(f"  wrote {training.save(policy, log)}")

    typer.echo("")
    typer.echo(training.table(training.compare(resolved, policy, device=device)))
