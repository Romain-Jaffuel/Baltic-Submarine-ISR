"""The two demonstration films, end to end: record, render, verify.

`demonstration.mp4` is the main one -- the same fleet before and after
training, side by side on one clock, forty-eight hours without a cut.
It carries no word beyond what the map needs to be read; the prose belongs
on the page, where somebody reads it faster than a video delivers it and
where it can be corrected without a re-render.

Self-contained on purpose. An earlier version depended on traces left behind
by a different script that rendered films nobody wanted any more, so deleting
the dead film also deleted the input to the live one. Everything the two films
need is recorded here, from the checkpoint and the config, and skipped if it
is already on disk.

    uv run python scripts/films.py             # record, render, verify
    uv run python scripts/films.py --measure   # re-measure the numbers only
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from baltic_isr import config
from baltic_isr.learn.env import OBSERVATION, Environment
from baltic_isr.learn.policy import Decentralised
from baltic_isr.learn.train import OUT, Greedy, evaluate, load
from baltic_isr.metrics.outcome import UPRIGHT, upright
from baltic_isr.render import trace as trace_module
from baltic_isr.render.replay import Replay
from baltic_isr.render.story import alone_fraction, demonstration, facts

SEEDS = (80_001, 80_002, 80_003, 80_004)
WORLDS = 8
UNTRAINED_SEED = 0
FILM_SEED = 777_001

DATA = Path("out/data/before_after.json")
TRACES = Path("out/traces")

# H.264 Level 5.0. A render that returns zero is not a deliverable: the first
# multi-panel film came out valid at 4416x1952, needed four times the
# macroblocks of a 1080p stream, and no consumer player would open it. The
# exit code said nothing was wrong.
CEILING = 22_080

# What is scored. Reported through `metrics.outcome.upright`, which restates
# these as successes so the page never has to say which way is good.
METRICS = ("deviations_missed", "cable_exposure_h", "picture_accuracy")


def measure() -> dict[str, dict[str, float]]:
    """Score the untrained network and the learned policy on the same seeds.

    Returns
    -------
    dict
        ``{"before": {...}, "after": {...}}``, each a metric name to its mean
        and its standard error.
    """
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(UNTRAINED_SEED)
    fresh = Decentralised(OBSERVATION).to(device)
    learned, _ = load(OUT / "decentralised.pt", device, OBSERVATION)

    out: dict[str, dict[str, float]] = {}
    for label, policy in (("before", fresh), ("after", learned)):
        started = time.perf_counter()
        runs = [evaluate(resolved, policy, None, seed, WORLDS, device, "local") for seed in SEEDS]
        scored: dict[str, float] = {}
        for key in METRICS:
            values = [getattr(run.outcome, key, None) or getattr(run, key, 0.0) for run in runs]
            scored[key] = statistics.fmean(values)
            scored[f"{key}_err"] = (
                statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            )
        out[label] = scored
        print(
            f"  {label:>6}  missed {scored['deviations_missed']:.3f}  "
            f"exposure {scored['cable_exposure_h']:.2f} h  "
            f"picture {scored['picture_accuracy']:.3f}  "
            f"({time.perf_counter() - started:.0f} s)"
        )
    return out


def record() -> tuple[Path, Path]:
    """Record one episode of each policy, on the same scenario and seed.

    The two films compare behaviour, so the traffic has to be identical and
    only the blue may differ. Skipped if both traces are already on disk --
    each is about thirteen megabytes and takes a couple of minutes.

    Returns
    -------
    tuple of Path
        The untrained trace and the learned one.
    """
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wanted = {"untrained": TRACES / "untrained.npz", "learned": TRACES / "learned.npz"}
    if all(path.is_file() for path in wanted.values()):
        print("  traces already recorded")
        return wanted["untrained"], wanted["learned"]

    torch.manual_seed(UNTRAINED_SEED)
    for name, destination in wanted.items():
        if destination.is_file():
            continue
        started = time.perf_counter()
        environment = Environment(resolved, 1, device)
        network = (
            Decentralised(OBSERVATION).to(device)
            if name == "untrained"
            else load(OUT / "decentralised.pt", device, OBSERVATION)[0]
        )
        taken = trace_module.record(environment, Greedy(network, environment), seed=FILM_SEED)
        taken.save(destination)
        regimes = np.asarray(taken.vehicle_regime)
        shares = " / ".join(f"{100 * (regimes == index).mean():.0f}%" for index in range(3))
        print(
            f"  {name:>9}  {taken.frames} frames in "
            f"{time.perf_counter() - started:.0f} s, regimes {shares}"
        )
    return wanted["untrained"], wanted["learned"]


def playable(path: Path) -> tuple[int, int, int]:
    """Return a film's width, height and macroblock count."""
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
    ).stderr
    found = re.search(r"(\d{3,5})x(\d{3,5})", probe)
    if found is None:
        return 0, 0, 0
    width, height = int(found.group(1)), int(found.group(2))
    return width, height, (width // 16) * (height // 16)


def report(scores: dict[str, dict[str, float]]) -> None:
    """Print the before/after table the page quotes, with more always better."""
    was, now = upright(scores["before"]), upright(scores["after"])
    for key, label in UPRIGHT:
        print(f"  {label:<26} {was[key]:7.3f} -> {now[key]:7.3f}   {now[key] / was[key]:5.1f}x")


def page_numbers(drawing: Replay) -> None:
    """Print the rest of what the README states, derived rather than typed.

    Every figure in the README's technical half comes from here. A number
    typed into prose goes stale the first time a constant moves and no test
    catches a stale sentence, so the page is written from this output and
    rechecked against it whenever the model changes.
    """
    numbers = facts(config.load("baseline"))
    print()
    print(f"  fleet {numbers['fleet']:.0f} vehicles")
    for name in ("drift", "cruise", "sprint"):
        hours = numbers[f"{name}_h"]
        endurance = f"{hours / 24.0:.1f} days" if hours > 72.0 else f"{hours:.1f} h"
        print(
            f"  {name:<8} hears {numbers[f'{name}_km']:.2f} km   "
            f"{numbers[f'{name}_w']:6.1f} W   {endurance}"
        )
    print(f"  sprint costs {numbers['deaf_ratio']:.1f}x the hearing of a cruise")
    print(f"  a vehicle has nobody in modem range {alone_fraction(drawing):.0%} of the time")


if __name__ == "__main__":
    for folder in ("data", "traces", "videos"):
        Path(f"out/{folder}").mkdir(parents=True, exist_ok=True)

    if "--measure" in sys.argv or not DATA.is_file():
        print(f"{len(SEEDS)} seeds x {WORLDS} worlds\n")
        DATA.write_text(json.dumps(measure(), indent=2), encoding="utf-8")
        print(f"\n  wrote {DATA}")
        if "--measure" in sys.argv:
            raise SystemExit(0)
    print()
    report(json.loads(DATA.read_text(encoding="utf-8")))

    print()
    untrained, learned = record()

    resolved = config.load("baseline")
    print()
    started = time.perf_counter()
    main = demonstration(
        Replay(resolved, dict(np.load(untrained))), Replay(resolved, dict(np.load(learned)))
    )
    print(f"  wrote {main} in {(time.perf_counter() - started) / 60:.1f} min")

    page_numbers(Replay(resolved, dict(np.load(learned))))

    print()
    for film in (main,):
        width, height, blocks = playable(film)
        verdict = "plays everywhere" if blocks <= CEILING else "LEVEL TOO HIGH FOR MOST PLAYERS"
        print(
            f"  {film.name:22s} {film.stat().st_size / 1e6:6.1f} MB  "
            f"{width}x{height}  {blocks} macroblocks  {verdict}"
        )
