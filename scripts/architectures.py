"""Compare every way of piloting the fleet, against chance.

Five architectures, same seeds, same scenario, same metrics. They form an
**ordered** sequence rather than a set of alternatives -- chance decides
nothing, the lawnmower follows a plan, the patrol follows a plan that knows
what it guards, the learned policy decides, and the oracle decides knowing
everything -- so the figures use a sequential ramp, not categorical hues. A
reader should see the order without being told it.

Writes `out/architectures.json`, then the figures.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from baltic_isr import config
from baltic_isr.baselines.cable_patrol import CablePatrol
from baltic_isr.baselines.chance import Chance
from baltic_isr.baselines.lawnmower import Lawnmower
from baltic_isr.learn.env import OBSERVATION, Environment
from baltic_isr.learn.train import OUT, evaluate, load
from baltic_isr.render import figures

SEEDS = (80_001, 80_002, 80_003, 80_004)
WORLDS = 8

# Everything measured, in its raw sense. The figure does not draw these: it
# draws `metrics.outcome.upright`, which restates three of them so a longer
# bar is better in all of them. The raw five stay in the JSON, because a
# table can carry a unit and a direction in words where a chart cannot.
METRICS = (
    "deviations_missed",
    "cable_exposure_h",
    "picture_accuracy",
    "false_alarm_rate",
    "verification_cost",
)


def collect() -> list[dict[str, object]]:
    """Evaluate every architecture on the same seeds."""
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle_width = Environment(resolved, 1, device, sees="oracle").observation_width

    learned, _ = load(OUT / "decentralised.pt", device, OBSERVATION)
    oracle, _ = load(OUT / "oracle.pt", device, oracle_width)

    architectures = (
        ("random", None, Chance, "local"),
        ("sweep", None, Lawnmower, "local"),
        ("patrol", None, CablePatrol, "local"),
        ("learned", learned, None, "local"),
        ("full_information", oracle, None, "oracle"),
    )

    rows: list[dict[str, object]] = []
    for name, policy, scripted, sees in architectures:
        started = time.perf_counter()
        runs = [evaluate(resolved, policy, scripted, seed, WORLDS, device, sees) for seed in SEEDS]
        row: dict[str, object] = {"name": name}
        for key in METRICS:
            values = [getattr(run.outcome, key, None) or getattr(run, key, 0.0) for run in runs]
            row[key] = statistics.fmean(values)
            row[f"{key}_err"] = (
                statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            )
        row["drift_share"] = statistics.fmean([run.drift_share for run in runs])
        row["sprint_share"] = statistics.fmean([run.sprint_share for run in runs])
        rows.append(row)
        print(
            f"  {name:>11}  missed {row['deviations_missed']:.3f}  "
            f"exposure {row['cable_exposure_h']:.2f} h  "
            f"({time.perf_counter() - started:.0f} s)"
        )
    return rows


if __name__ == "__main__":
    for folder in ("data", "traces", "models", "figures", "videos"):
        Path(f"out/{folder}").mkdir(parents=True, exist_ok=True)
    print(f"{len(SEEDS)} seeds x {WORLDS} worlds\n")
    rows = collect()
    Path("out/data/architectures.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("  wrote out/data/architectures.json")
    drawn = figures.architectures(rows, Path("out/figures/architectures.png"))
    print(f"  wrote {drawn}")
