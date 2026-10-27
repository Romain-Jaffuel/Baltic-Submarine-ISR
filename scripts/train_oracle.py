"""Train the global oracle, and measure what decentralisation costs.

The oracle is not a better algorithm, it is the same algorithm told the truth.
Everything downstream of the observation is identical -- network, PPO, reward,
evaluation -- so whatever separates the two policies is the price of seeing
through a sonar instead of through a window, and of a 250-byte acoustic
budget. That number is the only reason to run this at all.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from baltic_isr import config
from baltic_isr.learn import ppo, train
from baltic_isr.learn.env import Environment

WORLDS = 128
UPDATES = 150

if __name__ == "__main__":
    for folder in ("data", "traces", "models", "figures", "videos"):
        Path(f"out/{folder}").mkdir(parents=True, exist_ok=True)
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = ppo.Settings(rollout=32)
    width = Environment(resolved, 1, device, sees="oracle").observation_width

    print(f"oracle sur {device}, {WORLDS} worlds, {UPDATES} mises a jour")
    print(f"  observation {width} contre 213 pour le decentralise")
    started = time.perf_counter()
    policy, log = train.train(
        resolved,
        worlds=WORLDS,
        updates=UPDATES,
        device=device,
        settings=settings,
        log_every=10,
        sees="oracle",
    )
    print(f"\nentraine en {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  wrote {train.save(policy, log, name='oracle')}")

    print("\nevaluation sur des seeds jamais vues, politique gloutonne\n")
    reports = train.compare(resolved, policy, device=device, sees="oracle")
    print(train.table(reports))

    Path("out/data/oracle.json").write_text(
        json.dumps(
            [
                {
                    "name": item.name,
                    "cable_exposure_h": item.cable_exposure_h,
                    "deviations_missed": item.outcome.deviations_missed,
                    "picture_accuracy": item.outcome.picture_accuracy,
                    "false_alarm_rate": item.outcome.false_alarm_rate,
                    "drift_share": item.drift_share,
                    "reward": item.reward,
                }
                for item in reports
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
