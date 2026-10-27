"""Train the decentralised policy and judge it against both floors.

Run from the project root::

    uv run python scripts/train_decentralised.py

More updates matter more here than more parallel worlds. Forty vehicles in one
world already give forty transitions a decision, so widening the batch buys
precision on a gradient the policy only takes a couple of hundred of; what it
needs is to take more of them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from baltic_isr import config
from baltic_isr.learn import ppo, train

WORLDS = 128
UPDATES = 150

if __name__ == "__main__":
    for folder in ("data", "traces", "models", "figures", "videos"):
        Path(f"out/{folder}").mkdir(parents=True, exist_ok=True)
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = ppo.Settings(rollout=32)

    print(f"training on {device}, {WORLDS} worlds, {UPDATES} updates")
    print(f"  {WORLDS * resolved.fleet.count * settings.rollout:,} transitions per update")
    started = time.perf_counter()
    policy, log = train.train(
        resolved, worlds=WORLDS, updates=UPDATES, device=device, settings=settings, log_every=10
    )
    print(f"\nentraine en {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  {log[-1]['transitions'] / 1e6:.2f} M transitions")
    print(f"  wrote {train.save(policy, log)}")

    print("\nevaluation sur des seeds jamais vues, politique gloutonne\n")
    reports = train.compare(resolved, policy, device=device)
    print(train.table(reports))

    Path("out/data/comparison.json").write_text(
        json.dumps(
            [
                {
                    "name": item.name,
                    "cable_exposure_h": item.cable_exposure_h,
                    "deviations_missed": item.outcome.deviations_missed,
                    "picture_accuracy": item.outcome.picture_accuracy,
                    "false_alarm_rate": item.outcome.false_alarm_rate,
                    "drift_share": item.drift_share,
                    "sprint_share": item.sprint_share,
                    "reward": item.reward,
                }
                for item in reports
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
