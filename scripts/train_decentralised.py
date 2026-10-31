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
from baltic_isr.learn.env import OBSERVATION

# 400, and the number is now measured rather than guessed. It went 150 -> 250
# when the escort proved to be a rare state -- on-station pairs are 1.6 % of
# experience -- and 250 was not enough either.
#
# What says so is not the reward curve alone, though that was still climbing at
# the end (0.684 at update 240). It is that the reward was independently
# verified correct first: a vehicle standing in a suspect's radius with no
# claim earns -1.051 a decision against 0.881 for going back on patrol, a
# penalty of 1.93 measured term by term, and 3.8 % of deployed decisions still
# did it. When the exam is confirmed right and the answer is still wrong, the
# remedy is gradient steps.
#
# That distinction cost this project four training runs in the other direction
# -- three fixes aimed at the policy when the defect was in the reward. The
# rule that came out of it: show that a conforming trajectory earns more BEFORE
# training, and only then spend a run.
#
# 800, because the deviation rate went back to a credible 6 an episode from the
# 24 it had been raised to for exactly this signal. A quarter of the deviations
# is a quarter of the escort experience per update, and the cost of realism in
# the scenario is paid here rather than by pretending it is free: 400 updates
# at 6 would be 2 400 deviation-episodes of escort experience against the 6 000
# that were already found insufficient once.
#
# It does not buy the whole difference back -- matching 400 updates at 24 would
# take 1 600 -- and that is deliberate. The reward is a great deal more
# learnable than it was when 250 proved too few: the credit has a gradient at
# every range now instead of going silent at the rim, the claim is stable, and
# the shaping reaches 12 km. So this is the budget that tests whether the
# earlier shortfall was the rate or the reward. If the escort comes out thin,
# the rate is the answer and the number goes up.
WORLDS = 32
# A cap of 400 updates and a nine-minute ceiling, whichever comes first.
# That is a different kind of number from the ones above: it is a budget,
# not an estimate of what the problem needs. The run below no longer
# searches for the escort behaviour, it adjusts
# one that already works -- see START.
#
# What was cut to get here, said plainly rather than hidden in the arithmetic:
# 128 worlds became 32, so each gradient is estimated on 51 200 transitions
# instead of 201 600. That is a noisier gradient. It is the right thing to cut
# because this file's own docstring is right -- updates matter more here than
# batch width -- and because a warm start needs far fewer of both.
UPDATES = 400
BUDGET_S = 540.0

# The adjustment, rather than the search. `lot10u` already holds station at
# 0.31 km, shadows for a median 195 min and never exceeds the paid cap of
# five; what it has never seen is a fleet of 50 instead of 63, or a credible
# six deviations an episode instead of twenty-four. Those are changes to move
# a working policy through, not reasons to find it again from noise.
#
# This is legitimate because the network is per-vehicle: its observation width
# does not contain the fleet count, so the weights transfer across fleet sizes
# unchanged.
# `lot10w` rather than `lot10u`: it is the same behaviour already moved onto a
# fleet of 50 at six deviations an episode, so this run has only the spacing
# charge to absorb rather than the fleet size as well.
START = "out/models/lot10w.pt"

if __name__ == "__main__":
    for folder in ("data", "traces", "models", "figures", "videos"):
        Path(f"out/{folder}").mkdir(parents=True, exist_ok=True)
    resolved = config.load("baseline")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = ppo.Settings(rollout=32)

    print(f"training on {device}, {WORLDS} worlds, {UPDATES} updates")
    print(f"  {WORLDS * resolved.fleet.count * settings.rollout:,} transitions per update")
    started = time.perf_counter()
    started_from, _ = train.load(START, device=device, width=OBSERVATION)
    print(f"  continuing from {START}")
    policy, log = train.train(
        resolved,
        worlds=WORLDS,
        updates=UPDATES,
        device=device,
        settings=settings,
        log_every=10,
        start=started_from,
        budget_s=BUDGET_S,
    )
    print(f"\ntrained in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  {log[-1]['transitions'] / 1e6:.2f} M transitions")
    print(f"  wrote {train.save(policy, log)}")

    print("\nevaluated on unseen seeds, acting greedily\n")
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
