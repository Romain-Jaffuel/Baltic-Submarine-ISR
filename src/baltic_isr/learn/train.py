"""Train the decentralised policy, and judge it against the two floors.

The judgement matters as much as the training. A learned policy is only
interesting against something, and the something has to be honest: the
lawnmower sweeps the box and the cable patrol already walks the right ground,
so beating the first proves the policy found the cables and beating the second
proves it found *when* and *with whom*, which is the part that needs
coordination.

Evaluation is run greedily and on seeds the policy never trained on. Reporting
the training reward instead would measure how lucky the exploration was.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from baltic_isr.baselines.cable_patrol import CablePatrol
from baltic_isr.baselines.lawnmower import Lawnmower
from baltic_isr.config import Config
from baltic_isr.learn import ppo
from baltic_isr.learn.env import OBSERVATION, Environment
from baltic_isr.learn.policy import Decentralised
from baltic_isr.metrics.outcome import Outcome, Scorer

OUT = Path(__file__).resolve().parents[3] / "out" / "models"


@dataclass(frozen=True)
class Report:
    """What one evaluated policy scored.

    Attributes
    ----------
    name : str
        Which policy.
    outcome : Outcome
        The numbers of specification section 7.
    cable_exposure_h : float
        Mean hours since each cable sample was last heard. This is what the
        map shades, so it is the number the film is judged on.
    drift_share, sprint_share : float
        How much of the fleet's time went to each end of the regime table.
    reward : float
        Mean reward per decision, for comparison with the training curve only.
    """

    name: str
    outcome: Outcome
    cable_exposure_h: float
    drift_share: float
    sprint_share: float
    reward: float


class Greedy:
    """A learned policy dressed as a scripted one, so it can be filmed.

    The renderer drives the episode itself and hands a policy the fleet state;
    a learned policy needs the whole observation, which only the environment
    can assemble. This is the one adapter between the two, and it takes the
    argmax rather than sampling -- a film of a policy rolling dice is a film of
    the dice.

    Parameters
    ----------
    policy : Decentralised
        The trained network.
    environment : Environment
        The environment being driven, for its observation.
    """

    wants_state = True

    def __init__(self, policy: Decentralised, environment: Environment) -> None:
        self._policy = policy
        self._environment = environment

    def act(self, state: object, t: float) -> torch.Tensor:
        """Return the greedy action for every vehicle."""
        del t
        with torch.no_grad():
            logits, _ = self._policy(self._environment.observe(state))  # type: ignore[arg-type]
            chosen: torch.Tensor = logits.argmax(dim=-1)
        return chosen


def evaluate(
    config: Config,
    policy: Decentralised | None,
    scripted: type | None,
    seed: int,
    worlds: int,
    device: str,
    sees: str = "local",
    fleet_block: bool = False,
) -> Report:
    """Run one policy over a full episode and score it.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    policy : Decentralised, optional
        The learned policy, run greedily. ``None`` for a scripted one.
    scripted : type, optional
        A scripted policy class, instead of ``policy``.
    seed : int
        Episode seed.
    worlds : int
        How many worlds to average over.
    device : str
        Where to run.

    Returns
    -------
    Report
        The score.
    """
    environment = Environment(config, worlds, device, sees=sees, oracle_fleet_block=fleet_block)
    scorer = Scorer(config, environment._picture)
    state, observation = environment.reset(seed=seed)
    walker = scripted(config, environment.statics) if scripted is not None else None

    decisions = int(config.simulation.episode_s / config.simulation.decision_interval_s)
    exposure = 0.0
    reward = 0.0
    regimes = torch.zeros(3, device=device)
    for _ in range(decisions):
        if walker is not None:
            actions = walker.act(state.fleet, state.t)
        else:
            assert policy is not None
            with torch.no_grad():
                logits, _ = policy(observation)
                actions = logits.argmax(dim=-1)
        state, result = environment.step(state, actions)
        observation = result.observation
        scorer.observe(state, config.simulation.decision_interval_s)
        # Summed over worlds, not averaged: the scorer divides this by
        # `_held_s`, which also sums over worlds, so the ratio comes out
        # independent of how many worlds the evaluation ran.
        scorer.add_contact_time(float(result.contact_s.sum()))
        exposure += float((state.t - state.cable_heard_s).clamp_min(0.0).mean())
        reward += float(result.reward.mean())
        regimes += torch.bincount(state.fleet.regime.flatten(), minlength=3).float()

    share = regimes / regimes.sum().clamp_min(1.0)
    return Report(
        name=scripted.__name__ if scripted is not None else "appris",
        outcome=scorer.score(state),
        cable_exposure_h=exposure / decisions / 3600.0,
        drift_share=float(share[0]),
        sprint_share=float(share[2]),
        reward=reward / decisions,
    )


def train(
    config: Config,
    *,
    worlds: int = 256,
    updates: int = 200,
    device: str = "cuda",
    settings: ppo.Settings | None = None,
    seed: int = 0,
    log_every: int = 10,
    sees: str = "local",
    fleet_block: bool = False,
) -> tuple[Decentralised, list[dict[str, float]]]:
    """Run PPO and return the policy and the training log.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    worlds : int
        Parallel worlds. Transitions per update are
        ``worlds x vehicles x rollout``.
    updates : int
        How many rollout-and-update cycles.
    device : str
        Where to run.
    settings : Settings, optional
        PPO hyper-parameters.
    seed : int
        Seeds both torch and the first episode.
    log_every : int
        How often to print.

    Returns
    -------
    tuple
        The trained policy and one record per update.
    """
    settings = settings or ppo.Settings()
    torch.manual_seed(seed)
    environment = Environment(config, worlds, device, sees=sees, oracle_fleet_block=fleet_block)
    policy = Decentralised(environment.observation_width).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate, eps=1e-5)

    decisions = int(config.simulation.episode_s / config.simulation.decision_interval_s)
    per_update = worlds * config.fleet.count * settings.rollout
    state, observation = environment.reset(seed=seed)
    log: list[dict[str, float]] = []
    started = time.perf_counter()
    walked = 0

    for step in range(updates):
        state, observation, rollout = ppo.collect(environment, policy, state, observation, settings)
        losses = ppo.update(policy, optimiser, rollout, settings)
        walked += settings.rollout
        # One episode is 192 decisions. Restarting on the boundary keeps every
        # rollout inside one episode, which is what lets the advantage ignore
        # the terminal flag entirely.
        if walked >= decisions - settings.rollout:
            state, observation = environment.reset(seed=seed + step + 1)
            walked = 0

        record = {
            "update": float(step + 1),
            "transitions": float((step + 1) * per_update),
            "reward": float(rollout.reward.mean()),
            "seconds": time.perf_counter() - started,
            **losses,
        }
        log.append(record)
        if (step + 1) % log_every == 0 or step == 0:
            print(
                f"  {step + 1:4d}/{updates}  {record['transitions'] / 1e6:5.2f} M pas  "
                f"recompense {record['reward']:+.3f}  "
                f"kl {record['kl']:+.4f}  entropie {record['entropy']:.3f}  "
                f"{record['seconds']:4.0f} s"
            )

    return policy, log


def compare(
    config: Config,
    policy: Decentralised,
    *,
    seeds: tuple[int, ...] = (10_001, 10_002),
    worlds: int = 16,
    device: str = "cuda",
    sees: str = "local",
    fleet_block: bool = False,
) -> list[Report]:
    """Score the learned policy and both floors on the same unseen seeds."""
    reports: list[Report] = []
    for name, learned, scripted in (
        ("tondeuse", None, Lawnmower),
        ("patrouille", None, CablePatrol),
        ("appris", policy, None),
    ):
        # The floors are scripted, so they always run on the local
        # environment; only the learned policy follows `sees`.
        mode = sees if learned is not None else "local"
        runs = [evaluate(config, learned, scripted, seed, worlds, device, mode) for seed in seeds]
        reports.append(
            Report(
                name=name,
                outcome=runs[0].outcome,
                cable_exposure_h=sum(r.cable_exposure_h for r in runs) / len(runs),
                drift_share=sum(r.drift_share for r in runs) / len(runs),
                sprint_share=sum(r.sprint_share for r in runs) / len(runs),
                reward=sum(r.reward for r in runs) / len(runs),
            )
        )
    return reports


def table(reports: list[Report]) -> str:
    """Render the comparison the way the lot is judged."""
    header = (
        f"{'politique':>12} {'exposition':>12} {'ratees':>8} {'carte':>7} "
        f"{'fausses':>8} {'derive':>7} {'sprint':>7} {'recompense':>11}"
    )
    lines = [header, "-" * len(header)]
    for item in reports:
        lines.append(
            f"{item.name:>12} {item.cable_exposure_h:10.2f} h "
            f"{item.outcome.deviations_missed:8.3f} {item.outcome.picture_accuracy:7.3f} "
            f"{item.outcome.false_alarm_rate:8.2f} {item.drift_share:6.0%} "
            f"{item.sprint_share:6.0%} {item.reward:11.3f}"
        )
    return "\n".join(lines)


def save(policy: Decentralised, log: list[dict[str, float]], name: str = "decentralised") -> Path:
    """Write the weights and the training log to ``out/``."""
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{name}.pt"
    torch.save({"state": policy.state_dict(), "log": log}, destination)
    return destination


def load(
    path: Path, device: str = "cuda", width: int = OBSERVATION
) -> tuple[Decentralised, list[dict[str, float]]]:
    """Read back what :func:`save` wrote."""
    blob = torch.load(path, map_location=device, weights_only=False)
    policy = Decentralised(width).to(device)
    policy.load_state_dict(blob["state"])
    policy.eval()
    return policy, blob["log"]


def _settings_as_dict(settings: ppo.Settings) -> dict[str, float]:
    """Return the hyper-parameters, for the record."""
    return {key: float(value) for key, value in asdict(settings).items()}
