"""The suite's own guard rails: no test may wait on something that never comes.

This file exists because of one line. ``test_transit_out_costs_the_predicted_share``
waited for the fleet's MEAN distance to base to reach 39 km::

    while float(fleet.distance_to_base(state).mean()) < 39_000.0:

The mean cannot reach 39 km. The fleet spreads out and cycles through its
sorties, so the mean tops out at 32.2 km and oscillates there -- measured over
167 simulated hours. The test did not fail, it stopped returning, and because
pytest reports nothing until a test finishes, the whole suite simply hung with
a process pinned at one core. Several hours went into believing the machine was
slow.

A failure is a message. A hang is a mystery, and a much more expensive one, so
the two guards here are worth their keep:

* every test runs under a hard timeout, configured in ``pyproject.toml``;
* no test may contain an unbounded loop in the first place.

The second is the one that matters. A timeout turns a hang into a failure after
sixty wasted seconds; forbidding the loop means the mistake cannot be written.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parents[1]


def _sources() -> list[Path]:
    """Return every test module, this one included."""
    return sorted(TESTS.rglob("test_*.py"))


def _is_bounded(node: ast.While) -> bool:
    """Return whether a ``while`` is one that provably stops.

    ``while True`` with a ``break`` is bounded only in the sense that it might
    be; what is accepted here is the narrower, checkable thing -- a loop whose
    condition counts against a bound that the body changes, written as a
    comparison on a name assigned in the function. Rather than attempt that
    analysis, this accepts nothing: a test that needs to repeat something uses
    ``for ... in range(n)``, which cannot run forever, and breaks out early if
    it finishes sooner.
    """
    del node
    return False


def test_no_test_waits_forever() -> None:
    """No test may contain a ``while`` loop.

    ``for index in range(n)`` with a ``break`` says the same thing and carries
    its own ceiling. The replacement for the loop that prompted this reads::

        for _ in range(600):
            if float(reach.max()) >= 39_000.0:
                break
            ...
        else:
            raise AssertionError("no vehicle reached the operational radius")

    which fails with a sentence instead of hanging, and names the thing that
    did not happen.
    """
    offences = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.While) and not _is_bounded(node):
                offences.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not offences, (
        "unbounded loops in the suite: "
        + ", ".join(offences)
        + ". Use `for _ in range(n)` with a `break`, so the test fails with a "
        "message instead of hanging."
    )


def test_every_test_runs_under_a_timeout() -> None:
    """The suite configures a hard per-test ceiling.

    Checked in the file rather than through ``pytest`` itself, because a test
    that asked the running session for its own timeout would pass just as well
    with the plugin absent and the ceiling infinite.
    """
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = settings["tool"]["pytest"]["ini_options"]
    assert "timeout" in options, "pyproject.toml sets no pytest timeout"
    assert 0 < options["timeout"] <= 300, options["timeout"]
    # Windows has no SIGALRM, so the signal method silently does nothing here.
    assert options["timeout_method"] == "thread", options["timeout_method"]
    assert any("pytest-timeout" in package for package in settings["dependency-groups"]["dev"]), (
        "the timeout is configured but the plugin that enforces it is not a dependency"
    )


@pytest.mark.parametrize("path", _sources(), ids=lambda path: path.stem)
def test_every_test_module_parses(path: Path) -> None:
    """A module that cannot be parsed is a module whose tests never ran.

    Collection errors are easy to skim past in a long run, and a test file that
    silently stopped being collected is indistinguishable from one that passes.
    """
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
