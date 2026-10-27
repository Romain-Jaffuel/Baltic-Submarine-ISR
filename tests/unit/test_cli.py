"""The budget command, which regenerates the figures docs/constants.md quotes."""

from typer.testing import CliRunner

from baltic_isr.cli import app

runner = CliRunner()


def test_budget_runs_and_reports_the_calibration() -> None:
    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0
    assert "calibration" in result.stdout
    assert "23.96 h at 3 kt" in result.stdout


def test_budget_shows_the_sprint_cannot_return() -> None:
    result = runner.invoke(app, ["budget"])
    assert "IMPOSSIBLE" in result.stdout


def test_budget_accepts_a_scenario() -> None:
    result = runner.invoke(app, ["budget", "--scenario", "quiet"])
    assert result.exit_code == 0
    assert "quiet" in result.stdout


def test_budget_rejects_an_unknown_target() -> None:
    result = runner.invoke(app, ["budget", "--target", "submarine"])
    assert result.exit_code != 0


def test_scenarios_lists_all_five() -> None:
    result = runner.invoke(app, ["scenarios"])
    assert result.exit_code == 0
    for name in ("baseline", "dense", "quiet", "noisy", "far_base"):
        assert name in result.stdout
