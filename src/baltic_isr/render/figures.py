"""Figures comparing the ways of piloting the fleet.

**The five architectures are an ordered sequence, not a set of alternatives.**
Chance decides nothing; the lawnmower follows a plan; the patrol follows a
plan that knows what it guards; the learned policy decides; the oracle decides
knowing everything. So the colour is a sequential ramp, light to dark, and a
reader sees the order without being told it. Spending five categorical hues
here would also have collided with the map, where grey, amber and red are
reserved for what a vessel is doing.

The ramp is **validated, not chosen by eye**, with the same OKLab delta-E and
Machado colour-blindness simulation that `tests/unit/test_palette.py` runs over
the map palette. Measured: lightness strictly decreasing, adjacent steps 8.6 to
19.0 apart under simulated colour blindness against a target of 8, and the
palest step 11.1 from the chart surface. A second candidate with wider interior
steps was rejected because its palest step sat 7.3 from the surface -- the bar
would have been hard to see for the readers the check exists for.

Identity never rests on colour anyway: every bar is named on the axis and
carries its own value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from baltic_isr.metrics.outcome import UPRIGHT, upright
from baltic_isr.render.palette import Palette, palette

if TYPE_CHECKING:  # matplotlib is imported inside the drawing functions
    from matplotlib.axes import Axes

OUT = Path(__file__).resolve().parents[3] / "out" / "figures"

# Light to dark. Validated; see the module docstring.
SEQUENCE = ("#c3d9f7", "#86b6ef", "#5494e3", "#2a78d6", "#104281")

ORDER = ("random", "sweep", "patrol", "learned", "full_information")

# Short, plain, and one thing each. An earlier set read "chance - no decision
# at all" and "oracle, sees everything": a name, a comma, and a gloss, which
# is not how anybody labels a chart. "Oracle" in particular is a word from
# the reinforcement-learning literature that says nothing to a reader who
# does not already have it; "perfect information" is the standard term and
# means what it says.
LABELS = {
    "random": "Random",
    "sweep": "Fixed sweep",
    "patrol": "Cable patrol",
    "learned": "Learned policy",
    "full_information": "Perfect information",
}

# The row the project is about. Drawn bold, because a chart with five rows
# and no emphasis makes the reader find the answer for themselves.
HIGHLIGHT = "learned"


def ramp(count: int) -> list[str]:
    """Return ``count`` steps of the validated sequence, light to dark.

    Interpolated rather than sliced, because slicing a five-step ramp for six
    bars silently drops a step and hands the last bar the palest colour --
    which is exactly backwards when the colour is carrying the order.

    Parameters
    ----------
    count : int
        How many steps are wanted.

    Returns
    -------
    list of str
        Hex colours, light first.
    """
    from matplotlib.colors import to_hex, to_rgb

    if count <= 1:
        return [SEQUENCE[-1]]
    anchors = np.array([to_rgb(step) for step in SEQUENCE])
    wanted = np.linspace(0.0, len(SEQUENCE) - 1.0, count)
    return [
        to_hex(
            anchors[int(np.floor(spot))]
            + (spot - np.floor(spot))
            * (
                anchors[min(int(np.floor(spot)) + 1, len(SEQUENCE) - 1)]
                - anchors[int(np.floor(spot))]
            )
        )
        for spot in wanted
    ]


def _number(value: float) -> str:
    """Format a value for a direct label, at the precision a reader wants.

    A general format string cannot do this: ``:.3g`` turns 2350 into
    ``2.35e+03``, which is three characters shorter and unreadable. The
    precision a number deserves depends on its size, so it is chosen here.
    """
    if abs(value) >= 1000.0:
        # A narrow no-break space is the French thousands separator; written
        # as an escape because the literal character is invisible in a diff.
        return f"{value:,.0f}".replace(",", "\N{NARROW NO-BREAK SPACE}")
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _frame(axes: Axes, colours: Palette) -> None:
    """Strip an axes down to the data.

    Grid and spines recede to a hairline; the marks are the only thing with
    weight. Four spines around a bar chart are four lines nobody reads.
    """
    axes.set_facecolor(colours.surface)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(colours.hairline)
    axes.tick_params(colors=colours.ink_soft, length=0, labelsize=9)
    axes.xaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.set_axisbelow(True)


def architectures(
    rows: list[dict[str, Any]],
    path: Path | None = None,
    mode: str = "light",
) -> Path:
    """Draw the comparison as one grouped bar chart: three measures, one scale.

    It was five panels once, each captioned "lower is better" or "higher is
    better" because the measurements disagreed about which way was good. Five
    charts and five directions is not a comparison. `metrics.outcome.upright`
    restates three of them as percentages of a success, which puts them on
    one axis and lets them share a chart -- so there is one scale, one legend,
    and nothing to caption.

    Parameters
    ----------
    rows : list of dict
        One record per architecture, as `scripts/architectures.py` writes.
    path : Path, optional
        Destination. Defaults to ``out/architectures.png``.
    mode : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    Path
        Where it was written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = palette(mode)
    scored = {str(row["name"]): upright(row) for row in rows}
    present = [name for name in ORDER if name in scored]
    # Colour now says which measure, not which architecture, so the three
    # steps are taken from the ends and middle of the same ramp.
    shades = (SEQUENCE[1], SEQUENCE[3], SEQUENCE[4])

    figure, axes = plt.subplots(figsize=(8.6, 4.3), constrained_layout=True)
    figure.patch.set_facecolor(colours.surface)
    axes.set_facecolor(colours.surface)

    spots = np.arange(len(present))
    height = 0.26
    for index, ((key, label), shade) in enumerate(zip(UPRIGHT, shades, strict=True)):
        offset = (index - 1) * height
        values = [scored[name][key] for name in present]
        axes.barh(spots + offset, values, height=height, color=shade, label=label, zorder=3)
        for spot, value in zip(spots, values, strict=True):
            axes.text(
                value + 1.4,
                spot + offset,
                f"{value:.0f}",
                va="center",
                ha="left",
                color=colours.ink_soft,
                fontsize=8.5,
                zorder=4,
            )

    axes.set_yticks(spots)
    axes.set_yticklabels([LABELS[name] for name in present], fontsize=9.5)
    for tick, name in zip(axes.get_yticklabels(), present, strict=True):
        if name == HIGHLIGHT:
            tick.set_fontweight("bold")
            tick.set_color(colours.ink)
    axes.invert_yaxis()
    axes.set_xlim(0.0, 100.0)
    axes.set_xticks([0, 25, 50, 75, 100])
    axes.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9)
    axes.tick_params(colors=colours.ink, length=0)
    axes.xaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_title(
        "How well each way of piloting the fleet works",
        color=colours.ink,
        fontsize=12,
        loc="left",
        pad=12,
    )
    legend = axes.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncols=3,
        frameon=False,
        fontsize=9.5,
    )
    for text in legend.get_texts():
        text.set_color(colours.ink_soft)

    destination = path or (OUT / "architectures.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100, facecolor=colours.surface)
    plt.close(figure)
    return destination


def predictability(
    config: Any, statics: Any, worlds: int = 16, path: Path | None = None, mode: str = "light"
) -> Path:
    """Draw how fast each vessel class's prediction goes wrong.

    The lot 4 figure. `cli.py` has been calling `figures.predictability` since
    that lot against a module that did not exist, so `baltic-isr predictability
    --figure` has never once worked; creating this file is what surfaced it.

    One line per class, position error against elapsed time, with the
    staleness gate drawn across. Where a line crosses the gate is the useful
    prediction horizon, which is the whole point: a lane-bound cargo and a
    free-running trawler cross it hours apart.

    Parameters
    ----------
    config : Config
        Resolved configuration.
    statics : Statics
        Device geometry.
    worlds : int
        How many worlds to average over.
    path : Path, optional
        Destination. Defaults to ``out/predictability.png``.
    mode : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    Path
        Where it was written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from baltic_isr.picture import prediction

    colours = palette(mode)
    horizons = prediction.measure_horizons(config, statics, worlds=worlds, samples=36)
    gate = float(config.fleet.gate_m)

    # Bars, not lines. Each class's error grows linearly from the origin, so
    # five lines are five slopes -- five numbers -- and a line chart of them
    # crowds every fast class into the left corner while the axis runs on
    # empty for the slow one. The quantity a reader wants is "how long until
    # this prediction is worthless", which is one magnitude per category.
    ordered = sorted(horizons, key=lambda item: item.horizon_s)
    minutes = [item.horizon_s / 60.0 for item in ordered]
    names = [f"{item.name} ({item.rule.replace('_', ' ')})" for item in ordered]
    shades = ramp(len(ordered))

    figure, axes = plt.subplots(figsize=(8.0, 0.62 * len(ordered) + 1.9), constrained_layout=True)
    figure.patch.set_facecolor(colours.surface)

    spots = np.arange(len(ordered))
    axes.barh(
        spots,
        minutes,
        height=0.62,
        color=shades,
        edgecolor=colours.surface,
        linewidth=2.0,
        zorder=3,
    )
    # The number that decides whether any of this is actionable.
    decision = float(config.simulation.decision_interval_s) / 60.0
    widest = max(minutes) if minutes else 1.0
    for spot, value in zip(spots, minutes, strict=True):
        # A bar shorter than the reference line would have its label sitting
        # on top of the line; push those past it instead.
        anchor = max(value, decision) + widest * 0.02
        axes.text(
            anchor,
            float(spot),
            f"{value:.0f} min" if value < 120 else f"{value / 60:.1f} h",
            va="center",
            ha="left",
            color=colours.ink_soft,
            fontsize=9,
        )

    axes.axvline(decision, color=colours.threat, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    axes.text(
        decision + widest * 0.012,
        -0.72,
        f"one decision, {decision:.0f} min",
        color=colours.threat,
        fontsize=9,
        va="center",
    )

    axes.set_yticks(spots)
    axes.set_yticklabels(names, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(0.0, widest * 1.18)
    axes.set_xlabel(
        f"useful horizon: out to the {gate / 1000:.0f} km gate (min)", color=colours.ink_soft
    )
    axes.set_title(
        "knowing the lane, or not knowing it",
        color=colours.ink,
        fontsize=12,
        loc="left",
        pad=30,
    )
    _frame(axes, colours)

    destination = path or (OUT / "predictability.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100, facecolor=colours.surface)
    plt.close(figure)
    return destination


# --- the six figures that walk through the project's choices -----------------
#
# Each title names the quantities plotted and the conditions they were taken
# under, and stops there. A title that announces the conclusion -- "a drone
# that hurries goes deaf" -- is an editorial sitting where a caption belongs:
# it tells the reader what to think before they have looked, and it cannot be
# checked against the figure. The reading is the reader's.


def _page(
    title: str, subtitle: str, colours: Palette, size: tuple[float, float] = (11.0, 5.4)
) -> Any:
    """Open a figure with the title block these six share."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=size, dpi=110, facecolor=colours.surface)
    figure.text(0.035, 0.945, title, color=colours.ink, fontsize=15.5, va="top", ha="left")
    figure.text(0.035, 0.878, subtitle, color=colours.ink_soft, fontsize=9.5, va="top", ha="left")
    return figure


def _save(figure: Any, colours: Palette, name: str, path: Path | None) -> Path:
    import matplotlib.pyplot as plt

    destination = path or (OUT / name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110, facecolor=colours.surface)
    plt.close(figure)
    return destination


def detection_range_against_speed(
    speeds: np.ndarray,
    hearing_km: np.ndarray,
    draw_w: np.ndarray,
    marks: dict[str, float],
    path: Path | None = None,
    mode: str = "light",
) -> Path:
    """Draw two panels on one speed axis: what hurrying costs a drone."""
    colours = palette(mode)
    figure = _page(
        "Detection range and power draw, against speed",
        "Passive sonar against a cargo contact, baseline scenario. Markers at the three "
        "operating regimes.",
        colours,
    )
    left = figure.add_axes((0.065, 0.16, 0.38, 0.60))
    right = figure.add_axes((0.575, 0.16, 0.38, 0.60))

    for axes, values, label in (
        (left, hearing_km, "Detection range, km"),
        (right, draw_w, "Power draw, W"),
    ):
        _frame(axes, colours)
        axes.yaxis.grid(True, color=colours.hairline, linewidth=0.8)
        axes.xaxis.grid(False)
        axes.plot(speeds, values, color=colours.regime[1], linewidth=2.0)
        axes.set_xlabel("Speed through the water, m/s", color=colours.ink_soft, fontsize=9)
        axes.set_title(label, color=colours.ink, fontsize=10, loc="left", pad=8)

    # Each label is pushed away from whichever edge it is near, so none of
    # them lands on a panel title or runs off the plot.
    for name, speed in marks.items():
        for axes, values in ((left, hearing_km), (right, draw_w)):
            value = float(np.interp(speed, speeds, values))
            axes.plot([speed], [value], "o", color=colours.regime[2], markersize=6, zorder=5)
            low, high = axes.get_ylim()
            near_top = value > low + 0.80 * (high - low)
            near_right = speed > speeds[-1] * 0.80
            axes.annotate(
                f"{name.capitalize()}\n{_number(value)}",
                xy=(speed, value),
                xytext=(-8 if near_right else 8, -22 if near_top else 10),
                textcoords="offset points",
                color=colours.ink,
                fontsize=8.5,
                linespacing=1.35,
                ha="right" if near_right else "left",
                va="top" if near_top else "bottom",
            )
    return _save(figure, colours, "detection-range-against-speed.png", path)


def distance_to_nearest_drone(
    gaps_km: np.ndarray, modem_km: float, path: Path | None = None, mode: str = "light"
) -> Path:
    """Draw how far the nearest neighbour is, against the modem's reach."""
    colours = palette(mode)
    alone = float((gaps_km > modem_km).mean())
    figure = _page(
        "Distance to the nearest other drone",
        "All drones at sea, sampled over one 48-hour episode. The acoustic modem reaches "
        f"{modem_km:.1f} km; it also fails with relative speed, which this does not count.",
        colours,
        size=(11.0, 4.8),
    )
    axes = figure.add_axes((0.065, 0.18, 0.90, 0.58))
    _frame(axes, colours)
    axes.yaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.xaxis.grid(False)

    edges = np.linspace(0.0, float(np.percentile(gaps_km, 99)), 46)
    counts, edges = np.histogram(gaps_km, bins=edges)
    share = counts / counts.sum()
    middles = 0.5 * (edges[:-1] + edges[1:])
    within = middles <= modem_km
    axes.bar(
        middles[within],
        share[within],
        width=np.diff(edges)[0] * 0.88,
        color=colours.regime[1],
        linewidth=0,
    )
    axes.bar(
        middles[~within],
        share[~within],
        width=np.diff(edges)[0] * 0.88,
        color=colours.hairline,
        linewidth=0,
    )
    axes.axvline(modem_km, color=colours.ink_soft, linewidth=1.2)
    axes.annotate(
        "Modem range",
        xy=(modem_km, share.max() * 0.92),
        xytext=(9, 0),
        textcoords="offset points",
        color=colours.ink,
        fontsize=9,
        va="center",
    )
    axes.annotate(
        f"Beyond modem range\n{alone:.0%} of samples",
        xy=(middles[~within].mean(), share.max() * 0.55),
        color=colours.ink_soft,
        fontsize=9,
        ha="center",
        linespacing=1.35,
    )
    axes.set_xlabel("Distance to the nearest other drone, km", color=colours.ink_soft, fontsize=9)
    axes.set_yticks([])
    return _save(figure, colours, "distance-to-nearest-drone.png", path)


def gulf_within_detection_range(
    hours: np.ndarray,
    share: np.ndarray,
    at_sea: np.ndarray,
    count: int,
    path: Path | None = None,
    mode: str = "light",
) -> Path:
    """Draw the share of the gulf within detection range, hour by hour.

    Measured as the union of the detection discs over the water, not as their
    sum: forty discs of six kilometres add up to more than the gulf, and what
    a fleet covers is where they do not overlap.
    """
    colours = palette(mode)
    typical = float(np.median(share))
    figure = _page(
        "Share of the gulf within detection range, over an episode",
        f"Union of the detection discs over water, on a 1 km grid. {count} drones, of which "
        f"{at_sea.mean():.0f} at sea at a time; overlap counted once.",
        colours,
        size=(11.0, 4.8),
    )
    axes = figure.add_axes((0.070, 0.18, 0.895, 0.58))
    _frame(axes, colours)
    axes.yaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.xaxis.grid(False)
    axes.fill_between(hours, 0.0, share, color=colours.regime[0], alpha=0.45, linewidth=0)
    axes.plot(hours, share, color=colours.regime[1], linewidth=2.0)
    axes.axhline(typical, color=colours.ink_soft, linewidth=1.0)
    axes.annotate(
        f"{typical:.0%}",
        xy=(hours[-1], typical),
        xytext=(-4, 7),
        textcoords="offset points",
        color=colours.ink,
        fontsize=10,
        ha="right",
    )
    axes.set_xlabel("Hours into the episode", color=colours.ink_soft, fontsize=9)
    axes.set_ylabel("Gulf within detection range", color=colours.ink_soft, fontsize=9)
    axes.set_ylim(0.0, max(float(share.max()) * 1.25, 0.05))
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return _save(figure, colours, "gulf-within-detection-range.png", path)


def drawn_against_surveyed(
    drawn: dict[str, np.ndarray],
    surveyed: tuple[np.ndarray, ...],
    base_then: np.ndarray,
    base_now: np.ndarray,
    extent: np.ndarray,
    path: Path | None = None,
    mode: str = "light",
) -> Path:
    """Draw the coast as it was drawn, over the coast as it was surveyed."""
    from matplotlib.collections import LineCollection

    colours = palette(mode)
    moved = float(np.hypot(*(base_now - base_then))) / 1000.0
    width, height = float(extent[0]), float(extent[1])
    figure = _page(
        "Drawn coastline against surveyed coastline",
        "Five-vertex polylines over the EMODnet terrain model, with the Tallinn base "
        f"position under each. The two base positions lie {moved:.1f} km apart.",
        colours,
        size=(11.0, 11.0 * height / width * 0.93 + 1.1),
    )
    axes = figure.add_axes((0.035, 0.045, 0.93, 0.80))
    axes.set_facecolor(colours.water[-1])
    axes.set_xlim(0.0, width)
    axes.set_ylim(0.0, height)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for side in axes.spines.values():
        side.set_color(colours.hairline)

    # The survey, filled, so it reads as land rather than as scribble.
    for line in surveyed:
        if len(line) > 40:
            axes.fill(line[:, 0], line[:, 1], color=colours.land, linewidth=0, zorder=2)
    axes.add_collection(
        LineCollection(
            [line for line in surveyed if len(line) > 40],
            colors=colours.ink_soft,
            linewidths=0.6,
            zorder=3,
        )
    )
    for line in drawn.values():
        axes.plot(line[:, 0], line[:, 1], color=colours.threat, linewidth=2.2, zorder=5)

    axes.plot(
        [base_then[0], base_now[0]],
        [base_then[1], base_now[1]],
        color=colours.threat,
        linewidth=1.0,
        zorder=6,
    )
    axes.plot([base_then[0]], [base_then[1]], "o", color=colours.threat, markersize=7, zorder=7)
    axes.plot([base_now[0]], [base_now[1]], "o", color=colours.ink, markersize=7, zorder=7)
    axes.annotate(
        f"Tallinn base, {moved:.1f} km apart",
        xy=(float(base_then[0]), float(base_then[1])),
        xytext=(14, -18),
        textcoords="offset points",
        color=colours.ink,
        fontsize=9.5,
        zorder=7,
    )

    # A key in the corner, never on the lines it names.
    for row, (label, colour) in enumerate(
        (("Drawn coastline", colours.threat), ("Surveyed coastline", colours.ink_soft))
    ):
        y = height * (0.955 - row * 0.055)
        axes.plot([width * 0.035, width * 0.075], [y, y], color=colour, linewidth=2.2, zorder=8)
        axes.annotate(
            label,
            xy=(width * 0.085, y),
            color=colours.ink,
            fontsize=9.5,
            va="center",
            zorder=8,
        )
    return _save(figure, colours, "drawn-against-surveyed.png", path)


def regime_share(shares: np.ndarray, path: Path | None = None, mode: str = "light") -> Path:
    """Draw the share of time in each regime, before and after training."""
    colours = palette(mode)
    figure = _page(
        "Share of time in each regime, before and after training",
        "Drones at sea only. One 48-hour episode per condition, same scenario and seed.",
        colours,
        size=(10.0, 4.6),
    )
    axes = figure.add_axes((0.090, 0.17, 0.87, 0.58))
    _frame(axes, colours)
    axes.xaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.yaxis.grid(False)

    names = ("Drift", "Cruise", "Sprint")
    spots = np.arange(len(names))
    height = 0.34
    for row, (label, colour) in enumerate(
        (("Before training", colours.hairline), ("After training", colours.regime[1]))
    ):
        axes.barh(
            spots + (height / 2 if row else -height / 2),
            shares[row],
            height=height * 0.92,
            color=colour,
            linewidth=0,
            label=label,
        )
        for spot, value in zip(spots, shares[row], strict=True):
            axes.annotate(
                f"{value:.0%}",
                xy=(value, spot + (height / 2 if row else -height / 2)),
                xytext=(6, 0),
                textcoords="offset points",
                color=colours.ink_soft,
                fontsize=9,
                va="center",
            )
    axes.set_yticks(spots, names, color=colours.ink)
    axes.invert_yaxis()
    axes.set_xlim(0.0, max(shares.max() * 1.18, 0.1))
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    legend = axes.legend(loc="lower right", frameon=False, fontsize=9.5)
    for entry in legend.get_texts():
        entry.set_color(colours.ink_soft)
    return _save(figure, colours, "regime-share-before-and-after.png", path)


def time_since_patrolled(
    before_h: np.ndarray, after_h: np.ndarray, path: Path | None = None, mode: str = "light"
) -> Path:
    """Draw how long cable stretches go unheard, before and after."""
    colours = palette(mode)
    figure = _page(
        "Time since each cable stretch was last patrolled",
        f"Cumulative share of the cable network. One 48-hour episode per condition; "
        f"median {np.median(before_h):.0f} h before training, {np.median(after_h):.0f} h "
        "after.",
        colours,
        size=(11.0, 4.8),
    )
    axes = figure.add_axes((0.065, 0.18, 0.90, 0.58))
    _frame(axes, colours)
    axes.yaxis.grid(True, color=colours.hairline, linewidth=0.8)
    axes.xaxis.grid(False)

    # Cumulative, not a histogram. The raw distribution is one spike at zero
    # with a long flat tail, so the two curves lie on top of each other and
    # the difference -- which is the whole figure -- cannot be seen.
    top = float(np.percentile(np.concatenate([before_h, after_h]), 98))
    grid = np.linspace(0.0, top, 160)
    for values, label, colour in (
        (before_h, "Before training", colours.hairline),
        (after_h, "After training", colours.regime[1]),
    ):
        reached = (values[None, :] <= grid[:, None]).mean(axis=1)
        axes.plot(grid, reached, color=colour, linewidth=2.2, label=label)
    axes.set_xlabel("Hours since last patrolled", color=colours.ink_soft, fontsize=9)
    axes.set_ylabel("Share of the cable network", color=colours.ink_soft, fontsize=9)
    axes.set_ylim(0.0, 1.0)
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    legend = axes.legend(loc="lower right", frameon=False, fontsize=9.5)
    for entry in legend.get_texts():
        entry.set_color(colours.ink_soft)
    return _save(figure, colours, "time-since-cable-last-patrolled.png", path)
