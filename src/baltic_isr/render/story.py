"""The two films: a demonstration of the fleet, not a deck about it.

An earlier version of this module built an explainer -- eleven chapters, a
script column down the left, title cards, a built diagram of the three
regimes. It was a slideshow with a map in the corner, and a slideshow is the
wrong artefact: a reader who wants prose can read prose faster than any video
delivers it, and the one thing a video can do that a page cannot is **show the
thing running**. So the words moved to the page and the films became what they
should have been, which is footage.

What is left in frame is only what the footage needs to be legible: who each
panel is, the clock, two running counts, and a key along the foot naming every
mark. Nothing else.

`demonstration.mp4` is the main one -- the same fleet before and after
training, side by side on one clock, running the whole forty-eight hours
without a cut.
policy, where the hearing discs and the predictions are big enough to read.

The map they draw is also simpler than it was. Amber used to mark a vessel
behaving oddly and red the same vessel over a cable -- a distinction that is
real in the metrics and invisible to anybody watching for the first time. The
two collapsed into one red mark meaning "not behaving", which freed amber to
mean exactly one thing of its own: a cable nobody has listened to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from baltic_isr.config import Config
from baltic_isr.fleet import power
from baltic_isr.render.palette import Palette, palette
from baltic_isr.render.replay import AUV, SHIP, VIDEOS, Replay, _encode
from baltic_isr.sensing import acoustics

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

FPS = 24

# 1920x1080 at 100 dpi: divisible by 16 both ways, so no encoder resamples the
# hairlines, and the size every site expects of a hero video.
CANVAS = (19.2, 10.8)
DPI = 100

# How much of the map's height the single-panel film keeps. The gulf is 138 km
# by 110, but the top and bottom fifths are land, and a full-width map of it
# would be letterboxed into a strip with two beige bands. Trimming to the
# middle takes the aspect from 1.26 to about 1.7, near enough 16:9 to fill the
# frame. It cannot go much below this: the two shore bases sit at 0.16 and
# 0.85 of the height and are the first thing a tighter crop would cut.
KEEP = 0.88

STRIP = (0.012, 0.018, 0.976, 0.105)


@dataclass
class Key:
    """One entry of the map key: how to draw the mark, and what it means."""

    label: str
    kind: str
    colour: str
    size: float = 90.0
    marker: Any = "o"
    extra: dict[str, Any] = field(default_factory=dict)


GROUPS = ("Seabed and shore", "Surface traffic", "Our drones")


def keys(colours: Palette) -> tuple[tuple[Key, str], ...]:
    """Return the key, in reading order, for the films' bottom strip.

    Eight entries, which is every mark the map draws and no more. The list is
    the whole vocabulary: if something appears on screen that is not here, the
    film has asked the viewer to guess.
    """
    return (
        (Key("", "line", colours.asset, extra={"linewidth": 3.4}), "Cable, patrolled recently"),
        (Key("", "line", colours.watch, extra={"linewidth": 3.4}), "Cable, unpatrolled for 6 h"),
        (Key("", "dot", colours.ink_soft, size=55.0, marker="s"), "Shore base"),
        (Key("", "line", colours.lane, extra={"linewidth": 7.0}), "Shipping lane"),
        (Key("", "dot", colours.traffic, size=95.0, marker=SHIP), "Merchant vessel"),
        (Key("", "dot", colours.threat, size=190.0, marker=SHIP), "Suspicious vessel"),
        (Key("", "dot", colours.regime[1], size=150.0, marker=AUV), "Submarine drone"),
        (Key("", "disc", colours.regime[0], size=460.0), "Sonar detection range"),
    )


def facts(config: Config) -> dict[str, float]:
    """Derive the numbers that describe the fleet, from the resolved config.

    Nothing in the films quotes these any more -- the page does. They stay
    here because the page's numbers have the same problem the films' did: a
    figure typed into prose goes stale the first time a constant moves, and no
    test catches a stale sentence. These come from the same two functions the
    simulator and the renderer call.

    Parameters
    ----------
    config : Config
        Resolved configuration.

    Returns
    -------
    dict
        Ranges in kilometres, endurances in hours, power in watts.
    """
    # The **reference** vessel, which is the first traffic class, because this
    # is what the page states and `baltic-isr budget` prints the same numbers
    # from the same function. The hearing discs the films draw use the
    # loudest class instead -- a disc shows the best case a vehicle could
    # manage, a table states the typical one -- and the two differ by about
    # eight per cent. Quoting one and drawing the other without saying so is
    # how a page and a picture come to disagree.
    reference = config.traffic.classes[0].source_level_db
    speeds = (
        config.auv.regimes.drift_ms,
        config.auv.regimes.cruise_ms,
        config.auv.regimes.sprint_ms,
    )
    out: dict[str, float] = {}
    for name, speed in zip(("drift", "cruise", "sprint"), speeds, strict=True):
        out[f"{name}_km"] = (
            acoustics.detection_range_m(speed, reference, config.acoustics, config.environment)
            / 1000.0
        )
        out[f"{name}_h"] = power.endurance_s(speed, config.auv) / 3600.0
        out[f"{name}_w"] = power.power_w(speed, config.auv)
    out["deaf_ratio"] = out["cruise_km"] / out["sprint_km"]
    out["fleet"] = float(config.fleet.count)
    return out


def alone_fraction(drawing: Replay, every: int = 10) -> float:
    """Measure how much of the time a vehicle has nobody within modem range.

    Parameters
    ----------
    drawing : Replay
        The loaded run.
    every : int
        Sample one frame in this many.

    Returns
    -------
    float
        Fraction in ``[0, 1]``.
    """
    count = int(drawing._trace["vehicle_xy"].shape[1])
    alone = 0
    total = 0
    for frame in range(0, drawing.frames, every):
        seen = np.zeros(count, dtype=bool)
        pairs = drawing.links_at(frame)
        if pairs.size:
            seen[pairs[:, 0]] = True
            seen[pairs[:, 1]] = True
        alone += int((~seen).sum())
        total += count
    return alone / max(total, 1)


def _blank(axes: Axes, colours: Palette) -> None:
    """Strip an axes to a plain panel in surface colour."""
    axes.clear()
    axes.set_facecolor(colours.surface)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks([])
    axes.set_yticks([])
    for side in axes.spines.values():
        side.set_visible(False)


def _crop(axes: Axes, extent: np.ndarray, keep: float) -> None:
    """Trim the map's empty land, keeping the middle fraction of its height."""
    height = float(extent[1])
    margin = 0.5 * height * (1.0 - keep)
    axes.set_ylim(margin, height - margin)


def _key_at(axes: Axes, key: Key, label: str, x: float, y: float, colours: Palette) -> None:
    """Draw one key entry at a spot: the mark, then the words."""
    if key.kind == "line":
        axes.plot([x, x + 0.020], [y, y], color=key.colour, solid_capstyle="round", **key.extra)
        mark = x + 0.010
    else:
        mark = x + 0.009
        if key.kind == "ring":
            axes.scatter([mark], [y], s=key.size, facecolors="none", edgecolors=key.colour)
        elif key.kind == "disc":
            axes.scatter([mark], [y], s=key.size, color=key.colour, alpha=0.22, linewidths=0)
            axes.scatter([mark], [y], s=70.0, color=colours.regime[1], linewidths=0)
        else:
            axes.scatter(
                [mark],
                [y],
                s=key.size,
                marker=key.marker,
                color=key.colour,
                linewidths=0.0,
            )
    axes.text(x + 0.028, y, label, color=colours.ink_soft, fontsize=10.5, va="center", ha="left")


def _legend(axes: Axes, colours: Palette) -> None:
    """Draw the key as three named groups, not a row of loose marks.

    A flat strip of eight symbols makes the reader hold eight unrelated facts
    at once. Grouped under what KIND of thing each one is -- the seabed, the
    surface, our own fleet -- it becomes three facts with detail hanging off
    them, which is how anybody reads a chart legend.
    """
    _blank(axes, colours)

    # The bathymetry is published under CC-BY, which requires attribution
    # wherever the work is shown. It used to sit under each map panel, where
    # it read as a caption nobody asked for; once, small, at the foot of the
    # key is the smallest placement that still discharges the licence.
    axes.text(
        0.988,
        0.06,
        "Bathymetry: EMODnet Bathymetry Consortium (CC-BY 4.0)  ·  ETRS89 / TM35FIN (EPSG:3067)",
        color=colours.hairline,
        fontsize=5.5,
        va="bottom",
        ha="right",
    )

    entries = keys(colours)
    grouped = (entries[0:3], entries[3:6], entries[6:8])

    for column, (heading, block) in enumerate(zip(GROUPS, grouped, strict=True)):
        left = 0.012 + column * 0.335
        axes.text(
            left,
            0.88,
            heading,
            color=colours.ink,
            fontsize=8.5,
            fontweight="bold",
            va="top",
            ha="left",
        )
        axes.plot(
            [left, left + 0.305],
            [0.80, 0.80],
            color=colours.hairline,
            linewidth=1.0,
            zorder=1,
        )
        for row, (key, label) in enumerate(block):
            _key_at(axes, key, label, left, 0.60 - row * 0.235, colours)


def _header(figure: Figure, colours: Palette, title: str, standfirst: str | None = None) -> Axes:
    """Draw the standing title block, and return the band it sits on.

    The title names the SUBJECT, not the experiment. "The same fleet before
    and after it learned" described how the film was made to somebody who
    already knew what the fleet was -- and the two panels say before and
    after in their own titles, so it was also saying it twice. Somebody
    meeting this cold needs to be told what they are looking at.

    The standfirst under it carries the conditions a viewer cannot read off
    the map: what the vehicles sense with, how long this runs, and that the
    two sides are the same sea. It is information, not branding; the product
    name that used to sit there was branding, and it went.
    """
    band = figure.add_axes((0.0, 0.905, 1.0, 0.085))
    _blank(band, colours)
    band.text(
        0.012,
        0.66 if standfirst else 0.46,
        title,
        color=colours.ink,
        fontsize=17,
        va="center",
        ha="left",
    )
    if standfirst:
        band.text(
            0.012,
            0.24,
            standfirst,
            color=colours.ink_soft,
            fontsize=9.5,
            va="center",
            ha="left",
        )
    band.plot([0.012, 0.988], [0.02, 0.02], color=colours.hairline, linewidth=1.2)
    return band


def _clock(band: Axes, drawing: Replay, frame: int, colours: Palette) -> list[Any]:
    """Show how far into the episode we are, and return the artists to remove.

    The map has no other way of saying that time is passing: the traffic
    moves, but nothing says whether that is an hour or a day, and the whole
    point of the fleet's trade is what it costs over a sortie. One figure and
    a bar -- the figure for the reading, the bar for the proportion.

    Two counters used to sit beside it, contacts held and open questions.
    They were numbers nobody could act on while watching, and they made the
    header look like a dashboard.
    """
    hours = float(drawing._trace["times_s"][frame]) / 3600.0
    total = float(drawing._trace["times_s"][-1]) / 3600.0
    left, width, y = 0.880, 0.106, 0.30
    drawn: list[Any] = [
        band.text(
            left,
            0.62,
            f"{hours:.0f} h",
            color=colours.ink,
            fontsize=20,
            va="center",
            ha="left",
        ),
        band.text(
            left + width,
            0.62,
            f"of {total:.0f}",
            color=colours.ink_soft,
            fontsize=10,
            va="center",
            ha="right",
        ),
        band.plot(
            [left, left + width],
            [y, y],
            color=colours.hairline,
            linewidth=2.6,
            solid_capstyle="butt",
        )[0],
        band.plot(
            [left, left + width * min(hours / max(total, 1e-6), 1.0)],
            [y, y],
            color=colours.regime[1],
            linewidth=2.6,
            solid_capstyle="butt",
        )[0],
    ]
    return drawn


def _render(figure: Figure, writer: Any) -> None:
    """Push the current figure into the video as one frame."""
    figure.canvas.draw()
    canvas = cast("FigureCanvasAgg", figure.canvas)
    writer.append_data(np.asarray(canvas.buffer_rgba())[:, :, :3])  # type: ignore[no-untyped-call]


def demonstration(
    before: Replay,
    after: Replay,
    path: Path | None = None,
    mode: str = "light",
    every: int = 4,
) -> Path:
    """Render the main film: the same fleet before and after training.

    Both panels share a frame index and run the whole episode without a cut.
    They are different episodes of the same scenario and seed, so the traffic
    is identical; what the viewer compares is the behaviour of the blue.

    Side by side rather than one after the other, because the eye is good at
    "these two differ" and poor at "this differs from something ninety seconds
    ago".

    Parameters
    ----------
    before, after : Replay
        The untrained network and the learned policy.
    path : Path, optional
        Destination. Defaults to ``out/videos/demonstration.mp4``.
    mode : str
        ``"light"`` or ``"dark"``.
    every : int
        Trace frames per film frame.

    Returns
    -------
    Path
        Where it was written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = palette(mode)
    destination = path or (VIDEOS / "demonstration.mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=CANVAS, dpi=DPI, facecolor=colours.surface)
    _legend(figure.add_axes(STRIP), colours)
    band = _header(
        figure,
        colours,
        "40 submarine drones guarding the subsea cables of the Gulf of Finland",
    )
    left = figure.add_axes((0.012, 0.140, 0.480, 0.745))
    right = figure.add_axes((0.508, 0.140, 0.480, 0.745))
    for drawing in (before, after):
        drawing.show_legend = False
        drawing.style = "exposition"

    frames = min(before.frames, after.frames)
    writer = _encode(destination, FPS)
    for frame in range(0, frames, every):
        ticking = _clock(band, after, frame, colours)
        for axes, drawing, title in (
            (left, before, "Before training"),
            (right, after, "After training"),
        ):
            drawing.draw(axes, frame)
            drawing.furniture(axes)
            axes.set_title(title, color=colours.ink, fontsize=14, loc="left", pad=10)
        _render(figure, writer)
        for artist in ticking:
            artist.remove()
    writer.close()
    plt.close(figure)
    return destination
