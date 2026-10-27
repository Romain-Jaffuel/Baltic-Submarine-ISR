"""The map. One frame at a time, then sixty seconds of them.

Everything here is a reading of the trace, never of the simulation: the renderer
cannot reach into a running environment, which keeps it honest about what the
fleet actually knew.

The layering is from specification section 9.4, and the order matters because
the eye reads the top layer first. Fog, then land, then the lanes and the
cables, then the traffic, then the fleet's belief, then the fleet, then the red.
Nothing warm appears below the threat.

The chart underneath all of it -- the bathymetry, the coastline, the graticule,
the scale and the credits -- belongs to :mod:`baltic_isr.render.chart`, which
builds it once from the measured seabed. This module draws what moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from matplotlib import patheffects as path_effects
from matplotlib.path import Path as MarkerPath

from baltic_isr import geo
from baltic_isr.config import Config
from baltic_isr.fleet import auv as auv_module
from baltic_isr.picture.presence import PRESENCE_CELL_M
from baltic_isr.render import assets as assets_module
from baltic_isr.render.chart import Chart
from baltic_isr.render.palette import Palette, palette
from baltic_isr.world import field
from baltic_isr.world import statics as statics_module

if TYPE_CHECKING:  # matplotlib is imported inside the methods that draw
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg

# out/ is sorted by what a file IS, because fifty-three files at one level
# is not an output directory, it is a pile.
OUT = Path(__file__).resolve().parents[3] / "out"
VIDEOS = OUT / "videos"
STILLS = OUT / "figures"

TRAIL_FRAMES = {0: 2, 1: 20, 2: 45}

# How finely the fleet's heading is quantised for drawing. Twenty-four sectors
# is fifteen degrees, which at this mark size is under half a pixel of tip
# travel -- invisible, and it keeps the draw count near the sector count
# instead of near the vehicle count.
HEADING_SECTORS = 24

# Below this the belief ring sits on top of the grey dot it describes and says
# nothing the dot has not already said. Half the staleness gate.
SHOW_ELLIPSE_ABOVE_M = 1500.0

# A cable stretch nobody has heard for this long is drawn fully amber.
# Six hours is how long a cargo takes to cross the box.
EXPOSURE_FULL_S = 6.0 * 3600.0

# Points per at-sea stretch of cable. Constant, so the exposure table is a
# plain rectangular array indexed by stretch.
RUN_SAMPLES = 48
VESSEL_SIZE = 26.0
VEHICLE_SIZE = 95.0

# Dots said "something is here" and nothing else, and the first viewers asked
# which dots were theirs. So: two marks, both seen from above and both
# pointing right. A hull with a bow and a square stern for a vessel, and a
# chevron for one of ours.
#
# A chevron rather than a torpedo, and the reason is scale. The fleet is drawn
# at 95 points on a 138 km map, and at that size a torpedo's tail fins simply
# are not there -- it reduces to a dash, which is what a first viewer saw. A
# chevron is the one shape in this family that keeps its identity small,
# because its meaning is carried by an angle rather than by an outline. It
# gives up looking like a submarine, which is a real loss and a smaller one
# than being unreadable.
#
# The vessel is not rotated -- it reads as a *kind*, and a merchant on a lane
# is going the way the lane goes. The chevron is, because a chevron that does
# not point anywhere is a chevron wasted: its whole advantage over a hull
# silhouette is that it carries a direction at four pixels.
#
# Matplotlib cannot rotate a marker per point inside one scatter, and forty
# separate draws a frame is a different film. So headings are binned into
# HEADING_SECTORS and each sector is drawn once with its own rotated path --
# a couple of dozen draws, and fifteen degrees of error nobody can see.
SHIP = MarkerPath(
    [(-0.45, -0.30), (0.10, -0.30), (0.55, 0.0), (0.10, 0.30), (-0.45, 0.30), (-0.45, -0.30)],
    closed=True,
)
AUV = MarkerPath(
    [(-0.45, -0.45), (0.5, 0.0), (-0.45, 0.45), (-0.2, 0.0), (-0.45, -0.45)],
    closed=True,
)

# The map drawn one piece at a time. A replay shows all of them at once, which
# is right for somebody who already knows what the marks mean and wrong for
# somebody meeting them for the first time -- nine kinds of mark arriving
# together is not a map, it is a wall. The explainer film turns them on in the
# order it names them, so nothing is on screen that has not been introduced.
LAYERS = frozenset(
    {
        "fog",
        "land",
        "lanes",
        "cables",
        "bases",
        "traffic",
        "beliefs",
        "fleet",
        "status",
        "questions",
        "names",
    }
)


@dataclass(frozen=True)
class Scene:
    """The fixed background, projected into local metres once."""

    extent: np.ndarray
    rails: list[np.ndarray]
    corridors: list[np.ndarray]
    bases: np.ndarray
    fog_shape: tuple[int, int]


def build_scene(config: Config) -> Scene:
    """Project everything the background needs.

    Parameters
    ----------
    config : Config
        Resolved configuration.

    Returns
    -------
    Scene
        Background geometry in local metres.
    """
    geometry = geo.build(config)
    x_min, y_min, x_max, y_max = geometry.frame.bounds
    origin = np.array([x_min, y_min])

    def local(array: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, array - origin)

    extent = np.array([x_max - x_min, y_max - y_min])
    return Scene(
        extent=extent,
        rails=[local(rail.xy) for rail in geometry.rails],
        corridors=[local(item.xy) for item in geometry.corridors],
        bases=local(geometry.bases),
        fog_shape=(
            int(extent[1] // PRESENCE_CELL_M) + 1,
            int(extent[0] // PRESENCE_CELL_M) + 1,
        ),
    )


def _ellipse_shape(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return width, height and angle in degrees from packed covariances."""
    xx, xy, yy = cov[:, 0], cov[:, 1], cov[:, 2]
    half = 0.5 * (xx + yy)
    spread = np.sqrt(np.maximum((0.5 * (xx - yy)) ** 2 + xy * xy, 0.0))
    major = np.sqrt(np.maximum(half + spread, 0.0))
    minor = np.sqrt(np.maximum(half - spread, 0.0))
    angle = 0.5 * np.degrees(np.arctan2(2.0 * xy, xx - yy))
    return 2.0 * major, 2.0 * minor, angle


def _major_axis(cov: np.ndarray) -> np.ndarray:
    """Return the long semi-axis of each packed covariance, in metres."""
    xx, xy, yy = cov[:, 0], cov[:, 1], cov[:, 2]
    half = 0.5 * (xx + yy)
    spread = np.sqrt(np.maximum((0.5 * (xx - yy)) ** 2 + xy * xy, 0.0))
    return cast(np.ndarray, np.sqrt(np.maximum(half + spread, 0.0)))


def _encode(destination: Path, fps: int) -> Any:
    """Open a video writer that takes frames one at a time.

    Collecting every frame in a list and stacking at the end costs
    ``frames x height x width x 3`` bytes all at once: eight panels of a
    thirty-second film wanted **11.6 GiB** and the render died on the last
    line after twenty minutes of work. Streaming holds one frame.

    Parameters
    ----------
    destination : Path
        Where to write.
    fps : int
        Frames per second.

    Returns
    -------
    object
        An imageio writer; call ``append_data`` then ``close``.
    """
    import imageio.v2 as imageio

    destination.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(destination), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )


def _offset_band(line: np.ndarray, half_width: float) -> np.ndarray:
    """Return the closed polygon of a corridor around a polyline.

    A fill, because the corridor is a width in metres and a matplotlib
    linewidth is a width in points: the two only agree by accident, and at
    this scale six kilometres of corridor drew as a six-point hairline.
    """
    run = np.gradient(line, axis=0)
    length = np.linalg.norm(run, axis=1, keepdims=True).clip(min=1e-9)
    normal = np.stack([-run[:, 1], run[:, 0]], axis=-1) / length
    return np.concatenate([line + normal * half_width, (line - normal * half_width)[::-1]])


def _resample(line: np.ndarray, count: int) -> np.ndarray:
    """Return ``count`` points evenly spaced along a polyline."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    wanted = np.linspace(0.0, walked[-1], count)
    return np.stack(
        [np.interp(wanted, walked, line[:, 0]), np.interp(wanted, walked, line[:, 1])], axis=-1
    )


def _smooth(line: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline through a Catmull-Rom spline rather than its corners.

    The config gives a coastline five vertices, which is all the simulation
    needs and nothing like enough to draw. Straight segments meeting at hard
    angles are the single reason this map looked, in the first viewer's word,
    cheap -- no amount of colour fixes a five-point shore.
    """
    if line.shape[0] < 3:
        return _resample(line, count)
    padded = np.vstack([line[0] + (line[0] - line[1]), line, line[-1] + (line[-1] - line[-2])])
    out = []
    for index in range(1, padded.shape[0] - 2):
        p0, p1, p2, p3 = padded[index - 1 : index + 3]
        t = np.linspace(0.0, 1.0, 32, endpoint=False)[:, None]
        out.append(
            0.5
            * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
            )
        )
    out.append(padded[-2][None, :])
    return _resample(np.vstack(out), count)


def _blend(first: str, second: str, weight: float) -> tuple[float, float, float]:
    """Mix two hex colours, for shading a line by a scalar."""
    from matplotlib.colors import to_rgb

    a, b = np.array(to_rgb(first)), np.array(to_rgb(second))
    mixed = a + (b - a) * float(weight)
    return float(mixed[0]), float(mixed[1]), float(mixed[2])


def _verdict(on_cable: int, held: int, odd: int) -> str:
    """Say in one sentence what is happening to the cables.

    Rewritten once already. The first version led with deviations -- "2
    vessels deviating, the fleet sees none" -- which makes the detector the
    subject of the film. A vessel slowing down is a *reason to look*, not the
    event; the event is a vessel sitting over Balticconnector. So the sentence
    leads with the asset, and odd behaviour in open water is demoted to the
    clause it deserves.

    Parameters
    ----------
    on_cable : int
        Vessels behaving oddly inside a cable corridor, right now.
    held : int
        How many of those the fleet currently has a question about.
    odd : int
        Vessels behaving oddly away from any cable.

    Returns
    -------
    str
        One sentence, for the headline.
    """
    aside = f" {odd} more behaving oddly in open water." if odd else ""
    if on_cable == 0:
        return f"No vessel over a cable.{aside}"
    plural = "s" if on_cable > 1 else ""
    subject = "them" if on_cable > 1 else "it"
    what = f"{on_cable} vessel{plural} sitting over a cable"
    if held == 0:
        return f"{what} — the fleet cannot see {subject}.{aside}"
    if held < on_cable:
        return f"{what} — the fleet is tracking {held}.{aside}"
    return f"{what}, and the fleet has {subject}.{aside}"


class Replay:
    """Draws frames of a recorded episode.

    Parameters
    ----------
    config : Config
        Resolved configuration, for the background.
    trace : dict
        The arrays a :class:`~baltic_isr.render.trace.Trace` produced.
    mode : str
        ``"light"`` or ``"dark"``.
    """

    def __init__(
        self,
        config: Config,
        trace: dict[str, np.ndarray],
        mode: str = "light",
        show_reach: bool = True,
        style: str = "exposition",
    ) -> None:
        self._config = config
        self._trace = trace
        self.show_reach = show_reach
        # Off for the right panel of a side-by-side: one legend serves both.
        self.show_legend = True
        # Everything, unless a caller is building the map up piece by piece.
        self.layers = LAYERS
        # The unwatched-water wash. Faint by default so it never competes with
        # the marks; a chapter whose whole point IS the uncovered water turns
        # it up, because a caption that says "the grey is what nobody has
        # heard" over an invisible grey is a caption that lies.
        self.fog_alpha = 0.07
        self.style = style
        self._assets = assets_module.build_assets(config)
        self._statics = statics_module.build(config, "cpu")
        self._on_cable_cache: dict[int, np.ndarray] = {}
        self._cable_runs: list[tuple[np.ndarray, float]] | None = None
        self._exposure_table: np.ndarray | None = None
        self._colours: Palette = palette(mode)
        self._scene = build_scene(config)
        self._chart = Chart.build(config, self._colours)
        self._frame = geo.build(config).frame
        self._link_offsets = np.concatenate([[0], np.cumsum(trace["link_counts"])]).astype(np.int64)
        self._fog_index = {int(frame): row for row, frame in enumerate(trace["fog_frames"])}

    @property
    def frames(self) -> int:
        """Return how many frames the trace holds."""
        return int(self._trace["times_s"].shape[0])

    def links_at(self, frame: int) -> np.ndarray:
        """Return the acoustic pairs active on a frame."""
        start, stop = self._link_offsets[frame], self._link_offsets[frame + 1]
        return self._trace["link_pairs"][start:stop]

    def fog_at(self, frame: int) -> np.ndarray:
        """Return the most recent search picture at or before a frame."""
        row = max((value for key, value in self._fog_index.items() if key <= frame), default=0)
        return cast(np.ndarray, self._trace["fog_age_s"][row].reshape(self._scene.fog_shape))

    def draw(self, axes: Axes, frame: int) -> None:
        """Draw one frame onto an axes.

        Parameters
        ----------
        axes : Axes
            Where to draw.
        frame : int
            Which frame.
        """
        colours = self._colours
        scene = self._scene
        trace = self._trace
        width, height = float(scene.extent[0]), float(scene.extent[1])

        axes.clear()
        axes.set_facecolor(colours.water[-1])
        axes.set_xlim(0.0, width)
        axes.set_ylim(0.0, height)
        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])
        for side in axes.spines.values():
            side.set_visible(False)

        shown = self.layers

        # 1. the chart: bathymetry, the coastline, the fleet's own limit.
        # Built once in render.chart and redrawn here, because a frame starts
        # from a cleared axes and the alternative is contouring a 1.5 M cell
        # raster fourteen hundred times.
        if "land" in shown:
            self._chart.basemap(axes, zorder=0.0)
            self._chart.graticule(axes, self._frame, zorder=0.2)

        # 2. the fog: where nobody has looked lately.
        #
        # ON TOP of the chart, which is a change of order and not of taste.
        # The chart is an opaque raster, so a wash underneath it is a wash
        # nobody sees -- the fog went invisible the moment the basemap
        # arrived. It belongs between the sea and the marks anyway: what it
        # dulls is the water, and what it must never dull is the fleet.
        if "fog" in shown:
            fog = self.fog_at(frame)
            axes.imshow(
                np.clip(fog / (24.0 * 3600.0), 0.0, 1.0),
                extent=(0.0, width, 0.0, height),
                origin="lower",
                cmap="Greys",
                alpha=self.fog_alpha,
                vmin=0.0,
                vmax=1.4,
                interpolation="bilinear",
                zorder=0.5,
            )

        if "names" in shown:
            self._geography(axes)

        # 3. the lanes, then the cables
        if "lanes" in shown:
            for rail in scene.rails:
                self._lane(axes, _smooth(rail, 180))
        if "cables" in shown:
            self._assets_layer(axes, frame)
        if "bases" in shown:
            axes.scatter(
                scene.bases[:, 0],
                scene.bases[:, 1],
                s=70,
                marker="s",
                color=colours.ink_soft,
                zorder=4,
            )

        # 4. the traffic, which is context and must stay context
        vessels = trace["vessel_xy"][frame]
        deviating = trace["vessel_deviating"][frame]
        if "traffic" in shown:
            axes.scatter(
                vessels[~deviating, 0],
                vessels[~deviating, 1],
                s=VESSEL_SIZE,
                marker=SHIP,
                color=colours.traffic,
                alpha=0.70,
                linewidths=0,
                zorder=5,
            )

        # 5. what the fleet believes: the ellipse of every LIVE contact.
        # An expired one is drawn at its true size, which on this map is a
        # forty-kilometre circle swamping everything near it -- and it is also
        # a lie, because an expired prediction is not a belief, it is the
        # absence of one.
        known = trace["track_known"][frame]
        alarmed = trace["track_alarmed"][frame]
        cov = trace["track_cov"][frame]
        major = _major_axis(cov)
        live = known & (major < self._config.fleet.gate_m)
        # Only the ones worth drawing. A hundred and seventy-six contacts each
        # got a ring, most of them tight around a grey dot that already said
        # where the vessel was -- the same fact twice, a hundred and seventy-six
        # times. Below this the ring tells the viewer nothing the dot has not.
        quiet = live & ~alarmed & (major > SHOW_ELLIPSE_ABOVE_M)
        if quiet.any() and "beliefs" in shown:
            self._ellipses(
                axes, trace["track_xy"][frame][quiet], cov[quiet], colours.prediction, 0.6, 0.45, 6
            )

        # 6. the fleet, shaded by regime, trailing by regime
        if "fleet" in shown:
            self._fleet(axes, frame)

        # 7. the two status colours, last, on top of everything.
        #
        # The rule the first version got wrong: **hue says what a thing IS,
        # blue says what the fleet THINKS.** A red ring used to mean "the
        # fleet is unsure about this", and a viewer read it as "something
        # serious here" -- the opposite. Red is now spent on one thing only.
        #
        #   grey dot      a vessel going about its business
        #   amber dot     a vessel behaving oddly, in open water. Odd, not grave
        #   RED diamond   a vessel behaving oddly OVER A CABLE. The thing the
        #                 fleet exists to catch
        #   blue ring     the fleet has a question about this contact. Blue,
        #                 because every belief on this map is the fleet's, and
        #                 the fleet is blue
        #
        # So a red diamond inside a blue ring is a target the fleet has found;
        # a red diamond alone is one nobody is looking at; and a blue ring on
        # empty water is the fleet chasing a ghost. No legend needed for that.
        if "status" not in shown:
            if self.show_legend:
                self._legend(axes)
            return

        # **Red means one thing: a vessel that is not behaving.** It used to
        # mean two, with amber for "odd, in open water" and red for "odd, over
        # a cable" -- a distinction that is real in the metrics and invisible
        # to anybody watching for the first time. Two marks for one idea is
        # one mark too many, so the pair collapsed and amber went back to
        # meaning exactly one thing of its own: a cable nobody is listening to.
        #
        # The mark keeps the shape of what it is. Every vessel on this map is
        # a dot; a red one is a vessel in trouble. Changing the shape as well
        # would say it is a different kind of object, and it is not.
        odd = deviating
        if odd.any():
            # A glow under the dot. On a 138 km map a lone mark is four pixels
            # of fourteen hundred, and the eye has to be sent somewhere before
            # it can read anything.
            axes.scatter(
                vessels[odd, 0],
                vessels[odd, 1],
                s=1800,
                color=colours.threat,
                alpha=0.15,
                linewidths=0,
                zorder=11,
            )
            axes.scatter(
                vessels[odd, 0],
                vessels[odd, 1],
                s=260,
                marker=SHIP,
                color=colours.threat,
                edgecolors=colours.surface,
                linewidths=1.4,
                zorder=12,
            )

        if self.show_legend:
            self._legend(axes)

    def _lane(self, axes: Axes, line: np.ndarray) -> None:
        """Draw a shipping lane as a chart draws one, not as a grey smear.

        A traffic separation scheme on a nautical chart is a *corridor with
        edges*: two dashed boundaries, a tinted body, and arrows saying which
        way the traffic runs. Drawn as a thick soft line instead, it read as a
        smudge -- nobody looking at the map could say what it was, which was
        the complaint.
        """
        colours = self._colours
        # Roughly a nautical mile each side of the centreline, which is what
        # a real traffic separation scheme runs. The first attempt drew 6.8 km
        # of corridor and swamped the water it was meant to sit in.
        half = 1_850.0
        band = _offset_band(line, half)
        axes.fill(band[:, 0], band[:, 1], color=colours.lane, alpha=0.85, zorder=2, linewidth=0)
        for side in (half, -half):
            edge = _offset_band(line, side)[: line.shape[0]]
            axes.plot(
                edge[:, 0],
                edge[:, 1],
                color=colours.corridor,
                linewidth=0.9,
                linestyle=(0, (6, 4)),
                alpha=0.55,
                zorder=3,
            )
        # Arrows along the lane. Four is enough to say "traffic runs this
        # way" and few enough not to compete with the vessels on it.
        for at in np.linspace(0.12, 0.88, 4):
            index = int(at * (line.shape[0] - 2))
            here, ahead = line[index], line[index + 1]
            step = ahead - here
            step = step / max(float(np.linalg.norm(step)), 1e-9) * 5_200.0
            axes.annotate(
                "",
                xy=tuple(here + step),
                xytext=tuple(here),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colours.corridor,
                    "linewidth": 1.1,
                    "alpha": 0.5,
                },
                zorder=3,
            )

    def _geography(self, axes: Axes) -> None:
        """Name the water, the two countries and the two ports.

        A map of an unfamiliar sea is a pattern until somebody says where it
        is. The caricature needed two landmass colours to make the point that
        this was a strait between two countries; the measured coastline makes
        it by itself, so the names do the rest and the land is one tone, the
        way a chart has always drawn it.
        """
        colours = self._colours
        scene = self._scene
        width = float(scene.extent[0])
        height = float(scene.extent[1])

        halo = [path_effects.withStroke(linewidth=3.2, foreground=colours.water[-1])]

        axes.text(
            width * 0.56,
            height * 0.42,
            "Gulf of Finland",
            color=colours.ink_soft,
            fontsize=15,
            alpha=0.45,
            ha="center",
            va="center",
            style="italic",
            path_effects=halo,
            zorder=2,
        )
        # On the land, at the point render.chart found furthest from the
        # water on each side. Written down, the southern one landed in a bay.
        anchors = self._chart.anchors
        for name, side in (("Finland", "north"), ("Estonia", "south")):
            if side not in anchors:
                continue
            x, y = anchors[side]
            axes.text(
                x,
                y,
                name,
                color=colours.ink_soft,
                fontsize=11,
                alpha=0.75,
                ha="center",
                va="center",
                path_effects=[path_effects.withStroke(linewidth=3.0, foreground=colours.land)],
                # Clipped, which text is not by default. The films trim the
                # empty land, and an unclipped country name goes on being
                # drawn below the neatline where it reads as a smudge.
                clip_on=True,
                zorder=3,
            )
        for (x, y), port in zip(scene.bases, ("Helsinki", "Tallinn"), strict=False):
            axes.annotate(
                port,
                xy=(float(x), float(y)),
                xytext=(11, 9),
                textcoords="offset points",
                color=colours.ink,
                fontsize=10.5,
                path_effects=halo,
                zorder=14,
            )

    def furniture(self, axes: Axes) -> None:
        """Draw what tells a reader this is a chart and not a picture of one.

        The neatline, the bar scale, true north and the source credit. All of
        it comes from render.chart, and all of it is measured against the axes
        limits rather than the box.

        **Called after any crop, never inside :meth:`draw`.** Every one of
        these marks describes the view -- where its edge is, how far across it
        is, which way is up -- so placing them before the caller trims the
        empty land puts the scale bar and the neatline somewhere off screen.
        :meth:`still` does it; the films in render.story do it after
        ``_crop``; a caller that does neither gets a chart without furniture,
        which is a drawing, not a map.
        """
        self._chart.neatline(axes)
        # Right of centre, clear of the legend, which owns the bottom left.
        self._chart.scale_bar(axes, anchor=(0.615, 0.042))
        self._chart.north_arrow(axes, anchor=(0.955, 0.055))

    def _sea_runs(self, line: np.ndarray) -> list[np.ndarray]:
        """Split a cable into the stretches that are actually at sea.

        A cable makes landfall, so part of every one of them is drawn across
        the beige. Left whole they read as roads over Finland.
        """
        samples = _resample(line, 120)
        import torch

        wet = (
            field.distance_to_coast(self._statics, torch.from_numpy(samples).float()).numpy() > 0.0
        )
        runs, current = [], []
        for point, at_sea in zip(samples, wet, strict=True):
            if at_sea:
                current.append(point)
            elif len(current) > 1:
                runs.append(np.array(current))
                current = []
            else:
                current = []
        if len(current) > 1:
            runs.append(np.array(current))
        return runs or [samples]

    def _on_cable(self, frame: int) -> np.ndarray:
        """Return which vessels are inside a cable corridor, this frame."""
        cached = self._on_cable_cache.get(frame)
        if cached is None:
            cached = assets_module.over_a_cable(self._assets, self._trace["vessel_xy"][frame])
            self._on_cable_cache[frame] = cached
        return cached

    def _assets_layer(self, axes: Axes, frame: int) -> None:
        """Draw the cables and the places they are exposed.

        The cables used to be four grey dashes, which is the wrong weight for
        the only reason the fleet is out here. Red on a vessel cannot mean
        "serious" until the map has said what is at stake, so the asset gets a
        hue of its own, a visible protection corridor, and a mark on each of
        the five points where it passes under a shipping lane.

        In the ``exposition`` style the cable is shaded along its length by how
        long since any vehicle could hear that stretch, which turns the map
        into a picture of what is *unwatched* rather than of what is moving.
        """
        colours = self._colours
        exposed = self.style == "exposition"
        for index, (run, _half_width) in enumerate(self._runs()):
            # No protection corridor. It was drawn as a wide soft band at the
            # real six-kilometre width, to say what "over a cable" means --
            # but a halo around the one line that already carries a colour of
            # its own adds no fact and costs the water it sits in. The cable
            # itself is the mark.
            if exposed:
                self._exposed_cable(axes, index, frame)
            else:
                axes.plot(
                    run[:, 0], run[:, 1], color=colours.asset, linewidth=2.2, alpha=0.9, zorder=3
                )

        if self.style != "epure":
            axes.scatter(
                self._assets.landfalls[:, 0],
                self._assets.landfalls[:, 1],
                s=36,
                marker="|",
                color=colours.asset,
                linewidths=2.0,
                alpha=0.8,
                zorder=4,
            )

        # The chokepoints are not marked. They were drawn as rings, and a ring
        # on a map of rings is one more thing to decode: the crossing is
        # already visible where a cable meets a lane, and `assets.crossings`
        # still carries them for the metrics that count them.

    def _runs(self) -> list[tuple[np.ndarray, float]]:
        """Return every at-sea stretch of cable, with its corridor width.

        Fixed geometry, so it is found once. Each stretch is resampled to a
        constant count, which is what lets the exposure table below be a plain
        rectangular array.

        Drawn as laid in the configuration, which is to say straight between
        landfalls. An earlier version added a deterministic wander, on the
        grounds that a real cable is laid with slack and does not run ruled.
        That was true and it still looked better, but it is the one piece of
        geometry here that nobody measured, and inventing detail for it on a
        chart whose seabed is surveyed invites the reader to believe the route
        as well. The straight line says what is actually known: two landfalls.
        """
        if self._cable_runs is None:
            self._cable_runs = [
                (_resample(run, RUN_SAMPLES), half_width)
                for _, line, half_width in self._assets.cables
                for run in self._sea_runs(line)
            ]
        return self._cable_runs

    def _exposed_cable(self, axes: Axes, index: int, frame: int) -> None:
        """Shade one stretch of cable by how long it has gone unheard."""
        from matplotlib.collections import LineCollection

        run = self._runs()[index][0]
        samples = _resample(run, RUN_SAMPLES)
        age = self._exposure()[frame, index * RUN_SAMPLES : (index + 1) * RUN_SAMPLES]
        shade = np.clip(age / EXPOSURE_FULL_S, 0.0, 1.0)
        points = samples.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Violet to AMBER, never to red. The first attempt ran it to red and a
        # fully unwatched cable came out as a red line across the map, which
        # collides head-on with the one meaning red is allowed to have here. An
        # unwatched stretch is a caution, not a target.
        collection = LineCollection(
            list(segments),
            linewidths=5.0,
            colors=[_blend(self._colours.asset, self._colours.watch, value) for value in shade],
            zorder=4,
            capstyle="round",
        )
        axes.add_collection(collection)

    def _exposure(self) -> np.ndarray:
        """Return seconds since each cable sample was last heard, per frame.

        One forward pass over the whole trace, carrying the last time each
        sample was within somebody's detection range. Computing it per frame
        instead meant walking back over the history for every cable of every
        frame: about 1.3 billion distances for a film, against 22 million for
        this, and the answer is the same.
        """
        if self._exposure_table is not None:
            return self._exposure_table

        samples = np.concatenate([_resample(run, RUN_SAMPLES) for run, _ in self._runs()])
        times = self._trace["times_s"]
        # Nothing has been heard before the trace starts, and claiming
        # otherwise would paint the first minutes of the film falsely calm.
        last_heard = np.full(samples.shape[0], float(times[0]) - EXPOSURE_FULL_S)
        table = np.empty((len(times), samples.shape[0]), dtype=np.float32)
        for frame, now in enumerate(times):
            gap = np.linalg.norm(
                samples[:, None, :] - self._trace["vehicle_xy"][frame][None, :, :], axis=2
            )
            heard = (gap <= self._trace["vehicle_reach_m"][frame][None, :]).any(axis=1)
            last_heard = np.where(heard, float(now), last_heard)
            table[frame] = float(now) - last_heard
        self._exposure_table = table
        return table

    def _legend(self, axes: Axes) -> None:
        """Name every mark on the map, in the corner, once.

        The acceptance criterion of this project is that someone who knows
        nothing about it understands the film *without explanation*. A legend
        is the cheapest way to buy that, and the first viewer of the map asked
        what the red circles were -- which is the question it answers.
        """
        from matplotlib.lines import Line2D

        colours = self._colours
        cruise = colours.regime[1]

        def dot(colour: str, size: float, marker: str = "o", edge: str | None = None) -> Line2D:
            return Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markersize=size,
                markerfacecolor=colour,
                markeredgecolor=edge or colour,
                markeredgewidth=1.4 if edge else 0.0,
            )

        cable = (
            [
                (Line2D([], [], color=colours.asset, linewidth=4.0), "Cable, patrolled recently"),
                (
                    Line2D([], [], color=colours.watch, linewidth=4.0),
                    f"Cable, unpatrolled for {EXPOSURE_FULL_S / 3600:.0f} h",
                ),
            ]
            if self.style == "exposition"
            else [(Line2D([], [], color=colours.asset, linewidth=3.0), "Subsea cable")]
        )
        entries = [
            *cable,
            (dot(colours.traffic, 5.0), "Merchant vessel"),
            (dot(colours.threat, 8.0, edge=colours.surface), "Suspicious vessel"),
            (dot(cruise, 7.0), "Submarine drone"),
        ]
        legend = axes.legend(
            [handle for handle, _ in entries],
            [label for _, label in entries],
            loc="lower left",
            frameon=True,
            framealpha=0.88,
            facecolor=colours.surface,
            edgecolor=colours.hairline,
            fontsize=11,
            borderpad=0.9,
            labelspacing=0.75,
            handletextpad=1.0,
        )
        for text in legend.get_texts():
            text.set_color(colours.ink_soft)
        legend.set_zorder(20)

    def _ellipses(
        self,
        axes: Axes,
        centres: np.ndarray,
        cov: np.ndarray,
        colour: str,
        linewidth: float,
        alpha: float,
        zorder: int,
        fill: float = 0.0,
    ) -> None:
        """Draw a set of uncertainty ellipses.

        ``fill`` washes the inside in at that opacity. It is what separates a
        *belief* from a *thing* on this map: a wash is an area the fleet is
        unsure about, an opaque marker is a vessel that is really there. Drawn
        both as outlines, the two were indistinguishable.
        """
        from matplotlib.collections import EllipseCollection

        width, height, angle = _ellipse_shape(cov)
        axes.add_collection(
            EllipseCollection(
                width,
                height,
                angle,
                units="xy",
                offsets=centres,
                offset_transform=axes.transData,
                facecolors=colour if fill > 0.0 else "none",
                edgecolors=colour,
                linewidths=linewidth,
                alpha=fill if fill > 0.0 else alpha,
                zorder=zorder,
            )
        )
        if fill > 0.0:
            axes.add_collection(
                EllipseCollection(
                    width,
                    height,
                    angle,
                    units="xy",
                    offsets=centres,
                    offset_transform=axes.transData,
                    facecolors="none",
                    edgecolors=colour,
                    linewidths=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
            )

    def _headings(self, frame: int) -> np.ndarray:
        """Return each vehicle's course, as a sector index.

        Taken from where it has actually been rather than from a heading
        channel, because the trace does not carry one and the course made
        good is the honest thing to draw anyway: it is what an observer
        plotting the vehicle would write down.

        A vehicle that has not moved keeps the last course it had, so a
        drifting or charging vehicle does not spin while it sits.
        """
        track = self._trace["vehicle_xy"]
        past = max(frame - 3, 0)
        step = track[frame] - track[past]
        angle = np.arctan2(step[:, 0], step[:, 1])
        sector = np.round(angle / (2.0 * np.pi) * HEADING_SECTORS).astype(int)
        still = np.hypot(step[:, 0], step[:, 1]) < 1.0
        if still.any() and frame > 0:
            # fall back to the longest recent run, so a pause does not reset it
            older = track[frame] - track[max(frame - 30, 0)]
            angle_old = np.arctan2(older[:, 0], older[:, 1])
            sector[still] = np.round(angle_old[still] / (2.0 * np.pi) * HEADING_SECTORS).astype(int)
        return np.asarray(sector % HEADING_SECTORS)

    @staticmethod
    def _turned(sector: int) -> MarkerPath:
        """Return the chevron, rotated to one heading sector."""
        from matplotlib.transforms import Affine2D

        # The path points east at rest; a sector of zero is due north.
        degrees = 90.0 - sector * 360.0 / HEADING_SECTORS
        return MarkerPath(AUV.vertices @ Affine2D().rotate_deg(degrees).get_matrix()[:2, :2].T)

    def _fleet(self, axes: Axes, frame: int) -> None:
        """Draw the vehicles, their hearing and their links."""
        colours = self._colours
        trace = self._trace
        position = trace["vehicle_xy"][frame]
        regime = trace["vehicle_regime"][frame]
        mode = trace["vehicle_mode"][frame]
        reach = trace["vehicle_reach_m"][frame]
        at_sea = (mode == auv_module.MODE_DEPLOYED) | (mode == auv_module.MODE_RETURNING)
        course = self._headings(frame)

        # The hearing radius, which collapses when a vehicle sprints. A very
        # faint FILL, and the faintness is the whole design: it has to read as
        # texture, not as forty objects. Drawing it as an outline instead was
        # tried and is worse -- crisp rings pull the eye to forty places at
        # once, and on a fleet that only ever cruises those forty circles are
        # identical, so they pull it there to say nothing. The layer earns its
        # place once a policy varies the regime, which is why `show_reach`
        # exists rather than deleting it.
        from matplotlib.collections import EllipseCollection

        if self.show_reach and at_sea.any():
            axes.add_collection(
                EllipseCollection(
                    2.0 * reach[at_sea],
                    2.0 * reach[at_sea],
                    np.zeros(int(at_sea.sum())),
                    units="xy",
                    offsets=position[at_sea],
                    offset_transform=axes.transData,
                    facecolors=colours.regime[1],
                    edgecolors="none",
                    alpha=0.06,
                    zorder=4,
                )
            )

        for pair in self.links_at(frame):
            first, second = position[pair[0]], position[pair[1]]
            axes.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color=colours.regime[2],
                linewidth=0.9,
                alpha=0.65,
                zorder=8,
            )

        for index in range(3):
            selected = at_sea & (regime == index)
            if not selected.any():
                continue
            span = TRAIL_FRAMES[index]
            if span > 1 and frame >= span:
                history = trace["vehicle_xy"][frame - span : frame + 1][:, selected]
                for lane in range(history.shape[1]):
                    axes.plot(
                        history[:, lane, 0],
                        history[:, lane, 1],
                        color=colours.regime[index],
                        linewidth=1.1,
                        alpha=0.5,
                        zorder=8,
                    )
            for sector in np.unique(course[selected]):
                facing = selected & (course == sector)
                axes.scatter(
                    position[facing, 0],
                    position[facing, 1],
                    s=VEHICLE_SIZE,
                    marker=self._turned(int(sector)),
                    color=colours.regime[index],
                    linewidths=0,
                    zorder=9,
                )

        alongside = ~at_sea
        if alongside.any():
            for sector in np.unique(course[alongside]):
                facing = alongside & (course == sector)
                axes.scatter(
                    position[facing, 0],
                    position[facing, 1],
                    s=VEHICLE_SIZE * 0.6,
                    marker=self._turned(int(sector)),
                    color=colours.regime[0],
                    alpha=0.35,
                    linewidths=0,
                    zorder=9,
                )

    def still(self, frame: int, path: Path | None = None) -> Path:
        """Draw one frame to a file, for looking at while designing.

        Parameters
        ----------
        frame : int
            Which frame.
        path : Path, optional
            Destination. Defaults to ``out/frame_<n>.png``.

        Returns
        -------
        Path
            Where it was written.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(14.4, 11.52), constrained_layout=True)
        self.draw(axes, frame)
        # A still is read at leisure, so it gets the depths written on. A film
        # is read at a glance, and a page of numbers across the gulf is not
        # something anybody reads at a glance.
        self._chart.label_isobaths(axes, zorder=3.0)
        self.furniture(axes)
        self._caption(axes, frame)
        destination = path or (STILLS / f"frame_{frame:05d}.png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=110, facecolor=self._colours.surface)
        plt.close(figure)
        return destination

    def animate(
        self,
        path: Path | None = None,
        start: int = 0,
        stop: int | None = None,
        every: int = 1,
        fps: int = 24,
        dpi: int = 100,
    ) -> Path:
        """Write a stretch of the trace to an MP4.

        Parameters
        ----------
        path : Path, optional
            Destination. Defaults to ``out/replay.mp4``.
        start, stop : int
            Frame range. ``stop`` defaults to the end of the trace.
        every : int
            Keep one frame in this many. The trace holds 48 frames for each
            second of film, so dropping one in two still gives 24.
        fps : int
            Frames per second of the finished film.
        dpi : int
            Raster resolution.

        Returns
        -------
        Path
            Where it was written.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        last = self.frames if stop is None else stop
        indices = range(start, last, every)
        destination = path or (VIDEOS / "replay.mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)

        figure, axes = plt.subplots(figsize=(14.4, 11.52), constrained_layout=True)
        writer = _encode(destination, fps)
        for frame in indices:
            self.draw(axes, frame)
            self._caption(axes, frame)
            self._timebar(axes, frame)
            figure.canvas.draw()
            # buffer_rgba is unannotated in the matplotlib stubs; the Agg
            # canvas is the one in use, set two lines above.
            canvas = cast("FigureCanvasAgg", figure.canvas)
            writer.append_data(np.asarray(canvas.buffer_rgba())[:, :, :3])  # type: ignore[no-untyped-call]
        writer.close()
        plt.close(figure)
        return destination

    def _timebar(self, axes: Axes, frame: int) -> None:
        """Draw where we are in the episode, so a varying time base is visible."""
        colours = self._colours
        width = float(self._scene.extent[0])
        height = float(self._scene.extent[1])
        fraction = frame / max(self.frames - 1, 1)
        axes.plot(
            [0.0, width],
            [-0.02 * height] * 2,
            color=colours.hairline,
            linewidth=3.0,
            solid_capstyle="butt",
            clip_on=False,
            zorder=12,
        )
        axes.plot(
            [0.0, fraction * width],
            [-0.02 * height] * 2,
            color=colours.regime[1],
            linewidth=3.0,
            solid_capstyle="butt",
            clip_on=False,
            zorder=13,
        )

    def _caption(self, axes: Axes, frame: int, prefix: str | None = None) -> None:
        """Put the clock and a count where a viewer will find them.

        ``prefix`` names the panel on its own line above the verdict, for a
        grid of several runs. On one line the two were clipped: a verdict runs
        to 68 characters and a panel name to 36, which together overflow a
        1104 px panel and silently lose the end of the sentence -- the part
        that says whether the fleet saw anything.
        """
        trace = self._trace
        hours = float(trace["times_s"][frame]) / 3600.0
        held = int(trace["track_known"][frame].sum())
        red = int(trace["track_alarmed"][frame].sum())
        # The sentence is the headline and the counts are its evidence, in
        # that order, because a viewer who knows nothing about the project
        # should not have to cross-reference numbers and a legend to see what
        # is wrong. The track table is indexed by vessel, so the fleet's
        # opinion of a given vessel is one elementwise AND away.
        deviating = trace["vessel_deviating"][frame]
        on_cable = deviating & self._on_cable(frame)
        alarmed = trace["track_alarmed"][frame] & trace["track_known"][frame]
        headline = _verdict(
            int(on_cable.sum()),
            int((alarmed & on_cable).sum()),
            int((deviating & ~on_cable).sum()),
        )
        axes.set_title(
            headline if prefix is None else f"{prefix}\n{headline}",
            color=self._colours.ink,
            fontsize=16 if prefix is None else 14,
            loc="left",
            pad=32,
        )
        axes.text(
            0.0,
            1.006,
            f"t + {hours:.1f} h      {len(self._assets.crossings)} chokepoints"
            f"      {held} contacts held      {red} open questions",
            transform=axes.transAxes,
            color=self._colours.ink_soft,
            fontsize=11,
            va="bottom",
            ha="left",
        )
