"""The chart the data sits on: bathymetry, coastline, graticule, furniture.

Everything in this module is **static**. It is built once from the measured
seabed and from the projection, and every frame of a film draws the same
objects again. Nothing here reads the trace, and nothing here knows what the
fleet is doing.

The idiom is a sea chart rather than a GIS screenshot, and that is a decision
with a measurement behind it. A hypsometric ramp -- dark blue for deep water --
is what a GIS produces by default, and against this fleet it does not work: the
widest such ramp that still separates from every mark on the map has a
lightness span of 0.087, which is nine shades nobody can tell apart. Charts
have solved this for two centuries by inverting it. The shallows carry the
tint, the deep water is nearly white, depth is carried by **labelled
isobaths**, and the whole saturation budget is left for the data.

It also happens to suit the fleet: no vehicle may be in less than fifteen
metres of water, so the darkest band is water none of them can occupy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from baltic_isr import geo
from baltic_isr.config import Config
from baltic_isr.render.palette import Palette
from baltic_isr.world import seabed as seabed_module

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Where the water changes tone, and where the contours run: the same depths,
# because a tint edge and a contour that disagree read as two different maps.
# 10, 20, 50 and 100 is the set every chart of a shelf sea uses, and it is not
# arbitrary -- it is one step per order of consequence.
BANDS_M = (10.0, 20.0, 50.0)
ISOBATHS_M = (10.0, 20.0, 50.0, 100.0)

# Labelling all of them turns the gulf into a page of numbers. These are the
# ones a reader navigates by.
LABELLED_M = (20.0, 50.0, 100.0)

# The fleet's own limit is drawn separately from the depth contours, because
# it is a fact about the vehicles rather than about the sea -- without it a
# viewer watches the fleet turn away from open water for no visible reason.
# Taken from the configuration, never written down here: the two drifting
# apart would put the drawn wall somewhere the fleet does not stop.

# Contours are taken from a SMOOTHED copy of the depth field, never from the
# one the simulation gathers from. At 100 m the terrain model resolves every
# rock in the archipelago, and contouring that produces a thicket nobody can
# read -- three hundred-odd closed rings around individual boulders. Charts
# generalise, and this is the generalisation: a Gaussian of three cells, and
# then anything still shorter than two kilometres is a rock, not a contour.
CONTOUR_SMOOTH_CELLS = 3.0
MIN_CONTOUR_M = 2000.0

# Graticule spacing. The box is 2.4 degrees of longitude and 0.95 of latitude,
# so a half degree of longitude and a quarter of latitude give a grid of
# roughly square cells at this latitude.
LON_STEP_DEG = 0.5
LAT_STEP_DEG = 0.25

# Tick length as a fraction of the box, and how far the labels sit outside it.
TICK = 0.008
LABEL_PAD = 0.016

# How far inland a country's name sits. Far enough off the shore to be clearly
# on the land, near enough that it reads as a label for the coast below it.
INLAND_M = 9000.0


def _dms(value: float, axis: str) -> str:
    """Return a degrees-and-minutes label, the way a chart writes one."""
    hemisphere = ("E" if value >= 0 else "W") if axis == "lon" else ("N" if value >= 0 else "S")
    value = abs(value)
    degrees = int(value)
    minutes = (value - degrees) * 60.0
    if abs(minutes) < 0.05:
        return f"{degrees}°{hemisphere}"
    # U+2032 PRIME, which is what a chart prints for minutes of arc. An
    # apostrophe is a different character that merely looks like one.
    return f"{degrees}°{minutes:02.0f}′{hemisphere}"  # noqa: RUF001


@dataclass(frozen=True)
class Chart:
    """The static background of every frame.

    Attributes
    ----------
    extent : tuple of float
        Box width and height in local metres.
    tint : ndarray
        The sea and the land already coloured, ``(rows, columns, 3)``, ready
        for a single ``imshow``. Built once: contouring a 1 380 x 1 099 raster
        on every one of fourteen hundred frames is a different film.
    coast : tuple of ndarray
        The waterline as polylines in local metres.
    isobaths : dict
        Depth in metres to the polylines at that depth.
    navigable : tuple of ndarray
        The fifteen-metre line: the edge of the water the fleet may enter.
    convergence_deg : float
        How far true north lies from grid north at the centre of the box.
        TM35FIN is a transverse Mercator on 27 degrees east and this box is
        not, so the two differ and a north arrow that ignores it is wrong.
    credit : str
        The attribution the licence requires, and the projection, as one line.
    anchors : dict
        Where the two country names go, in local metres. Found on the signed
        distance field rather than written down: the hard-coded pair put
        ESTONIA in a bay, because a coastline with bays in it had arrived.
    origin : ndarray
        Projected coordinates of the box's south-west corner, shape ``(2,)``.
        Everything drawn is in metres from here.
    bounds : tuple of float
        ``(lon_min, lon_max, lat_min, lat_max)`` of the box, for the
        graticule.
    """

    extent: tuple[float, float]
    tint: np.ndarray
    coast: tuple[np.ndarray, ...]
    isobaths: dict[float, tuple[np.ndarray, ...]]
    navigable: tuple[np.ndarray, ...]
    convergence_deg: float
    credit: str
    anchors: dict[str, tuple[float, float]]
    origin: np.ndarray
    bounds: tuple[float, float, float, float]
    colours: Palette

    # ----------------------------------------------------------------- build

    @staticmethod
    def build(config: Config, colours: Palette) -> Chart:
        """Project, colour and contour the seabed once.

        Parameters
        ----------
        config : Config
            Resolved configuration, for the projection.
        colours : Palette
            The chart's colours.

        Returns
        -------
        Chart
            Ready to draw.
        """
        seabed = seabed_module.load()
        frame = geo.build(config).frame
        depth = seabed.depth_m

        band = np.digitize(depth, BANDS_M)
        water = np.array([_rgb(step) for step in colours.water], dtype=np.float32)
        tint = water[np.clip(band, 0, len(water) - 1)]
        tint[depth <= 0.0] = _rgb(colours.land)

        return Chart(
            extent=seabed.extent_m,
            tint=tint,
            coast=seabed.coast,
            isobaths=_contours(depth, seabed.cell_m),
            navigable=_contours(
                depth,
                seabed.cell_m,
                levels=(config.fleet.min_water_m,),
                minimum_m=6000.0,
            )[config.fleet.min_water_m],
            convergence_deg=_convergence(frame, config),
            anchors=_anchors(seabed.coast_distance_m, seabed.cell_m),
            origin=seabed.origin,
            bounds=(
                config.geo.lon_min,
                config.geo.lon_max,
                config.geo.lat_min,
                config.geo.lat_max,
            ),
            credit=(
                f"Bathymetry: {seabed.credit}"
                f"  ·  Projection: ETRS89 / TM35FIN ({seabed.metadata['projected_crs']})"
            ),
            colours=colours,
        )

    # ------------------------------------------------------------------ draw

    def basemap(self, axes: Axes, zorder: float = 0.0) -> None:
        """Draw the sea, the land and the depth contours.

        Parameters
        ----------
        axes : Axes
            Where to draw.
        zorder : float
            The bottom of the stack; everything is drawn just above it.
        """
        from matplotlib.collections import LineCollection

        width, height = self.extent
        axes.imshow(
            self.tint,
            extent=(0.0, width, 0.0, height),
            origin="lower",
            interpolation="bilinear",
            zorder=zorder,
        )

        # Isobaths, thin and unlabelled here: a frame of a film is read at a
        # glance and a page of numbers across the gulf is not glanceable. The
        # still, which is read at leisure, labels them.
        for metres, lines in self.isobaths.items():
            if not lines:
                continue
            axes.add_collection(
                LineCollection(
                    list(lines),
                    colors=self.colours.isobath,
                    linewidths=0.9 if metres in LABELLED_M else 0.45,
                    alpha=0.55,
                    zorder=zorder + 0.1,
                )
            )

        # The fleet's limit. Dashed, because it is a rule rather than a
        # feature: the sea does not change at fifteen metres, the fleet does.
        if self.navigable:
            axes.add_collection(
                LineCollection(
                    list(self.navigable),
                    colors=self.colours.isobath,
                    linewidths=0.8,
                    linestyles=(0, (5, 3)),
                    alpha=0.85,
                    zorder=zorder + 0.15,
                )
            )

        # The waterline. One weight, drawn over everything in the basemap,
        # because on a chart the coast is the one line that is never in doubt.
        axes.add_collection(
            LineCollection(
                list(self.coast),
                colors=self.colours.ink_soft,
                linewidths=0.7,
                zorder=zorder + 0.2,
            )
        )

    def label_isobaths(self, axes: Axes, zorder: float = 2.0) -> None:
        """Write the depth on the three isobaths worth navigating by.

        For stills. A film draws the lines and leaves them unlabelled.
        """
        width, height = self.extent
        for metres in LABELLED_M:
            for line in self.isobaths.get(metres, ()):
                if len(line) < 120:
                    continue
                point = line[len(line) // 2]
                if not (0.08 * width < point[0] < 0.92 * width):
                    continue
                if not (0.08 * height < point[1] < 0.92 * height):
                    continue
                axes.text(
                    float(point[0]),
                    float(point[1]),
                    f"{metres:.0f}",
                    color=self.colours.isobath,
                    fontsize=6.5,
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": self.colours.water[-1],
                        "edgecolor": "none",
                        "pad": 0.8,
                    },
                    zorder=zorder,
                )
                break

    # ------------------------------------------------------------- furniture

    def graticule(self, axes: Axes, frame: geo.Frame, zorder: float = 1.0) -> None:
        """Draw and label the meridians and parallels.

        The box is a rectangle in metres and therefore **not** one in degrees,
        so a meridian is not a vertical line on this page: it leans by the
        grid convergence. Drawing it as a column of ticks would be drawing a
        different projection from the one everything else is in, so each one
        is projected point by point and comes out curved, as it should.
        """
        from matplotlib.collections import LineCollection

        lon_min, lon_max, lat_min, lat_max = self.bounds
        # The labels belong on the edge of the VIEW. A cropped film keeps the
        # middle of the box, and a label pinned to the box would sit outside
        # the picture -- or worse, inside it but in the wrong place.
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        span_x, span_y = high_x - low_x, high_y - low_y

        lines: list[np.ndarray] = []
        for lon in self._ticks(lon_min, lon_max, LON_STEP_DEG):
            lat = np.linspace(lat_min, lat_max, 64)
            lines.append(self._local(frame, np.full_like(lat, lon), lat))
        for lat in self._ticks(lat_min, lat_max, LAT_STEP_DEG):
            lon = np.linspace(lon_min, lon_max, 64)
            lines.append(self._local(frame, lon, np.full_like(lon, lat)))

        axes.add_collection(
            LineCollection(
                lines,
                colors=self.colours.hairline,
                linewidths=0.5,
                linestyles=(0, (1, 4)),
                zorder=zorder,
            )
        )

        # The labels go on the neatline, where a chart puts them, which means
        # finding where each meridian actually crosses the edge of the box.
        for lon in self._ticks(lon_min, lon_max, LON_STEP_DEG):
            # Sampled past the corners, because the box is a rectangle in
            # metres and a meridian only touches its southern edge at one
            # point -- stopping at lat_min leaves most of them never crossing
            # it, and so never labelled.
            lat = np.linspace(lat_min - 0.25, lat_max + 0.25, 400)
            line = self._local(frame, np.full_like(lat, lon), lat)
            crossing = self._crosses(line, axis=1, value=low_y)
            if crossing is None or not low_x <= crossing <= high_x:
                continue
            axes.text(
                crossing,
                low_y - LABEL_PAD * span_y,
                _dms(lon, "lon"),
                color=self.colours.ink_soft,
                fontsize=7.5,
                ha="center",
                va="top",
                zorder=zorder,
            )
        for lat in self._ticks(lat_min, lat_max, LAT_STEP_DEG):
            lon = np.linspace(lon_min - 0.5, lon_max + 0.5, 400)
            line = self._local(frame, lon, np.full_like(lon, lat))
            crossing = self._crosses(line, axis=0, value=low_x)
            if crossing is None or not low_y <= crossing <= high_y:
                continue
            axes.text(
                low_x - LABEL_PAD * span_x * 0.5,
                crossing,
                _dms(lat, "lat"),
                color=self.colours.ink_soft,
                fontsize=7.5,
                ha="right",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                zorder=zorder,
            )

    def neatline(self, axes: Axes, zorder: float = 20.0) -> None:
        """Draw the double rule that closes the chart.

        Two weights, the outer heavier, which is what tells the eye that the
        map has an edge rather than merely running out of pixels.

        Drawn against the axes limits, not the box: the films trim the empty
        land to fill the frame, and a neatline on the box would leave its top
        and bottom somewhere off screen with two verticals floating.
        """
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        span_y = high_y - low_y
        for inset, weight in ((0.0, 1.6), (0.004 * span_y, 0.5)):
            axes.plot(
                [
                    low_x + inset,
                    high_x - inset,
                    high_x - inset,
                    low_x + inset,
                    low_x + inset,
                ],
                [
                    low_y + inset,
                    low_y + inset,
                    high_y - inset,
                    high_y - inset,
                    low_y + inset,
                ],
                color=self.colours.ink,
                linewidth=weight,
                solid_joinstyle="miter",
                clip_on=False,
                zorder=zorder,
            )

    def scale_bar(self, axes: Axes, anchor: tuple[float, float], zorder: float = 21.0) -> None:
        """Draw an alternating bar scale, the kind that is read rather than measured.

        Parameters
        ----------
        axes : Axes
            Where to draw.
        anchor : tuple of float
            Left end, in axes fraction.
        zorder : float
            Stack position.
        """
        from matplotlib.patches import Rectangle

        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        span_x, span_y = high_x - low_x, high_y - low_y

        segment_km = 5.0
        segments = 4
        left = low_x + anchor[0] * span_x
        bottom = low_y + anchor[1] * span_y
        thickness = 0.008 * span_y
        step = segment_km * 1000.0

        for index in range(segments):
            axes.add_patch(
                Rectangle(
                    (left + index * step, bottom),
                    step,
                    thickness,
                    facecolor=self.colours.ink if index % 2 == 0 else self.colours.surface,
                    edgecolor=self.colours.ink,
                    linewidth=0.6,
                    zorder=zorder,
                )
            )
        for index in range(segments + 1):
            axes.text(
                left + index * step,
                bottom + thickness * 1.6,
                f"{index * segment_km:.0f}",
                color=self.colours.ink_soft,
                fontsize=6.5,
                ha="center",
                va="bottom",
                zorder=zorder,
            )
        axes.text(
            left + segments * step + 0.006 * span_x,
            bottom + thickness * 0.5,
            "km",
            color=self.colours.ink_soft,
            fontsize=6.5,
            ha="left",
            va="center",
            zorder=zorder,
        )

    def north_arrow(self, axes: Axes, anchor: tuple[float, float], zorder: float = 21.0) -> None:
        """Draw true north, leaning off grid north by the convergence.

        A north arrow drawn straight up a transverse Mercator is wrong
        everywhere except its own central meridian. This box sits two degrees
        west of TM35FIN's, which puts true north about 1.7 degrees off the
        top of the page -- small, and exactly the sort of small that a reader
        who knows charts will check.
        """
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        span_x, span_y = high_x - low_x, high_y - low_y
        x = low_x + anchor[0] * span_x
        y = low_y + anchor[1] * span_y
        length = 0.045 * span_y
        tilt = np.radians(self.convergence_deg)

        tip = (x + length * np.sin(tilt), y + length * np.cos(tilt))
        axes.annotate(
            "",
            xy=tip,
            xytext=(x, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": self.colours.ink,
                "linewidth": 1.1,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=zorder,
        )
        axes.text(
            tip[0],
            tip[1] + 0.012 * span_y,
            "N",
            color=self.colours.ink,
            fontsize=8.5,
            ha="center",
            va="bottom",
            zorder=zorder,
        )

    def credits(self, axes: Axes, zorder: float = 21.0) -> None:
        """Print where the seabed came from and what it is projected in.

        Required by the licence, and the thing that separates a chart from a
        picture of one: a reader can go and check.
        """
        high_x = axes.get_xlim()[1]
        low_y, high_y = axes.get_ylim()
        axes.text(
            high_x,
            low_y - LABEL_PAD * (high_y - low_y) * 2.0,
            self.credit,
            color=self.colours.ink_soft,
            fontsize=6.0,
            ha="right",
            va="top",
            zorder=zorder,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _ticks(low: float, high: float, step: float) -> np.ndarray:
        """Return the graticule values strictly inside a span."""
        first = np.ceil(low / step) * step
        return np.arange(first, high + 1e-9, step)

    def _local(self, frame: geo.Frame, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Project degrees into the box's own metres."""
        projected: np.ndarray = frame.to_xy(lon, lat) - self.origin
        return projected

    @staticmethod
    def _crosses(line: np.ndarray, axis: int, value: float) -> float | None:
        """Return where a projected graticule line crosses an edge of the box."""
        other = 1 - axis
        side = line[:, axis] - value
        for index in range(len(side) - 1):
            if side[index] == 0.0 or side[index] * side[index + 1] < 0.0:
                weight = abs(side[index]) / (abs(side[index]) + abs(side[index + 1]))
                return float(
                    line[index, other] + weight * (line[index + 1, other] - line[index, other])
                )
        return None


def _rgb(colour: str) -> np.ndarray:
    """Return a hex colour as a float RGB triple."""
    raw = colour.lstrip("#")
    return np.array([int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _blur(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Return a Gaussian-smoothed copy, separably and without scipy.

    Only ever applied to the drawing copy. The simulation gathers from the
    raster as measured.
    """
    radius = int(np.ceil(3.0 * sigma_cells))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(field.astype(np.float64), radius, mode="edge")
    rows = np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), 1, padded)
    return np.apply_along_axis(lambda column: np.convolve(column, kernel, "valid"), 0, rows)


def _length(line: np.ndarray) -> float:
    """Return the length of a polyline in metres."""
    return float(np.hypot(*np.diff(line, axis=0).T).sum())


def _contours(
    depth: np.ndarray,
    cell_m: float,
    levels: tuple[float, ...] = ISOBATHS_M,
    minimum_m: float = MIN_CONTOUR_M,
) -> dict[float, tuple[np.ndarray, ...]]:
    """Return generalised isobaths as polylines in local metres, computed once.

    Through contourpy rather than ``pyplot.contour``: a module that draws has
    no business creating a figure to compute with, and on a machine without a
    display the default backend turns that into a Tk error at import time.
    """
    from contourpy import contour_generator

    smooth = _blur(depth, CONTOUR_SMOOTH_CELLS)
    rows, columns = depth.shape
    generate = contour_generator(
        x=(np.arange(columns) + 0.5) * cell_m,
        y=(np.arange(rows) + 0.5) * cell_m,
        z=smooth,
        line_type="SeparateCode",
    )
    result: dict[float, tuple[np.ndarray, ...]] = {}
    for metres in levels:
        lines: list[np.ndarray] = []
        for vertices in generate.lines(float(metres))[0]:
            line = np.asarray(vertices, dtype=np.float32)
            if len(line) >= 8 and _length(line) >= minimum_m:
                lines.append(line)
        result[metres] = tuple(lines)
    return result


def _anchors(distance: np.ndarray, cell_m: float) -> dict[str, tuple[float, float]]:
    """Return where each country's name sits, in local metres.

    A fixed **margin inland of its own coast**, which is where a chart has
    always put a country name: close enough that the name obviously belongs
    to the shore below it, far enough that it is not sitting in the water.

    Two rules were tried before this one and both were worse. Written down,
    ESTONIA landed in a bay. Taken as the deepest point inland -- the minimum
    of the signed distance field -- it landed at the bottom edge of the box,
    on top of the scale bar, because the land goes on getting more inland
    past the frame and the deepest point of a cropped country is its corner.

    So the target is a distance, not an extremum: the land cell whose
    inland-ness is nearest ``INLAND_M``, inside a window that keeps each name
    clear of the furniture that owns that part of the frame.
    """
    rows, columns = distance.shape

    def band(low: float, high: float, size: int) -> slice:
        return slice(int(low * size), int(high * size))

    anchors: dict[str, tuple[float, float]] = {}
    # The two sides look in different places, because the two sides of the
    # frame are not equally free. The legend owns the bottom left and the bar
    # scale the bottom centre, so ESTONIA is sought east of both; FINLAND
    # keeps the west, where the single-panel crop is least likely to reach.
    for name, down, across in (
        ("north", (0.72, 0.99), (0.05, 0.32)),
        ("south", (0.01, 0.28), (0.74, 0.97)),
    ):
        rows_of = band(*down, rows)
        span = band(*across, columns)
        window = distance[rows_of, span]
        ashore = window < 0.0
        if not ashore.any():
            continue
        # Nearest to the wanted margin, among land cells only.
        score = np.where(ashore, np.abs(window + INLAND_M), np.inf)
        row, column = np.unravel_index(int(score.argmin()), score.shape)
        anchors[name] = (
            float((column + span.start + 0.5) * cell_m),
            float((row + rows_of.start + 0.5) * cell_m),
        )
    return anchors


def _convergence(frame: geo.Frame, config: Config) -> float:
    """Return the angle from grid north to true north at the box centre.

    Measured off the projection rather than taken from the closed form, so it
    cannot disagree with the transformer that places everything else.
    """
    lon = 0.5 * (config.geo.lon_min + config.geo.lon_max)
    lat = 0.5 * (config.geo.lat_min + config.geo.lat_max)
    here = frame.to_xy(lon, lat)
    north = frame.to_xy(lon, lat + 0.05)
    delta = north - here
    return float(np.degrees(np.arctan2(float(delta[0]), float(delta[1]))))
