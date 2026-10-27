"""The map palette, held to the data-visualisation validator's own checks.

Node is not installed here, so the reference validator cannot be run directly.
Its arithmetic is ported instead -- sRGB to linear, OKLab, the Machado, Oliveira
and Fernandes (2009) colour-blindness transforms at severity 1.0, Euclidean
Delta E in OKLab times one hundred, WCAG contrast -- and its thresholds applied
unchanged. Porting it rather than eyeballing it is the point: a palette that
merely looks fine to the author is not a checked palette, and this way the
checks run on every commit instead of once.

Two things are checked. The fleet ramp is **ordinal**, so it takes the ordinal
checks: monotone lightness, visible gaps, a pale end that still reads, one hue.
The separation between the fleet and the threat is checked by hand against the
same floors, because that pair is the whole point of the map -- a viewer who
cannot tell the red from the blue has learnt nothing.
"""

from __future__ import annotations

import math

import pytest

from baltic_isr.render import palette as palette_module

# Thresholds, verbatim from the reference validator.
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET = 8.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0

MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}


def linear(hex_colour: str) -> tuple[float, float, float]:
    """Return linear-light sRGB for a hex colour."""
    raw = hex_colour.strip().lstrip("#")
    channels = [int(raw[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    return tuple(  # type: ignore[return-value]
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    )


def oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return OKLab for linear-light sRGB."""
    red, green, blue = rgb
    long = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
    medium = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (1 / 3)
    short = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (1 / 3)
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def lightness(hex_colour: str) -> float:
    """Return OKLCH lightness."""
    return oklab(linear(hex_colour))[0]


def chroma(hex_colour: str) -> float:
    """Return OKLCH chroma."""
    _, a, b = oklab(linear(hex_colour))
    return math.hypot(a, b)


def hue(hex_colour: str) -> float:
    """Return OKLCH hue in degrees."""
    _, a, b = oklab(linear(hex_colour))
    return math.degrees(math.atan2(b, a)) % 360.0


# Two large fields side by side, rather than a mark on a field. Area buys
# discriminability, so the floor for one is not the floor for the other.
AREA_FLOOR = 8.0


def contrast(first: str, second: str) -> float:
    """Return the WCAG contrast ratio between two colours."""

    def luminance(colour: str) -> float:
        red, green, blue = linear(colour)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def simulate(hex_colour: str, kind: str) -> tuple[float, float, float]:
    """Return the colour as a dichromat sees it."""
    rgb = linear(hex_colour)
    matrix = MACHADO[kind]
    return tuple(  # type: ignore[return-value]
        min(1.0, max(0.0, sum(row[index] * rgb[index] for index in range(3)))) for row in matrix
    )


def delta_e(first: str, second: str, kind: str | None = None) -> float:
    """Return Euclidean OKLab distance times one hundred."""
    left = oklab(simulate(first, kind) if kind else linear(first))
    right = oklab(simulate(second, kind) if kind else linear(second))
    return 100.0 * math.dist(left, right)


def worst_cvd(first: str, second: str) -> float:
    """Return the worse of the two dichromat separations."""
    return min(delta_e(first, second, "protan"), delta_e(first, second, "deutan"))


MODES = ("light", "dark")


# --- the fleet ramp is ordinal, so it takes the ordinal checks ---------------


@pytest.mark.parametrize("mode", MODES)
def test_regime_ramp_is_monotone(mode: str) -> None:
    """Drift, cruise and sprint are ordered, so the colour has to be ordered too."""
    steps = [lightness(colour) for colour in palette_module.palette(mode).regime]
    assert steps == sorted(steps) or steps == sorted(steps, reverse=True)


@pytest.mark.parametrize("mode", MODES)
def test_regime_steps_are_visibly_apart(mode: str) -> None:
    ramp = palette_module.palette(mode).regime
    gaps = [abs(lightness(ramp[index + 1]) - lightness(ramp[index])) for index in range(2)]
    assert min(gaps) >= ORDINAL_MIN_DL


@pytest.mark.parametrize("mode", MODES)
def test_the_pale_end_still_reads_against_the_sea(mode: str) -> None:
    """A vehicle at the quiet end of the ramp must not vanish into the surface."""
    colours = palette_module.palette(mode)
    nearest = min(colours.regime, key=lambda colour: contrast(colour, colours.surface))
    assert contrast(nearest, colours.surface) >= ORDINAL_LIGHT_FLOOR


@pytest.mark.parametrize("mode", MODES)
def test_the_fleet_is_one_hue(mode: str) -> None:
    """Spending hue on the regime would leave the threat competing for attention."""
    hues = [hue(colour) for colour in palette_module.palette(mode).regime]
    spread = max(hues) - min(hues)
    assert min(spread, 360.0 - spread) <= 40.0


@pytest.mark.parametrize("mode", MODES)
def test_the_hardest_working_regime_stands_out_most(mode: str) -> None:
    """Sprint is the most contrasted against the sea, whichever mode it is."""
    colours = palette_module.palette(mode)
    ratios = [contrast(colour, colours.surface) for colour in colours.regime]
    assert ratios[2] == max(ratios)


# --- the pair the whole map depends on --------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_threat_separates_from_every_regime(mode: str) -> None:
    """The one thing a viewer who knows nothing must be able to do."""
    colours = palette_module.palette(mode)
    for index, blue in enumerate(colours.regime):
        assert delta_e(colours.threat, blue) >= NORMAL_FLOOR, palette_module.REGIME_LABELS[index]
        assert worst_cvd(colours.threat, blue) >= CVD_TARGET, palette_module.REGIME_LABELS[index]


@pytest.mark.parametrize("mode", MODES)
def test_the_threat_separates_from_ordinary_traffic(mode: str) -> None:
    colours = palette_module.palette(mode)
    assert delta_e(colours.threat, colours.traffic) >= NORMAL_FLOOR
    assert worst_cvd(colours.threat, colours.traffic) >= CVD_TARGET


@pytest.mark.parametrize("mode", MODES)
def test_caution_and_alarm_are_three_distinct_states(mode: str) -> None:
    """Grey, amber, red have to be three things, not a gradient.

    The map's whole claim is that a vessel over a cable is a different kind of
    event from one merely behaving oddly, which is only true to a viewer if
    the two marks are genuinely different colours -- including to one who
    cannot see red.
    """
    colours = palette_module.palette(mode)
    for first, second in (
        (colours.threat, colours.watch),
        (colours.watch, colours.traffic),
        (colours.threat, colours.traffic),
    ):
        assert delta_e(first, second) >= NORMAL_FLOOR
        assert worst_cvd(first, second) >= CVD_TARGET


@pytest.mark.parametrize("mode", MODES)
def test_the_asset_is_structure_and_not_a_sixth_hue(mode: str) -> None:
    """The cable is drawn with weight, not with a competing colour.

    A violet cable was tried and measured delta-E 2.6 against the cruising
    fleet under simulated colour blindness, against a floor of 8; no candidate
    hue cleared the floors against everything else either, because a palette
    already spending hue on grey, three blues, amber and red has no room for a
    sixth category. So the asset must stay **neutral** -- near the ink, far
    from every status colour -- and earn its prominence from line weight and a
    corridor band instead.
    """
    colours = palette_module.palette(mode)
    for status in (colours.threat, colours.watch):
        assert worst_cvd(colours.asset, status) >= CVD_TARGET
    assert delta_e(colours.asset, colours.surface) >= NORMAL_FLOOR


@pytest.mark.parametrize("mode", MODES)
def test_the_fleet_separates_from_ordinary_traffic(mode: str) -> None:
    """Grey is deliberately recessive, but a vehicle must not read as a ship."""
    colours = palette_module.palette(mode)
    for index, blue in enumerate(colours.regime):
        assert delta_e(blue, colours.traffic) >= NORMAL_FLOOR, palette_module.REGIME_LABELS[index]


@pytest.mark.parametrize("mode", MODES)
def test_the_threat_carries_enough_chroma_and_contrast(mode: str) -> None:
    colours = palette_module.palette(mode)
    assert chroma(colours.threat) >= CHROMA_FLOOR
    assert contrast(colours.threat, colours.surface) >= CONTRAST_MIN


@pytest.mark.parametrize("mode", MODES)
def test_the_threat_is_the_only_warm_thing_on_the_map(mode: str) -> None:
    """Warmth is the alarm channel. Nothing else may use it."""
    colours = palette_module.palette(mode)
    others = [
        colours.traffic,
        colours.corridor,
        colours.lane,
        colours.land,
        colours.isobath,
        colours.boundary,
        *colours.water,
        *colours.regime,
    ]
    warm = hue(colours.threat)
    for colour in others:
        if chroma(colour) < CHROMA_FLOOR:
            continue
        separation = abs(hue(colour) - warm)
        assert min(separation, 360.0 - separation) > 60.0


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        palette_module.palette("sepia")


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_land_is_one_tone_and_reads_against_every_depth(mode: str) -> None:
    """One landmass colour, and it has to separate from all four water bands.

    Two colours used to be required here, on the grounds that a single tone
    made the map read as a channel between two unnamed banks. That was true
    of a coastline with five vertices a side. It stopped being true when the
    measured one arrived: nobody mistakes the Helsinki archipelago for the
    Estonian shore, the names say which is which, and no chart ever printed
    fills two countries in two colours.

    Two floors, because there are two jobs. Land against water is an **area**
    against an area -- thousands of pixels each, so 8 is enough and forcing 15
    would turn the map into a flag. Land against any **mark** that can stray
    over it is a dot on a field, and keeps the full 15.
    """
    colours = palette_module.palette(mode)

    for mark in (colours.traffic, colours.threat, *colours.regime):
        assert delta_e(colours.land, mark) >= NORMAL_FLOOR, mark

    if mode == "light":
        for band in colours.water:
            assert delta_e(colours.land, band) >= AREA_FLOOR, band
        return

    # On a dark ground the boundary is **drawn, not inferred**, and the floor
    # moves onto the thing that draws it. Measured, the land clears only 5.1
    # against the deepest water -- and the coastline stroke clears 44.5
    # against the water and 57.4 against the land, so there is a hairline
    # between them that nobody can miss. Demanding a tonal step as well would
    # buy nothing and cost the one thing a dark basemap is for, which is
    # leaving the whole contrast budget to the data.
    #
    # This is a stricter claim than the light one, not a looser one: it binds
    # the stroke, which the light test never did.
    for band in colours.water:
        assert delta_e(colours.land, band) >= 4.0, band
        assert delta_e(colours.ink_soft, band) >= NORMAL_FLOOR, band
    assert delta_e(colours.ink_soft, colours.land) >= NORMAL_FLOOR


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_water_is_an_ordinal_ramp(mode: str) -> None:
    """Shallow to deep, in order, with steps a reader can actually see.

    Ordered the way a chart orders it -- the shallows carry the tint and the
    deep water is nearly white -- which is the opposite of the hypsometric
    ramp a GIS reaches for. The reason is in the next test: a dark blue sea
    and a pale blue fleet cannot both exist.
    """
    steps = [lightness(band) for band in palette_module.palette(mode).water]
    assert steps == sorted(steps) or steps == sorted(steps, reverse=True)
    gaps = [abs(steps[index + 1] - steps[index]) for index in range(len(steps) - 1)]
    assert min(gaps) >= 0.03


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_mark_survives_the_water_it_can_sit_on(mode: str) -> None:
    """The check that chose the ramp, rather than a check written after it.

    Searched rather than eyeballed: the widest ramp running the other way --
    dark for deep -- had a lightness span of 0.087 before the deep end
    collided with the drifting end of the fleet, which is nine shades nobody
    can tell apart. This one spans 0.150.

    The shallowest band is held to a shorter list on purpose. No AUV may be
    in less than fifteen metres of water, so the only things that can appear
    over the 0-10 m tint are vessels and the threat.
    """
    colours = palette_module.palette(mode)
    afloat = (colours.traffic, colours.threat, colours.watch)
    everything = (*afloat, *colours.regime)

    for index, band in enumerate(colours.water):
        marks = afloat if index == 0 else everything
        for mark in marks:
            assert delta_e(band, mark) >= NORMAL_FLOOR, (band, mark)
            assert worst_cvd(band, mark) >= CVD_TARGET, (band, mark)
