"""The colours of the map, and the one rule behind them.

**Hue carries the story, lightness carries the physics.** A viewer who knows
nothing about the project has to read the map in one pass, so hue is spent on
three states and nothing else:

```
grey   traffic behaving itself      recessive, it is the backdrop
blue   an AUV                       one hue, so the fleet reads as one thing
red    a contact that has deviated  reserved status, never used for anything else
```

Within blue, *lightness* carries the variable the project is actually about --
which regime a vehicle is in. That is an **ordinal** ramp, not a categorical
one: drift, cruise and sprint are ordered, and the reader should see the order
in the colour. Pale means idling, dark means running hard.

Spending hue on the regime instead would have put six hues on a map of two
hundred marks, and the red would have stopped being the only warm thing on
screen. The regime also gets a second channel -- trail length -- so it never
depends on lightness alone.

Every value here comes from the documented reference palette, and
``tests/unit/test_palette.py`` re-runs the validator's own checks over them:
monotone lightness, a visible gap between steps, a pale end that still reads
against the surface, one hue, and enough separation between the fleet and the
threat under simulated colour blindness. Nothing here is eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- surfaces and chrome, from the reference palette -------------------------

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"

INK_PRIMARY = {"light": "#0b0b0b", "dark": "#ffffff"}
INK_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
INK_MUTED = {"light": "#898781", "dark": "#898781"}
HAIRLINE = {"light": "#e1e0d9", "dark": "#2c2c2a"}


@dataclass(frozen=True)
class Palette:
    """Every colour the map uses, for one mode.

    Attributes
    ----------
    surface : str
        The chart surface the validator measures against, and the paper the
        map sits on.
    water : str
        The sea. Separate from `surface` because a near-white sea against a
        beige shore does not read as water: the first viewers of the films
        could not tell which side of the coastline they were looking at.
        Pale enough that two hundred marks still sit on top of it, blue
        enough that nobody has to be told what it is.
    land : str
        Finland and Estonia, in one tone.

        One, not two. Colouring the two countries differently was how the
        caricature stopped the map reading as a channel between two unnamed
        banks -- with a five-vertex shore there was nothing else to tell them
        apart. Measured, the coastline does that work by itself: nobody who
        has seen the Helsinki archipelago mistakes it for the Estonian shore,
        and no chart ever printed fills two countries in two colours. The
        names and the maritime boundary say which is which.

        It clears 8 against every water band as an **area**, and the full 15
        against every **mark** that can stray over it, because that is a dot
        on a field and it has to be found.
    water : tuple of str
        The sea in four steps, shallow to deep, by the chart convention:
        the shallows carry the tint and the deep water is nearly white.

        This is the opposite of the hypsometric ramp a GIS would reach for,
        and it is the only version that works here. Measured against the
        fleet, the widest blue ramp that still clears every mark had a
        lightness span of 0.087 -- nine steps nobody could tell apart --
        because a dark blue sea collides with a pale blue vehicle. Inverted,
        the span is 0.150 and every band clears its marks, helped by a fact
        about the fleet rather than about colour: no AUV may be in less than
        fifteen metres of water, so the darkest band is water none of them
        can occupy.
    isobath : str
        Depth contours. The channel that actually carries depth here.
    corridor : str
        Seabed cable and pipe corridors.
    lane : str
        Traffic separation lanes.
    traffic : str
        A vessel behaving itself. Deliberately the muted ink: it is context.
    threat : str
        A vessel that is **over a cable and not behaving** -- a target, not a
        doubt. Status critical, reserved, never used for anything else.
    watch : str
        A contact the fleet has questions about. Amber, because the first
        viewer of this map read a red ring as "something serious here" and it
        meant "the fleet is unsure", which is the opposite of serious. Caution
        and alarm are different states and traffic lights have taught everyone
        the difference already.
    asset : str
        The cables and the chokepoints on them: the thing being protected.

        Deliberately **not a sixth hue**. A violet cable was tried and
        measured at delta-E 2.6 against the cruising fleet under simulated
        colour blindness, against a floor of 8 -- and no candidate hue
        (teal, green, magenta, brown, slate) cleared the floors against every
        other colour, because a map already spending hue on grey, three blues,
        amber and red has no room for another category.

        That is the signal that colour is the wrong tool here. A cable is
        structure, not a state: it gets **weight and form** -- a continuous
        line with a corridor band, where every other mark is a dot -- and the
        hue budget stays spent on what things are doing.
    regime : tuple of str
        Drift, cruise, sprint. An ordinal blue ramp, most contrast at the
        hardest-working end.
    prediction : str
        The uncertainty ellipse around a held contact.
    ink, ink_soft, hairline : str
        Text and rules.
    """

    surface: str
    water: tuple[str, str, str, str]
    land: str
    isobath: str
    boundary: str
    corridor: str
    lane: str
    traffic: str
    threat: str
    watch: str
    asset: str
    regime: tuple[str, str, str]
    prediction: str
    ink: str
    ink_soft: str
    hairline: str


# Blue steps 250 / 450 / 650 of the reference sequential ramp. On light the
# hardest-working regime is the darkest; on dark it is the lightest, because
# what the eye reads is distance from the surface, not absolute lightness.
LIGHT = Palette(
    surface=SURFACE_LIGHT,
    # 0-15 m, 15-30 m, 30-60 m, over 60 m. Solved rather than picked: see
    # tests/unit/test_palette.py, which re-runs the separations.
    water=("#a6d2e7", "#c7e3f1", "#e2f0f7", "#f9fcff"),
    land="#e0ceb7",
    isobath="#8fb4c9",
    boundary="#6f7f8c",
    corridor="#898781",
    lane="#e1e0d9",
    traffic=INK_MUTED["light"],
    threat="#d03b3b",
    watch="#f59e0b",
    asset="#3d4652",
    regime=("#86b6ef", "#2a78d6", "#104281"),
    prediction="#b7d3f6",
    ink=INK_PRIMARY["light"],
    ink_soft=INK_SECONDARY["light"],
    hairline=HAIRLINE["light"],
)

DARK = Palette(
    # The dark basemap idiom, which is what every legible maritime traffic
    # picture uses -- Kepler.gl, MarineTraffic, the AIS density maps. The
    # reason is mechanical rather than stylistic: on a dark ground the
    # land/water boundary needs almost no contrast to read, bright marks
    # separate from the background without an outline, and dozens of
    # translucent coverage discs add up into density instead of washing each
    # other out. The light map needed halos and a contrast floor for every
    # label to fight exactly those problems.
    #
    # So the ground is nearly colourless and the whole saturation budget goes
    # to the data: the fleet glows, the threat glows, and the two countries
    # are told apart by a hint of warm against a hint of cool.
    surface="#0b0f14",
    # Searched under the same floors as the light ramp, against the dark
    # marks. It is darker than it looks necessary: the binding constraint is
    # the pale end of the fleet, which on a dark ground is nearly white.
    water=("#23393f", "#1a2b2f", "#121d20", "#0a1112"),
    # Warm, and that is not decoration: the water is cool by construction, so
    # a cool land sits on top of it (1.9 against the deepest band, measured)
    # and a warm one separates. Charts have put brown beside blue for the same
    # reason for as long as they have been printed in colour.
    land="#22110c",
    isobath="#2f4b63",
    boundary="#55646f",
    corridor="#7e8794",
    lane="#1b2531",
    traffic="#9a968c",
    threat="#ff4d4d",
    watch="#f2ac44",
    asset="#cfdbe8",
    regime=("#2f74c8", "#4f9dfb", "#9ccfff"),
    prediction="#2d5f9e",
    ink="#ffffff",
    ink_soft="#aeb6c0",
    hairline="#232c38",
)

REGIME_LABELS = ("drift", "cruise", "sprint")


def palette(mode: str = "light") -> Palette:
    """Return the palette for a mode.

    Parameters
    ----------
    mode : str
        ``"light"`` or ``"dark"``.

    Returns
    -------
    Palette
        The colours.
    """
    if mode not in ("light", "dark"):
        raise ValueError(f"unknown mode {mode!r}")
    return LIGHT if mode == "light" else DARK
