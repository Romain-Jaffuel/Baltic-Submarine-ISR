"""Cut README-sized loops out of the two films.

GitHub plays an animated GIF in a README on its own, with no click and no
player, which is the only way a repository page shows motion. It also serves
the whole file before showing a single frame, so the size is the feature: a
fifteen-megabyte hero is a blank rectangle for the first few seconds on
anything but a fast connection.

So this targets a budget rather than hoping for one. It encodes, measures, and
if the file is over budget it drops quality a step and tries again -- frame
rate first, because the maps read fine at eight frames a second, then colours,
and width last, since width is what makes the two panels legible at all.

Writes into `assets/`, which is version controlled, because a README cannot
show a file that lives in an ignored directory.

    uv run python scripts/gif.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import imageio_ffmpeg

ASSETS = Path("assets")
BUDGET_BYTES = 4_500_000

# Start part way in. Both films open with the fleet leaving its bases, which
# looks much the same either side; the difference is established once a day of
# simulated traffic has gone by.
#
# 11.0 s is measured, not chosen. **The film is 30 s for 48 h of simulation,
# so one second is 1.6 h** -- it keeps one trace frame in four, which is the
# arithmetic to redo rather than the frame numbers. The twelve seconds from
# here cover 17.6 h to 36.8 h and hold four separate suspects with no gap
# between them: a red point is on screen in every second from 11 to 21, and
# each draws vehicles inside the three kilometres the escort is paid for --
# four on the first at 20.8 h, two on the second at the same moment, one on
# the third from 27.2 h and one on the last at 33.6 h.
#
# The counts here are inside the PAID radius, not inside hearing. Earlier cuts
# quoted six and nine drones on a vessel; those were counted at 6 km, which
# was the old paid radius and is now only the range at which a vehicle can
# hear. Three kilometres is what the reward asks for and what the eye reads as
# an escort.
#
# An earlier cut began at 12.0 s and caught a dispersal, so the loop in the
# README showed none of the behaviour it was there to show. Re-derive this
# after any retraining -- the window moves with the seed and with the policy,
# and it has moved on every one of them so far.
CUTS = ((Path("out/videos/demonstration.mp4"), ASSETS / "demonstration.gif", 11.0, 12.0),)

# Tried in order, stopping at the first that fits. Width moves last.
LADDER = (
    {"width": 860, "fps": 10, "colours": 128},
    {"width": 860, "fps": 8, "colours": 96},
    {"width": 760, "fps": 8, "colours": 64},
    {"width": 680, "fps": 7, "colours": 48},
    {"width": 600, "fps": 6, "colours": 32},
)


def encode(
    source: Path,
    destination: Path,
    start_s: float,
    seconds: float,
    width: int,
    fps: int,
    colours: int,
) -> int:
    """Write one GIF with a two-pass palette and return its size in bytes."""
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    palette = destination.with_suffix(".palette.png")
    common = ["-ss", str(start_s), "-t", str(seconds), "-i", str(source)]
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(
        [
            binary,
            "-y",
            "-loglevel",
            "error",
            *common,
            "-vf",
            f"{scale},palettegen=stats_mode=diff:max_colors={colours}",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            binary,
            "-y",
            "-loglevel",
            "error",
            *common,
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return destination.stat().st_size


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"budget {BUDGET_BYTES / 1e6:.1f} MB each\n")

    for source, destination, start_s, seconds in CUTS:
        if not source.is_file():
            raise SystemExit(f"{source} missing: run scripts/films.py first")
        print(f"{destination.name}  ({seconds:.0f} s from {start_s:.0f} s)")
        for step in LADDER:
            started = time.perf_counter()
            size = encode(source, destination, start_s, seconds, **step)
            fits = size <= BUDGET_BYTES
            print(
                f"  {step['width']}px {step['fps']}fps {step['colours']} colours -> "
                f"{size / 1e6:5.1f} MB  ({time.perf_counter() - started:.0f} s)  "
                + ("within budget" if fits else "too big, stepping down")
            )
            if fits:
                break
        else:
            print("  still over budget; shorten the cut")
        print()
