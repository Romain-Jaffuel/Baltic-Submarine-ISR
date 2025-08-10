import json
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from config_area import AREA_POLY

def poly_bbox(poly):
    lats = [p[0] for p in poly]; lons = [p[1] for p in poly]
    return min(lats), max(lats), min(lons), max(lons)

def main():
    Path("out").mkdir(parents=True, exist_ok=True)
    tracks = json.load(open("data/lura_tracks.json","r",encoding="utf-8"))
    reports = json.load(open("data/lura_reports.json","r",encoding="utf-8"))

    tiler = OSM(); crs_tiles = tiler.crs
    lat_min, lat_max, lon_min, lon_max = poly_bbox(AREA_POLY)
    mlat = (lat_max - lat_min) * 0.4
    mlon = (lon_max - lon_min) * 0.4
    extent = [lon_min-mlon, lon_max+mlon, lat_min-mlat, lat_max+mlat]

    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection=crs_tiles)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiler, 8)

    # corridor
    xs = [p[1] for p in AREA_POLY] + [AREA_POLY[0][1]]
    ys = [p[0] for p in AREA_POLY] + [AREA_POLY[0][0]]
    ax.plot(xs, ys, color="#222", lw=1.3, alpha=0.9, transform=ccrs.PlateCarree())
    ax.fill(xs, ys, facecolor="#2c7fb8", alpha=0.06, transform=ccrs.PlateCarree())

    # tracks grouped by LURA id
    by_id = {}
    for r in tracks:
        by_id.setdefault(r["lura_id"], []).append((r["lon"], r["lat"]))
    for pts in by_id.values():
        ax.plot([p[0] for p in pts],[p[1] for p in pts], lw=0.8, alpha=0.7,
                color="#00bcd4", transform=ccrs.PlateCarree())

    # reports
    if reports:
        ax.scatter([r["lon"] for r in reports], [r["lat"] for r in reports],
                   s=6, alpha=0.6, transform=ccrs.PlateCarree(), label="reports", zorder=5)

    ax.set_title("LURA patrols & reports – Gulf of Finland")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("out/figs/lura_patrols.png", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    main()