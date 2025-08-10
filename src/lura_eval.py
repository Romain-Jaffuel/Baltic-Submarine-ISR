import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

from config_area import AREA_POLY, point_in_poly, _lerp, side_west, side_east

GRID_NX, GRID_NY = 12, 8

def grid_cells(poly, nx, ny):
    def bilinear(s,u):
        w0,w1 = side_west(poly); e0,e1 = side_east(poly)
        pw = _lerp(w0, w1, s);    pe = _lerp(e0, e1, s)
        return _lerp(pw, pe, u)
    out = []
    for ix in range(nx):
        s0,s1 = ix/nx, (ix+1)/nx
        for iy in range(ny):
            u0,u1 = iy/ny, (iy+1)/ny
            out.append([bilinear(s0,u0), bilinear(s1,u0),
                        bilinear(s1,u1), bilinear(s0,u1)])
    return out

def point_in_quad(lat, lon, quad):
    # réutilise ray casting
    xs = [p[1] for p in quad]; ys = [p[0] for p in quad]
    xs, ys = xs + [xs[0]], ys + [ys[0]]
    inside = False
    for i in range(4):
        x1,y1 = xs[i],ys[i]; x2,y2 = xs[i+1],ys[i+1]
        if (y1 > lat) != (y2 > lat):
            xin = (x2 - x1)*(lat - y1)/(y2 - y1 + 1e-12) + x1
            if lon < xin: inside = not inside
    return inside

def main():
    Path("out").mkdir(parents=True, exist_ok=True)
    ev = pd.read_json("data/events_simulated.json")
    rp = pd.read_json("data/lura_reports.json")

    # inside corridor
    ev = ev[ev.apply(lambda r: point_in_poly(r["lat"], r["lon"], AREA_POLY), axis=1)].copy()

    # recall (track-level)
    seen_tracks = set(rp["target_track_id"].dropna().unique())
    truth_tracks = set(ev["track_id"].unique())
    by_cls = ev.groupby("cls")["track_id"].nunique().to_dict()
    seen_by_cls = ev[ev["track_id"].isin(seen_tracks)].groupby("cls")["track_id"].nunique().to_dict()
    rows = []
    for c,n in by_cls.items():
        s = seen_by_cls.get(c,0)
        rows.append({"cls":c, "tracks_total":int(n), "tracks_seen":int(s), "recall_tracks": (s/n if n>0 else 0.0)})
    df_cls = pd.DataFrame(rows).sort_values("recall_tracks")

    # point-level recall (@ exact timestamp)
    ev["ts"] = (
        pd.to_datetime(ev["timestamp"], utc=True)
        .dt.tz_convert(None)
        .dt.floor("s")
        .values
        .astype("datetime64[s]")
    )

    rp["ts"] = (
        pd.to_datetime(rp["timestamp"], utc=True)
        .dt.tz_convert(None)
        .dt.floor("s")
        .values
        .astype("datetime64[s]")
    )

    rp_nonan = rp.dropna(subset=["target_track_id"]).copy()
    rp_keys = set(zip(rp_nonan["ts"], rp_nonan["target_track_id"]))
    ev["seen"] = ev.apply(lambda r: (r["ts"], r["track_id"]) in rp_keys, axis=1)

    point_recall = float(ev["seen"].mean())

    print(f"TRACK recall: {len(seen_tracks)}/{len(truth_tracks)} = {len(seen_tracks)/max(1,len(truth_tracks)):.3f}")
    print(f"POINT recall: {point_recall:.3f}")
    print(df_cls.to_string(index=False))

    # miss heatmap + cell weights
    cells = grid_cells(AREA_POLY, GRID_NX, GRID_NY)
    miss = ev[~ev["seen"]]
    cell_counts = np.zeros(len(cells), dtype=float)
    if len(miss):
        miss_pts = miss[["lat","lon"]].to_numpy()
        for i, quad in enumerate(cells):
            lats = [p[0] for p in quad]; lons = [p[1] for p in quad]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            cand = miss_pts[(miss_pts[:,0]>=lat_min)&(miss_pts[:,0]<=lat_max)&
                            (miss_pts[:,1]>=lon_min)&(miss_pts[:,1]<=lon_max)]
            cnt = 0
            for (la,lo) in cand:
                if point_in_quad(la,lo,quad): cnt += 1
            cell_counts[i] = cnt

        w = cell_counts + cell_counts.mean()*0.05 + 1e-6
        with open("out/cell_weights.json","w",encoding="utf-8") as f:
            json.dump(w.tolist(), f, indent=2)

        # petite carte de chaleur (optionnelle)
        try:
            tiler = OSM(); crs_tiles = tiler.crs
            lat_min, lat_max = ev["lat"].min(), ev["lat"].max()
            lon_min, lon_max = ev["lon"].min(), ev["lon"].max()
            mlat = (lat_max - lat_min) * 0.4; mlon = (lon_max - lon_min) * 0.4
            extent = [lon_min-mlon, lon_max+mlon, lat_min-mlat, lat_max+mlat]

            fig = plt.figure(figsize=(10,7))
            ax = plt.axes(projection=crs_tiles)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_image(tiler, 8)
            xs = [p[1] for p in AREA_POLY] + [AREA_POLY[0][1]]
            ys = [p[0] for p in AREA_POLY] + [AREA_POLY[0][0]]
            ax.plot(xs, ys, color="#222", lw=1.3, alpha=0.9, transform=ccrs.PlateCarree())
            vmax = max(1.0, cell_counts.max())
            for i, quad in enumerate(cells):
                col = cell_counts[i] / vmax
                poly_x = [p[1] for p in quad] + [quad[0][1]]
                poly_y = [p[0] for p in quad] + [quad[0][0]]
                ax.fill(poly_x, poly_y, transform=ccrs.PlateCarree(),
                        facecolor=(1.0, 0.2, 0.2, 0.25*col), edgecolor=None, zorder=3)
            ax.set_title("LURA miss per cell (red = more misses)")
            fig.tight_layout()
            fig.savefig("out/lura_miss_cells.png", dpi=180)
            plt.close(fig)
        except Exception as e:
            print("heatmap draw skipped:", e)

    summary = {
        "tracks_total": int(len(truth_tracks)),
        "tracks_seen": int(len(seen_tracks)),
        "recall_tracks": float(len(seen_tracks)/max(1,len(truth_tracks))),
        "point_recall": float(point_recall),
        "by_class": rows,
        "cells_miss_sum": float(cell_counts.sum())
    }
    with open("out/lura_eval_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()