import json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from config_area import AREA_POLY, clamp, _lerp, side_west, side_east, point_in_poly, heading_to, kinematic_step, EARTH_R

RNG = np.random.default_rng(42); random.seed(42)

@dataclass
class LuraCfg:
    n_lura: int = 100
    grid_nx: int = 12
    grid_ny: int = 8
    speed_ms: Tuple[float,float] = (1.5, 2.8)
    sensor_R_m: float = 8000.0
    detect_base: float = 0.85
    false_pos_rate: float = 1e-4
    dwell_s: int = 1800
    dt_s: int = 15

@dataclass
class Lura:
    lura_id: str
    lat: float
    lon: float
    speed_ms: float
    heading_deg: float
    waypoints: List[Tuple[float,float]]
    mode: str = "patrol"      # patrol|investigate
    target_id: str = ""
    dwell_left_s: int = 0
    patrol_seed: List[Tuple[float,float]] = None

def bilinear(poly, s, u):
    w0,w1 = side_west(poly); e0,e1 = side_east(poly)
    pw = _lerp(w0, w1, s);    pe = _lerp(e0, e1, s)
    return _lerp(pw, pe, u)

def grid_cells(poly, nx, ny):
    out = []
    for ix in range(nx):
        s0, s1 = ix/nx, (ix+1)/nx
        for iy in range(ny):
            u0, u1 = iy/ny, (iy+1)/ny
            out.append([bilinear(poly,s0,u0), bilinear(poly,s1,u0),
                        bilinear(poly,s1,u1), bilinear(poly,s0,u1)])
    return out

def lawnmower_path(cell, legs=6):
    a,b,c,d = cell
    path = []
    for k in range(legs+1):
        t = k/legs
        L = _lerp(a,d,t); R = _lerp(b,c,t)
        path += [L,R] if k%2==0 else [R,L]
    return path

def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*EARTH_R*math.asin(math.sqrt(a))

def class_bias(cls: str) -> float:
    if cls == "merchant_tanker": return 1.05
    if cls == "merchant_cargo":  return 1.00
    if cls == "military_vessel": return 0.95
    if cls == "small_boat":      return 0.85
    if cls == "usv":             return 0.75
    return 0.80

def make_swarm(cfg: LuraCfg, cell_weights=None) -> List[Lura]:
    cells = grid_cells(AREA_POLY, cfg.grid_nx, cfg.grid_ny)
    m = len(cells)

    if cell_weights is None or len(cell_weights) != m:
        # uniforme (ancienne logique)
        idx = np.linspace(0, m-1, cfg.n_lura, dtype=int)
    else:
        # échantillonnage pondéré
        p = np.asarray(cell_weights, dtype=float)
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        idx = np.random.choice(np.arange(m), size=cfg.n_lura, replace=True, p=p)

    swarm = []
    for i, ci in enumerate(idx):
        wps = lawnmower_path(cells[ci], legs=6)
        spd = RNG.uniform(*cfg.speed_ms)
        hdg = heading_to(wps[0], wps[1])
        swarm.append(Lura(f"L{i:03d}", wps[0][0], wps[0][1], spd, hdg, wps[1:], "patrol", "", 0, wps[1:].copy()))
    return swarm


def patrol_and_collect(events, t_start, t_end, cfg: LuraCfg):
    cell_weights = None
    try:
        with open("out/cell_weights.json","r",encoding="utf-8") as f:
            cell_weights = json.load(f)
    except FileNotFoundError:
        pass
    swarm = make_swarm(cfg, cell_weights=cell_weights)
    tracks, reports = [], []
    sea_state = 2.5  

    def tstr(t64):  # np.datetime64 -> "YYYY-mm-ddTHH:MM:SSZ"
        return np.datetime_as_string(t64, unit="s") + "Z"

    def ts64(s):    # "YYYY-mm-ddTHH:MM:SSZ" -> np.datetime64[s]
        return np.datetime64(s.replace("Z",""), "s")

    ev_idx = defaultdict(list)
    for e in events:
        # si tes events ne sont pas pile sur la seconde, tronquer à la seconde
        ev_idx[ts64(e["timestamp"])].append(e)
        
    t = t_start.astype("datetime64[s]")
    t_end = t_end.astype("datetime64[s]")

    while t <= t_end:
        # tolérance ±dt_s pour mismatch simulateur/patrouille
        cand = []
        for off in (0, -cfg.dt_s, cfg.dt_s):
            cand.extend(ev_idx.get(t + np.timedelta64(off, "s"), []))
        current = [e for e in cand if point_in_poly(e["lat"], e["lon"], AREA_POLY)]

        for L in swarm:
            # move
            L.lat, L.lon, L.heading_deg, L.speed_ms = kinematic_step(L.lat, L.lon, L.speed_ms, L.heading_deg, 0.03, 1.2)

            # follow path
            if L.waypoints:
                tgt = L.waypoints[0]
                brg = heading_to((L.lat, L.lon), tgt)
                err = ((brg - L.heading_deg + 540) % 360) - 180
                L.heading_deg = (L.heading_deg + clamp(err, -6, 6)) % 360
                if abs(L.lat - tgt[0]) < 0.005 and abs(L.lon - tgt[1]) < 0.005:
                    L.waypoints.pop(0)
            else:
                if L.mode == "patrol" and L.patrol_seed:
                    L.waypoints = L.patrol_seed.copy()

            # sense
            for a in current:
                r = haversine_m(L.lat, L.lon, a["lat"], a["lon"])
                if r > cfg.sensor_R_m: continue
                # Pd = base * bias * (1 - (r/R)^2)^+ * sea
                sea_pen = clamp(1.0 - 0.06*sea_state, 0.65, 1.0)
                Pd = clamp(cfg.detect_base * class_bias(a["cls"]) * sea_pen * max(0.0, 1.0 - (r/cfg.sensor_R_m)**2), 0.0, 0.99)
                if RNG.uniform() < Pd:
                    reports.append({
                        "obs_id": f"O{RNG.integers(10**9)}",
                        "timestamp": tstr(t),
                        "lura_id": L.lura_id,
                        "lat": round(L.lat,6), "lon": round(L.lon,6),
                        "range_m": round(r,1),
                        "bearing_deg": round(heading_to((L.lat,L.lon),(a["lat"],a["lon"])),1),
                        "target_track_id": a["track_id"],
                        "est_cls": a["cls"] if RNG.uniform()<0.85 else random.choice(["merchant_cargo","merchant_tanker","small_boat","usv","military_vessel","unknown"]),
                        "est_confidence": float(np.clip(RNG.normal(0.75,0.12), 0.2, 0.98)),
                        "ghost_hint": int(a.get("cls")=="merchant_tanker" and a.get("ais_anomaly",0)==1),
                    })
                    # reactive investigate
                    if L.mode == "patrol" and (a["cls"] in ("military_vessel","usv") or (a["cls"]=="merchant_tanker" and a.get("ais_anomaly",0)==1)):
                        L.mode = "investigate"; L.target_id = a["track_id"]; L.dwell_left_s = cfg.dwell_s
                        L.patrol_seed = L.waypoints.copy() if L.waypoints else L.patrol_seed
                        L.waypoints = [(a["lat"], a["lon"])] + L.waypoints

            if L.mode == "investigate":
                L.dwell_left_s = max(0, L.dwell_left_s - cfg.dt_s)
                if L.dwell_left_s == 0 and L.patrol_seed:
                    L.mode = "patrol"
                    nxt = L.patrol_seed[0]
                    L.waypoints = [nxt] + L.patrol_seed[1:]

            tracks.append({
                "lura_id": L.lura_id, "timestamp": tstr(t),
                "lat": round(L.lat,6), "lon": round(L.lon,6), "mode": L.mode
            })


        t = t + np.timedelta64(cfg.dt_s, "s")

    tracks.sort(key=lambda x: x["timestamp"]); reports.sort(key=lambda x: x["timestamp"])
    return tracks, reports

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/events_simulated.json","r",encoding="utf-8") as f:
        events = json.load(f)

    # time window from events
    ts = np.array([np.datetime64(e["timestamp"].replace("Z","")) for e in events])
    t_start, t_end = ts.min(), ts.max()

    cfg = LuraCfg()  # tweak if needed
    tracks, reports = patrol_and_collect(events, t_start, t_end, cfg)

    with open("data/lura_tracks.json","w",encoding="utf-8") as f: json.dump(tracks, f, indent=2)
    with open("data/lura_reports.json","w",encoding="utf-8") as f: json.dump(reports, f, indent=2)

if __name__ == "__main__":
    main()