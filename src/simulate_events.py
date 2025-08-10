import json, math, datetime, random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

RNG = np.random.default_rng(42)
random.seed(42)

# --- Geo helpers ---
EARTH_R = 6371000.0
def meters_to_deg(lat_deg: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    dlat = (dy_m / EARTH_R) * (180.0 / math.pi)
    dlon = (dx_m / (EARTH_R * math.cos(lat_rad))) * (180.0 / math.pi)
    return dlat, dlon

def clamp(v, a, b):
    return max(a, min(b, v))

AREA = dict(
    lat_min=59.64733026547654,
    lat_max=60.13277611865003,
    lon_min=24.75925878084836,
    lon_max=27.65033872959603,
)
LAT_MID = 0.5 * (AREA["lat_min"] + AREA["lat_max"])
LAT_SIG = (AREA["lat_max"] - AREA["lat_min"]) / 6.0

def sample_lat_normal() -> float:
    lat = RNG.normal(LAT_MID, LAT_SIG)
    return clamp(lat, AREA["lat_min"], AREA["lat_max"])

def sample_lon_uniform() -> float:
    return RNG.uniform(AREA["lon_min"], AREA["lon_max"])

def random_point() -> Tuple[float, float]:
    return (sample_lat_normal(), sample_lon_uniform())

# --- Config ---
CLASSES = ["merchant_vessel", "small_boat", "uav", "usv", "unknown"]

@dataclass
class SimConfig:
    t_start: datetime.datetime
    t_end: datetime.datetime
    dt_s: int = 30
    n_routes: int = 4
    n_spurious: int = 160
    sea_state_mu: float = 2.5 
    sea_state_sigma: float = 1.0

@dataclass
class Agent:
    track_id: str
    cls: str
    lat: float
    lon: float
    speed_ms: float
    heading_deg: float
    waypoints: List[Tuple[float, float]]
    pri_hint: int

def route_between(a: Tuple[float,float], b: Tuple[float,float], bends: int = 0) -> List[Tuple[float,float]]:
    pts = [a]
    if bends > 0:
        for _ in range(bends):
            la = RNG.uniform(min(a[0], b[0]), max(a[0], b[0]))
            lo = RNG.uniform(min(a[1], b[1]), max(a[1], b[1]))
            pts.append((la, lo))
    pts.append(b)
    return pts

def kinematic_step(lat, lon, v_ms, hdg_deg, jitter_std_ms=0.1, turn_std_deg=2.0):
    v = max(0.0, RNG.normal(v_ms, jitter_std_ms * v_ms))
    hdg = (hdg_deg + RNG.normal(0.0, turn_std_deg)) % 360.0
    dx = v * math.sin(math.radians(hdg))
    dy = v * math.cos(math.radians(hdg))
    dlat, dlon = meters_to_deg(lat, dx, dy)
    return lat + dlat, lon + dlon, hdg

def heading_to(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brng

def speed_for_class(cls: str) -> float:
    if cls == "merchant_vessel": return RNG.uniform(4.0, 8.0)
    if cls == "small_boat":     return RNG.uniform(2.0, 6.0)
    if cls == "uav":            return RNG.uniform(10.0, 30.0)
    if cls == "usv":            return RNG.uniform(1.0, 2.0)
    return RNG.uniform(0.5, 5.0)

def pri_for_class(cls: str) -> int:
    if cls in ("uav", "usv"): return int(clamp(int(RNG.normal(2, 0.8)), 1, 3))
    if cls == "unknown":      return int(clamp(int(RNG.normal(2, 1.0)), 1, 4))
    return int(clamp(int(RNG.normal(3, 1.0)), 2, 5))

def confidence_model(base: float, sea_state: float, cls: str) -> float:
    sea_pen = clamp(1.0 - 0.08 * sea_state, 0.6, 1.0)
    cls_bias = 0.9 if cls in ("uav","usv") else (0.95 if cls in ("merchant_vessel","small_boat") else 0.8)
    return clamp(RNG.normal(base * sea_pen * cls_bias, 0.08), 0.05, 0.99)

def make_agents(n_routes: int) -> List[Agent]:
    agents = []
    lon_w = AREA["lon_min"] + 0.02
    lon_e = AREA["lon_max"] - 0.02
    forced_classes = CLASSES.copy()
    random.shuffle(forced_classes)

    for rid in range(n_routes):
        if forced_classes:
            cls = forced_classes.pop()
        else:
            cls = random.choices(CLASSES, weights=[0.45,0.2,0.12,0.15,0.08])[0]

        lat_w = sample_lat_normal()
        lat_e = sample_lat_normal()
        start = (lat_w, lon_w) if rid % 2 == 0 else (lat_e, lon_e)
        end   = (lat_e, lon_e) if rid % 2 == 0 else (lat_w, lon_w)
        bends = int(RNG.integers(0, 2))
        wps   = route_between(start, end, bends=bends)
        v_ms  = speed_for_class(cls)
        hdg   = heading_to(wps[0], wps[1])
        agents.append(Agent(
            track_id=f"T{rid}",
            cls=cls,
            lat=wps[0][0],
            lon=wps[0][1],
            speed_ms=v_ms,
            heading_deg=hdg,
            waypoints=wps[1:],
            pri_hint=pri_for_class(cls)
        ))
    return agents

def advance_agent(a: Agent) -> None:
    if not a.waypoints:
        a.lat, a.lon, a.heading_deg = kinematic_step(a.lat, a.lon, a.speed_ms*0.4, a.heading_deg, 0.2, 10.0)
        return
    tgt = a.waypoints[0]
    brg = heading_to((a.lat, a.lon), tgt)
    err = ((brg - a.heading_deg + 540) % 360) - 180
    a.heading_deg = (a.heading_deg + clamp(err, -6, 6)) % 360
    a.lat, a.lon, a.heading_deg = kinematic_step(a.lat, a.lon, a.speed_ms, a.heading_deg)
    if abs(a.lat - tgt[0]) < 0.01 and abs(a.lon - tgt[1]) < 0.01:
        a.waypoints.pop(0)

def in_bounds(lat, lon) -> bool:
    return (AREA["lat_min"] <= lat <= AREA["lat_max"]) and (AREA["lon_min"] <= lon <= AREA["lon_max"])

def simulate(cfg: SimConfig) -> List[Dict]:
    sea_state = clamp(RNG.normal(cfg.sea_state_mu, cfg.sea_state_sigma), 0.0, 6.0)
    agents = make_agents(cfg.n_routes)
    t = cfg.t_start
    events: List[Dict] = []

    step_spurious = int(cfg.n_spurious * cfg.dt_s / ((cfg.t_end - cfg.t_start).total_seconds()))

    mandatory_spurious = []
    for c in CLASSES:
        la, lo = random_point()
        base = 0.7 if c in ("uav","usv") else 0.6
        conf = confidence_model(base, sea_state, c)
        mandatory_spurious.append({
            "track_id": f"S{RNG.integers(10**9)}",
            "timestamp": t.replace(microsecond=0).isoformat() + "Z",
            "lat": round(la, 6),
            "lon": round(lo, 6),
            "cls": c,
            "confidence": round(conf, 3),
            "priority_hint": pri_for_class(c)
        })

    while t <= cfg.t_end:
        for a in agents:
            advance_agent(a)
            if not in_bounds(a.lat, a.lon):
                continue
            base = 0.78 if a.cls != "unknown" else 0.58
            conf = confidence_model(base, sea_state, a.cls)
            events.append({
                "track_id": a.track_id,
                "timestamp": t.replace(microsecond=0).isoformat() + "Z",
                "lat": round(a.lat, 6),
                "lon": round(a.lon, 6),
                "cls": a.cls,
                "confidence": round(conf, 3),
                "priority_hint": a.pri_hint
            })

        spurious_batch = mandatory_spurious if t == cfg.t_start else []
        for _ in range(step_spurious):
            la, lo = random_point()
            c = random.choices(CLASSES, weights=[0.25,0.25,0.2,0.2,0.1])[0]
            base = 0.7 if c in ("uav","usv") else 0.6
            conf = confidence_model(base, sea_state, c)
            spurious_batch.append({
                "track_id": f"S{RNG.integers(10**9)}",
                "timestamp": t.replace(microsecond=0).isoformat() + "Z",
                "lat": round(la, 6),
                "lon": round(lo, 6),
                "cls": c,
                "confidence": round(conf, 3),
                "priority_hint": pri_for_class(c)
            })

        events.extend(spurious_batch)
        t += datetime.timedelta(seconds=cfg.dt_s)

    events.sort(key=lambda x: x["timestamp"])
    return events

def main():
    cfg = SimConfig(
        t_start=datetime.datetime(2026, 4, 1, 6, 0, 0),
        t_end=datetime.datetime(2026, 4, 1, 18, 0, 0),
        dt_s=30,
        n_routes=4,
        n_spurious=160
    )
    evts = simulate(cfg)
    with open("data/events_simulated.json", "w", encoding="utf-8") as f:
        json.dump(evts, f, indent=2)

if __name__ == "__main__":
    main()
