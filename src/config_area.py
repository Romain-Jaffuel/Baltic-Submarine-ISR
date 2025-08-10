import math

EARTH_R = 6371000.0

# oriented corridor (parallelogram)
AREA_POLY = [
    (59.35, 23.50),  # SW
    (59.80, 26.50),  # SE
    (60.30, 26.50),  # NE
    (59.85, 23.50),  # NW
]

def clamp(v, a, b): return max(a, min(b, v))

def _lerp(a, b, t):
    return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))

def side_west(poly): return (poly[3], poly[0])  # NW->SW
def side_east(poly): return (poly[2], poly[1])  # NE->SE

def point_in_poly(lat, lon, poly):
    # ray casting in (lon,lat)
    x, y = lon, lat
    pts = [(p[1], p[0]) for p in poly]
    inside = False
    for i in range(len(pts)):
        x1, y1 = pts[i]; x2, y2 = pts[(i+1) % len(pts)]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin: inside = not inside
    return inside

def meters_to_deg(lat_deg, dx_m, dy_m):
    lat_rad = math.radians(lat_deg)
    dlat = (dy_m / EARTH_R) * (180.0 / math.pi)
    dlon = (dx_m / (EARTH_R * math.cos(lat_rad))) * (180.0 / math.pi)
    return dlat, dlon

def heading_to(a, b):
    import math
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def kinematic_step(lat, lon, v_ms, hdg_deg, jitter_std_ms=0.08, turn_std_deg=1.5):
    import numpy as np, math
    RNG = np.random.default_rng()
    v = max(0.0, RNG.normal(v_ms, jitter_std_ms * v_ms))
    hdg = (hdg_deg + RNG.normal(0.0, turn_std_deg)) % 360.0
    dx = v * math.sin(math.radians(hdg))
    dy = v * math.cos(math.radians(hdg))
    dlat, dlon = meters_to_deg(lat, dx, dy)
    return lat + dlat, lon + dlon, hdg, v
