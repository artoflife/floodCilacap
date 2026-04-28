"""
Flood Evacuation System — Flask App (Production v6)
====================================================
Gabungan versi lama + v5:
  - Backbone  : v5 (BMKG, ORS→OSRM→Haversine chain, logging, health)
  - Dari lama : RF predict per shelter top-3, ORS Matrix jarak_km_matrix,
                Dijkstra via osmnx, Folium heatmap
  - Fitur RF  : jarak_km_matrix, risk_weight, RH, RR (versi lama)

Deploy ke Railway via GitHub.
"""

import os, json, math, logging, csv, time, pickle
import numpy as np
import pandas as pd
import joblib
import requests
import folium
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx
from datetime import datetime, timezone, timedelta
from threading import Lock
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS

# ── App Init ─────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
ORS_KEY      = os.environ.get("ORS_KEY", "")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model", "rf_model.pkl")
META_PATH    = os.path.join(os.path.dirname(__file__), "model", "metadata.json")
FLOOD_CSV    = os.environ.get("FLOOD_CSV",   "data/Desa_Rawan_Banjir.csv")
SHELTER_CSV  = os.environ.get("SHELTER_CSV", "data/Tempat_Evakuasi_Final.csv")
LOG_PATH     = "prediction_log.csv"

CILACAP_BBOX   = (-7.79, 108.52, -7.17, 109.41)  # south, west, north, east
CILACAP_CENTER = (-7.44, 108.96)

# ── Constants ─────────────────────────────────────────────────
DELAY_MAP    = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.5}
RISK_LABELS  = {1: "Normal", 2: "Moderate", 3: "Caution", 4: "Critical"}
ROUTE_LABELS = {0: "Fastest", 1: "Safest", 2: "Balanced"}

KECAMATAN = [
    {"adm4":"33.01.01.2001","nama":"Dayeuhluhur","lat":-7.218,"lon":108.405},
    {"adm4":"33.01.02.2001","nama":"Wanareja","lat":-7.333,"lon":108.687},
    {"adm4":"33.01.03.2001","nama":"Majenang","lat":-7.300,"lon":108.760},
    {"adm4":"33.01.04.2001","nama":"Cimanggu","lat":-7.245,"lon":108.895},
    {"adm4":"33.01.05.2001","nama":"Karangpucung","lat":-7.336,"lon":108.834},
    {"adm4":"33.01.06.2001","nama":"Cipari","lat":-7.395,"lon":108.727},
    {"adm4":"33.01.07.2001","nama":"Sidareja","lat":-7.470,"lon":108.821},
    {"adm4":"33.01.08.2001","nama":"Kedungreja","lat":-7.548,"lon":108.775},
    {"adm4":"33.01.09.2001","nama":"Patimuan","lat":-7.578,"lon":108.777},
    {"adm4":"33.01.10.2001","nama":"Gandrungmangu","lat":-7.530,"lon":108.700},
    {"adm4":"33.01.11.2001","nama":"Bantarsari","lat":-7.570,"lon":108.890},
    {"adm4":"33.01.12.2001","nama":"Kawunganten","lat":-7.598,"lon":108.930},
    {"adm4":"33.01.13.2001","nama":"Kampung Laut","lat":-7.660,"lon":108.820},
    {"adm4":"33.01.14.2001","nama":"Jeruklegi","lat":-7.598,"lon":109.010},
    {"adm4":"33.01.15.2001","nama":"Kesugihan","lat":-7.630,"lon":109.060},
    {"adm4":"33.01.16.2001","nama":"Adipala","lat":-7.680,"lon":109.080},
    {"adm4":"33.01.17.2001","nama":"Maos","lat":-7.620,"lon":109.120},
    {"adm4":"33.01.18.2001","nama":"Sampang","lat":-7.580,"lon":109.145},
    {"adm4":"33.01.19.2001","nama":"Kroya","lat":-7.630,"lon":109.245},
    {"adm4":"33.01.20.2001","nama":"Binangun","lat":-7.610,"lon":109.310},
    {"adm4":"33.01.21.2001","nama":"Nusawungu","lat":-7.660,"lon":109.365},
    {"adm4":"33.01.22.2001","nama":"Cilacap Selatan","lat":-7.740,"lon":109.010},
    {"adm4":"33.01.23.2001","nama":"Cilacap Tengah","lat":-7.720,"lon":109.015},
    {"adm4":"33.01.24.2001","nama":"Cilacap Utara","lat":-7.690,"lon":109.005},
]

# ── Load Model ───────────────────────────────────────────────
logger.info("Loading RF model...")
try:
    with open(MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    P25 = metadata["P25"]
    P50 = metadata["P50"]
    P75 = metadata["P75"]
    SHELTER_DATA = metadata["shelters"]  # [{nama, lat, lon}]
    logger.info(f"RF model loaded. P75={P75:.2f}, shelters={len(SHELTER_DATA)}")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    rf_model = None
    P25 = P50 = P75 = 0
    SHELTER_DATA = []

# ── Load CSV Data ─────────────────────────────────────────────
try:
    df_flood = pd.read_csv(FLOOD_CSV)
    df_flood.columns = df_flood.columns.str.strip()
    logger.info(f"Flood CSV loaded: {len(df_flood)} rows")
except Exception as e:
    logger.error(f"Flood CSV load failed ({FLOOD_CSV}): {e}")
    df_flood = pd.DataFrame()

try:
    df_shelters_csv = pd.read_csv(SHELTER_CSV)
    df_shelters_csv.columns = df_shelters_csv.columns.str.strip()
    logger.info(f"Shelter CSV loaded: {len(df_shelters_csv)} rows")
except Exception as e:
    logger.error(f"Shelter CSV load failed ({SHELTER_CSV}): {e}")
    df_shelters_csv = pd.DataFrame()

# ── Load OSMnx Graph ──────────────────────────────────────────
logger.info("Loading OSMnx road graph Cilacap (first boot may take 2-3 min)...")
try:
    G = ox.graph_from_bbox(
        north=CILACAP_BBOX[2], south=CILACAP_BBOX[0],
        east=CILACAP_BBOX[3],  west=CILACAP_BBOX[1],
        network_type='drive'
    )
    logger.info(f"OSMnx graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
except Exception as e:
    logger.error(f"OSMnx graph load failed: {e}")
    G = None

# ── Route Cache ───────────────────────────────────────────────
_route_cache = {}
_log_lock    = Lock()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = sin(dlat/2)**2 + cos(la1)*cos(la2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def hitung_rolling3(rr_today, rr_yesterday=0, rr_2days=0):
    return (rr_today or 0) + (rr_yesterday or 0) + (rr_2days or 0)


def hitung_risk_level(rolling3, rh, kejadian_banjir=0):
    """Risk level 1-4 berbasis rolling3 + RH + KejadianBanjir (novelty)."""
    if rolling3 >= P75 or kejadian_banjir >= 1:
        return 4
    elif rolling3 >= P50 and rh >= 80:
        return 3
    elif rolling3 >= P25 and rh >= 75:
        return 2
    return 1


def hitung_risk_weight(risk_level):
    return (risk_level - 1) / 3


def find_nearest_kecamatan(lat, lon):
    return min(KECAMATAN, key=lambda k: haversine_km(lat, lon, k["lat"], k["lon"]))


def find_nearby_flood_points(lat, lon, radius_km=5.0):
    if df_flood.empty:
        return []
    nearby = []
    for _, fp in df_flood.iterrows():
        d = haversine_km(lat, lon, fp['Latitude'], fp['Longitude'])
        if 0.1 < d <= radius_km:
            nearby.append({
                "lat": fp['Latitude'], "lon": fp['Longitude'],
                "dist": round(d, 2), "nama": fp['Nama']
            })
    return sorted(nearby, key=lambda x: x['dist'])[:10]


# ═══════════════════════════════════════════════════════════════
# ORS MATRIX (jarak_km_matrix — fitur RF)
# ═══════════════════════════════════════════════════════════════

def get_ors_matrix(lon_src, lat_src, destinations):
    """
    Ambil jarak_km_matrix dari ORS Matrix API.
    Dipakai sebagai fitur RF (bukan untuk routing).
    Fallback ke Haversine jika ORS gagal.
    """
    if not ORS_KEY:
        return [haversine_km(lat_src, lon_src, d["lat"], d["lon"]) for d in destinations]

    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    body = {
        "locations": [[lon_src, lat_src]] + [[d["lon"], d["lat"]] for d in destinations],
        "sources": [0],
        "destinations": list(range(1, len(destinations) + 1)),
        "metrics": ["distance"],
        "units": "km"
    }
    headers = {"Authorization": ORS_KEY, "Content-Type": "application/json"}
    try:
        r = requests.post(url, json=body, headers=headers, timeout=15)
        if r.status_code == 200:
            dists = r.json().get("distances", [[]])[0]
            # Fallback per-shelter jika None
            return [
                d if d is not None else haversine_km(lat_src, lon_src, destinations[i]["lat"], destinations[i]["lon"])
                for i, d in enumerate(dists)
            ]
    except Exception as e:
        logger.warning(f"ORS Matrix error: {e}")
    return [haversine_km(lat_src, lon_src, d["lat"], d["lon"]) for d in destinations]


# ═══════════════════════════════════════════════════════════════
# ROUTING: ORS → OSRM → Haversine (v5 chain)
# ═══════════════════════════════════════════════════════════════

ORS_DIR_URL  = "https://api.openrouteservice.org/v2/directions/driving-car"
OSRM_URL     = "https://router.project-osrm.org/route/v1/driving"
ROAD_FACTOR  = 1.4
AVG_SPEED    = 40  # km/h


def _make_polyline(coords_lonlat):
    """[[lon,lat],...] → [[lat,lon],...] untuk Leaflet."""
    return [[c[1], c[0]] for c in coords_lonlat]


def _decode_polyline(encoded):
    """Decode Google encoded polyline → [[lon,lat],...]."""
    coords = []; idx = 0; lat = 0; lng = 0
    while idx < len(encoded):
        for var in ['lat', 'lng']:
            shift = result = 0
            while True:
                b = ord(encoded[idx]) - 63; idx += 1
                result |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if var == 'lat': lat += delta
            else: lng += delta
        coords.append([lng / 1e5, lat / 1e5])
    return coords


def _ors_directions(olat, olon, dlat, dlon, preference="fastest", avoid_polygons=None):
    if not ORS_KEY:
        return None
    body = {
        "coordinates": [[olon, olat], [dlon, dlat]],
        "preference": preference,
        "geometry": True,
        "instructions": False,
    }
    if avoid_polygons:
        body["options"] = {"avoid_polygons": avoid_polygons}
    try:
        resp = requests.post(ORS_DIR_URL, json=body, timeout=12,
                             headers={"Authorization": ORS_KEY, "Content-Type": "application/json"})
        if resp.status_code == 200:
            routes = resp.json().get("routes", [])
            if routes:
                r = routes[0]; summary = r.get("summary", {})
                geom = r.get("geometry", "")
                coords = _decode_polyline(geom) if isinstance(geom, str) else geom.get("coordinates", [])
                return {
                    "distance_km": round(summary.get("distance", 0) / 1000, 2),
                    "travel_time_min": round(summary.get("duration", 0) / 60, 1),
                    "polyline": _make_polyline(coords),
                    "source": f"ORS_{preference}",
                }
        elif resp.status_code == 429:
            logger.warning("ORS rate limit")
    except Exception as e:
        logger.debug(f"ORS directions error: {e}")
    return None


def _osrm_directions(olat, olon, dlat, dlon, alternatives=False):
    try:
        params = "overview=full&geometries=geojson"
        if alternatives:
            params += "&alternatives=true"
        url = f"{OSRM_URL}/{olon},{olat};{dlon},{dlat}?{params}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.json().get("code") == "Ok":
            results = []
            for r in resp.json().get("routes", []):
                results.append({
                    "distance_km": round(r["distance"] / 1000, 2),
                    "travel_time_min": round(r["duration"] / 60, 1),
                    "polyline": _make_polyline(r["geometry"]["coordinates"]),
                    "source": "OSRM",
                })
            return results if alternatives else (results[0] if results else None)
    except Exception as e:
        logger.debug(f"OSRM error: {e}")
    return None


def _haversine_fallback(olat, olon, dlat, dlon, multiplier=1.0):
    d = haversine_km(olat, olon, dlat, dlon) * ROAD_FACTOR * multiplier
    return {
        "distance_km": round(d, 2),
        "travel_time_min": round(d / AVG_SPEED * 60, 1),
        "polyline": [[olat, olon], [dlat, dlon]],
        "source": "haversine",
    }


def get_fastest_route(olat, olon, dlat, dlon):
    ck = f"fast:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}"
    if ck in _route_cache:
        return _route_cache[ck]
    r = (_ors_directions(olat, olon, dlat, dlon, "fastest") or
         _osrm_directions(olat, olon, dlat, dlon) or
         _haversine_fallback(olat, olon, dlat, dlon))
    _route_cache[ck] = r
    return r


def get_safest_route(olat, olon, dlat, dlon, flood_points):
    ck = f"safe:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}:{len(flood_points)}"
    if ck in _route_cache:
        return _route_cache[ck]

    # Coba ORS dengan avoid_polygons (zona banjir)
    if ORS_KEY and flood_points:
        avoid_coords = []
        for fp in flood_points[:5]:
            r_deg = 0.008
            lat_, lon_ = fp["lat"], fp["lon"]
            avoid_coords.append([
                [lon_-r_deg, lat_-r_deg], [lon_+r_deg, lat_-r_deg],
                [lon_+r_deg, lat_+r_deg], [lon_-r_deg, lat_+r_deg], [lon_-r_deg, lat_-r_deg]
            ])
        avoid_poly = {"type": "MultiPolygon", "coordinates": [[c] for c in avoid_coords]}
        r = _ors_directions(olat, olon, dlat, dlon, "recommended", avoid_poly)
        if r:
            r["source"] = "ORS_safest_avoid"
            r["avoided_zones"] = len(avoid_coords)
            _route_cache[ck] = r
            return r

    r = (_ors_directions(olat, olon, dlat, dlon, "recommended") or
         _haversine_fallback(olat, olon, dlat, dlon, multiplier=1.4))
    _route_cache[ck] = r
    return r


def get_balanced_route(olat, olon, dlat, dlon):
    ck = f"bal:{olat:.5f},{olon:.5f}:{dlat:.5f},{dlon:.5f}"
    if ck in _route_cache:
        return _route_cache[ck]
    r = (_ors_directions(olat, olon, dlat, dlon, "shortest") or
         _osrm_directions(olat, olon, dlat, dlon) or
         _haversine_fallback(olat, olon, dlat, dlon, multiplier=1.15))
    _route_cache[ck] = r
    return r


# ═══════════════════════════════════════════════════════════════
# DIJKSTRA via OSMnx (fitur versi lama — top-3 shelter)
# ═══════════════════════════════════════════════════════════════

def apply_risk_weights_to_graph(G_copy, risk_weight):
    """Mutasi bobot graf berdasarkan risk_weight saat ini."""
    for u, v, k, data in G_copy.edges(data=True, keys=True):
        length = data.get("length", 100)
        G_copy[u][v][k]["weight"] = length * (1 + risk_weight)
    return G_copy


def find_route_dijkstra(G_weighted, lat_src, lon_src, lat_dst, lon_dst):
    """Dijkstra pada graf OSMnx berbobot risiko."""
    if G_weighted is None:
        return None, None
    try:
        node_src = ox.nearest_nodes(G_weighted, lon_src, lat_src)
        node_dst = ox.nearest_nodes(G_weighted, lon_dst, lat_dst)
        path = nx.shortest_path(G_weighted, node_src, node_dst, weight="weight")
        coords = [(G_weighted.nodes[n]["y"], G_weighted.nodes[n]["x"]) for n in path]
        total_km = sum(
            G_weighted[path[i]][path[i+1]][0].get("length", 0)
            for i in range(len(path) - 1)
        ) / 1000
        return coords, round(total_km, 2)
    except Exception as e:
        logger.debug(f"Dijkstra error: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# RF PREDICT (fitur: jarak_km_matrix, risk_weight, RH, RR)
# ═══════════════════════════════════════════════════════════════

def predict_route_strategy(jarak_km_matrix, risk_weight, rh, rr):
    """RF prediksi strategi rute — return (label_str, proba_list)."""
    X = np.array([[jarak_km_matrix, risk_weight, rh, rr]])
    pred  = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0]
    label_map = {0: "Fastest", 1: "Safest", 2: "Balanced"}
    return label_map[pred], proba.tolist()


# ═══════════════════════════════════════════════════════════════
# BMKG Weather
# ═══════════════════════════════════════════════════════════════

BMKG_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"

def fetch_bmkg_weather(adm4):
    try:
        resp = requests.get(BMKG_URL, params={"adm4": adm4}, timeout=10,
                            headers={"User-Agent": "FloodEvac/6.0"})
        if resp.status_code != 200:
            return None
        bmkg = resp.json(); data_list = bmkg.get("data", [])
        if not data_list:
            return None
        all_fc = []
        for g in data_list[0].get("cuaca", []):
            if isinstance(g, list): all_fc.extend(g)
            elif isinstance(g, dict): all_fc.append(g)
        if not all_fc:
            return None
        all_fc.sort(key=lambda x: x.get("datetime", ""))
        cur = all_fc[0]
        tp24 = round(sum(f.get("tp", 0) or 0 for f in all_fc[:8]), 1)
        lok  = bmkg.get("lokasi", data_list[0].get("lokasi", {}))
        return {
            "source": "BMKG", "adm4": adm4,
            "kecamatan": lok.get("kecamatan", ""),
            "rainfall_24h": tp24,
            "humidity_now": cur.get("hu", 80),
            "temperature": cur.get("t", 0),
            "wind_speed": cur.get("ws", 0),
            "weather_desc": cur.get("weather_desc", ""),
            "forecast_24h": [
                {"local_datetime": f.get("local_datetime", ""),
                 "weather_desc": f.get("weather_desc", ""),
                 "temperature": f.get("t", 0),
                 "humidity": f.get("hu", 0),
                 "rainfall_3h": f.get("tp", 0),
                 "wind_speed": f.get("ws", 0)}
                for f in all_fc[:8]
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.warning(f"BMKG error: {e}")
        return None


def simulated_weather():
    import random; random.seed(int(time.time()) % 10000)
    tp = round(random.uniform(0, 40), 1) if random.random() < 0.4 else 0
    return {
        "source": "simulated", "rainfall_24h": tp,
        "humidity_now": round(random.uniform(70, 98)),
        "temperature": round(random.uniform(24, 32)),
        "wind_speed": round(random.uniform(1, 12)),
        "weather_desc": "Hujan Ringan" if tp > 0 else "Berawan",
        "forecast_24h": [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

_LOG_FIELDS = [
    "timestamp", "lat", "lon", "shelter", "jarak_km_matrix",
    "strategy_rf", "risk_level", "risk_weight", "rolling3",
    "rr", "rh", "route_source", "jarak_aktual_km", "estimasi_waktu_menit",
    "prob_fastest", "prob_safest", "prob_balanced"
]

def log_pred(row):
    with _log_lock:
        exists = os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)


# ═══════════════════════════════════════════════════════════════
# FOLIUM HEATMAP helper
# ═══════════════════════════════════════════════════════════════

def build_heatmap(routes_detail, lat_src, lon_src):
    """
    Buat Folium map dengan:
    - HeatMap dari titik rawan banjir terdekat
    - Polyline rute top-3 shelter
    - Marker sumber dan shelter
    """
    m = folium.Map(location=[lat_src, lon_src], zoom_start=12, tiles="OpenStreetMap")

    # HeatMap — titik rawan banjir
    flood_pts = find_nearby_flood_points(lat_src, lon_src, radius_km=10)
    if flood_pts:
        heat_data = [[fp["lat"], fp["lon"]] for fp in flood_pts]
        HeatMap(heat_data, radius=20, blur=15, min_opacity=0.4).add_to(m)

    # Marker sumber evakuasi
    folium.Marker(
        [lat_src, lon_src],
        tooltip="Titik Evakuasi",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    colors = {"Fastest": "red", "Safest": "green", "Balanced": "orange"}

    for item in routes_detail:
        color = colors.get(item["strategy"], "blue")
        # Polyline rute Dijkstra (jika ada)
        if item.get("route_coords"):
            folium.PolyLine(
                item["route_coords"], color=color, weight=4, opacity=0.8,
                tooltip=f"{item['strategy']} → {item['shelter']}"
            ).add_to(m)
        # Marker shelter
        folium.Marker(
            [item["lat_shelter"], item["lon_shelter"]],
            tooltip=f"[{item['strategy']}] {item['shelter']}",
            icon=folium.Icon(color=color, icon="star")
        ).add_to(m)

    return m._repr_html_()


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/evacuate", methods=["POST"])
def evacuate():
    """
    Endpoint utama evakuasi.

    Request JSON:
    {
        "lat": -7.5, "lon": 108.9,
        "rr": 22.0, "rh": 87,
        "rr_yesterday": 35.0, "rr_2days": 28.0,
        "kejadian_banjir": 0
    }

    Flow:
      1. Ambil cuaca BMKG (fallback simulated)
      2. Hitung rolling3 → risk_level → risk_weight
      3. ORS Matrix → jarak_km_matrix per shelter (fitur RF)
      4. RF predict strategi per shelter → top-3 terbaik
      5. Dijkstra OSMnx → rute aktual top-3
      6. ORS/OSRM → polyline per strategi (fastest/safest/balanced)
      7. Folium heatmap HTML
      8. Log prediksi
    """
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    lat          = float(data["lat"])
    lon          = float(data["lon"])
    rr           = float(data.get("rr", 0))
    rh           = float(data.get("rh", 80))
    rr_yesterday = float(data.get("rr_yesterday", 0))
    rr_2days     = float(data.get("rr_2days", 0))
    kejadian     = int(data.get("kejadian_banjir", 0))

    # ── Step 1: Cuaca BMKG ───────────────────────────────────
    kec     = find_nearest_kecamatan(lat, lon)
    weather = fetch_bmkg_weather(kec["adm4"])
    if weather is None:
        weather = simulated_weather()
        weather["kecamatan"] = kec["nama"]
        weather["adm4"]      = kec["adm4"]

    # Override RR/RH dari input user (lebih real-time dari BMKG prakiraan)
    rr_used = rr if rr > 0 else weather.get("rainfall_24h", 0)
    rh_used = rh if rh > 0 else weather.get("humidity_now", 80)

    # ── Step 2: Risk ─────────────────────────────────────────
    rolling3    = hitung_rolling3(rr_used, rr_yesterday, rr_2days)
    risk_level  = hitung_risk_level(rolling3, rh_used, kejadian)
    risk_weight = hitung_risk_weight(risk_level)
    delay_factor = DELAY_MAP[risk_level]

    # ── Step 3: Pre-filter shelter (Haversine top-10) ────────
    shelters_sorted = sorted(
        SHELTER_DATA,
        key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"])
    )[:10]

    # ── Step 4: ORS Matrix → jarak_km_matrix (fitur RF) ─────
    matrix_dists = get_ors_matrix(lon, lat, shelters_sorted)

    # ── Step 5: RF predict per shelter ───────────────────────
    rf_results = []
    for i, shelter in enumerate(shelters_sorted):
        jarak_km_matrix = matrix_dists[i]
        strategy, proba = predict_route_strategy(jarak_km_matrix, risk_weight, rh_used, rr_used)
        rf_results.append({
            "shelter":          shelter["nama"],
            "lat_shelter":      shelter["lat"],
            "lon_shelter":      shelter["lon"],
            "jarak_km_matrix":  round(jarak_km_matrix, 3),
            "strategy":         strategy,
            "proba": {
                "Fastest":  round(proba[0], 3),
                "Safest":   round(proba[1], 3),
                "Balanced": round(proba[2], 3),
            }
        })

    # Sort: Safest > Balanced > Fastest, lalu jarak
    priority = {"Safest": 0, "Balanced": 1, "Fastest": 2}
    rf_results.sort(key=lambda x: (priority[x["strategy"]], x["jarak_km_matrix"]))
    top3 = rf_results[:3]

    # ── Step 6: Dijkstra OSMnx → rute aktual top-3 ──────────
    flood_pts   = find_nearby_flood_points(lat, lon)
    G_weighted  = apply_risk_weights_to_graph(G.copy(), risk_weight) if G else None
    routes_detail = []

    for item in top3:
        slat, slon = item["lat_shelter"], item["lon_shelter"]

        # Dijkstra (rute berbobot risiko)
        coords, dist_dijkstra = find_route_dijkstra(G_weighted, lat, lon, slat, slon)

        # ORS/OSRM polyline sesuai strategi RF
        strategy = item["strategy"]
        if strategy == "Fastest":
            route_data = get_fastest_route(lat, lon, slat, slon)
        elif strategy == "Safest":
            route_data = get_safest_route(lat, lon, slat, slon, flood_pts)
        else:
            route_data = get_balanced_route(lat, lon, slat, slon)

        # Gunakan jarak Dijkstra jika tersedia (lebih akurat karena berbobot risiko)
        jarak_aktual = dist_dijkstra or route_data.get("distance_km")
        estimasi_waktu = None
        if jarak_aktual:
            speed = 40 if risk_level <= 2 else 20
            estimasi_waktu = round((jarak_aktual / speed) * 60 * delay_factor, 1)

        row = {
            **item,
            "route_coords":          coords,             # Dijkstra (osmnx)
            "polyline_ors":          route_data.get("polyline"),  # ORS/OSRM
            "route_source":          route_data.get("source"),
            "jarak_aktual_km":       jarak_aktual,
            "estimasi_waktu_menit":  estimasi_waktu,
            "risk_level":            risk_level,
            "risk_weight":           round(risk_weight, 3),
            "rolling3":              round(rolling3, 1),
            "delay_factor":          delay_factor,
            "avoided_zones":         route_data.get("avoided_zones", 0),
        }
        routes_detail.append(row)

        # Log
        log_pred({
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "lat": lat, "lon": lon,
            "shelter":              item["shelter"],
            "jarak_km_matrix":      item["jarak_km_matrix"],
            "strategy_rf":          strategy,
            "risk_level":           risk_level,
            "risk_weight":          round(risk_weight, 3),
            "rolling3":             round(rolling3, 1),
            "rr": rr_used, "rh": rh_used,
            "route_source":         route_data.get("source"),
            "jarak_aktual_km":      jarak_aktual,
            "estimasi_waktu_menit": estimasi_waktu,
            "prob_fastest":         item["proba"]["Fastest"],
            "prob_safest":          item["proba"]["Safest"],
            "prob_balanced":        item["proba"]["Balanced"],
        })

    # ── Step 7: Folium Heatmap ───────────────────────────────
    heatmap_html = build_heatmap(routes_detail, lat, lon)

    return jsonify({
        "status": "ok",
        "input": {
            "lat": lat, "lon": lon,
            "rr": rr_used, "rh": rh_used,
            "rolling3": round(rolling3, 1),
            "risk_level": risk_level,
            "risk_label": RISK_LABELS[risk_level],
            "risk_weight": round(risk_weight, 3),
            "delay_factor": delay_factor,
        },
        "weather": weather,
        "kecamatan": kec["nama"],
        "nearby_flood_points": flood_pts[:5],
        "rekomendasi": routes_detail,
        "heatmap_html": heatmap_html,
        "routing_provider": "ORS" if ORS_KEY else "OSRM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/weather")
def api_weather():
    adm4 = request.args.get("adm4", "33.01.22.2001")
    w = fetch_bmkg_weather(adm4)
    if w is None:
        w = simulated_weather(); w["adm4"] = adm4
    return jsonify(w)


@app.route("/api/flood-points")
def api_flood_points():
    if df_flood.empty:
        return jsonify({"error": "No data"}), 503
    return jsonify(df_flood.to_dict(orient="records"))


@app.route("/api/shelters")
def api_shelters():
    return jsonify(SHELTER_DATA)


@app.route("/api/kecamatan")
def api_kecamatan():
    nama = request.args.get("nama")
    adm4 = request.args.get("adm4")
    if nama:
        return jsonify([k for k in KECAMATAN if nama.lower() in k["nama"].lower()])
    if adm4:
        hits = [k for k in KECAMATAN if k["adm4"] == adm4]
        return jsonify(hits[0] if hits else {"error": "not found"}), (200 if hits else 404)
    return jsonify({"count": len(KECAMATAN), "data": KECAMATAN})


@app.route("/api/prediction-log")
def api_prediction_log():
    if not os.path.exists(LOG_PATH):
        return jsonify({"rows": 0})
    df = pd.read_csv(LOG_PATH)
    if request.args.get("format") == "csv":
        return Response(
            df.to_csv(index=False), mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=prediction_log.csv"}
        )
    return jsonify({
        "rows": len(df),
        "latest": df.tail(5).to_dict(orient="records"),
        "retrain_ready": len(df) >= 1000
    })


@app.route("/health")
def health():
    log_n = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log_n = max(0, sum(1 for _ in f) - 1)
    return jsonify({
        "status":          "ok",
        "model":           rf_model is not None,
        "osmnx_graph":     G is not None,
        "flood_pts":       len(df_flood),
        "shelters":        len(SHELTER_DATA),
        "ors_key":         bool(ORS_KEY),
        "routing_chain":   "ORS → OSRM → Haversine",
        "prediction_log":  log_n,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
