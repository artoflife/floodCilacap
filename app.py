"""
Flood Evacuation System — Flask App
Deploy ke Railway via GitHub: https://github.com/artoflife/flood
"""

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import HeatMap
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# ── CONFIG ───────────────────────────────────────────────────
ORS_KEY    = os.environ.get("ORS_KEY", "your_ors_key_here")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "rf_model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "model", "metadata.json")

# Bounding box Cilacap
CILACAP_BBOX = (-7.79, 108.52, -7.17, 109.41)  # south, west, north, east
CILACAP_CENTER = (-7.44, 108.96)

# ── LOAD MODEL ───────────────────────────────────────────────
print("Loading RF model...")
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

with open(DATA_PATH, "r") as f:
    metadata = json.load(f)

P25 = metadata["P25"]
P50 = metadata["P50"]
P75 = metadata["P75"]
SHELTER_DATA = metadata["shelters"]  # list of {nama, lat, lon}

print(f"Model loaded. Threshold P75={P75:.2f}")

# ── GRAPH JALAN ──────────────────────────────────────────────
print("Loading road graph Cilacap...")
G = ox.graph_from_bbox(
    north=CILACAP_BBOX[2], south=CILACAP_BBOX[0],
    east=CILACAP_BBOX[3], west=CILACAP_BBOX[1],
    network_type='drive'
)
print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")


# ── HELPER FUNCTIONS ─────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def hitung_rolling3(rr_today, rr_yesterday=None, rr_2days=None):
    """Hitung rolling3 dari data RR 3 hari."""
    r1 = rr_today or 0
    r2 = rr_yesterday or 0
    r3 = rr_2days or 0
    return r1 + r2 + r3


def hitung_risk_level(rolling3, rh, kejadian_banjir=0):
    """Novelty: risk level berbasis rolling3 + KejadianBanjir."""
    if rolling3 >= P75 or kejadian_banjir >= 1:
        return 4
    elif rolling3 >= P50 and rh >= 80:
        return 3
    elif rolling3 >= P25 and rh >= 75:
        return 2
    else:
        return 1


def hitung_risk_weight(risk_level):
    return (risk_level - 1) / 3


def get_ors_matrix(lon_src, lat_src, destinations):
    """Ambil jarak_km_matrix dari ORS Matrix API."""
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
            return r.json().get("distances", [[]])[0]
    except Exception as e:
        print(f"ORS Matrix error: {e}")
    return [None] * len(destinations)


def predict_route_strategy(jarak_km_matrix, risk_weight, rh, rr):
    """RF prediksi strategi rute."""
    X = np.array([[jarak_km_matrix, risk_weight, rh, rr]])
    pred = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0]
    label_map = {0: "Fastest", 1: "Safest", 2: "Balanced"}
    return label_map[pred], proba.tolist()


def apply_risk_weights_to_graph(G_copy, risk_weight):
    """Mutasi bobot graf berdasarkan risk_weight saat ini."""
    for u, v, k, data in G_copy.edges(data=True, keys=True):
        length = data.get("length", 100)  # meter
        # edge_weight = panjang × (1 + risk_weight)
        G_copy[u][v][k]["weight"] = length * (1 + risk_weight)
    return G_copy


def find_route_dijkstra(G_weighted, lat_src, lon_src, lat_dst, lon_dst):
    """Cari rute optimal dengan Dijkstra pada graf berbobot."""
    try:
        node_src = ox.nearest_nodes(G_weighted, lon_src, lat_src)
        node_dst = ox.nearest_nodes(G_weighted, lon_dst, lat_dst)
        path = nx.shortest_path(G_weighted, node_src, node_dst, weight="weight")
        coords = [(G_weighted.nodes[n]["y"], G_weighted.nodes[n]["x"]) for n in path]
        total_length = sum(
            G_weighted[path[i]][path[i+1]][0].get("length", 0)
            for i in range(len(path)-1)
        ) / 1000  # convert to km
        return coords, round(total_length, 2)
    except Exception as e:
        print(f"Dijkstra error: {e}")
        return None, None


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           center_lat=CILACAP_CENTER[0],
                           center_lon=CILACAP_CENTER[1])


@app.route("/api/evacuate", methods=["POST"])
def evacuate():
    """
    Endpoint utama: terima koordinat + kondisi cuaca,
    return rekomendasi rute evakuasi.

    Request JSON:
    {
        "lat": -7.5,
        "lon": 108.9,
        "rr": 22.0,
        "rh": 87,
        "rr_yesterday": 35.0,
        "rr_2days": 28.0,
        "kejadian_banjir": 0
    }
    """
    data = request.get_json()
    lat = float(data["lat"])
    lon = float(data["lon"])
    rr  = float(data.get("rr", 0))
    rh  = float(data.get("rh", 80))
    rr_yesterday = float(data.get("rr_yesterday", 0))
    rr_2days     = float(data.get("rr_2days", 0))
    kejadian     = int(data.get("kejadian_banjir", 0))

    # ── Step 1: Hitung kondisi risiko ────────────────────────
    rolling3    = hitung_rolling3(rr, rr_yesterday, rr_2days)
    risk_level  = hitung_risk_level(rolling3, rh, kejadian)
    risk_weight = hitung_risk_weight(risk_level)

    delay_map = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.5}
    delay_factor = delay_map[risk_level]

    # ── Step 2: Cari top-5 shelter terdekat (Haversine pre-filter) ──
    shelters_sorted = sorted(
        SHELTER_DATA,
        key=lambda s: haversine(lat, lon, s["lat"], s["lon"])
    )[:10]

    # ── Step 3: ORS Matrix untuk jarak_km_matrix ─────────────
    matrix_dists = get_ors_matrix(lon, lat, shelters_sorted)

    # ── Step 4: Prediksi RF per shelter ──────────────────────
    results = []
    for i, shelter in enumerate(shelters_sorted):
        jarak_km_matrix = matrix_dists[i] if matrix_dists[i] else \
                          haversine(lat, lon, shelter["lat"], shelter["lon"])

        strategy, proba = predict_route_strategy(
            jarak_km_matrix, risk_weight, rh, rr
        )

        results.append({
            "shelter": shelter["nama"],
            "lat_shelter": shelter["lat"],
            "lon_shelter": shelter["lon"],
            "jarak_km_matrix": round(jarak_km_matrix, 3),
            "strategy": strategy,
            "proba": {
                "Fastest": round(proba[0], 3),
                "Safest":  round(proba[1], 3),
                "Balanced":round(proba[2], 3)
            }
        })

    # Sort: Safest dulu, lalu Fastest, lalu Balanced
    priority = {"Safest": 0, "Balanced": 1, "Fastest": 2}
    results.sort(key=lambda x: (priority[x["strategy"]], x["jarak_km_matrix"]))
    top3 = results[:3]

    # ── Step 5: Dijkstra untuk top-3 shelter ─────────────────
    G_weighted = apply_risk_weights_to_graph(G.copy(), risk_weight)

    routes_detail = []
    for item in top3:
        coords, dist_km = find_route_dijkstra(
            G_weighted, lat, lon,
            item["lat_shelter"], item["lon_shelter"]
        )
        waktu_menit = None
        if dist_km:
            speed_kmh = 40 if risk_level <= 2 else 20
            waktu_menit = round((dist_km / speed_kmh) * 60 * delay_factor, 1)

        routes_detail.append({
            **item,
            "route_coords": coords,
            "jarak_aktual_km": dist_km,
            "estimasi_waktu_menit": waktu_menit,
            "risk_level": risk_level,
            "risk_weight": round(risk_weight, 3),
            "rolling3": round(rolling3, 1),
            "delay_factor": delay_factor
        })

    return jsonify({
        "status": "ok",
        "input": {
            "lat": lat, "lon": lon,
            "rr": rr, "rh": rh,
            "rolling3": round(rolling3, 1),
            "risk_level": risk_level,
            "risk_weight": round(risk_weight, 3)
        },
        "rekomendasi": routes_detail
    })


@app.route("/api/weather_today")
def weather_today():
    """
    Placeholder: return kondisi cuaca hari ini.
    Di production, koneksi ke API BMKG atau database cuaca.
    """
    return jsonify({
        "tanggal": datetime.now().strftime("%Y-%m-%d"),
        "rr": 15.0,
        "rh": 85,
        "rr_yesterday": 22.0,
        "rr_2days": 18.0,
        "sumber": "BMKG Tunggul Wulung (simulasi)"
    })


@app.route("/api/shelters")
def shelters():
    return jsonify(SHELTER_DATA)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "rf_loaded"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
