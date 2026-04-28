"""
JALANKAN DI NOTEBOOK SETELAH TRAINING RF
Simpan model + metadata ke folder model/
Lalu push ke GitHub
"""

import pickle
import json
import os

# ── 1. Pastikan folder ada ───────────────────────────────────
os.makedirs("model", exist_ok=True)

# ── 2. Simpan model RF ───────────────────────────────────────
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("✅ model/rf_model.pkl tersimpan")

# ── 3. Simpan metadata ───────────────────────────────────────
# Baca shelter dari df_rute
shelters = (
    df_rute_ok[["shelter", "lat_shelter", "lon_shelter"]]
    .drop_duplicates(subset=["shelter"])
    .rename(columns={"shelter": "nama", "lat_shelter": "lat", "lon_shelter": "lon"})
    .to_dict("records")
)

metadata = {
    "P25": float(P25),
    "P50": float(P50),
    "P75": float(P75),
    "features": ["jarak_km_matrix", "risk_weight", "RH", "RR"],
    "classes": {0: "Fastest", 1: "Safest", 2: "Balanced"},
    "accuracy": float(acc),
    "kappa": float(kappa),
    "n_shelters": len(shelters),
    "shelters": shelters
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ model/metadata.json tersimpan ({len(shelters)} shelter)")
print("\nFile siap di-push ke GitHub:")
print("  model/rf_model.pkl")
print("  model/metadata.json")
