# Flood Evacuation System — Cilacap
Adaptive Geospatial Routing menggunakan Random Forest

## Cara Deploy ke Railway

### STEP 1 — Simpan model dari notebook

```python
# Jalankan save_model.py di notebook setelah training selesai
exec(open('save_model.py').read())
```

File yang dihasilkan:
- `model/rf_model.pkl`
- `model/metadata.json`

### STEP 2 — Struktur folder GitHub

```
flood/
├── app.py
├── requirements.txt
├── Procfile
├── railway.json
├── .gitignore
├── model/
│   ├── rf_model.pkl       ← dari notebook
│   └── metadata.json      ← dari notebook
└── templates/
    └── index.html
```

### STEP 3 — Push ke GitHub

```bash
git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/artoflife/flood.git
git push -u origin main
```

### STEP 4 — Deploy Railway

1. Buka https://railway.app
2. New Project → Deploy from GitHub repo
3. Pilih repo `artoflife/flood`
4. Set Environment Variables:
   ```
   ORS_KEY = your_openrouteservice_api_key
   ```
5. Railway otomatis detect Procfile dan deploy

### Environment Variables

| Variable | Keterangan |
|---|---|
| `ORS_KEY` | API key dari openrouteservice.org (gratis) |
| `PORT` | Diset otomatis oleh Railway |

### Endpoints

| Endpoint | Method | Keterangan |
|---|---|---|
| `/` | GET | Web-GIS interface |
| `/api/evacuate` | POST | Prediksi rute evakuasi |
| `/api/shelters` | GET | List semua shelter |
| `/api/weather_today` | GET | Kondisi cuaca hari ini |
| `/health` | GET | Health check Railway |

### Contoh Request API

```bash
curl -X POST https://your-app.railway.app/api/evacuate \
  -H "Content-Type: application/json" \
  -d '{
    "lat": -7.7267,
    "lon": 109.0153,
    "rr": 22.0,
    "rh": 87,
    "rr_yesterday": 35.0,
    "rr_2days": 28.0,
    "kejadian_banjir": 0
  }'
```
