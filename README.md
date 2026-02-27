# Bulls Eye — NBA Predictive Engine

> MSADS Time Series Analysis (31006) — Final Project
> Live dashboard: [https://gsx-2.com/bullseye](https://gsx-2.com/bullseye)

A GPU-accelerated machine-learning pipeline that forecasts **Chicago Bulls player points** and **win probability** for the next scheduled game. The pipeline runs nightly and publishes results to a public Streamlit dashboard.

---

## Overview

The system has three main layers:

1. **Data ingestion** — incrementally fetches NBA game logs from the NBA CDN (no API key required) for the 2022–26 seasons.
2. **Prediction models** — trains XGBoost models (GPU) to predict player points with calibrated prediction intervals (q10 / q50 / q90) and a binary win-probability classifier.
3. **Dashboard** — a Streamlit web application that displays per-player point predictions, historical accuracy, and win probability for the next Bulls game.

---

## Project Structure

```
FINAL_PROJECT_VF/
├── app.py                          # Streamlit dashboard (main entry point)
├── download_nba_model_data.py      # Step 1 — data fetch + XGBoost point model
├── model_data_quantile.py          # Step 2 — quantile regression (prediction intervals)
├── total_next_game_prediction.py   # Step 3 — win-probability XGBoost classifier
├── ensemble.py                     # Alternate Step 3 — XGBoost + LightGBM ensemble
├── run_pipeline.sh                 # Orchestrates Steps 1-3 + ntfy notifications
├── healthcheck.sh                  # Service health check
├── setup_service.sh                # One-time systemd service setup
├── bullseye.service                # systemd unit (Streamlit app)
├── bullseye-notify-failure.service # systemd unit (failure alerting)
├── nginx_bullseye.conf             # nginx reverse-proxy config
├── requirements.txt                # Full Python dependency list
├── test_GPU.py                     # GPU/CUDA sanity check
├── images/                         # Static assets (logo, team images)
├── static/                         # Additional static files
├── model/                          # Model artifacts and data (auto-generated)
│   ├── df_raw.parquet
│   ├── df_model.parquet
│   ├── df_model_quantile.parquet
│   ├── nba_xgb_model.json
│   ├── nba_xgb_q10_quantile.json
│   ├── nba_xgb_q50_quantile.json
│   ├── nba_xgb_q90_quantile.json
│   ├── calibration_factors.json
│   ├── nba_win_model.json
│   ├── predictions_history.csv
│   ├── predictions_history_quantile.csv
│   ├── bulls_dashboard.json
│   ├── bulls_dashboard_quantile.json
│   ├── win_probability.json
│   ├── meta.json
│   └── meta_quantile.json
└── logs/                           # Daily pipeline logs (auto-generated)
```

---

## Pipeline

The pipeline is run daily via cron at **04:00** by `run_pipeline.sh`. Each step sends a push notification via [ntfy](https://ntfy.sh).

### Step 1 — `download_nba_model_data.py`
- Incrementally fetches new game logs from `cdn.nba.com` using parallel threads.
- Builds rolling features (points, minutes, usage, volatility) for each player.
- Retrains an XGBoost regressor (GPU, `tree_method=hist`) on all historical data.
- Scans the next 7 days of the schedule to find the next Bulls game.
- Writes `bulls_dashboard.json` and `predictions_history.csv`.

### Step 2 — `model_data_quantile.py`
- Trains three XGBoost quantile regressors: **q10** (floor), **q50** (median), **q90** (ceiling).
- Adds richer volatility features: `PTS_STD_3/10`, `PTS_CV_5`, `MIN_STD_5`, `PTS_IQR_10`.
- Runs per-player binary-search calibration to achieve ~80% empirical interval coverage.
- Writes `bulls_dashboard_quantile.json`, `predictions_history_quantile.csv`, and per-quantile model files.

### Step 3 — `total_next_game_prediction.py`
- Reads team-level rolling features from `df_model.parquet`.
- Trains an XGBoost binary classifier (GPU) on historical game outcomes.
- Predicts `P(CHI wins)` for the next scheduled game.
- Writes `win_probability.json`.

> **Ensemble alternative**: `ensemble.py` runs the same classification task using a weighted average of XGBoost + LightGBM (both GPU-accelerated).

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (project developed on RTX 5080, 16.6 GB VRAM)
- CUDA Toolkit 12.x

### Software
- Python 3.12.3
- Virtual environment at `tsenvi/`

### Setup

```bash
# Create and activate virtual environment
python3 -m venv tsenvi
source tsenvi/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python test_GPU.py
```

---

## Running the Pipeline

```bash
# Activate environment
source /path/to/tsenvi/bin/activate

# Run all three steps sequentially (recommended)
bash run_pipeline.sh

# Or run individual steps
python download_nba_model_data.py
python model_data_quantile.py
python total_next_game_prediction.py
```

---

## Running the Dashboard Locally

```bash
source tsenvi/bin/activate
streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502` in your browser.

---

## Deployment

The production deployment uses the following stack:

| Component | Details |
|-----------|---------|
| App server | Streamlit on port `8502` |
| Process manager | systemd (`bullseye.service`) |
| Reverse proxy | nginx (`location ^~ /bullseye`) |
| DNS / TLS | Cloudflare Tunnel → nginx port 80 |
| Push alerts | ntfy.sh (`legion-9f3a2c-alerts` topic) |
| Static docs | nginx alias → `images/` directory |

### One-time server setup

```bash
bash setup_service.sh
```

This installs and enables the `bullseye.service` systemd unit.

### Service management

```bash
sudo systemctl status bullseye
sudo systemctl restart bullseye
sudo journalctl -u bullseye -f
```

### nginx configuration

Copy `nginx_bullseye.conf` to `/etc/nginx/sites-available/` and enable it:

```bash
sudo ln -s /etc/nginx/sites-available/nginx_bullseye.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo nginx -s reload
```

### Cron schedule

```cron
0 4 * * * /path/to/FINAL_PROJECT_VF/run_pipeline.sh
```

---

## Key Design Decisions

- **No NBA Stats API key required** — all data is fetched from `cdn.nba.com` public JSON endpoints, which bypass Akamai CDN bot-detection that blocks `stats.nba.com`.
- **ESPN API fallback** — used when the CDN schedule endpoint returns 403.
- **Incremental updates** — only new games are fetched each run; existing parquet files are extended, not rebuilt from scratch.
- **Per-player interval calibration** — a binary search adjusts each player's q10/q90 scale factors until ~80% of historical games fall inside the predicted interval.
- **GPU-first** — XGBoost and LightGBM are configured with `device='cuda'` and `tree_method='hist'`. PyTorch is used for device detection and CUDA verification.
- **Cache TTL** — `@st.cache_data(ttl=1800)` prevents stale data from persisting more than 30 minutes after a pipeline run.

---

## Data

| Field | Details |
|-------|---------|
| Source | NBA CDN (cdn.nba.com) |
| Seasons | 2022-23, 2023-24, 2024-25, 2025-26 |
| Granularity | Player × game |
| Size | ~97,000+ rows |
| Target variable | `PTS` (points per game) |
| Teams | All 30 NBA teams |
| Focus team | Chicago Bulls (`CHI`) |

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| XGBoost | 3.2.0 | Point prediction + win probability (GPU) |
| LightGBM | 4.6.0 | Ensemble win probability (GPU) |
| PyTorch | 2.10.0+cu128 | GPU detection / CUDA verification |
| Streamlit | 1.54.0 | Dashboard UI |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.8.0 | Metrics and preprocessing |
| statsmodels | 0.14.6 | Statistical analysis |

---

## License

This project was developed as a final project for MSADS Time Series Analysis (31006). All NBA data is consumed from publicly accessible endpoints and used for academic purposes only.
