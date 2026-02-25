#!/usr/bin/env python3
"""
NBA Points Prediction — Quantile XGBoost Regression (v2 — Calibrated Intervals)
Predicts three quantiles per player per game:
  q10 → floor   (10th percentile — low estimate)
  q50 → median  (50th percentile — central estimate, used as MAE benchmark)
  q90 → ceiling (90th percentile — high estimate)

KEY IMPROVEMENTS over v1:
  1. Richer volatility features: PTS_STD_3/10, PTS_CV_5, MIN_STD_5, MIN_CV_5, PTS_IQR_10
  2. Player-level interval calibration (Phase 3.5): binary-search per player to
     achieve ~80% empirical coverage instead of relying on global quantile widths.
  3. Calibration applied consistently in Phase 4 (history) and Phase 5 (tomorrow).

Inputs  : NBA CDN endpoints (no API key required)
Outputs :
  model/df_raw.parquet                    — raw player-game rows
  model/df_model_quantile.parquet         — feature-engineered dataset (quantile version)
  model/nba_xgb_q10_quantile.json         — XGBoost q10 model
  model/nba_xgb_q50_quantile.json         — XGBoost q50 model (central)
  model/nba_xgb_q90_quantile.json         — XGBoost q90 model
  model/calibration_factors.json          — per-player scale factors
  model/predictions_history_quantile.csv  — actual vs [q10, q50, q90] per player/game
  model/bulls_dashboard_quantile.json     — web-ready JSON with prediction intervals
  model/meta_quantile.json                — run metadata + metrics

Run:
  python download_nba_model_data_quantile.py
"""

# ── 0. GPU SETUP ───────────────────────────────────────────────────────────────
import os
import time
import json
import requests
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device : {device}")
if device.type == "cuda":
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA version  : {torch.version.cuda}")

USE_GPU = device.type == "cuda"
XGB_GPU_PARAMS = {"device": "cuda", "tree_method": "hist"} if USE_GPU else {}

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Quantiles to predict
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

# Target empirical coverage for calibration
TARGET_COVERAGE = 0.80

# ── 1. PATHS ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
MODEL_DIR   = PROJECT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

RAW_PATH          = MODEL_DIR / "df_raw.parquet"
DATA_PATH         = MODEL_DIR / "df_model_quantile.parquet"
HISTORY_PATH      = MODEL_DIR / "predictions_history_quantile.csv"
DASHBOARD_PATH    = MODEL_DIR / "bulls_dashboard_quantile.json"
META_PATH         = MODEL_DIR / "meta_quantile.json"
CALIBRATION_PATH  = MODEL_DIR / "calibration_factors.json"

MODEL_PATHS = {
    q: MODEL_DIR / f"nba_xgb_{q}_quantile.json"
    for q in QUANTILES
}

print(f"\nModel dir : {MODEL_DIR}")

# ── 2. DATA SOURCE CONFIG ──────────────────────────────────────────────────────
CDN_BOXSCORE   = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
CDN_SCHEDULE   = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
CDN_TIMEOUT    = 20
CDN_WORKERS    = 8
SEASONS        = ["2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = SEASONS[-1]
BULLS_ABBR     = "CHI"


def _fetch_boxscore(game_id: str) -> dict | None:
    for attempt in range(1, 4):
        try:
            r = requests.get(CDN_BOXSCORE.format(game_id), timeout=CDN_TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json().get("game")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(attempt * 5)
        except Exception:
            return None
    return None


def _parse_boxscore(game: dict) -> list[dict]:
    game_id   = game["gameId"]
    game_date = game.get("gameTimeUTC", "")[:10]
    home      = game["homeTeam"]
    away      = game["awayTeam"]
    home_tri  = home["teamTricode"]
    away_tri  = away["teamTricode"]
    home_pts  = int(home.get("score") or 0)
    away_pts  = int(away.get("score") or 0)

    rows: list[dict] = []
    for team, opp_tri, is_home in [(home, away_tri, True), (away, home_tri, False)]:
        team_tri = team["teamTricode"]
        if home_pts == away_pts:
            wl = None
        elif is_home:
            wl = "W" if home_pts > away_pts else "L"
        else:
            wl = "W" if away_pts > home_pts else "L"

        matchup = f"{team_tri} vs. {opp_tri}" if is_home else f"{team_tri} @ {opp_tri}"

        for p in team.get("players", []):
            if p.get("played") == "0":
                continue
            s = p.get("statistics", {})
            min_str = s.get("minutesCalculated", "PT0M")
            try:
                minutes = int(min_str[2: min_str.index("M")])
            except (ValueError, AttributeError):
                minutes = 0
            if minutes == 0:
                continue

            rows.append({
                "PLAYER_ID":         p["personId"],
                "PLAYER_NAME":       p["name"],
                "TEAM_ABBREVIATION": team_tri,
                "GAME_ID":           game_id,
                "GAME_DATE":         game_date,
                "MATCHUP":           matchup,
                "WL":                wl,
                "MIN":               minutes,
                "PTS":               int(s.get("points") or 0),
                "FGM":               int(s.get("fieldGoalsMade") or 0),
                "FGA":               int(s.get("fieldGoalsAttempted") or 0),
                "FG_PCT":            float(s.get("fieldGoalsPercentage") or 0),
                "FG3M":              int(s.get("threePointersMade") or 0),
                "FG3A":              int(s.get("threePointersAttempted") or 0),
                "FG3_PCT":           float(s.get("threePointersPercentage") or 0),
                "FTM":               int(s.get("freeThrowsMade") or 0),
                "FTA":               int(s.get("freeThrowsAttempted") or 0),
                "FT_PCT":            float(s.get("freeThrowsPercentage") or 0),
                "OREB":              int(s.get("reboundsOffensive") or 0),
                "DREB":              int(s.get("reboundsDefensive") or 0),
                "REB":               int(s.get("reboundsTotal") or 0),
                "AST":               int(s.get("assists") or 0),
                "STL":               int(s.get("steals") or 0),
                "BLK":               int(s.get("blocks") or 0),
                "TOV":               int(s.get("turnovers") or 0),
                "PF":                int(s.get("foulsPersonal") or 0),
                "PLUS_MINUS":        float(s.get("plusMinusPoints") or 0),
            })
    return rows


def fetch_season(season: str, known_game_ids: set[str] | None = None) -> pd.DataFrame:
    yy         = season[:4][2:]
    candidates = [f"002{yy}{i:05d}" for i in range(1, 1251)]
    if known_game_ids:
        candidates = [g for g in candidates if g not in known_game_ids]

    all_rows: list[dict] = []
    done = 0

    def fetch_one(gid: str) -> list[dict]:
        game = _fetch_boxscore(gid)
        if game is None or game.get("gameStatus") != 3:
            return []
        return _parse_boxscore(game)

    with ThreadPoolExecutor(max_workers=CDN_WORKERS) as pool:
        futures = {pool.submit(fetch_one, gid): gid for gid in candidates}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            done += 1
            if done % 200 == 0:
                print(f"      {done}/{len(candidates)} probed  |  "
                      f"{len(all_rows):,} player-game rows so far")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError(f"No completed games found for season {season}")
    return df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])


# ── 3. LOAD / UPDATE RAW DATA ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 1 — DATA INGESTION")
print("=" * 55)

if RAW_PATH.exists():
    df_raw = pd.read_parquet(RAW_PATH)
    df_raw["GAME_DATE"] = pd.to_datetime(df_raw["GAME_DATE"])
    last_date = df_raw["GAME_DATE"].max()
    print(f"Existing data: {len(df_raw):,} rows — last game: {last_date.date()}")

    existing_ids = set(df_raw["GAME_ID"].astype(str).unique())
    print(f"Checking for new {CURRENT_SEASON} games (skipping {len(existing_ids):,} known)...")
    new_df = fetch_season(CURRENT_SEASON, known_game_ids=existing_ids)
    new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])
    new_games = new_df[new_df["GAME_DATE"] > last_date]

    if new_games.empty:
        print("  → No new games since last update.")
    else:
        print(f"  → {len(new_games):,} new player-game rows")
        df_raw = (
            pd.concat([df_raw, new_games], ignore_index=True)
            .drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
        )
        df_raw.to_parquet(RAW_PATH, index=False)
        print(f"  → Raw data updated: {len(df_raw):,} total rows")
else:
    print("First run — fetching all seasons from scratch...")
    frames = []
    for season in SEASONS:
        print(f"  → {season} (probing up to 1,250 game IDs in parallel)...")
        s_df = fetch_season(season)
        frames.append(s_df)
        print(f"  → {season}: {len(s_df):,} player-game rows")
    df_raw = pd.concat(frames, ignore_index=True)
    df_raw["GAME_DATE"] = pd.to_datetime(df_raw["GAME_DATE"])
    df_raw.to_parquet(RAW_PATH, index=False)
    print(f"  → {len(df_raw):,} total rows saved")

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 2 — FEATURE ENGINEERING")
print("=" * 55)

df = df_raw.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

KEY_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]
available = [c for c in KEY_COLS if c in df.columns]
df_model  = df[available].copy()

df_model["IS_HOME"]  = df_model["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)
df_model["OPPONENT"] = df_model["MATCHUP"].apply(
    lambda x: str(x).split(" ")[-1] if pd.notna(x) else None
)
df_model["GAME_NUM"]  = df_model.groupby("PLAYER_ID").cumcount() + 1
df_model["DAYS_REST"] = (
    df_model.groupby("PLAYER_ID")["GAME_DATE"]
    .diff().dt.days.fillna(3)
)

print("  Building rolling player features...")

# ── Rolling means (original) ──────────────────────────────────────────────────
for window in [3, 5, 10]:
    df_model[f"PTS_ROLL_{window}"] = (
        df_model.groupby("PLAYER_ID")["PTS"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_model[f"MIN_ROLL_{window}"] = (
        df_model.groupby("PLAYER_ID")["MIN"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# ── Lag features (original) ───────────────────────────────────────────────────
for lag in [1, 2, 3]:
    df_model[f"PTS_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["PTS"].shift(lag)
    df_model[f"MIN_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["MIN"].shift(lag)

df_model["PTS_SEASON_AVG"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# ── Volatility features (v2 — NEW) ───────────────────────────────────────────
print("  Building volatility features (v2)...")

# Standard deviation at multiple windows
df_model["PTS_STD_5"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
)
df_model["PTS_STD_3"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
)
df_model["PTS_STD_10"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
)

# Coefficient of variation — normalises volatility by scoring level.
# A bench player with std=3 around a 6-pt avg is far more volatile than
# a star with std=3 around a 28-pt avg.
df_model["PTS_CV_5"] = df_model["PTS_STD_5"] / (df_model["PTS_SEASON_AVG"] + 1e-6)

# Minutes volatility — the single biggest driver of point variance for role
# players whose usage swings based on game-script or foul trouble.
df_model["MIN_STD_5"] = (
    df_model.groupby("PLAYER_ID")["MIN"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
)
df_model["MIN_CV_5"] = df_model["MIN_STD_5"] / (df_model["MIN_ROLL_5"] + 1e-6)

# Inter-quartile range over 10 games — more robust than std to outlier blowout
# games; captures the *typical* spread excluding extreme observations.
df_model["PTS_IQR_10"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(
        lambda x: (
            x.shift(1).rolling(10, min_periods=4).quantile(0.75)
            - x.shift(1).rolling(10, min_periods=4).quantile(0.25)
        )
    )
)

# ── Team defensive metrics (original) ────────────────────────────────────────
print("  Building team defensive metrics...")

df_team = (
    df_model.groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"])["PTS"]
    .sum().reset_index()
)
merged = df_team.merge(
    df_team[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]],
    on="GAME_ID", suffixes=("", "_OPP"),
)
merged = merged[merged["TEAM_ABBREVIATION"] != merged["TEAM_ABBREVIATION_OPP"]].copy()
merged["PTS_ALLOWED"] = merged["PTS_OPP"]

team_def = (
    merged[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS", "PTS_ALLOWED"]]
    .copy()
    .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    .reset_index(drop=True)
)

team_def["DEF_PTS_ALLOWED_AVG"] = (
    team_def.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
for window in [5, 10]:
    team_def[f"DEF_PTS_ALLOWED_ROLL_{window}"] = (
        team_def.groupby("TEAM_ABBREVIATION")["PTS_ALLOWED"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
    )
team_def["TEAM_GAME_NUM"] = team_def.groupby("TEAM_ABBREVIATION").cumcount() + 1

league_avg_pts = team_def["PTS_ALLOWED"].mean()
PRIOR_WEIGHT   = 5


def compute_regularized_def(group, league_avg, prior_weight):
    shifted   = group["PTS_ALLOWED"].shift(1)
    cum_sum   = shifted.expanding().sum()
    cum_count = shifted.expanding().count()
    return (cum_sum + prior_weight * league_avg) / (cum_count + prior_weight)


_reg_parts = []
for _, _grp in team_def.groupby("TEAM_ABBREVIATION", sort=False):
    _reg_parts.append(compute_regularized_def(_grp, league_avg_pts, PRIOR_WEIGHT))
team_def["DEF_PTS_ALLOWED_REGULARIZED"] = pd.concat(_reg_parts).sort_index()

opp_defense = team_def[[
    "GAME_DATE", "TEAM_ABBREVIATION",
    "DEF_PTS_ALLOWED_AVG", "DEF_PTS_ALLOWED_ROLL_5", "DEF_PTS_ALLOWED_ROLL_10",
    "DEF_PTS_ALLOWED_REGULARIZED", "TEAM_GAME_NUM",
]].rename(columns={
    "TEAM_ABBREVIATION":           "OPPONENT",
    "DEF_PTS_ALLOWED_AVG":         "OPP_DEF_AVG",
    "DEF_PTS_ALLOWED_ROLL_5":      "OPP_DEF_ROLL_5",
    "DEF_PTS_ALLOWED_ROLL_10":     "OPP_DEF_ROLL_10",
    "DEF_PTS_ALLOWED_REGULARIZED": "OPP_DEF_REG",
    "TEAM_GAME_NUM":               "OPP_GAMES_PLAYED",
})
opp_defense["OPPONENT"] = opp_defense["OPPONENT"].astype("object")
df_model["OPPONENT"]    = df_model["OPPONENT"].astype("object")

df_model = df_model.merge(opp_defense, on=["OPPONENT", "GAME_DATE"], how="left")

for col in ["OPP_DEF_REG", "OPP_DEF_AVG", "OPP_DEF_ROLL_5", "OPP_DEF_ROLL_10"]:
    df_model[col] = df_model[col].fillna(league_avg_pts)

df_model.to_parquet(DATA_PATH, index=False)
print(f"  → Feature dataset: {df_model.shape}")

# ── 5. TRAIN QUANTILE MODELS ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 — MODEL TRAINING (Quantile XGBoost)")
print("=" * 55)

# v2: FEATURE_COLS now includes six new volatility signals
FEATURE_COLS = [
    # Rolling scoring means
    "PTS_ROLL_3",     "PTS_ROLL_5",    "PTS_ROLL_10",
    # Lag scoring
    "PTS_LAG_1",      "PTS_LAG_2",     "PTS_LAG_3",
    # Rolling minutes means
    "MIN_ROLL_3",     "MIN_ROLL_5",    "MIN_ROLL_10",
    # Lag minutes
    "MIN_LAG_1",      "MIN_LAG_2",     "MIN_LAG_3",
    # Season-level scoring
    "PTS_SEASON_AVG",
    # ── NEW: volatility signals ───────────────────────────────────────────────
    # These teach q10/q90 models to produce tighter intervals for consistent
    # players and wider intervals for volatile ones, instead of using a single
    # global width learned from the full dataset.
    "PTS_STD_3",      "PTS_STD_5",     "PTS_STD_10",   # std at multiple horizons
    "PTS_CV_5",       # coefficient of variation — volatility relative to avg pts
    "MIN_STD_5",      "MIN_CV_5",      # minute volatility (usage uncertainty)
    "PTS_IQR_10",     # robust spread measure over last 10 games
    # ─────────────────────────────────────────────────────────────────────────
    # Game context
    "IS_HOME",        "DAYS_REST",     "GAME_NUM",
    # Opponent defense
    "OPP_DEF_AVG",    "OPP_DEF_ROLL_5", "OPP_DEF_ROLL_10",
    "OPP_DEF_REG",    "OPP_GAMES_PLAYED",
]
TARGET = "PTS"

df_clean = df_model.dropna(subset=FEATURE_COLS + [TARGET]).copy()
print(f"Training rows : {len(df_clean):,}  (dropped {len(df_model) - len(df_clean):,} NaN rows)")

split_idx = int(len(df_clean) * 0.80)
X_train = df_clean[FEATURE_COLS].iloc[:split_idx]
y_train = df_clean[TARGET].iloc[:split_idx]
X_val   = df_clean[FEATURE_COLS].iloc[split_idx:]
y_val   = df_clean[TARGET].iloc[split_idx:]

print(f"  Train : {len(X_train):,} rows")
print(f"  Val   : {len(X_val):,} rows  (chronological 80/20)\n")

models      = {}
val_preds   = {}
val_metrics = {}

for q_name, q_alpha in QUANTILES.items():
    print(f"  ── Training {q_name} (α={q_alpha}) ──")
    m = xgb.XGBRegressor(
        n_estimators          = 500,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        early_stopping_rounds = 30,
        objective             = "reg:quantileerror",
        quantile_alpha        = q_alpha,
        eval_metric           = "quantile",
        **XGB_GPU_PARAMS,
    )
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds = m.get_booster().inplace_predict(X_val.to_numpy()).clip(0)
    val_preds[q_name] = preds

    if q_name == "q50":
        mae  = float(mean_absolute_error(y_val, preds))
        rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
        r2   = float(1 - np.sum((y_val.values - preds)**2) /
                         np.sum((y_val.values - y_val.mean())**2))
        val_metrics["mae"]  = mae
        val_metrics["rmse"] = rmse
        val_metrics["r2"]   = r2
        print(f"  q50  → MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")

    if q_name == "q90" and "q10" in val_preds:
        raw_coverage = float(np.mean(
            (y_val.values >= val_preds["q10"]) &
            (y_val.values <= val_preds["q90"])
        ))
        raw_width = float(np.mean(val_preds["q90"] - val_preds["q10"]))
        val_metrics["raw_interval_coverage_80pct"] = round(raw_coverage, 3)
        val_metrics["raw_avg_interval_width_pts"]  = round(raw_width, 2)
        print(f"\n  Raw [q10–q90] coverage : {raw_coverage:.1%}  "
              f"(target ~80%)  avg width: {raw_width:.1f} pts  ← before calibration")

    models[q_name] = m
    m.save_model(str(MODEL_PATHS[q_name]))
    print(f"  → Saved: {MODEL_PATHS[q_name]}\n")

# ── 6. PLAYER-LEVEL INTERVAL CALIBRATION ─────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3.5 — PLAYER-LEVEL INTERVAL CALIBRATION (v2)")
print("=" * 55)
print(
    "  Rationale: the global q10/q90 models learn average dataset variance.\n"
    "  A consistent role player (e.g. Yabusele) ends up with the same\n"
    "  interval width as a volatile star. Per-player calibration shrinks\n"
    "  intervals for consistent players and widens them for volatile ones\n"
    "  so that each player individually achieves ~80% empirical coverage.\n"
)

val_df = df_clean.iloc[split_idx:].copy()
X_val_np = val_df[FEATURE_COLS].to_numpy()

val_df["q10_raw"] = models["q10"].get_booster().inplace_predict(X_val_np).clip(0)
val_df["q50_raw"] = models["q50"].get_booster().inplace_predict(X_val_np).clip(0)
val_df["q90_raw"] = models["q90"].get_booster().inplace_predict(X_val_np).clip(0)


def compute_player_scale_factor(
    actual: np.ndarray,
    q50: np.ndarray,
    q10_raw: np.ndarray,
    q90_raw: np.ndarray,
    target: float = TARGET_COVERAGE,
) -> float:
    """
    Binary-search for the multiplicative scale factor `s` such that:
        coverage( actual in [q50 - s*hw, q50 + s*hw] ) ≈ target
    where hw = (q90_raw - q10_raw) / 2.

    The half-width is anchored symmetrically around q50 so that the median
    prediction (the best point estimate) is never shifted — only the interval
    width changes.

    Returns s ∈ [0.1, 5.0].
    """
    half_width = (q90_raw - q10_raw) / 2.0

    lo, hi = 0.1, 5.0
    for _ in range(30):          # 30 iterations → precision < 1e-8
        mid = (lo + hi) / 2.0
        coverage = np.mean(
            (actual >= q50 - mid * half_width) &
            (actual <= q50 + mid * half_width)
        )
        if coverage < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


# ── Global fallback (players without enough val data) ────────────────────────
global_factor = compute_player_scale_factor(
    actual   = y_val.values,
    q50      = val_df["q50_raw"].values,
    q10_raw  = val_df["q10_raw"].values,
    q90_raw  = val_df["q90_raw"].values,
)
print(f"  Global fallback scale factor : {global_factor:.4f}")

# ── Per-player calibration (require ≥ 10 validation games for stability) ─────
MIN_VAL_GAMES = 10
player_counts  = val_df.groupby("PLAYER_ID").size()
players_enough = player_counts[player_counts >= MIN_VAL_GAMES].index

calibration_factors: dict[str, float] = {}   # keyed by str(PLAYER_ID)

for pid in players_enough:
    grp = val_df[val_df["PLAYER_ID"] == pid]
    factor = compute_player_scale_factor(
        actual   = grp["PTS"].values,
        q50      = grp["q50_raw"].values,
        q10_raw  = grp["q10_raw"].values,
        q90_raw  = grp["q90_raw"].values,
    )
    calibration_factors[str(pid)] = factor

print(f"  Players calibrated : {len(calibration_factors)}  "
      f"(≥{MIN_VAL_GAMES} val games each)")
print(f"  Players using global fallback : "
      f"{val_df['PLAYER_ID'].nunique() - len(calibration_factors)}")

# ── Show sample factors ───────────────────────────────────────────────────────
print("\n  Sample calibration factors (factor < 1.0 → narrow, > 1.0 → wide):")
sample_pids = list(calibration_factors.keys())[:10]
for pid in sample_pids:
    name = val_df[val_df["PLAYER_ID"] == int(pid)]["PLAYER_NAME"].iloc[0]
    f    = calibration_factors[pid]
    tag  = "← narrowed" if f < 0.95 else ("← widened" if f > 1.05 else "≈ unchanged")
    print(f"    {name:<28}  factor: {f:.4f}  {tag}")

# ── Verify post-calibration coverage on validation set ───────────────────────
factors_arr = (
    val_df["PLAYER_ID"]
    .astype(str)
    .map(calibration_factors)
    .fillna(global_factor)
    .values
)
hw_val        = (val_df["q90_raw"].values - val_df["q10_raw"].values) / 2.0
q10_cal_val   = (val_df["q50_raw"].values - factors_arr * hw_val).clip(0)
q90_cal_val   = (val_df["q50_raw"].values + factors_arr * hw_val).clip(0)
cal_coverage  = float(np.mean(
    (y_val.values >= q10_cal_val) & (y_val.values <= q90_cal_val)
))
cal_width     = float(np.mean(q90_cal_val - q10_cal_val))

val_metrics["cal_interval_coverage_80pct"] = round(cal_coverage, 3)
val_metrics["cal_avg_interval_width_pts"]  = round(cal_width, 2)

print(f"\n  Post-calibration [q10–q90] coverage : {cal_coverage:.1%}  (target ~80%)")
print(f"  Post-calibration avg interval width : {cal_width:.1f} pts  "
      f"(was {val_metrics['raw_avg_interval_width_pts']:.1f} pts)")

# ── Persist calibration factors ───────────────────────────────────────────────
cal_payload = {
    "global_factor":  global_factor,
    "target_coverage": TARGET_COVERAGE,
    "min_val_games":   MIN_VAL_GAMES,
    "player_factors":  calibration_factors,
}
CALIBRATION_PATH.write_text(json.dumps(cal_payload, indent=2))
print(f"\n  → Calibration factors saved : {CALIBRATION_PATH}")


# ── Helper: apply calibration to any DataFrame ───────────────────────────────
def predict_with_calibration(
    df_in: pd.DataFrame,
    models: dict,
    feature_cols: list[str],
    calibration_factors: dict[str, float],
    global_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (q10_cal, q50, q90_cal) arrays aligned with df_in rows.

    The q50 (median) is unchanged — it is the best point prediction.
    Only the interval half-width is scaled by the per-player factor so
    that each player's interval reflects their own historical volatility.
    """
    X = df_in[feature_cols].to_numpy()

    q10_raw = models["q10"].get_booster().inplace_predict(X).clip(0)
    q50     = models["q50"].get_booster().inplace_predict(X).clip(0)
    q90_raw = models["q90"].get_booster().inplace_predict(X).clip(0)

    hw = (q90_raw - q10_raw) / 2.0

    factors = (
        df_in["PLAYER_ID"]
        .astype(str)
        .map(calibration_factors)
        .fillna(global_factor)
        .values
    )

    q10_cal = (q50 - factors * hw).clip(0)
    q90_cal = (q50 + factors * hw).clip(0)

    return q10_cal.round(1), q50.round(1), q90_cal.round(1)


# ── 7. PREDICTIONS HISTORY ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 4 — PREDICTIONS HISTORY")
print("=" * 55)

df_pred = df_clean.copy()

q10_all, q50_all, q90_all = predict_with_calibration(
    df_in               = df_pred,
    models              = models,
    feature_cols        = FEATURE_COLS,
    calibration_factors = calibration_factors,
    global_factor       = global_factor,
)

df_pred["PTS_Q10"]       = q10_all
df_pred["PTS_Q50"]       = q50_all
df_pred["PTS_Q90"]       = q90_all
df_pred["PTS_PREDICTED"] = df_pred["PTS_Q50"]
df_pred["PTS_ERROR"]     = (df_pred["PTS_Q50"] - df_pred[TARGET]).round(1)
df_pred["IN_INTERVAL"]   = (
    (df_pred[TARGET] >= df_pred["PTS_Q10"]) &
    (df_pred[TARGET] <= df_pred["PTS_Q90"])
).astype(int)
df_pred["IS_VALIDATION"] = np.arange(len(df_pred)) >= split_idx

history_log = df_pred[[
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "OPPONENT", "IS_HOME",
    TARGET, "PTS_Q10", "PTS_Q50", "PTS_Q90",
    "PTS_PREDICTED", "PTS_ERROR", "IN_INTERVAL", "IS_VALIDATION",
]].sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

history_log["GAME_DATE"] = history_log["GAME_DATE"].dt.date
history_log.to_csv(HISTORY_PATH, index=False)

val_coverage = history_log[history_log["IS_VALIDATION"] == True]["IN_INTERVAL"].mean()
val_width    = (
    history_log[history_log["IS_VALIDATION"] == True]["PTS_Q90"]
    - history_log[history_log["IS_VALIDATION"] == True]["PTS_Q10"]
).mean()
print(f"  → {len(history_log):,} rows saved to {HISTORY_PATH}")
print(f"  → Val interval coverage [q10–q90] : {val_coverage:.1%}  (target ~80%)")
print(f"  → Val avg interval width           : {val_width:.1f} pts")

# ── 8. TOMORROW'S SCHEDULE (Chicago Bulls) ────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 — TOMORROW'S PREDICTIONS (CHI)")
print("=" * 55)

tomorrow      = pd.Timestamp.now() + pd.Timedelta(days=1)
tomorrow_str  = tomorrow.strftime("%Y-%m-%d")
tomorrow_date = tomorrow.date()
chi_next_game = None

# ESPN abbreviation → NBA tricode (only entries that differ)
ESPN_TO_NBA = {
    "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "GS": "GSW", "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS",
}

def _find_chi_game_espn(date_str: str) -> tuple[int, str] | None:
    """Query ESPN public API. Returns (is_home, opponent_tricode) or None."""
    date_key = date_str.replace("-", "")
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        f"/scoreboard?dates={date_key}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    for event in r.json().get("events", []):
        comp = event["competitions"][0]
        home_raw = next(
            t for t in comp["competitors"] if t["homeAway"] == "home"
        )["team"]["abbreviation"]
        away_raw = next(
            t for t in comp["competitors"] if t["homeAway"] == "away"
        )["team"]["abbreviation"]
        home = ESPN_TO_NBA.get(home_raw, home_raw)
        away = ESPN_TO_NBA.get(away_raw, away_raw)
        if home == BULLS_ABBR:
            return 1, away
        if away == BULLS_ABBR:
            return 0, home
    return None

chi_is_home  = None
chi_opponent = None

# ── 1. Primary: CDN schedule ──────────────────────────────────────────────────
try:
    print("Fetching season schedule from cdn.nba.com...")
    sched_r = requests.get(CDN_SCHEDULE, timeout=30)
    sched_r.raise_for_status()
    all_games_sched = [
        g
        for day in sched_r.json()["leagueSchedule"]["gameDates"]
        for g in day["games"]
        if g["gameId"].startswith("002")
    ]
    chi_games_sched = [
        g for g in all_games_sched
        if g.get("gameDateEst", "")[:10] == tomorrow_str
        and (
            g["homeTeam"]["teamTricode"] == BULLS_ABBR
            or g["awayTeam"]["teamTricode"] == BULLS_ABBR
        )
    ]
    if chi_games_sched:
        g            = chi_games_sched[0]
        chi_is_home  = int(g["homeTeam"]["teamTricode"] == BULLS_ABBR)
        chi_opponent = (
            g["awayTeam"]["teamTricode"] if chi_is_home
            else g["homeTeam"]["teamTricode"]
        )
        print(f"  → CHI {'(HOME)' if chi_is_home else '(AWAY)'} vs {chi_opponent}  [CDN]")
    else:
        print(f"  → CHI has no game on {tomorrow_str} [CDN]")
except Exception as exc:
    print(f"  → CDN schedule failed: {exc}")

# ── 2. Fallback: ESPN public API ──────────────────────────────────────────────
if chi_is_home is None:
    try:
        print("  → Trying ESPN API fallback...")
        espn_result = _find_chi_game_espn(tomorrow_str)
        if espn_result:
            chi_is_home, chi_opponent = espn_result
            print(f"  → CHI {'(HOME)' if chi_is_home else '(AWAY)'} vs {chi_opponent}  [ESPN]")
        else:
            print(f"  → CHI has no game on {tomorrow_str} [ESPN]")
    except Exception as exc2:
        print(f"  → ESPN fallback also failed: {exc2}")

if chi_is_home is not None and chi_opponent is not None:
    chi_players = df_clean[df_clean["TEAM_ABBREVIATION"] == BULLS_ABBR].copy()
    latest_chi  = (
        chi_players
        .sort_values("GAME_DATE")
        .groupby("PLAYER_ID")
        .last()
        .reset_index()
    )
    cutoff     = df_model["GAME_DATE"].max() - pd.Timedelta(days=30)
    latest_chi = latest_chi[latest_chi["GAME_DATE"] >= cutoff].copy()

    latest_chi["IS_HOME"]   = chi_is_home
    latest_chi["OPPONENT"]  = chi_opponent
    latest_chi["DAYS_REST"] = (
        pd.Timestamp(tomorrow_date) - pd.to_datetime(latest_chi["GAME_DATE"])
    ).dt.days

    opp_def_latest = (
        team_def[team_def["TEAM_ABBREVIATION"] == chi_opponent]
        .sort_values("GAME_DATE")
        .iloc[-1]
    )
    latest_chi["OPP_DEF_AVG"]      = opp_def_latest.get("DEF_PTS_ALLOWED_AVG",        league_avg_pts)
    latest_chi["OPP_DEF_ROLL_5"]   = opp_def_latest.get("DEF_PTS_ALLOWED_ROLL_5",      league_avg_pts)
    latest_chi["OPP_DEF_ROLL_10"]  = opp_def_latest.get("DEF_PTS_ALLOWED_ROLL_10",     league_avg_pts)
    latest_chi["OPP_DEF_REG"]      = opp_def_latest.get("DEF_PTS_ALLOWED_REGULARIZED", league_avg_pts)
    latest_chi["OPP_GAMES_PLAYED"] = opp_def_latest.get("TEAM_GAME_NUM", 0)

    for col in FEATURE_COLS:
        latest_chi[col] = latest_chi[col].fillna(df_clean[col].mean())

    # Apply calibrated predictions for tomorrow
    q10_tmrw, q50_tmrw, q90_tmrw = predict_with_calibration(
        df_in               = latest_chi,
        models              = models,
        feature_cols        = FEATURE_COLS,
        calibration_factors = calibration_factors,
        global_factor       = global_factor,
    )
    latest_chi["PTS_Q10"] = q10_tmrw
    latest_chi["PTS_Q50"] = q50_tmrw
    latest_chi["PTS_Q90"] = q90_tmrw

    chi_next_game = {
        "game_date": tomorrow_str,
        "opponent":  chi_opponent,
        "is_home":   chi_is_home,
        "predictions": (
            latest_chi[[
                "PLAYER_NAME", "PLAYER_ID",
                "PTS_Q10", "PTS_Q50", "PTS_Q90",
                "PTS_ROLL_5", "PTS_SEASON_AVG", "DAYS_REST",
            ]]
            .rename(columns={
                "PTS_Q10":        "pts_floor",
                "PTS_Q50":        "pts_predicted",
                "PTS_Q90":        "pts_ceiling",
                "PTS_ROLL_5":     "pts_roll_5",
                "PTS_SEASON_AVG": "pts_season_avg",
                "DAYS_REST":      "days_rest",
            })
            .assign(
                pts_floor      = lambda x: x["pts_floor"].apply(lambda v: round(float(v), 1)),
                pts_predicted  = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
                pts_ceiling    = lambda x: x["pts_ceiling"].apply(lambda v: round(float(v), 1)),
                pts_roll_5     = lambda x: x["pts_roll_5"].apply(lambda v: round(float(v), 1)),
                pts_season_avg = lambda x: x["pts_season_avg"].apply(lambda v: round(float(v), 1)),
                days_rest      = lambda x: x["days_rest"].astype(int),
                interval_width = lambda x: (x["pts_ceiling"] - x["pts_floor"]).round(1),
                cal_factor     = lambda x: x["PLAYER_ID"].astype(str).map(
                    calibration_factors
                ).fillna(global_factor).round(4),
            )
            .drop(columns=["PLAYER_ID"])
            .rename(columns={"PLAYER_NAME": "player_name"})
            .sort_values("pts_predicted", ascending=False)
            .to_dict(orient="records")
        ),
    }
    print(f"  → {len(chi_next_game['predictions'])} CHI players to predict")
    print(f"\n  Sample predictions (calibrated):")
    print(f"  {'Player':<28}  {'Floor':>5}  {'Median':>6}  {'Ceiling':>7}  {'Width':>5}  {'Scale':>5}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}")
    for p in chi_next_game["predictions"][:8]:
        print(
            f"  {p['player_name']:<28}  "
            f"{p['pts_floor']:>5}  "
            f"{p['pts_predicted']:>6}  "
            f"{p['pts_ceiling']:>7}  "
            f"{p['interval_width']:>5}  "
            f"{p['cal_factor']:>5}"
        )
    chi_next_game = None

# ── 9. BULLS DASHBOARD JSON ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 6 — BULLS DASHBOARD JSON")
print("=" * 55)

chi_history = history_log[history_log["TEAM_ABBREVIATION"] == BULLS_ABBR].copy()
chi_history["GAME_DATE"] = pd.to_datetime(chi_history["GAME_DATE"])

last_10_game_ids = (
    chi_history[["GAME_ID", "GAME_DATE"]]
    .drop_duplicates()
    .sort_values("GAME_DATE", ascending=False)
    .head(10)["GAME_ID"]
    .tolist()
)

recent_games = []
for game_id in last_10_game_ids:
    game_rows = chi_history[chi_history["GAME_ID"] == game_id].copy()
    if game_rows.empty:
        continue
    first  = game_rows.iloc[0]
    result = None
    if "WL" in df_model.columns:
        wl_rows = df_model[
            (df_model["GAME_ID"] == game_id) &
            (df_model["TEAM_ABBREVIATION"] == BULLS_ABBR)
        ]["WL"]
        if not wl_rows.empty:
            result = str(wl_rows.iloc[0])

    players_in_game = (
        game_rows[game_rows["PTS"] > 0][[
            "PLAYER_NAME", "PTS",
            "PTS_Q10", "PTS_Q50", "PTS_Q90",
            "PTS_ERROR", "IN_INTERVAL"
        ]]
        .rename(columns={
            "PLAYER_NAME": "player_name",
            "PTS":         "pts_actual",
            "PTS_Q10":     "pts_floor",
            "PTS_Q50":     "pts_predicted",
            "PTS_Q90":     "pts_ceiling",
            "PTS_ERROR":   "pts_error",
            "IN_INTERVAL": "in_interval",
        })
        .assign(
            pts_floor      = lambda x: x["pts_floor"].apply(lambda v: round(float(v), 1)),
            pts_predicted  = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
            pts_ceiling    = lambda x: x["pts_ceiling"].apply(lambda v: round(float(v), 1)),
            pts_error      = lambda x: x["pts_error"].apply(lambda v: round(float(v), 1)),
            interval_width = lambda x: (x["pts_ceiling"] - x["pts_floor"]).round(1),
        )
        .sort_values("pts_actual", ascending=False)
        .to_dict(orient="records")
    )

    recent_games.append({
        "game_id":   game_id,
        "game_date": str(first["GAME_DATE"].date()),
        "opponent":  str(first["OPPONENT"]),
        "is_home":   int(first["IS_HOME"]),
        "result":    result,
        "players":   players_in_game,
    })

dashboard = {
    "team":         "Chicago Bulls",
    "abbreviation": BULLS_ABBR,
    "last_updated": str(pd.Timestamp.now().date()),
    "model_version": "v2_calibrated",
    "model_metrics": {
        "q50_mae":                         round(val_metrics.get("mae",  0), 3),
        "q50_rmse":                        round(val_metrics.get("rmse", 0), 3),
        "q50_r2":                          round(val_metrics.get("r2",   0), 3),
        # Raw (pre-calibration) interval stats
        "raw_interval_coverage_80pct":     val_metrics.get("raw_interval_coverage_80pct", None),
        "raw_avg_interval_width_pts":      val_metrics.get("raw_avg_interval_width_pts",  None),
        # Calibrated interval stats
        "cal_interval_coverage_80pct":     val_metrics.get("cal_interval_coverage_80pct", None),
        "cal_avg_interval_width_pts":      val_metrics.get("cal_avg_interval_width_pts",  None),
        "calibrated_players":              len(calibration_factors),
        "global_fallback_factor":          global_factor,
        "best_iter_q10":                   int(models["q10"].best_iteration),
        "best_iter_q50":                   int(models["q50"].best_iteration),
        "best_iter_q90":                   int(models["q90"].best_iteration),
    },
    "recent_games": recent_games,
    "next_game":    chi_next_game,
}

DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2, default=str))
print(f"  → Dashboard saved : {DASHBOARD_PATH}")
print(f"  → Recent games    : {len(recent_games)}")
print(f"  → Next game       : {chi_next_game['game_date'] if chi_next_game else 'N/A'}")

# ── 10. METADATA ──────────────────────────────────────────────────────────────
meta = {
    "model_version":   "v2_calibrated",
    "last_update":     str(pd.Timestamp.now().date()),
    "last_game_date":  str(df_model["GAME_DATE"].max().date()),
    "total_rows":      len(df_model),
    "train_rows":      len(df_clean),
    "split_idx":       split_idx,
    "q50_mae":         round(val_metrics.get("mae",  0), 3),
    "q50_rmse":        round(val_metrics.get("rmse", 0), 3),
    "q50_r2":          round(val_metrics.get("r2",   0), 3),
    "raw_interval_coverage_80pct":  val_metrics.get("raw_interval_coverage_80pct", None),
    "raw_avg_interval_width_pts":   val_metrics.get("raw_avg_interval_width_pts",  None),
    "cal_interval_coverage_80pct":  val_metrics.get("cal_interval_coverage_80pct", None),
    "cal_avg_interval_width_pts":   val_metrics.get("cal_avg_interval_width_pts",  None),
    "calibrated_players":           len(calibration_factors),
    "global_fallback_factor":       global_factor,
    "best_iter_q10":   int(models["q10"].best_iteration),
    "best_iter_q50":   int(models["q50"].best_iteration),
    "best_iter_q90":   int(models["q90"].best_iteration),
    "seasons":         SEASONS,
    "feature_cols":    FEATURE_COLS,
    "quantiles":       QUANTILES,
    "target_coverage": TARGET_COVERAGE,
    "min_val_games_for_calibration": MIN_VAL_GAMES,
}
META_PATH.write_text(json.dumps(meta, indent=2))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"  q50 MAE    : {val_metrics.get('mae',  0):.3f} pts  ← compare vs baseline 4.010")
print(f"  q50 RMSE   : {val_metrics.get('rmse', 0):.3f} pts")
print(f"  q50 R²     : {val_metrics.get('r2',   0):.3f}")
print()
print(f"  Interval [q10–q90] — RAW       : "
      f"coverage {val_metrics.get('raw_interval_coverage_80pct', 0):.1%}  "
      f"| width {val_metrics.get('raw_avg_interval_width_pts', 0):.1f} pts")
print(f"  Interval [q10–q90] — CALIBRATED: "
      f"coverage {val_metrics.get('cal_interval_coverage_80pct', 0):.1%}  "
      f"| width {val_metrics.get('cal_avg_interval_width_pts', 0):.1f} pts  ← target ~80%")
print()
print("  Output files:")
print(f"    {RAW_PATH}")
print(f"    {DATA_PATH}")
for q, p in MODEL_PATHS.items():
    print(f"    {p}  ← {q} model")
print(f"    {CALIBRATION_PATH}  ← per-player scale factors")
print(f"    {HISTORY_PATH}")
print(f"    {DASHBOARD_PATH}  ← web service JSON")
print(f"    {META_PATH}")