#!/usr/bin/env python3
"""
NBA Points Prediction — Quantile XGBoost Regression
Predicts three quantiles per player per game:
  q10 → floor  (10th percentile  — low estimate)
  q50 → median (50th percentile  — central estimate, used as MAE benchmark)
  q90 → ceiling (90th percentile — high estimate)

Inputs  : same as download_nba_model_data.py
Outputs :
  model/df_raw.parquet                    — raw player-game rows (shared)
  model/df_model.parquet                  — feature-engineered dataset (shared)
  model/nba_xgb_q10_quantile.json         — XGBoost q10 model
  model/nba_xgb_q50_quantile.json         — XGBoost q50 model (central)
  model/nba_xgb_q90_quantile.json         — XGBoost q90 model
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

# ── 1. PATHS ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
MODEL_DIR   = PROJECT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

RAW_PATH      = MODEL_DIR / "df_raw.parquet"
DATA_PATH     = MODEL_DIR / "df_model.parquet"
HISTORY_PATH  = MODEL_DIR / "predictions_history_quantile.csv"
DASHBOARD_PATH= MODEL_DIR / "bulls_dashboard_quantile.json"
META_PATH     = MODEL_DIR / "meta_quantile.json"

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
for window in [3, 5, 10]:
    df_model[f"PTS_ROLL_{window}"] = (
        df_model.groupby("PLAYER_ID")["PTS"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df_model[f"MIN_ROLL_{window}"] = (
        df_model.groupby("PLAYER_ID")["MIN"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

for lag in [1, 2, 3]:
    df_model[f"PTS_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["PTS"].shift(lag)
    df_model[f"MIN_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["MIN"].shift(lag)

df_model["PTS_SEASON_AVG"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
df_model["PTS_STD_5"] = (
    df_model.groupby("PLAYER_ID")["PTS"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
)

# ── 5. TEAM DEFENSIVE METRICS ─────────────────────────────────────────────────
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

# ── 6. TRAIN QUANTILE MODELS ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 — MODEL TRAINING (Quantile XGBoost GPU)")
print("=" * 55)

FEATURE_COLS = [
    "PTS_ROLL_3",     "PTS_ROLL_5",    "PTS_ROLL_10",
    "PTS_LAG_1",      "PTS_LAG_2",     "PTS_LAG_3",
    "MIN_ROLL_3",     "MIN_ROLL_5",    "MIN_ROLL_10",
    "MIN_LAG_1",      "MIN_LAG_2",     "MIN_LAG_3",
    "PTS_SEASON_AVG", "PTS_STD_5",
    "IS_HOME",        "DAYS_REST",     "GAME_NUM",
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

# Train one model per quantile
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

    # For q50 we compute MAE and R² (comparable to baseline)
    if q_name == "q50":
        mae  = float(mean_absolute_error(y_val, preds))
        rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
        r2   = float(1 - np.sum((y_val.values - preds)**2) /
                         np.sum((y_val.values - y_val.mean())**2))
        val_metrics["mae"]  = mae
        val_metrics["rmse"] = rmse
        val_metrics["r2"]   = r2
        print(f"  q50  → MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}")

    # Interval coverage: % of actual values inside [q10, q90]
    if q_name == "q90" and "q10" in val_preds:
        coverage = float(np.mean(
            (y_val.values >= val_preds["q10"]) &
            (y_val.values <= val_preds["q90"])
        ))
        interval_width = float(np.mean(val_preds["q90"] - val_preds["q10"]))
        val_metrics["interval_coverage_80pct"] = round(coverage, 3)
        val_metrics["avg_interval_width_pts"]  = round(interval_width, 2)
        print(f"\n  Interval [q10–q90] coverage : {coverage:.1%}  "
              f"(target ~80%)  avg width: {interval_width:.1f} pts")

    models[q_name] = m
    m.save_model(str(MODEL_PATHS[q_name]))
    print(f"  → Saved: {MODEL_PATHS[q_name]}\n")

# ── 7. PREDICTIONS HISTORY ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 4 — PREDICTIONS HISTORY")
print("=" * 55)

df_pred  = df_clean.copy()
X_all    = df_pred[FEATURE_COLS].to_numpy()

df_pred["PTS_Q10"]       = models["q10"].get_booster().inplace_predict(X_all).clip(0).round(1)
df_pred["PTS_Q50"]       = models["q50"].get_booster().inplace_predict(X_all).clip(0).round(1)
df_pred["PTS_Q90"]       = models["q90"].get_booster().inplace_predict(X_all).clip(0).round(1)
df_pred["PTS_PREDICTED"] = df_pred["PTS_Q50"]   # q50 = central prediction
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
print(f"  → {len(history_log):,} rows saved to {HISTORY_PATH}")
print(f"  → Val interval coverage [q10–q90]: {val_coverage:.1%}  (target ~80%)")

# ── 8. TOMORROW'S SCHEDULE (Chicago Bulls) ────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 — TOMORROW'S PREDICTIONS (CHI)")
print("=" * 55)

tomorrow      = pd.Timestamp.now() + pd.Timedelta(days=1)
tomorrow_str  = tomorrow.strftime("%Y-%m-%d")
tomorrow_date = tomorrow.date()
chi_next_game = None

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

    if not chi_games_sched:
        print(f"  → CHI has no game on {tomorrow_str}")
    else:
        g            = chi_games_sched[0]
        chi_is_home  = int(g["homeTeam"]["teamTricode"] == BULLS_ABBR)
        chi_opponent = (
            g["awayTeam"]["teamTricode"] if chi_is_home
            else g["homeTeam"]["teamTricode"]
        )
        print(f"  → CHI {'(HOME)' if chi_is_home else '(AWAY)'} vs {chi_opponent}")

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

        X_tomorrow = latest_chi[FEATURE_COLS]

        latest_chi["PTS_Q10"] = models["q10"].get_booster().inplace_predict(X_tomorrow.to_numpy()).clip(0).round(1)
        latest_chi["PTS_Q50"] = models["q50"].get_booster().inplace_predict(X_tomorrow.to_numpy()).clip(0).round(1)
        latest_chi["PTS_Q90"] = models["q90"].get_booster().inplace_predict(X_tomorrow.to_numpy()).clip(0).round(1)

        chi_next_game = {
            "game_date": tomorrow_str,
            "opponent":  chi_opponent,
            "is_home":   chi_is_home,
            "predictions": (
                latest_chi[[
                    "PLAYER_NAME",
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
                    pts_floor     = lambda x: x["pts_floor"].apply(lambda v: round(float(v), 1)),
                    pts_predicted = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
                    pts_ceiling   = lambda x: x["pts_ceiling"].apply(lambda v: round(float(v), 1)),
                    pts_roll_5    = lambda x: x["pts_roll_5"].apply(lambda v: round(float(v), 1)),
                    pts_season_avg= lambda x: x["pts_season_avg"].apply(lambda v: round(float(v), 1)),
                    days_rest     = lambda x: x["days_rest"].astype(int),
                )
                .rename(columns={"PLAYER_NAME": "player_name"})
                .sort_values("pts_predicted", ascending=False)
                .to_dict(orient="records")
            ),
        }
        print(f"  → {len(chi_next_game['predictions'])} CHI players to predict")
        print(f"\n  Sample predictions:")
        for p in chi_next_game["predictions"][:5]:
            print(f"    {p['player_name']:<25}  "
                  f"floor: {p['pts_floor']:>4}  "
                  f"median: {p['pts_predicted']:>4}  "
                  f"ceiling: {p['pts_ceiling']:>4}")

except Exception as exc:
    print(f"  → Could not fetch tomorrow's schedule: {exc}")
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
            pts_floor     = lambda x: x["pts_floor"].apply(lambda v: round(float(v), 1)),
            pts_predicted = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
            pts_ceiling   = lambda x: x["pts_ceiling"].apply(lambda v: round(float(v), 1)),
            pts_error     = lambda x: x["pts_error"].apply(lambda v: round(float(v), 1)),
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
    "model_metrics": {
        "q50_mae":                    round(val_metrics.get("mae",  0), 3),
        "q50_rmse":                   round(val_metrics.get("rmse", 0), 3),
        "q50_r2":                     round(val_metrics.get("r2",   0), 3),
        "interval_coverage_80pct":    val_metrics.get("interval_coverage_80pct", None),
        "avg_interval_width_pts":     val_metrics.get("avg_interval_width_pts",  None),
        "best_iter_q10":              int(models["q10"].best_iteration),
        "best_iter_q50":              int(models["q50"].best_iteration),
        "best_iter_q90":              int(models["q90"].best_iteration),
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
    "last_update":    str(pd.Timestamp.now().date()),
    "last_game_date": str(df_model["GAME_DATE"].max().date()),
    "total_rows":     len(df_model),
    "train_rows":     len(df_clean),
    "split_idx":      split_idx,
    "q50_mae":        round(val_metrics.get("mae",  0), 3),
    "q50_rmse":       round(val_metrics.get("rmse", 0), 3),
    "q50_r2":         round(val_metrics.get("r2",   0), 3),
    "interval_coverage_80pct": val_metrics.get("interval_coverage_80pct", None),
    "avg_interval_width_pts":  val_metrics.get("avg_interval_width_pts",  None),
    "best_iter_q10":  int(models["q10"].best_iteration),
    "best_iter_q50":  int(models["q50"].best_iteration),
    "best_iter_q90":  int(models["q90"].best_iteration),
    "seasons":        SEASONS,
    "feature_cols":   FEATURE_COLS,
    "quantiles":      QUANTILES,
}
META_PATH.write_text(json.dumps(meta, indent=2))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"  q50 MAE   : {val_metrics.get('mae',  0):.3f} pts  ← compare vs baseline 4.010")
print(f"  q50 RMSE  : {val_metrics.get('rmse', 0):.3f} pts")
print(f"  q50 R²    : {val_metrics.get('r2',   0):.3f}")
print(f"  [q10–q90] coverage : {val_metrics.get('interval_coverage_80pct', 0):.1%}  (target ~80%)")
print(f"  avg interval width : {val_metrics.get('avg_interval_width_pts', 0):.1f} pts")
print()
print("  Output files:")
print(f"    {RAW_PATH}")
print(f"    {DATA_PATH}")
for q, p in MODEL_PATHS.items():
    print(f"    {p}  ← {q} model")
print(f"    {HISTORY_PATH}")
print(f"    {DASHBOARD_PATH}  ← web service JSON")
print(f"    {META_PATH}")