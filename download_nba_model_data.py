#!/usr/bin/env python3
"""
NBA Points Prediction — Daily Update Script

Run daily (next day after games) to:
  1. Incrementally fetch only new game logs
  2. Rebuild feature engineering on full dataset
  3. Retrain XGBoost (GPU) on all historical data
  4. Save predictions_history.csv (actual vs. predicted per player/game)
  5. Fetch tomorrow's schedule and predict CHI player points
  6. Write bulls_dashboard.json for web service consumption
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

XGB_GPU_PARAMS = {"device": "cuda", "tree_method": "hist"}
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── 1. PATHS ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
MODEL_DIR   = PROJECT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

RAW_PATH         = MODEL_DIR / "df_raw.parquet"
DATA_PATH        = MODEL_DIR / "df_model.parquet"
MODEL_PATH       = MODEL_DIR / "nba_xgb_model.json"
HISTORY_PATH     = MODEL_DIR / "predictions_history.csv"
DASHBOARD_PATH   = MODEL_DIR / "bulls_dashboard.json"
META_PATH        = MODEL_DIR / "meta.json"

print(f"\nModel dir : {MODEL_DIR}")

# ── 2. DATA SOURCE CONFIG ──────────────────────────────────────────────────────
# stats.nba.com is blocked by Akamai CDN bot-detection on this network.
# cdn.nba.com serves the same boxscore data without restrictions.
CDN_BOXSCORE = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
CDN_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
CDN_TIMEOUT  = 20    # seconds per request
CDN_WORKERS  = 8     # parallel threads — keep low to avoid Akamai WAF rate-limit
SEASONS        = ["2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = SEASONS[-1]   # "2025-26"
BULLS_ABBR     = "CHI"


def _fetch_boxscore(game_id: str) -> dict | None:
    """Fetch one boxscore from cdn.nba.com. Returns None on 404 or error."""
    for attempt in range(1, 4):
        try:
            r = requests.get(CDN_BOXSCORE.format(game_id), timeout=CDN_TIMEOUT)
            if r.status_code == 404:
                return None          # game doesn't exist
            r.raise_for_status()
            return r.json().get("game")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(attempt * 5)
        except Exception:
            return None
    return None


def _parse_boxscore(game: dict) -> list[dict]:
    """Extract one row per player who played from a boxscore dict."""
    game_id   = game["gameId"]
    game_date = game.get("gameTimeUTC", "")[:10]   # YYYY-MM-DD
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
            wl = None          # shouldn't happen in finished games
        elif is_home:
            wl = "W" if home_pts > away_pts else "L"
        else:
            wl = "W" if away_pts > home_pts else "L"

        matchup = f"{team_tri} vs. {opp_tri}" if is_home else f"{team_tri} @ {opp_tri}"

        for p in team.get("players", []):
            if p.get("played") == "0":
                continue
            s = p.get("statistics", {})
            # minutesCalculated is "PT26M"; minutes is "PT25M39.99S"
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
    """Fetch all completed regular-season games for a season from cdn.nba.com.

    Game IDs follow the pattern 002{YY}{num:05d} where YY is the
    4-digit start year's last 2 digits (e.g. 2024-25 → '24').
    We probe 1-1 250 and skip 404s (postponed / non-existent games).
    Uses a thread pool so ~1 230 requests finish in ~2 minutes.
    """
    yy = season[:4][2:]   # "2024-25" → "24"
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

    # Only probe game IDs we haven't seen yet in the current season
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
        print(f"  → {season} (probing up to 1 250 game IDs in parallel)...")
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
df_model = df[available].copy()

# Context features
df_model["IS_HOME"]  = df_model["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)
df_model["OPPONENT"] = df_model["MATCHUP"].apply(
    lambda x: str(x).split(" ")[-1] if pd.notna(x) else None
)
df_model["GAME_NUM"]  = df_model.groupby("PLAYER_ID").cumcount() + 1
df_model["DAYS_REST"] = (
    df_model.groupby("PLAYER_ID")["GAME_DATE"]
    .diff().dt.days.fillna(3)
)

# Rolling player stats — shift(1) prevents data leakage
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

# Lag features
for lag in [1, 2, 3]:
    df_model[f"PTS_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["PTS"].shift(lag)
    df_model[f"MIN_LAG_{lag}"] = df_model.groupby("PLAYER_ID")["MIN"].shift(lag)

# Season-to-date average and volatility
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

# Rolling defensive metrics (shift(1) = no leakage)
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
    """Bayesian shrinkage: blend expanding avg with league average."""
    shifted   = group["PTS_ALLOWED"].shift(1)
    cum_sum   = shifted.expanding().sum()
    cum_count = shifted.expanding().count()
    return (cum_sum + prior_weight * league_avg) / (cum_count + prior_weight)


# Explicit loop avoids pandas 2.x MultiIndex misalignment from groupby.apply
_reg_parts = []
for _, _grp in team_def.groupby("TEAM_ABBREVIATION", sort=False):
    _reg_parts.append(compute_regularized_def(_grp, league_avg_pts, PRIOR_WEIGHT))
team_def["DEF_PTS_ALLOWED_REGULARIZED"] = pd.concat(_reg_parts).sort_index()

# Merge defensive features onto player rows
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

# ── 6. TRAIN XGBOOST MODEL ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 — MODEL TRAINING (XGBoost GPU)")
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

# Drop rows where rolling features are NaN (first few games per player)
df_clean = df_model.dropna(subset=FEATURE_COLS + [TARGET]).copy()
print(f"Training rows : {len(df_clean):,}  (dropped {len(df_model) - len(df_clean):,} NaN rows)")

# Chronological 80/20 split
split_idx = int(len(df_clean) * 0.80)
X_train = df_clean[FEATURE_COLS].iloc[:split_idx]
y_train = df_clean[TARGET].iloc[:split_idx]
X_val   = df_clean[FEATURE_COLS].iloc[split_idx:]
y_val   = df_clean[TARGET].iloc[split_idx:]

model = xgb.XGBRegressor(
    n_estimators          = 500,
    learning_rate         = 0.05,
    max_depth             = 6,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    early_stopping_rounds = 30,
    eval_metric           = "mae",
    **XGB_GPU_PARAMS,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

y_pred_val = model.get_booster().inplace_predict(X_val.to_numpy())
mae  = float(mean_absolute_error(y_val, y_pred_val))
rmse = float(np.sqrt(np.mean((y_val.values - y_pred_val) ** 2)))
print(f"\n  Val MAE  : {mae:.2f} pts")
print(f"  Val RMSE : {rmse:.2f} pts")

model.save_model(str(MODEL_PATH))
print(f"  → Model saved: {MODEL_PATH}")

# ── 7. PREDICTIONS HISTORY (global CSV) ───────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 4 — PREDICTIONS HISTORY")
print("=" * 55)

df_pred = df_clean.copy()
df_pred["PTS_PREDICTED"] = model.get_booster().inplace_predict(df_pred[FEATURE_COLS].to_numpy()).clip(0).round(1)
df_pred["PTS_ERROR"]     = (df_pred["PTS_PREDICTED"] - df_pred[TARGET]).round(1)
df_pred["IS_VALIDATION"] = np.arange(len(df_pred)) >= split_idx

history_log = df_pred[[
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "OPPONENT", "IS_HOME",
    TARGET, "PTS_PREDICTED", "PTS_ERROR", "IS_VALIDATION",
]].sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

history_log["GAME_DATE"] = history_log["GAME_DATE"].dt.date
history_log.to_csv(HISTORY_PATH, index=False)
print(f"  → {len(history_log):,} rows saved to {HISTORY_PATH}")

# ── 8. TOMORROW'S SCHEDULE (Chicago Bulls) ────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 — TOMORROW'S PREDICTIONS (CHI)")
print("=" * 55)

tomorrow       = (pd.Timestamp.now() + pd.Timedelta(days=1))
tomorrow_str   = tomorrow.strftime("%Y-%m-%d")
tomorrow_date  = tomorrow.date()
chi_next_game  = None

# ESPN abbreviation → NBA tricode (only entries that differ)
ESPN_TO_NBA = {
    "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "GS": "GSW", "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS",
}

def _find_chi_game_espn(date_str: str) -> tuple[int, str] | None:
    """Query ESPN public API for CHI game on date_str (YYYY-MM-DD).
    Returns (is_home, opponent_tricode) or None if no game found."""
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
    chi_games = [
        g for g in all_games_sched
        if g.get("gameDateEst", "")[:10] == tomorrow_str
        and (
            g["homeTeam"]["teamTricode"] == BULLS_ABBR
            or g["awayTeam"]["teamTricode"] == BULLS_ABBR
        )
    ]
    if chi_games:
        g = chi_games[0]
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
    # Get most recent features for each CHI player (at least 5 games this season)
    chi_players = df_clean[df_clean["TEAM_ABBREVIATION"] == BULLS_ABBR].copy()
    latest_chi  = (
        chi_players
        .sort_values("GAME_DATE")
        .groupby("PLAYER_ID")
        .last()
        .reset_index()
    )
    # Keep only players with recent activity (last 30 days)
    cutoff = df_model["GAME_DATE"].max() - pd.Timedelta(days=30)
    latest_chi = latest_chi[latest_chi["GAME_DATE"] >= cutoff].copy()

    # Override context features for tomorrow's game
    latest_chi["IS_HOME"]  = chi_is_home
    latest_chi["OPPONENT"] = chi_opponent
    latest_chi["DAYS_REST"] = (
        pd.Timestamp(tomorrow_date) - pd.to_datetime(latest_chi["GAME_DATE"])
    ).dt.days

    # Get opponent defensive features (most recent row for that team)
    opp_def_latest = (
        team_def[team_def["TEAM_ABBREVIATION"] == chi_opponent]
        .sort_values("GAME_DATE")
        .iloc[-1]
    )
    latest_chi["OPP_DEF_AVG"]      = opp_def_latest.get("DEF_PTS_ALLOWED_AVG", league_avg_pts)
    latest_chi["OPP_DEF_ROLL_5"]   = opp_def_latest.get("DEF_PTS_ALLOWED_ROLL_5", league_avg_pts)
    latest_chi["OPP_DEF_ROLL_10"]  = opp_def_latest.get("DEF_PTS_ALLOWED_ROLL_10", league_avg_pts)
    latest_chi["OPP_DEF_REG"]      = opp_def_latest.get("DEF_PTS_ALLOWED_REGULARIZED", league_avg_pts)
    latest_chi["OPP_GAMES_PLAYED"] = opp_def_latest.get("TEAM_GAME_NUM", 0)

    # Fill any remaining NaNs with column means from training data
    for col in FEATURE_COLS:
        latest_chi[col] = latest_chi[col].fillna(df_clean[col].mean())

    X_tomorrow = latest_chi[FEATURE_COLS]
    latest_chi["PTS_PREDICTED"] = model.get_booster().inplace_predict(X_tomorrow.to_numpy()).clip(0).round(1)

    chi_next_game = {
        "game_date": tomorrow_str,
        "opponent":  chi_opponent,
        "is_home":   chi_is_home,
        "predictions": (
            latest_chi[["PLAYER_NAME", "PTS_PREDICTED", "PTS_ROLL_5", "PTS_SEASON_AVG", "DAYS_REST"]]
            .rename(columns={
                "PTS_PREDICTED":  "pts_predicted",
                "PTS_ROLL_5":     "pts_roll_5",
                "PTS_SEASON_AVG": "pts_season_avg",
                "DAYS_REST":      "days_rest",
            })
            .assign(
                pts_predicted  = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
                pts_roll_5     = lambda x: x["pts_roll_5"].apply(lambda v: round(float(v), 1)),
                pts_season_avg = lambda x: x["pts_season_avg"].apply(lambda v: round(float(v), 1)),
                days_rest      = lambda x: x["days_rest"].astype(int),
            )
            .rename(columns={"PLAYER_NAME": "player_name"})
            .sort_values("pts_predicted", ascending=False)
            .to_dict(orient="records")
        ),
    }
    print(f"  → {len(chi_next_game['predictions'])} CHI players to predict")

# ── 9. BULLS DASHBOARD JSON ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 6 — BULLS DASHBOARD JSON")
print("=" * 55)

# Last 10 CHI games from history log
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
    first = game_rows.iloc[0]

    # Determine opponent and result from the WL column if available
    result = None
    if "WL" in df_model.columns:
        wl_rows = df_model[
            (df_model["GAME_ID"] == game_id) &
            (df_model["TEAM_ABBREVIATION"] == BULLS_ABBR)
        ]["WL"]
        if not wl_rows.empty:
            result = str(wl_rows.iloc[0])

    players_in_game = (
        game_rows[game_rows["PTS"] > 0][  # only players who had minutes
            ["PLAYER_NAME", "PTS", "PTS_PREDICTED", "PTS_ERROR"]
        ]
        .rename(columns={
            "PLAYER_NAME":    "player_name",
            "PTS":            "pts_actual",
            "PTS_PREDICTED":  "pts_predicted",
            "PTS_ERROR":      "pts_error",
        })
        .assign(
            pts_predicted = lambda x: x["pts_predicted"].apply(lambda v: round(float(v), 1)),
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
    "team":          "Chicago Bulls",
    "abbreviation":  BULLS_ABBR,
    "last_updated":  str(pd.Timestamp.now().date()),
    "model_metrics": {
        "val_mae":  round(mae, 2),
        "val_rmse": round(rmse, 2),
    },
    "recent_games": recent_games,   # last 10 games, desc by date
    "next_game":    chi_next_game,  # None if CHI doesn't play tomorrow
}

DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2, default=str))
print(f"  → Dashboard saved: {DASHBOARD_PATH}")
print(f"  → Recent games   : {len(recent_games)}")
print(f"  → Next game      : {chi_next_game['game_date'] if chi_next_game else 'N/A'}")

# ── 10. METADATA ──────────────────────────────────────────────────────────────
meta = {
    "last_update":    str(pd.Timestamp.now().date()),
    "last_game_date": str(df_model["GAME_DATE"].max().date()),
    "total_rows":     len(df_model),
    "train_rows":     len(df_clean),
    "val_mae":        round(mae, 3),
    "val_rmse":       round(rmse, 3),
    "best_iteration": int(model.best_iteration),
    "seasons":        SEASONS,
    "feature_cols":   FEATURE_COLS,
}
META_PATH.write_text(json.dumps(meta, indent=2))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"  Last game date : {meta['last_game_date']}")
print(f"  Total rows     : {meta['total_rows']:,}")
print(f"  Val MAE        : {meta['val_mae']} pts")
print(f"  Val RMSE       : {meta['val_rmse']} pts")
print(f"  Best XGB iter  : {meta['best_iteration']}")
print()
print("  Output files:")
print(f"    {RAW_PATH}")
print(f"    {DATA_PATH}")
print(f"    {MODEL_PATH}")
print(f"    {HISTORY_PATH}")
print(f"    {DASHBOARD_PATH}  ← web service JSON")
print(f"    {META_PATH}")
