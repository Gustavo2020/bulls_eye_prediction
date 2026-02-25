#!/usr/bin/env python3
"""
Win-Probability Model — Chicago Bulls Next Game

Reads from model/ (output of download_nba_model_data.py).
Trains an XGBoost binary classifier on team-level rolling features.
Predicts P(CHI wins) for the next scheduled game.

No external API calls — reads only from model/.

Inputs:
  model/df_model.parquet      — full player-level feature dataset (50 cols)
  model/bulls_dashboard.json  — next_game player predictions

Outputs:
  model/win_probability.json  — win/loss probability for next CHI game
  model/nba_win_model.json    — saved XGBoost binary classifier

Run:
  python total_next_game_prediction.py
"""

import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_absolute_error

# ── SETUP ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device : {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")

XGB_GPU_PARAMS = (
    {"device": "cuda", "tree_method": "hist"} if device.type == "cuda" else {}
)

PROJECT_DIR    = Path(__file__).parent
MODEL_DIR      = PROJECT_DIR / "model"
DATA_PATH      = MODEL_DIR / "df_model.parquet"
DASHBOARD_PATH = MODEL_DIR / "bulls_dashboard.json"
WIN_PROB_PATH  = MODEL_DIR / "win_probability.json"
WIN_MODEL_PATH = MODEL_DIR / "nba_win_model.json"

BULLS_ABBR = "CHI"

print(f"\nModel dir : {MODEL_DIR}\n")

# ── PHASE 1: LOAD DATA ────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 1 — LOAD PLAYER-GAME DATA")
print("=" * 55)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found.\nRun download_nba_model_data.py first."
    )

df = pd.read_parquet(DATA_PATH)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
print(f"  {len(df):,} player-game rows | last game: {df['GAME_DATE'].max().date()}")

# ── PHASE 2: AGGREGATE TO TEAM-GAME LEVEL ────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 2 — AGGREGATE TO TEAM-GAME LEVEL")
print("=" * 55)

team_games = (
    df.groupby(["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"])
    .agg(
        TEAM_PTS = ("PTS",      "sum"),
        IS_HOME  = ("IS_HOME",  "first"),
        WL       = ("WL",       "first"),
        OPPONENT = ("OPPONENT", "first"),
    )
    .reset_index()
    .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    .reset_index(drop=True)
)

# Self-join on GAME_ID to add opponent team PTS
opp_side = (
    team_games[["GAME_ID", "TEAM_ABBREVIATION", "TEAM_PTS"]]
    .rename(columns={"TEAM_ABBREVIATION": "_OPP", "TEAM_PTS": "OPP_PTS"})
)
team_games = team_games.merge(
    opp_side,
    left_on=["GAME_ID", "OPPONENT"],
    right_on=["GAME_ID", "_OPP"],
    how="left",
).drop(columns=["_OPP"])

team_games["WIN"]    = (team_games["TEAM_PTS"] > team_games["OPP_PTS"]).astype(int)
team_games["MARGIN"] = team_games["TEAM_PTS"] - team_games["OPP_PTS"]

print(f"  {len(team_games):,} team-game rows | {team_games['TEAM_ABBREVIATION'].nunique()} teams")

# ── PHASE 3: ROLLING FEATURES (shift-1, all 30 teams) ────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 — ROLLING FEATURES (shift-1 to prevent leakage)")
print("=" * 55)

team_games = (
    team_games
    .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    .reset_index(drop=True)
)

for stat, label in [
    ("WIN",      "WIN"),
    ("TEAM_PTS", "PTS_SCORED"),
    ("OPP_PTS",  "PTS_ALLOWED"),
    ("MARGIN",   "MARGIN"),
]:
    for w in [3, 5, 10]:
        team_games[f"{label}_ROLL_{w}"] = (
            team_games.groupby("TEAM_ABBREVIATION")[stat]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )

team_games["WIN_STREAK"] = (
    team_games.groupby("TEAM_ABBREVIATION")["WIN"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
)
team_games["TEAM_DAYS_REST"] = (
    team_games.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
    .diff().dt.days.fillna(3)
)

# OPP features (rename and join back onto CHI rows)
opp_features = (
    team_games[[
        "GAME_DATE", "TEAM_ABBREVIATION",
        "WIN_ROLL_5",        "WIN_ROLL_10",
        "PTS_SCORED_ROLL_5", "PTS_SCORED_ROLL_10",
        "PTS_ALLOWED_ROLL_5",
        "MARGIN_ROLL_5",
    ]]
    .rename(columns={
        "TEAM_ABBREVIATION":  "_OPP",
        "WIN_ROLL_5":         "OPP_WIN_ROLL_5",
        "WIN_ROLL_10":        "OPP_WIN_ROLL_10",
        "PTS_SCORED_ROLL_5":  "OPP_PTS_SCORED_ROLL_5",
        "PTS_SCORED_ROLL_10": "OPP_PTS_SCORED_ROLL_10",
        "PTS_ALLOWED_ROLL_5": "OPP_PTS_ALLOWED_ROLL_5",
        "MARGIN_ROLL_5":      "OPP_MARGIN_ROLL_5",
    })
)

# Merge OPP rolling features for ALL 30 teams (training uses all of them)
all_games = team_games.merge(
    opp_features,
    left_on=["GAME_DATE", "OPPONENT"],
    right_on=["GAME_DATE", "_OPP"],
    how="left",
).drop(columns=["_OPP"])

# Merge OPP defense quality from df_model (already computed in main script).
# df has OPP_DEF_* keyed by [GAME_DATE, OPPONENT] — valid for any team perspective.
opp_def = (
    df[["GAME_DATE", "OPPONENT", "OPP_DEF_AVG", "OPP_DEF_REG"]]
    .drop_duplicates(subset=["GAME_DATE", "OPPONENT"])
)
all_games = all_games.merge(opp_def, on=["GAME_DATE", "OPPONENT"], how="left")

# Fill NaN OPP features with league averages
league_means = team_games[["WIN", "TEAM_PTS", "OPP_PTS", "MARGIN"]].mean()
nan_fills = {
    "OPP_WIN_ROLL_5":         float(league_means["WIN"]),
    "OPP_WIN_ROLL_10":        float(league_means["WIN"]),
    "OPP_PTS_SCORED_ROLL_5":  float(league_means["TEAM_PTS"]),
    "OPP_PTS_SCORED_ROLL_10": float(league_means["TEAM_PTS"]),
    "OPP_PTS_ALLOWED_ROLL_5": float(league_means["OPP_PTS"]),
    "OPP_MARGIN_ROLL_5":      float(league_means["MARGIN"]),
    "OPP_DEF_AVG":            float(df["OPP_DEF_AVG"].mean()),
    "OPP_DEF_REG":            float(df["OPP_DEF_REG"].mean()),
}
for col, val in nan_fills.items():
    all_games[col] = all_games[col].fillna(val)

# CHI subset used for next-game prediction feature building
chi_games = all_games[all_games["TEAM_ABBREVIATION"] == BULLS_ABBR].copy()

print(f"  All team-game rows : {len(all_games):,}")
print(f"  CHI games          : {len(chi_games):,} | win rate: {chi_games['WIN'].mean():.1%}")

# ── PHASE 4: TRAIN WIN/LOSS CLASSIFIER ───────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 4 — TRAIN WIN/LOSS CLASSIFIER (XGBoost GPU)")
print("=" * 55)

WIN_FEATURES = [
    # CHI offensive form
    "PTS_SCORED_ROLL_3", "PTS_SCORED_ROLL_5", "PTS_SCORED_ROLL_10",
    # CHI defensive form
    "PTS_ALLOWED_ROLL_3", "PTS_ALLOWED_ROLL_5", "PTS_ALLOWED_ROLL_10",
    # CHI recent form
    "WIN_ROLL_3",  "WIN_ROLL_5",  "WIN_ROLL_10",
    "MARGIN_ROLL_3", "MARGIN_ROLL_5",
    "WIN_STREAK",
    # Game context
    "IS_HOME", "TEAM_DAYS_REST",
    # Opponent form
    "OPP_WIN_ROLL_5",        "OPP_WIN_ROLL_10",
    "OPP_PTS_SCORED_ROLL_5",
    "OPP_PTS_ALLOWED_ROLL_5",
    "OPP_MARGIN_ROLL_5",
    # Opponent defensive quality (Bayesian-regularized)
    "OPP_DEF_AVG", "OPP_DEF_REG",
]

# Use ALL 30 teams for training — win/loss relationships generalise across the league.
# CHI subset is used only for next-game prediction (Phase 5).
all_clean = (
    all_games.dropna(subset=WIN_FEATURES + ["WIN"])
    .sort_values("GAME_DATE")
    .reset_index(drop=True)
)
chi_clean = (
    all_clean[all_clean["TEAM_ABBREVIATION"] == BULLS_ABBR]
    .reset_index(drop=True)
)
print(f"  Training rows (all teams) : {len(all_clean):,}  "
      f"(dropped {len(all_games) - len(all_clean):,} NaN rows)")
print(f"  CHI games in clean set    : {len(chi_clean):,}")

# Chronological split by calendar date (temporal integrity across all teams)
all_dates  = all_clean["GAME_DATE"].sort_values().unique()
split_date = pd.Timestamp(all_dates[int(len(all_dates) * 0.80)])
train_mask = all_clean["GAME_DATE"] < split_date

X_train = all_clean[train_mask][WIN_FEATURES]
y_train = all_clean[train_mask]["WIN"]
X_val   = all_clean[~train_mask][WIN_FEATURES]
y_val   = all_clean[~train_mask]["WIN"]
print(f"  Train : {len(X_train):,} team-game rows (before {split_date.date()})")
print(f"  Val   : {len(X_val):,}  team-game rows (on/after {split_date.date()})")

win_model = xgb.XGBClassifier(
    n_estimators          = 300,
    learning_rate         = 0.05,
    max_depth             = 4,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    early_stopping_rounds = 30,
    eval_metric           = "logloss",
    objective             = "binary:logistic",
    **XGB_GPU_PARAMS,
)
win_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

y_prob_val  = win_model.get_booster().inplace_predict(X_val.to_numpy())
y_pred_val  = (y_prob_val > 0.5).astype(int)
val_acc     = float(accuracy_score(y_val, y_pred_val))
val_auc     = float(roc_auc_score(y_val, y_prob_val))
val_logloss = float(log_loss(y_val, y_prob_val))
val_mae     = float(mean_absolute_error(y_val, y_prob_val))

print(f"\n  Val Accuracy : {val_acc:.1%}")
print(f"  Val AUC      : {val_auc:.3f}")
print(f"  Val Log-Loss : {val_logloss:.3f}")
print(f"  Val MAE      : {val_mae:.3f}")

win_model.save_model(str(WIN_MODEL_PATH))
print(f"  → Model saved: {WIN_MODEL_PATH}")

# ── PHASE 5: PREDICT NEXT GAME WIN PROBABILITY ────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 — PREDICT NEXT GAME WIN PROBABILITY")
print("=" * 55)

if not DASHBOARD_PATH.exists():
    raise FileNotFoundError(
        f"{DASHBOARD_PATH} not found.\nRun download_nba_model_data.py first."
    )

with open(DASHBOARD_PATH) as f:
    dashboard = json.load(f)

next_game = dashboard.get("next_game")

model_metrics_block = {
    "val_accuracy": round(val_acc,     3),
    "val_auc":      round(val_auc,     3),
    "val_logloss":  round(val_logloss, 3),
    "val_mae":      round(val_mae,     3),
    "val_games":    len(X_val),
}

if next_game is None:
    print("  → No next game scheduled for CHI.")
    result = {"next_game": None, "model_metrics": model_metrics_block}
else:
    game_date = next_game["game_date"]
    opponent  = next_game["opponent"]
    is_home   = next_game["is_home"]

    # Projected CHI team score from player model (informational only — not a training feature)
    chi_projected_pts = round(
        sum(p["pts_predicted"] for p in next_game.get("predictions", [])), 1
    )

    # CHI current form — use actual last N games (no shift, we want stats thru the last game)
    chi_sorted = chi_clean.sort_values("GAME_DATE")

    def tail_mean(col: str, k: int) -> float:
        return float(chi_sorted[col].tail(k).mean())

    feat_row = {
        "PTS_SCORED_ROLL_3":   tail_mean("TEAM_PTS", 3),
        "PTS_SCORED_ROLL_5":   tail_mean("TEAM_PTS", 5),
        "PTS_SCORED_ROLL_10":  tail_mean("TEAM_PTS", 10),
        "PTS_ALLOWED_ROLL_3":  tail_mean("OPP_PTS",  3),
        "PTS_ALLOWED_ROLL_5":  tail_mean("OPP_PTS",  5),
        "PTS_ALLOWED_ROLL_10": tail_mean("OPP_PTS",  10),
        "WIN_ROLL_3":          tail_mean("WIN", 3),
        "WIN_ROLL_5":          tail_mean("WIN", 5),
        "WIN_ROLL_10":         tail_mean("WIN", 10),
        "MARGIN_ROLL_3":       tail_mean("MARGIN", 3),
        "MARGIN_ROLL_5":       tail_mean("MARGIN", 5),
        "WIN_STREAK":          float(chi_sorted["WIN"].tail(3).sum()),
        "IS_HOME":             float(is_home),
        "TEAM_DAYS_REST":      float(
            (pd.Timestamp(game_date) - chi_sorted["GAME_DATE"].iloc[-1]).days
        ),
    }

    # OPP current form
    opp_data = (
        team_games[team_games["TEAM_ABBREVIATION"] == opponent]
        .sort_values("GAME_DATE")
    )
    if len(opp_data) >= 5:
        feat_row["OPP_WIN_ROLL_5"]         = float(opp_data["WIN"].tail(5).mean())
        feat_row["OPP_WIN_ROLL_10"]        = float(opp_data["WIN"].tail(10).mean())
        feat_row["OPP_PTS_SCORED_ROLL_5"]  = float(opp_data["TEAM_PTS"].tail(5).mean())
        feat_row["OPP_PTS_ALLOWED_ROLL_5"] = float(opp_data["OPP_PTS"].tail(5).mean())
        feat_row["OPP_MARGIN_ROLL_5"]      = float(opp_data["MARGIN"].tail(5).mean())
    else:
        feat_row["OPP_WIN_ROLL_5"]         = float(league_means["WIN"])
        feat_row["OPP_WIN_ROLL_10"]        = float(league_means["WIN"])
        feat_row["OPP_PTS_SCORED_ROLL_5"]  = float(league_means["TEAM_PTS"])
        feat_row["OPP_PTS_ALLOWED_ROLL_5"] = float(league_means["OPP_PTS"])
        feat_row["OPP_MARGIN_ROLL_5"]      = float(league_means["MARGIN"])

    # OPP defensive quality from CHI's history vs that opponent
    chi_vs_opp = chi_clean[chi_clean["OPPONENT"] == opponent].sort_values("GAME_DATE")
    if not chi_vs_opp.empty:
        feat_row["OPP_DEF_AVG"] = float(chi_vs_opp["OPP_DEF_AVG"].iloc[-1])
        feat_row["OPP_DEF_REG"] = float(chi_vs_opp["OPP_DEF_REG"].iloc[-1])
    else:
        feat_row["OPP_DEF_AVG"] = float(df["OPP_DEF_AVG"].mean())
        feat_row["OPP_DEF_REG"] = float(df["OPP_DEF_REG"].mean())

    X_next      = pd.DataFrame([feat_row])[WIN_FEATURES]
    win_prob    = float(win_model.get_booster().inplace_predict(X_next.to_numpy())[0])
    pred_result = "W" if win_prob >= 0.5 else "L"
    confidence  = "high" if (win_prob > 0.65 or win_prob < 0.35) else "medium"

    print(f"  CHI vs {opponent} ({'HOME' if is_home else 'AWAY'}) — {game_date}")
    print(f"  CHI projected score : {chi_projected_pts} pts  (sum of player model)")
    print(f"  Win probability     : {win_prob:.1%}")
    print(f"  Predicted result    : {pred_result}  [{confidence} confidence]")

    # Recent form summary strings
    chi_last5 = chi_sorted.tail(5)
    chi_w5    = int(chi_last5["WIN"].sum())
    chi_record_5 = f"{chi_w5}-{5 - chi_w5}"

    if len(opp_data) >= 5:
        opp_w5 = int(opp_data["WIN"].tail(5).sum())
        opp_record_5 = f"{opp_w5}-{5 - opp_w5}"
    else:
        opp_record_5 = "N/A"

    result = {
        "game_date":          game_date,
        "opponent":           opponent,
        "is_home":            is_home,
        "chi_projected_pts":  chi_projected_pts,
        "win_probability":    round(win_prob, 3),
        "predicted_result":   pred_result,
        "confidence":         confidence,
        "chi_recent_form": {
            "last_5_record":     chi_record_5,
            "avg_pts_scored_5":  round(tail_mean("TEAM_PTS", 5), 1),
            "avg_pts_allowed_5": round(tail_mean("OPP_PTS",  5), 1),
            "avg_margin_5":      round(tail_mean("MARGIN",   5), 1),
        },
        "opponent_recent_form": {
            "team":              opponent,
            "last_5_record":     opp_record_5,
            "avg_pts_scored_5":  round(feat_row["OPP_PTS_SCORED_ROLL_5"],  1),
            "avg_pts_allowed_5": round(feat_row["OPP_PTS_ALLOWED_ROLL_5"], 1),
            "avg_margin_5":      round(feat_row["OPP_MARGIN_ROLL_5"],       1),
        },
        "model_metrics": model_metrics_block,
    }

# ── PHASE 6: SAVE OUTPUT ──────────────────────────────────────────────────────
WIN_PROB_PATH.write_text(json.dumps(result, indent=2))
print(f"\n  → Saved: {WIN_PROB_PATH}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"  Val Accuracy : {val_acc:.1%}")
print(f"  Val AUC      : {val_auc:.3f}")
print(f"  Val Log-Loss : {val_logloss:.3f}")
print(f"  Val MAE      : {val_mae:.3f}")
print()
print("  Output files:")
print(f"    {WIN_MODEL_PATH}")
print(f"    {WIN_PROB_PATH}  ← web service JSON")
