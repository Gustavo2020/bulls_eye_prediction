
#!/bin/bash
# Bulls Eye — Daily Pipeline Runner
# Runs the 3 scripts sequentially and sends ntfy notifications on success/error.
#
# Requires a .env file in the same directory. Copy .env.example and configure it:
#   cp .env.example .env && nano .env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"

VENV="${VENV_DIR}/bin/activate"
PROJECT="${PROJECT_DIR:-$SCRIPT_DIR}"
NTFY="${NTFY_TOPIC:?ERROR: NTFY_TOPIC not set. Copy .env.example to .env and configure it.}"
LOG_DIR="$PROJECT/logs"
DATE=$(date '+%Y-%m-%d')

mkdir -p "$LOG_DIR"

# ── ntfy helper ────────────────────────────────────────────────────────────────
notify() {
    local title="$1"
    local msg="$2"
    local tags="$3"
    local priority="${4:-default}"
    curl -s \
        -H "Title: $title" \
        -H "Tags: $tags" \
        -H "Priority: $priority" \
        -d "$msg" \
        "$NTFY" > /dev/null
}

source "$VENV"
cd "$PROJECT"

# ── Step 1: download_nba_model_data.py ────────────────────────────────────────
LOG1="$LOG_DIR/download_${DATE}.log"
notify "Bulls Eye" "Starting download_nba_model_data.py — $(date '+%H:%M')" "hourglass_flowing_sand" "default"

python "$PROJECT/download_nba_model_data.py" > "$LOG1" 2>&1
EXIT1=$?

if [ $EXIT1 -eq 0 ]; then
    notify "Bulls Eye ✓" "download_nba_model_data.py OK — $(date '+%H:%M')" "white_check_mark" "default"
else
    ERROR=$(tail -8 "$LOG1" | tr '\n' ' ')
    notify "Bulls Eye FAILED" "download_nba_model_data.py ERROR — ${ERROR}" "x,rotating_light" "high"
    exit 1
fi

# ── Step 2: model_data_quantile.py ────────────────────────────────────────────
LOG2="$LOG_DIR/quantile_${DATE}.log"
notify "Bulls Eye" "Starting model_data_quantile.py — $(date '+%H:%M')" "hourglass_flowing_sand" "default"

python "$PROJECT/model_data_quantile.py" > "$LOG2" 2>&1
EXIT2=$?

if [ $EXIT2 -eq 0 ]; then
    notify "Bulls Eye ✓" "model_data_quantile.py OK — $(date '+%H:%M')" "white_check_mark" "default"
else
    ERROR=$(tail -8 "$LOG2" | tr '\n' ' ')
    notify "Bulls Eye FAILED" "model_data_quantile.py ERROR — ${ERROR}" "x,rotating_light" "high"
    exit 1
fi

# ── Step 3: total_next_game_prediction.py ─────────────────────────────────────
LOG3="$LOG_DIR/win_prediction_${DATE}.log"
notify "Bulls Eye" "Starting total_next_game_prediction.py — $(date '+%H:%M')" "hourglass_flowing_sand" "default"

python "$PROJECT/total_next_game_prediction.py" > "$LOG3" 2>&1
EXIT3=$?

if [ $EXIT3 -eq 0 ]; then
    notify "Bulls Eye — Pipeline Complete" "All 3 scripts finished successfully. Dashboard updated! — $(date '+%H:%M')" "white_check_mark,tada" "default"
else
    ERROR=$(tail -8 "$LOG3" | tr '\n' ' ')
    notify "Bulls Eye FAILED" "total_next_game_prediction.py ERROR — ${ERROR}" "x,rotating_light" "high"
    exit 1
fi

# ── Restart Streamlit to clear cache and load fresh data ──────────────────────
sudo systemctl restart bullseye && \
    notify "Bulls Eye" "Streamlit restarted — cache cleared — $(date '+%H:%M')" "recycle" "default"
