#!/bin/bash
# Bulls Eye health check — runs every 5 min via cron
# Sends ntfy alert if the app is not responding
#
# Requires a .env file in the same directory. Copy .env.example and configure it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && source "$SCRIPT_DIR/.env"

NTFY="${NTFY_TOPIC:?ERROR: NTFY_TOPIC not set. Copy .env.example to .env and configure it.}"
PORT="${APP_PORT:-8502}"
URL="http://127.0.0.1:${PORT}/bullseye/healthz"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL")

if [ "$HTTP_CODE" != "200" ]; then
    # Check if process is running at all
    if systemctl is-active --quiet bullseye; then
        STATUS="El proceso corre pero no responde (HTTP $HTTP_CODE). Posible cuelgue."
        systemctl restart bullseye
    else
        STATUS="El servicio bullseye está detenido. Intentando reinicio..."
        systemctl start bullseye
    fi

    curl -s \
        -H "Title: Bulls Eye — Health Check Failed" \
        -H "Tags: warning,stethoscope" \
        -H "Priority: high" \
        -d "$STATUS — $(date '+%Y-%m-%d %H:%M')" \
        "$NTFY" > /dev/null
fi
