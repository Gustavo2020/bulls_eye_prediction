#!/bin/bash
# Bulls Eye — Service Setup
# Run with: sudo bash setup_service.sh

set -e
PROJECT="YOUR_HOME_DIR/MSADS_/Time_series_31006/FINAL_PROJECT_VF"

echo "==> Installing systemd services..."
cp "$PROJECT/bullseye.service"                /etc/systemd/system/bullseye.service
cp "$PROJECT/bullseye-notify-failure.service" /etc/systemd/system/bullseye-notify-failure.service

echo "==> Reloading systemd..."
systemctl daemon-reload

echo "==> Enabling bullseye on boot..."
systemctl enable bullseye

echo "==> Starting bullseye..."
systemctl start bullseye

echo "==> Adding nginx location block..."
NGINX_CONF="/etc/nginx/sites-available/gsx-2.com"

if grep -q "/bullseye" "$NGINX_CONF"; then
    echo "    [SKIP] /bullseye block already present in nginx config"
else
    python3 - <<'PYEOF'
import re, sys

conf   = "/etc/nginx/sites-available/gsx-2.com"
block  = open("YOUR_HOME_DIR/MSADS_/Time_series_31006/FINAL_PROJECT_VF/nginx_bullseye.conf").read()
text   = open(conf).read()

# Insert the location block before the last closing brace of the file
idx = text.rfind("}")
if idx == -1:
    sys.exit("ERROR: no closing brace found in nginx config")

new_text = text[:idx] + "\n" + block + "\n" + text[idx:]
open(conf, "w").write(new_text)
print("    [OK] nginx block inserted")
PYEOF
fi

echo "==> Testing nginx config..."
nginx -t

echo "==> Reloading nginx..."
systemctl reload nginx

echo ""
echo "===================================="
echo " Bulls Eye is live!"
echo " https://gsx-2.com/bullseye"
echo "===================================="
echo ""
echo "Useful commands:"
echo "  sudo systemctl status bullseye"
echo "  sudo journalctl -u bullseye -f"
echo "  tail -f $PROJECT/logs/streamlit.log"
