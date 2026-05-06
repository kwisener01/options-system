#!/usr/bin/env bash
# Run once on a fresh Oracle Cloud Ubuntu 22.04 instance (as ubuntu user with sudo).
# Tested on VM.Standard.E2.1.Micro (1 GB RAM, x86_64) — swap file added for LightGBM.
# Usage: bash setup.sh
set -e

APP_DIR="/home/ubuntu/alpaca"
SERVICE="alpaca-trader"

echo "=== 1. System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y

echo "=== 2. Add 2 GB swap (needed for LightGBM on 1 GB RAM instance) ==="
if ! swapon --show | grep -q '/swapfile'; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "Swap created and enabled"
else
  echo "Swap already exists — skipping"
fi

echo "=== 3. Install Python 3.12 + build tools ==="
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential git rsync

echo "=== 4. Create app directory ==="
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/reports"
mkdir -p "$APP_DIR/logs"

echo "=== 5. Create virtual environment ==="
python3.12 -m venv "$APP_DIR/.venv"

echo "=== 6. Install systemd service ==="
sudo cp "$APP_DIR/deploy/alpaca-trader.service" "/etc/systemd/system/${SERVICE}.service"
sudo systemd-analyze verify "/etc/systemd/system/${SERVICE}.service" 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload your code:  run deploy/deploy.sh from your local machine"
echo "  2. Upload your .env:  scp .env ubuntu@<YOUR_IP>:$APP_DIR/.env"
echo "  3. Install packages:  $APP_DIR/.venv/bin/pip install -r $APP_DIR/requirements.txt"
echo "  4. Start the trader:  sudo systemctl start alpaca-trader"
echo "  5. Check logs:        journalctl -u alpaca-trader -f"
