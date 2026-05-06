#!/usr/bin/env bash
# Push code updates from your local machine to Oracle Cloud.
# Run from Git Bash on Windows: bash deploy/deploy.sh
# Usage: bash deploy/deploy.sh <server-ip>
set -e

SERVER_IP="${1:-}"
REMOTE_USER="ubuntu"
APP_DIR="/home/ubuntu/alpaca"

if [[ -z "$SERVER_IP" ]]; then
  echo "Usage: bash deploy/deploy.sh <server-ip>"
  echo "Example: bash deploy/deploy.sh 132.145.20.55"
  exit 1
fi

REMOTE="${REMOTE_USER}@${SERVER_IP}"

echo "=== Syncing code to ${REMOTE}:${APP_DIR} ==="
rsync -avz --progress \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pkl' \
  --exclude='data/gex_chain/' \
  --exclude='data/options_paper_state.json' \
  --exclude='data/rotating_paper_state.json' \
  --exclude='reports/' \
  --exclude='logs/' \
  --exclude='.git/' \
  . "${REMOTE}:${APP_DIR}"

echo "=== Installing / updating packages ==="
ssh "${REMOTE}" "${APP_DIR}/.venv/bin/pip install -q -r ${APP_DIR}/requirements.txt"

echo "=== Restarting service ==="
ssh "${REMOTE}" "sudo systemctl restart alpaca-trader && sudo systemctl status alpaca-trader --no-pager"

echo ""
echo "=== Deploy complete. Tail logs with: ==="
echo "  ssh ${REMOTE} 'journalctl -u alpaca-trader -f'"
