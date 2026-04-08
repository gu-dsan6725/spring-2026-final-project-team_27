#!/bin/bash
# ============================================================
# session_start.sh — Run at the BEGINNING of each AWS lab session
# Usage: ./session_start.sh <EC2-IP> [path-to-key.pem]
# ============================================================

set -e

EC2_IP="${1}"
KEY="${2:-~/Downloads/labsuser.pem}"
REMOTE_USER="ec2-user"
REMOTE_DIR="~/polysignal-btc-2"
LOCAL_DB="./polysignal_btc.db"
HOURS=3.5  # Leave 30min buffer before lab closes

if [ -z "$EC2_IP" ]; then
  echo "Usage: ./session_start.sh <EC2-IP> [path-to-key.pem]"
  echo "Example: ./session_start.sh 54.123.45.67 ~/Downloads/labsuser.pem"
  exit 1
fi

SSH="ssh -i $KEY -o StrictHostKeyChecking=no $REMOTE_USER@$EC2_IP"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

echo "============================================"
echo " PolySignal Session Start"
echo " EC2: $EC2_IP"
echo " Run duration: ${HOURS} hours"
echo "============================================"

# 1. Wait for SSH to be ready
echo "[1/5] Waiting for EC2 to be ready..."
until $SSH "echo ok" &>/dev/null; do
  sleep 3
done
echo "      EC2 is up."

# 2. Install dependencies on EC2 (idempotent, fast if already done)
echo "[2/5] Installing dependencies..."
$SSH "pip install -q anthropic pytz requests websocket-client pydantic rich python-dotenv 2>&1 | tail -1"

# 3. Upload project files (rsync-style, only changed files)
echo "[3/5] Uploading project files..."
$SCP -r \
  analyzer.py collector.py ensemble.py evaluator.py \
  models.py reporter.py run.py simulator.py storage.py \
  requirements.txt \
  $REMOTE_USER@$EC2_IP:$REMOTE_DIR/ 2>/dev/null || \
$SSH "mkdir -p $REMOTE_DIR" && \
$SCP -r \
  analyzer.py collector.py ensemble.py evaluator.py \
  models.py reporter.py run.py simulator.py storage.py \
  requirements.txt \
  $REMOTE_USER@$EC2_IP:$REMOTE_DIR/

# 4. Upload .env (API keys)
if [ -f ".env" ]; then
  echo "[4/5] Uploading .env..."
  $SCP .env $REMOTE_USER@$EC2_IP:$REMOTE_DIR/.env
else
  echo "[4/5] WARNING: .env not found locally. Make sure it exists on EC2."
fi

# 5. Upload DB if it exists locally (skip on first session)
if [ -f "$LOCAL_DB" ]; then
  DB_SIZE=$(du -sh "$LOCAL_DB" | cut -f1)
  echo "[5/5] Uploading database ($DB_SIZE)..."
  $SCP "$LOCAL_DB" $REMOTE_USER@$EC2_IP:$REMOTE_DIR/polysignal_btc.db
  echo "      Database restored from previous session."
else
  echo "[5/5] No local database found — starting fresh (first session)."
fi

echo ""
echo "============================================"
echo " Setup complete! Starting prediction run..."
echo " Duration: ${HOURS} hours (~$(echo "$HOURS * 12" | bc | cut -d. -f1) predictions)"
echo " Run 'session_end.sh $EC2_IP $KEY' before lab closes!"
echo "============================================"
echo ""

# Run the pipeline (in foreground so you can watch it)
$SSH -t "cd $REMOTE_DIR && python run.py --live --hours $HOURS"

echo ""
echo "============================================"
echo " Run complete! Downloading database..."
echo "============================================"
$SCP $REMOTE_USER@$EC2_IP:$REMOTE_DIR/polysignal_btc.db "$LOCAL_DB"
DB_SIZE=$(du -sh "$LOCAL_DB" | cut -f1)
echo " Database saved locally ($DB_SIZE). Session done."
echo "============================================"
