#!/bin/bash
# ============================================================
# session_end.sh — Run EARLY (before lab auto-closes) to save DB
# Usage: ./session_end.sh <EC2-IP> [path-to-key.pem]
# ============================================================

EC2_IP="${1}"
KEY="${2:-~/Downloads/labsuser.pem}"
REMOTE_USER="ec2-user"
REMOTE_DIR="~/polysignal-btc-2"
LOCAL_DB="./polysignal_btc.db"
LOCAL_DB_BACKUP="./polysignal_btc_$(date +%Y%m%d_%H%M).db"

if [ -z "$EC2_IP" ]; then
  echo "Usage: ./session_end.sh <EC2-IP> [path-to-key.pem]"
  exit 1
fi

SCP="scp -i $KEY -o StrictHostKeyChecking=no"

echo "============================================"
echo " PolySignal Session End — Saving Database"
echo "============================================"

# Backup existing local DB before overwriting
if [ -f "$LOCAL_DB" ]; then
  cp "$LOCAL_DB" "$LOCAL_DB_BACKUP"
  echo " Backup of previous DB: $LOCAL_DB_BACKUP"
fi

# Download DB from EC2
echo " Downloading database from EC2..."
$SCP $REMOTE_USER@$EC2_IP:$REMOTE_DIR/polysignal_btc.db "$LOCAL_DB"

DB_SIZE=$(du -sh "$LOCAL_DB" | cut -f1)
echo " Saved: $LOCAL_DB ($DB_SIZE)"

# Quick stats
echo ""
echo " Quick stats:"
python3 -c "
import sqlite3
conn = sqlite3.connect('$LOCAL_DB')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM predictions')
preds = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM eval_records')
evals = cur.fetchone()[0]
cur.execute('SELECT balance FROM account')
row = cur.fetchone()
balance = row[0] if row else 'N/A'
cur.execute('SELECT AVG(CASE WHEN correct=1 THEN 1.0 ELSE 0.0 END) FROM eval_records')
acc = cur.fetchone()[0]
print(f'  Predictions: {preds}')
print(f'  Scored:      {evals}')
print(f'  Accuracy:    {acc*100:.1f}%' if acc else '  Accuracy:    N/A')
print(f'  Balance:     \${balance:,.2f}' if isinstance(balance, float) else f'  Balance:     {balance}')
conn.close()
" 2>/dev/null || echo "  (install python3 locally to see stats)"

echo ""
echo " Safe to close the lab now."
echo "============================================"
