# PMCC Scanner — Setup Guide
# Fynix Systems — Kevin

## QUICK START (5 steps)

### Step 1 — Install dependencies
```
pip install requests pandas gspread google-auth slack-sdk schedule
```

### Step 2 — Set up Alpaca API keys
1. Go to https://app.alpaca.markets
2. Click "API Keys" in the left menu
3. Generate Paper Trading keys (for testing)
4. Copy API Key and Secret Key
5. Paste into CONFIG in pmcc_scanner.py:
   ALPACA_API_KEY:    "your_key_here"
   ALPACA_SECRET_KEY: "your_secret_here"

### Step 3 — Set up Google Sheets access
1. Go to https://console.cloud.google.com
2. Create new project (name it anything)
3. Search for "Google Sheets API" → Enable it
4. Search for "Google Drive API" → Enable it
5. Go to "Credentials" → "Create Credentials" → "Service Account"
6. Name it "pmcc-scanner" → Create
7. Click your new service account → "Keys" tab
8. "Add Key" → "Create New Key" → JSON
9. Download the JSON file
10. Rename it to: google_credentials.json
11. Put it in the same folder as pmcc_scanner.py
12. Open the JSON file — copy the "client_email" value
13. Go to your Google Sheet → Share → paste that email → Editor

### Step 4 — Set up Slack bot
1. Go to https://api.slack.com/apps
2. "Create New App" → "From Scratch"
3. Name it "PMCC Scanner" → pick your workspace
4. Go to "OAuth & Permissions" → "Bot Token Scopes"
5. Add scope: chat:write
6. "Install to Workspace" → Authorize
7. Copy "Bot User OAuth Token" (starts with xoxb-)
8. Paste into CONFIG: SLACK_BOT_TOKEN
9. In Slack: go to your channel → /invite @PMCC Scanner
10. Set SLACK_CHANNEL to your channel name (e.g. #pmcc-scanner)

### Step 5 — Run it
```
# Run once right now
python3 pmcc_scanner.py

# Run every trading day at 9:35 AM automatically
python3 pmcc_scanner.py --schedule
```

---

## HOW TO READ THE OUTPUT

### Console output (example):
```
──────────────────────────────────────────────────
  SOUN @ $8.88
  ⚠️ CHECK GATES 2 & 7 MANUALLY (8/8 auto gates pass)
  Net Debit: $334 | ROC: 24.5%/cycle | Breakeven: $8.84
  ✅ Gate 1: Price $8.88 above EMA21 $8.45
  ⚠️ Gate 2: Verify at earningswhispers.com (MANUAL)
  ✅ Gate 3: IV Rank 32.8% — low enough
  ✅ Gate 4: Delta 0.86 — strong stock surrogate
  ✅ Gate 5: $334 < $450 — trade has edge
  ✅ Gate 6: $416 fits account sizing
  ⚠️ Gate 7: Research company fundamentals (MANUAL)
  ✅ Gate 8: $10.00 > $5.50 — correct structure
  ✅ Gate 9: 66 DTE — outside 30-60 day window ← fix this
  ✅ Gate 10: $1166 buffer available
```

### Slack output:
- 🟢 = All auto gates pass, check 2 & 7 manually then enter
- ⚠️ = Some gates need manual verification
- 🔴 = Failed one or more auto gates — skip this ticker

### Google Sheets:
- Opens "Daily Scanner Results" tab
- One row per ticker with all gate results
- Updated every time you run the script

---

## GATES THAT ARE MANUAL (cannot automate)

Gate 2 — Earnings:
  Alpaca does not provide earnings dates
  Check: https://earningswhispers.com
  Or: https://finance.yahoo.com/calendar/earnings

Gate 7 — Bullish Narrative:
  Requires human judgment
  Ask: Does this company have a real reason to be
       higher in 6-12 months? Real revenue? Growing?

---

## CUSTOMIZE YOUR WATCHLIST

Open pmcc_scanner.py and edit the WATCHLIST:
```python
WATCHLIST = [
    "SOUN",   # your active position
    "RIVN",   # pending verification
    "SOFI",   # waiting for gates to clear
    # add any ticker here
]
```

---

## ADVANCED — Run as a background service (Mac/Linux)

Create a file called run_scanner.sh:
```bash
#!/bin/bash
cd /path/to/your/script/folder
python3 pmcc_scanner.py --schedule >> scanner.log 2>&1
```

Run it in the background:
```bash
chmod +x run_scanner.sh
nohup ./run_scanner.sh &
```

---

## IMPORTANT NOTES

1. Paper trading keys work for account info
   but you NEED live data subscription for
   real-time options quotes on Alpaca
   ($9/month for Algo Trader Plus plan)

2. IV Rank is approximated from historical volatility
   For more accurate IV Rank, check barchart.com
   or add a Finnhub API key (free tier available)

3. Gate 9 (DTE) will show FAIL if your target
   expiration is outside 30-60 days from today
   This is correct — adjust your short call expiration

4. All results are logged to pmcc_scanner.log
   Review this if anything looks wrong

5. This script does NOT place trades
   It only scans and reports — you enter manually
