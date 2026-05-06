import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
IS_PAPER = "paper" in ALPACA_BASE_URL

# Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Trading parameters — tunable via env vars (change in Render dashboard, no redeploy needed)
ACCOUNT_RISK_PCT  = float(os.getenv("ACCOUNT_RISK_PCT", "0.08"))   # 8% per position
MAX_POSITIONS     = int(os.getenv("MAX_POSITIONS", "3"))            # max concurrent positions
MIN_STOCK_PRICE   = float(os.getenv("MIN_STOCK_PRICE", "10.0"))    # skip penny stocks
MIN_MARKET_CAP    = 1_000_000_000                                   # $1B+ market cap only

# Monitor
MONITOR_INTERVAL_MINUTES = 15
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
TIMEZONE = "America/New_York"

# Macro ETF universe for regime detection
MACRO_SYMBOLS = ["SPY", "QQQ", "IWM", "^VIX", "GLD", "TLT", "XLF", "XLK", "XLE", "XLV"]

# Signal thresholds (from monitor_config.json)
SIGNAL_THRESHOLDS = {
    "distance_to_max_pain_pct": 0.5,
    "distance_to_wall_pct": 0.5,
    "implied_move_pct": 0.8,
    "put_call_ratio_bearish": 1.2,
    "put_call_ratio_bullish": 0.7,
}

# ML model
ML_TARGET_RETURN_PCT = 0.02      # label positive if 5-day forward return > 2%
ML_LOOKBACK_DAYS = 504           # 2 years of training history
ML_TOP_N          = int(os.getenv("ML_TOP_N", "3"))   # stocks to select per cycle
ML_MIN_TRAIN_WEEKS = 26          # minimum weeks before first walk-forward fold

# Backtest
BACKTEST_INITIAL_CAPITAL = 10_000.0
BACKTEST_SLIPPAGE_PCT = 0.0005   # 5 bps per side

# Wheel strategy
WHEEL_TICKERS = ["XSP"]
WHEEL_PUT_OTM_PCT = 0.05       # sell put 5% below current price (~0.30 delta)
WHEEL_CALL_OTM_PCT = 0.03      # sell call 3% above cost basis
WHEEL_DTE = 30                 # target days to expiration
WHEEL_IV_PREMIUM = 1.15        # inflate realized vol to mimic IV premium sellers collect
WHEEL_VOL_WINDOW = 20          # trading days for realized-vol calculation
WHEEL_INITIAL_CAPITAL = 100_000.0   # SPY/XSP ~$500 => 1 contract needs ~$47-55k collateral

# GEX intraday trader
GEX_RISK_PCT    = float(os.getenv("GEX_RISK_PCT", "0.01"))              # % of equity per GEX trade
ALLOW_SHORT_GEX = os.getenv("ALLOW_SHORT_GEX", "false").lower() == "true"

# Paths
import pathlib
_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR         = str(_ROOT / "data")
REPORTS_DIR      = str(_ROOT / "reports")
SP500_CACHE_PATH = str(_ROOT / "data" / "sp500_tickers.csv")
PRICE_CACHE_PATH = str(_ROOT / "data" / "prices_cache.pkl")
MODEL_PATH       = str(_ROOT / "data" / "ml_model.joblib")
GEX_CHAIN_DIR    = str(_ROOT / "data" / "gex_chain")
