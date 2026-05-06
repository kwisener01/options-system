"""
Download and cache today's SPY options chain for GEX analysis.

Run once per morning (or add to scheduler) so the GEX scanner never
blocks on a live API call during the trading session:

    python download_gex_chain.py              # today
    python download_gex_chain.py --date 2026-04-30  # specific date (replay)

The file is saved to data/gex_chain/spy_chain_YYYY-MM-DD.pkl
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.analysis.gex_scanner import _fetch_chain_from_api, save_chain, load_chain, _spot_and_vix
from datetime import date


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None,
                   help="Date to cache (YYYY-MM-DD, default=today)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if cache already exists")
    p.add_argument("--n-expiries", type=int, default=4,
                   help="Number of weekly expiries to include (default 4)")
    return p.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args   = parse_args()

    as_of = date.fromisoformat(args.date) if args.date else date.today()

    if not args.force:
        existing = load_chain(as_of)
        if existing:
            print(f"Cache already exists for {as_of}: {len(existing)} contracts. Use --force to refresh.")
            return

    print(f"Fetching SPY options chain for {as_of} ({args.n_expiries} expiries)...")
    spot, vix, _ = _spot_and_vix()
    print(f"  SPY=${spot:.2f}  VIX={vix:.1f}")

    contracts = _fetch_chain_from_api(spot, n_expiries=args.n_expiries)
    if not contracts:
        print("ERROR: No contracts returned. Market may be closed or yfinance unavailable.")
        sys.exit(1)

    save_chain(contracts, as_of)
    print(f"Saved {len(contracts)} contracts -> data/gex_chain/spy_chain_{as_of}.pkl")


if __name__ == "__main__":
    main()
