"""
Download full SPY history and build the feature dataset.

Usage
-----
    python download_spy_data.py             # build + save parquet + CSV
    python download_spy_data.py --catalog   # print feature catalog after building
    python download_spy_data.py --info      # print info on existing cached dataset
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.logger import setup_logging
from src.data.spy_historical import build_dataset, load_dataset, feature_catalog, PARQUET_PATH


def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", action="store_true", help="Print feature catalog after build")
    p.add_argument("--info",    action="store_true", help="Print info on cached dataset (no rebuild)")
    p.add_argument("--no-save", action="store_true", help="Build but don't write to disk")
    args = p.parse_args()

    if args.info:
        if not os.path.exists(PARQUET_PATH):
            print("No cached dataset found. Run without --info to build it.")
            return
        df = load_dataset()
        _print_info(df)
        return

    df = build_dataset(save=not args.no_save)
    _print_info(df)

    if args.catalog:
        cat = feature_catalog(df)
        print("\n=== FEATURE CATALOG ===\n")
        print(cat.to_string(index=False))


def _print_info(df):
    print()
    print("=" * 60)
    print("  SPY HISTORICAL DATASET")
    print("=" * 60)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Date range    : {df.index.min().date()} to {df.index.max().date()}")
    warm = df.dropna(subset=["sma_200","rsi_14","adx"])
    print(f"  Warm rows     : {len(warm):,}  (after 200-day SMA lookback)")
    print()
    from src.data.spy_historical import feature_catalog
    cat = feature_catalog(df)
    for grp, sub in cat.groupby("group"):
        print(f"  {grp:<14} {len(sub):>3} features  "
              f"(earliest: {sub['from_date'].min()})")
    print()
    print(f"  Saved to: {PARQUET_PATH}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
