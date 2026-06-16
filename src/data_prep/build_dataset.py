"""
================================================================================
 File   : src/data_prep/build_dataset.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Joins processed prices + sentiment scores into final features.parquet
          that the temporal encoder reads.

 Input  : data/processed/prices/psx_prices_processed.csv
          data/processed/news/news_processed.csv  (after finbert scoring)

 Output : data/processed/features.parquet
          One row per (date, ticker) with all price features + sentiment score.

 Cache  : If features.parquet already exists -> return it directly
================================================================================
"""

import os, logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_valid(path):
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return False, None
        log.info("Cache hit: %s  (%d rows, %d tickers, %s -> %s)",
                 os.path.basename(path), len(df), df["ticker"].nunique(),
                 df["date"].min(), df["date"].max())
        return True, df
    except Exception as e:
        log.warning("Cache check failed: %s", e)
        return False, None


# ── Load prices ───────────────────────────────────────────────────────────────

def _load_prices(path):
    log.info("Loading prices: %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info("Prices: %d rows | %d tickers | %s -> %s",
             len(df), df["ticker"].nunique(),
             df["date"].min().date(), df["date"].max().date())
    return df


# ── Load sentiment ────────────────────────────────────────────────────────────

def _load_sentiment(path):
    """
    Loads sentiment CSV produced by finbert.py.
    Expected columns: date, sentiment_score
    One row per date. Scores in [-1, 1].
    If file missing -> returns empty DataFrame (sentiment filled with 0.0).
    """
    if not os.path.exists(path):
        log.warning("Sentiment file not found: %s — filling with 0.0", path)
        return pd.DataFrame(columns=["date", "sentiment_score"])
    df = pd.read_csv(path, parse_dates=["date"])
    log.info("Sentiment: %d dates loaded | score range [%.3f, %.3f]",
             len(df), df["sentiment_score"].min(), df["sentiment_score"].max())
    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def _merge(prices_df, sentiment_df):
    """
    Left join prices onto sentiment by date.
    Every (date, ticker) row gets the same daily sentiment score.
    Missing sentiment dates filled with 0.0 (neutral).
    """
    if sentiment_df.empty:
        prices_df["sentiment_score"] = 0.0
        return prices_df

    sentiment_df = sentiment_df[["date", "sentiment_score"]].copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    df = prices_df.merge(sentiment_df, on="date", how="left")
    missing = df["sentiment_score"].isna().sum()
    if missing > 0:
        log.info("Filling %d rows with no sentiment score -> 0.0", missing)
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    return df


# ── Train / test split check ──────────────────────────────────────────────────

def _log_split_info(df, train_end, test_start):
    train = df[df["date"] <= train_end]
    test  = df[df["date"] >= test_start]
    log.info("Train set: %d rows | %s -> %s",
             len(train), train["date"].min(), train["date"].max())
    log.info("Test  set: %d rows | %s -> %s",
             len(test),  test["date"].min(),  test["date"].max())
    overlap = df[(df["date"] > train_end) & (df["date"] < test_start)]
    if len(overlap) > 0:
        log.warning("Gap between train and test: %d rows", len(overlap))


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    prices_path    = os.path.join(_resolve(cfg["data"]["processed_prices_dir"]), "psx_prices_processed.csv")
    sentiment_path = os.path.join(_resolve(cfg["data"]["processed_news_dir"]),   "sentiment_scores.csv")
    features_path  = _resolve(cfg["data"]["features_path"])
    train_end      = pd.Timestamp(cfg["data"]["train_end"])
    test_start     = pd.Timestamp(cfg["data"]["test_start"])

    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    # cache check
    valid, cached = _cache_valid(features_path)
    if valid:
        return cached

    # load
    prices_df    = _load_prices(prices_path)
    sentiment_df = _load_sentiment(sentiment_path)

    # merge
    df = _merge(prices_df, sentiment_df)

    # sort and verify
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    _log_split_info(df, train_end, test_start)

    # final column order
    price_cols     = ["prev_close", "open", "high", "low", "close", "volume"]
    indicator_cols = ["macd", "rsi", "cci", "dmi_dx",
                      "ema_9", "ema_21", "ema_50", "ema_200",
                      "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct",
                      "turbulence"]
    sentiment_cols = ["sentiment_score"]
    all_cols       = ["date", "ticker"] + price_cols + indicator_cols + sentiment_cols
    df = df[all_cols]

    log.info("Final shape: %s | features per row: %d",
             df.shape, len(price_cols) + len(indicator_cols) + len(sentiment_cols))
    log.info("NaN check: %s", df.isna().sum()[df.isna().sum() > 0].to_dict() or "none")

    df.to_parquet(features_path, index=False)
    log.info("Saved -> %s  (%.1f MB)", features_path,
             os.path.getsize(features_path) / 1e6)
    return df


if __name__ == "__main__":
    run()
