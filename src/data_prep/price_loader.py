"""
================================================================================
 File   : src/data_prep/price_loader.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Loads psx_prices_raw.csv, cleans it, aligns to PSX trading calendar,
          computes all technical indicators and saves psx_prices_processed.csv.

 Input  : data/raw/prices/psx_prices_raw.csv
 Output : data/processed/prices/psx_prices_processed.csv

 Cache  : If processed CSV already exists -> return it directly, skip everything

 Fixes  : [v2]
          - drop_warmup: replaced groupby+apply+iloc with cumcount() filter
            to prevent pandas 2.x silently dropping the 'ticker' column
          - align_to_calendar: explicit reset_index() guard after reindex
          - load_raw: defensive ticker column rename for alternate column names
          - _cache_valid: validates 'ticker' and 'date' columns exist before use
          - All groupby operations use observed=True (pandas 2.x FutureWarning fix)
================================================================================
"""

import os, logging, warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EMA_PERIODS = [9, 21, 50, 200]
_WARMUP_ROWS = 252

# ── Required columns after full pipeline ─────────────────────────────────────
_REQUIRED_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


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


# ── Column guard ──────────────────────────────────────────────────────────────

def _assert_cols(df, cols, stage):
    """Raise immediately if any required column is missing — fail fast, fail loud."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "[" + stage + "] Missing columns: " + str(missing) + "\n"
            "  Present columns: " + str(list(df.columns))
        )


# ── Cache check ───────────────────────────────────────────────────────────────

def _cache_valid(path):
    if not os.path.exists(path):
        return False, None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return False, None
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            log.warning("Cache missing columns %s — rebuilding.", missing)
            return False, None
        log.info("Cache hit: %s  (%d rows, %d tickers, %s -> %s)",
                 os.path.basename(path), len(df), df["ticker"].nunique(),
                 df["date"].min().date(), df["date"].max().date())
        return True, df
    except Exception as e:
        log.warning("Cache check failed: %s", e)
        return False, None


# ── Load & clean raw ──────────────────────────────────────────────────────────

def load_raw(raw_path, train_start, test_end):
    log.info("Loading raw prices: %s", raw_path)
    df = pd.read_csv(raw_path, parse_dates=["date"])

    df = df.rename(columns={
        "symbol": "ticker", "Symbol": "ticker", "SYMBOL": "ticker",
        "TICKER": "ticker", "Ticker": "ticker",
        "stock":  "ticker", "Stock":  "ticker",
    }, errors="ignore")

    df = df.drop(columns=["change", "change_pct"], errors="ignore")
    df = df.rename(columns={"ldcp": "prev_close"}, errors="ignore")

    _assert_cols(df, ["date", "ticker", "open", "high", "low", "close", "volume"], stage="load_raw")

    df = df[(df["date"] >= train_start) & (df["date"] <= test_end)].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info("Raw loaded: %d rows | %d tickers | %s -> %s",
             len(df), df["ticker"].nunique(),
             df["date"].min().date(), df["date"].max().date())

    _assert_cols(df, _REQUIRED_COLS, stage="load_raw[post]")
    return df


# ── PSX calendar alignment ────────────────────────────────────────────────────

def align_to_calendar(df, n_stocks):
    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    tickers  = sorted(df["ticker"].unique())

    log.info("PSX calendar: %d trading days | %d tickers", len(calendar), len(tickers))

    if len(tickers) != n_stocks:
        log.warning("Expected %d tickers, found %d", n_stocks, len(tickers))

    full_idx = pd.MultiIndex.from_product([calendar, tickers], names=["date", "ticker"])
    df = df.set_index(["date", "ticker"]).sort_index()
    df = df.reindex(full_idx)

    price_cols = ["prev_close", "open", "high", "low", "close"]
    df[price_cols] = (df[price_cols]
                      .groupby(level="ticker", observed=True)
                      .transform(lambda s: s.ffill()))
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["close"])
    df = df.reset_index()

    _assert_cols(df, _REQUIRED_COLS, stage="align_to_calendar")
    log.info("After alignment: %d rows", len(df))
    return df


# ── Technical indicators ──────────────────────────────────────────────────────

def _turbulence_1d(window):
    if len(window) < 20 or np.std(window[:-1]) == 0:
        return 0.0
    mu  = np.mean(window[:-1])
    sig = np.std(window[:-1])
    return float(((window[-1] - mu) / sig) ** 2)


def add_indicators(df):
    log.info("Computing technical indicators for %d tickers ...", df["ticker"].nunique())
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    frames = []

    for ticker, grp in df.groupby("ticker", observed=True):
        g     = grp.copy().sort_values("date").reset_index(drop=True)
        close = g["close"]
        high  = g["high"]
        low   = g["low"]

        g["macd"] = (close.ewm(span=12, adjust=False).mean()
                     - close.ewm(span=26, adjust=False).mean())

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        g["rsi"] = 100 - (100 / (1 + gain / loss))

        tp  = (high + low + close) / 3
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        g["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))

        prev_close = close.shift(1)
        up_move    = high - high.shift(1)
        dn_move    = low.shift(1) - low
        pos_dm     = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        neg_dm     = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        tr         = pd.concat([high - low,
                                 (high - prev_close).abs(),
                                 (low  - prev_close).abs()], axis=1).max(axis=1)
        atr14  = tr.rolling(14).mean()
        pdi14  = (100 * pd.Series(pos_dm, index=g.index).rolling(14).mean()
                  / atr14.replace(0, np.nan))
        ndi14  = (100 * pd.Series(neg_dm, index=g.index).rolling(14).mean()
                  / atr14.replace(0, np.nan))
        g["dmi_dx"] = (100 * (pdi14 - ndi14).abs()
                       / (pdi14 + ndi14).replace(0, np.nan))

        for p in _EMA_PERIODS:
            g["ema_" + str(p)] = close.ewm(span=p, adjust=False).mean()

        bb_sma        = close.rolling(20).mean()
        bb_std        = close.rolling(20).std(ddof=0)
        g["bb_mid"]   = bb_sma
        g["bb_upper"] = bb_sma + 2 * bb_std
        g["bb_lower"] = bb_sma - 2 * bb_std
        bb_range      = (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)
        g["bb_width"] = bb_range / g["bb_mid"].replace(0, np.nan)
        g["bb_pct"]   = (close - g["bb_lower"]) / bb_range

        g["turbulence"] = (close.pct_change()
                               .rolling(252)
                               .apply(_turbulence_1d, raw=True)
                               .fillna(0.0))
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    _assert_cols(result, _REQUIRED_COLS, stage="add_indicators")
    return result


# ── Warmup drop ───────────────────────────────────────────────────────────────

def drop_warmup(df):
    before = len(df)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    mask = df.groupby("ticker", observed=True).cumcount() >= _WARMUP_ROWS
    df   = df[mask].reset_index(drop=True)
    df   = df.dropna().reset_index(drop=True)

    _assert_cols(df, _REQUIRED_COLS, stage="drop_warmup")
    log.info("Warmup drop: %d -> %d rows", before, len(df))
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def run(cfg=None):
    if cfg is None:
        cfg = load_config()

    raw_path       = os.path.join(_resolve(cfg["data"]["raw_prices_dir"]),       "psx_prices_raw.csv")
    processed_path = os.path.join(_resolve(cfg["data"]["processed_prices_dir"]), "psx_prices_processed.csv")
    train_start    = cfg["data"]["train_start"]
    test_end       = cfg["data"]["test_end"]
    n_stocks       = cfg["data"]["n_stocks"]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    valid, cached = _cache_valid(processed_path)
    if valid:
        return cached

    df = load_raw(raw_path, train_start, test_end)
    df = align_to_calendar(df, n_stocks)
    df = add_indicators(df)
    df = drop_warmup(df)

    _assert_cols(df, _REQUIRED_COLS, stage="run[pre-save]")
    log.info("Final shape: %d rows | %d tickers | cols: %s",
             len(df), df["ticker"].nunique(), list(df.columns))

    df.to_csv(processed_path, index=False)
    log.info("Saved -> %s  (%d rows, %.1f MB)",
             processed_path, len(df), os.path.getsize(processed_path) / 1e6)
    return df


if __name__ == "__main__":
    run()
