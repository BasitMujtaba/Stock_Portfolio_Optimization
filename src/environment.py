
"""
================================================================================
 File   : src/environment.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Gymnasium environment for PSX portfolio optimization.

 Observation : (n_tickers, window, n_features=21) z-score normalized
               using train-set mean/std only (no leakage into test).
 Action      : (n_stocks+1,) raw logits -> softmax -> portfolio weights
               index 0 = cash, indices 1..n_stocks = stock weights
 Reward      : log_return - lambda_mdd * drawdown_penalty
               - mu_turbulence * turbulence_flag
 Turbulence  : flag=1 if today turbulence > rolling 95th pct of train history

 Factory     : build_env(cfg, split='train'|'test')
================================================================================
"""

import os, logging
import numpy as np
import pandas as pd
import yaml
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# feature columns in exact parquet order (21 total)
FEATURE_COLS = [
    "prev_close", "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "dmi_dx",
    "ema_9", "ema_21", "ema_50", "ema_200",
    "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "turbulence", "sentiment_score",
]
N_FEATURES = len(FEATURE_COLS)   # 21


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def _resolve(cfg_path):
    if os.path.isabs(cfg_path):
        return cfg_path
    return os.path.join(PROJECT_ROOT, cfg_path)


# ── Normalisation stats (train only) ──────────────────────────────────────────

def _compute_norm_stats(train_df: pd.DataFrame):
    """
    Compute per-ticker per-feature mean and std from the TRAIN split only.
    Returns dicts keyed by ticker: stats[ticker] = (mean_arr, std_arr)
    both shape (n_features,).
    """
    stats = {}
    for ticker, grp in train_df.groupby("ticker"):
        vals = grp[FEATURE_COLS].values.astype(np.float32)
        mu   = vals.mean(axis=0)
        sig  = vals.std(axis=0)
        sig  = np.where(sig < 1e-8, 1.0, sig)   # avoid div-by-zero
        stats[ticker] = (mu, sig)
    return stats


# ── Turbulence threshold (train only) ─────────────────────────────────────────

def _compute_turbulence_threshold(train_df: pd.DataFrame) -> float:
    """95th percentile of daily turbulence over the train split."""
    # one turbulence value per (date, ticker); take mean across tickers per date
    daily = (train_df.groupby("date")["turbulence"]
                     .mean()
                     .values)
    threshold = float(np.percentile(daily, 95))
    log.info("Turbulence 95th pct threshold (train): %.4f", threshold)
    return threshold


# ── Data loader ────────────────────────────────────────────────────────────────

def _load_split(cfg, split: str):
    """
    Load features.parquet and slice to the requested split.
    Returns (train_df, split_df, tickers, dates).
    train_df always returned so norm stats can be computed.
    """
    features_path = _resolve(cfg["data"]["features_path"])
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    tickers = sorted(df["ticker"].unique())
    assert len(tickers) == cfg["data"]["n_stocks"], (
        f"Expected {cfg['data']['n_stocks']} tickers, got {len(tickers)}"
    )

    train_end   = pd.Timestamp(cfg["data"]["train_end"])
    test_start  = pd.Timestamp(cfg["data"]["test_start"])

    train_df = df[df["date"] <= train_end].copy()

    if split == "train":
        split_df = train_df
    elif split == "test":
        split_df = df[df["date"] >= test_start].copy()
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    dates = sorted(split_df["date"].unique())
    log.info("Split=%s | %d dates | %d tickers | %s -> %s",
             split, len(dates), len(tickers),
             dates[0].date(), dates[-1].date())
    return train_df, split_df, tickers, dates


# ── Environment ────────────────────────────────────────────────────────────────

class PSXPortfolioEnv(gym.Env):
    """
    PSX Portfolio Optimisation Environment.

    Observation space : Box (n_tickers, window, n_features) float32
    Action space      : Box (n_stocks + 1,) float32  [raw logits]
                        index 0 = cash
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, split: str = "train"):
        super().__init__()

        self.cfg    = cfg
        self.split  = split
        enc_cfg     = cfg["encoder"]
        env_cfg     = cfg["environment"]

        self.window          = enc_cfg["window"]            # 20
        self.n_stocks        = cfg["data"]["n_stocks"]      # 30
        self.initial_cash    = env_cfg["initial_cash"]      # 1_000_000
        self.lambda_mdd      = env_cfg["lambda_mdd"]        # 0.1
        self.mu_turb         = env_cfg["mu_turbulence"]     # 0.05

        # ── load data ────────────────────────────────────────────────────────
        train_df, split_df, tickers, dates = _load_split(cfg, split)

        self.tickers  = tickers
        self.dates    = dates
        self.n_dates  = len(dates)

        # build (dates x tickers x features) array for fast indexing
        self._build_arrays(split_df)

        # ── normalisation (train stats only) ─────────────────────────────────
        self._norm_stats       = _compute_norm_stats(train_df)
        self._turb_threshold   = _compute_turbulence_threshold(train_df)

        # ── spaces ───────────────────────────────────────────────────────────
        obs_shape = (self.n_stocks, self.window, N_FEATURES)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=obs_shape, dtype=np.float32
        )
        # raw logits for (cash + n_stocks)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_stocks + 1,),
            dtype=np.float32
        )

        # ── episode state (set in reset) ──────────────────────────────────────
        self._t            = None   # current timestep index into self.dates
        self._portfolio_v  = None   # current portfolio value
        self._peak_v       = None   # peak value for drawdown
        self._weights      = None   # current weights (n_stocks+1,)

        log.info(
            "PSXPortfolioEnv | split=%s | obs=%s | action=%s | dates=%d",
            split, obs_shape, (self.n_stocks + 1,), self.n_dates
        )

    # ── Internal array builder ─────────────────────────────────────────────────

    def _build_arrays(self, split_df: pd.DataFrame):
        """
        Build self._data: ndarray (n_dates, n_tickers, n_features)
        and self._close: ndarray (n_dates, n_tickers)  [raw close prices]
        and self._turb:  ndarray (n_dates,)             [mean turbulence per day]
        """
        ticker_idx = {t: i for i, t in enumerate(self.tickers)}
        date_idx   = {d: i for i, d in enumerate(self.dates)}

        n_d = len(self.dates)
        n_t = len(self.tickers)

        data  = np.zeros((n_d, n_t, N_FEATURES), dtype=np.float32)
        close = np.zeros((n_d, n_t),             dtype=np.float32)
        turb  = np.zeros((n_d,),                 dtype=np.float32)

        for _, row in split_df.iterrows():
            di = date_idx[row["date"]]
            ti = ticker_idx[row["ticker"]]
            data[di, ti, :]  = [row[c] for c in FEATURE_COLS]
            close[di, ti]    = row["close"]

        # mean turbulence across tickers per day
        turb_idx = FEATURE_COLS.index("turbulence")
        turb     = data[:, :, turb_idx].mean(axis=1)

        self._data  = data    # (n_dates, n_tickers, n_features)
        self._close = close   # (n_dates, n_tickers)
        self._turb  = turb    # (n_dates,)

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        """
        raw  : (n_tickers, window, n_features)
        out  : (n_tickers, window, n_features)  z-score using train stats
        """
        out = raw.copy()
        for i, ticker in enumerate(self.tickers):
            mu, sig     = self._norm_stats[ticker]   # (n_features,)
            out[i]      = (raw[i] - mu) / sig        # broadcast over window
        return out

    # ── Observation builder ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Returns z-score normalised window ending at self._t (inclusive).
        Shape: (n_tickers, window, n_features)
        """
        start = self._t - self.window + 1
        end   = self._t + 1
        raw   = self._data[start:end]              # (window, n_tickers, n_features)
        raw   = raw.transpose(1, 0, 2)            # (n_tickers, window, n_features)
        return self._normalise(raw).astype(np.float32)

    # ── gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start at window-1 so first obs has a full window
        self._t           = self.window - 1
        self._portfolio_v = float(self.initial_cash)
        self._peak_v      = float(self.initial_cash)
        # equal weight across cash + stocks initially
        self._weights     = np.ones(self.n_stocks + 1, dtype=np.float32) / (self.n_stocks + 1)

        obs  = self._get_obs()
        info = {"date": self.dates[self._t], "portfolio_value": self._portfolio_v}
        return obs, info

    def step(self, action: np.ndarray):
        # ── 1. softmax action -> new weights ─────────────────────────────────
        action      = np.asarray(action, dtype=np.float64)
        exp_a       = np.exp(action - action.max())
        new_weights = (exp_a / exp_a.sum()).astype(np.float32)  # (n_stocks+1,)

        # ── 2. price return for this step ────────────────────────────────────
        prev_t = self._t
        next_t = self._t + 1

        if next_t >= self.n_dates:
            # terminal: no price change
            returns = np.zeros(self.n_stocks, dtype=np.float32)
        else:
            prev_close = self._close[prev_t]   # (n_stocks,)
            next_close = self._close[next_t]   # (n_stocks,)
            # avoid div-by-zero on zero prices
            safe_prev  = np.where(prev_close > 0, prev_close, 1.0)
            returns    = (next_close - prev_close) / safe_prev   # (n_stocks,)

        # stock weights are indices 1..n_stocks; index 0 = cash (return=0)
        stock_weights   = new_weights[1:]                         # (n_stocks,)
        portfolio_ret   = float(np.dot(stock_weights, returns))   # scalar

        # ── 3. portfolio value update ─────────────────────────────────────────
        prev_value          = self._portfolio_v
        new_value           = prev_value * (1.0 + portfolio_ret)
        self._portfolio_v   = new_value
        self._peak_v        = max(self._peak_v, new_value)

        # ── 4. reward components ──────────────────────────────────────────────
        # log return (clipped for numerical safety)
        log_ret = float(np.log(new_value / prev_value + 1e-8))

        # drawdown penalty: current drawdown from peak
        drawdown        = (self._peak_v - new_value) / (self._peak_v + 1e-8)
        drawdown_pen    = max(0.0, drawdown)

        # turbulence flag
        turb_today      = float(self._turb[prev_t])
        turb_flag       = 1.0 if turb_today > self._turb_threshold else 0.0

        reward = (log_ret
                  - self.lambda_mdd  * drawdown_pen
                  - self.mu_turb     * turb_flag)

        # ── 5. advance timestep ───────────────────────────────────────────────
        self._t       = next_t
        self._weights = new_weights
        terminated    = (self._t >= self.n_dates - 1)
        truncated     = False

        # ── 6. next obs ───────────────────────────────────────────────────────
        if not terminated:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "date"            : self.dates[prev_t],
            "portfolio_value" : new_value,
            "log_return"      : log_ret,
            "drawdown"        : drawdown_pen,
            "turbulence_flag" : turb_flag,
            "portfolio_return": portfolio_ret,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        log.info("t=%d | date=%s | value=%.2f | peak=%.2f",
                 self._t, self.dates[min(self._t, self.n_dates-1)],
                 self._portfolio_v, self._peak_v)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_env(cfg: dict = None, split: str = "train") -> PSXPortfolioEnv:
    """
    Convenience factory.

    Parameters
    ----------
    cfg   : loaded config dict (loads from config.yaml if None)
    split : 'train' | 'test'
    """
    if cfg is None:
        cfg = load_config()
    return PSXPortfolioEnv(cfg, split=split)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    cfg = load_config()

    for split in ["train", "test"]:
        env  = build_env(cfg, split=split)
        obs, info = env.reset()
        print(f"\n[{split}] reset obs={obs.shape} | value={info['portfolio_value']:.0f}")

        total_reward = 0.0
        for step in range(5):
            action          = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward   += reward
            print(f"  step={step+1} | reward={reward:.6f} | "
                  f"value={info['portfolio_value']:.2f} | "
                  f"turb_flag={info['turbulence_flag']} | "
                  f"done={terminated}")
            if terminated:
                break

        print(f"  total_reward={total_reward:.6f}")
