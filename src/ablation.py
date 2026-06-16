
"""
================================================================================
 File   : src/ablation.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Ablation study over (encoder_mode x sentiment_on/off) for PPO.
          Trains a fresh PPO agent per config, evaluates on test split,
          and compares performance to isolate the contribution of the
          temporal encoder architecture and the sentiment feature.

 Configs : encoder_mode in {lstm, transformer, hybrid}
           sentiment    in {True, False}
           -> 6 total configs, PPO only

 Output  : results/tables/ablation_summary.csv
           results/figures/ablation_comparison.png

 Usage   : python -m src.ablation
           or: from src.ablation import main; main()
================================================================================
"""

import os
import logging

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import PSXPortfolioEnv, build_env, FEATURE_COLS
from src.agents import build_agent
from src.train import run_episode
from src.evaluate import run_rollout, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENCODER_MODES = ["lstm", "transformer", "hybrid"]
SENTIMENT_OPTIONS = [True, False]
AGENT_TYPE = "ppo"
SENTIMENT_COL_IDX = FEATURE_COLS.index("sentiment_score")


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


# ── Sentiment-ablated environment wrapper ──────────────────────────────────────

class NoSentimentEnv(PSXPortfolioEnv):
    """
    Identical to PSXPortfolioEnv except the sentiment_score feature is
    zeroed out post-normalization (neutral signal), isolating the
    contribution of sentiment without altering any other pipeline logic.
    """

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        obs[:, :, SENTIMENT_COL_IDX] = 0.0
        return obs


def build_ablation_env(cfg: dict, split: str, sentiment: bool):
    if sentiment:
        return build_env(cfg, split=split)
    return NoSentimentEnv(cfg, split=split)


# ── Train + evaluate one config ────────────────────────────────────────────────

def run_config(cfg: dict, encoder_mode: str, sentiment: bool):
    tag = f"{encoder_mode}_{'sent' if sentiment else 'nosent'}"
    log.info("=" * 70)
    log.info("Ablation config: %s", tag)
    log.info("=" * 70)

    np.random.seed(cfg["training"]["seed"])

    train_env = build_ablation_env(cfg, split="train", sentiment=sentiment)
    agent     = build_agent(AGENT_TYPE, cfg, encoder_mode=encoder_mode)

    episodes = cfg["training"]["episodes"]
    for ep in range(1, episodes + 1):
        total_reward, final_value, _ = run_episode(agent, train_env, AGENT_TYPE, explore=True)
        if ep % 10 == 0 or ep == episodes:
            log.info("[%s] ep=%03d/%03d | reward=%.4f | value=%.2f",
                     tag, ep, episodes, total_reward, final_value)

    test_env = build_ablation_env(cfg, split="test", sentiment=sentiment)
    rollout  = run_rollout(agent, test_env)
    metrics  = compute_metrics(rollout["values"], rollout["daily_returns"])
    metrics["encoder_mode"]  = encoder_mode
    metrics["sentiment"]     = sentiment
    metrics["config"]        = tag
    metrics["final_value"]   = float(rollout["values"][-1])

    log.info("[%s] DONE | cum_return=%.4f | sharpe=%.4f | final_value=%.2f",
             tag, metrics["cumulative_return"], metrics["sharpe_ratio"], metrics["final_value"])

    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ablation(summary_df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pivot_sharpe = summary_df.pivot(index="encoder_mode", columns="sentiment", values="sharpe_ratio")
    pivot_sharpe.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Sharpe Ratio by Encoder Mode & Sentiment")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].legend(title="Sentiment", labels=["Off", "On"])
    axes[0].grid(alpha=0.3)

    pivot_ret = summary_df.pivot(index="encoder_mode", columns="sentiment", values="cumulative_return")
    pivot_ret.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Cumulative Return by Encoder Mode & Sentiment")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].legend(title="Sentiment", labels=["Off", "On"])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Saved ablation comparison plot -> %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: dict = None):
    if cfg is None:
        cfg = load_config()

    table_dir  = _resolve("results/tables")
    figure_dir = _resolve("results/figures")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    rows = []
    for encoder_mode in ENCODER_MODES:
        for sentiment in SENTIMENT_OPTIONS:
            metrics = run_config(cfg, encoder_mode, sentiment)
            rows.append(metrics)

    summary_df = pd.DataFrame(rows)
    col_order  = ["config", "encoder_mode", "sentiment", "cumulative_return",
                  "annualized_return", "annualized_volatility", "sharpe_ratio",
                  "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate",
                  "final_value"]
    summary_df = summary_df[col_order]

    table_path = os.path.join(table_dir, "ablation_summary.csv")
    summary_df.to_csv(table_path, index=False)
    log.info("Saved ablation summary -> %s", table_path)

    figure_path = os.path.join(figure_dir, "ablation_comparison.png")
    plot_ablation(summary_df, figure_path)

    print("\n" + "=" * 100)
    print("ABLATION SUMMARY (PPO)")
    print("=" * 100)
    print(summary_df.round(4).to_string(index=False))
    print("=" * 100)

    return summary_df


if __name__ == "__main__":
    main()
