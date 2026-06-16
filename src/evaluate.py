
"""
================================================================================
 File   : src/evaluate.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Loads best checkpoints for DDPG, PPO, A2C, runs greedy (no-explore)
          rollouts on the TEST split, computes standard portfolio metrics,
          and saves a comparison table + equity curve plot.

 Input  : results/checkpoints/{agent}_best.pt
 Output : results/tables/evaluation_summary.csv
          results/figures/equity_curves.png

 Metrics: cumulative_return, annualized_return, annualized_volatility,
          sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate

 Usage  : python -m src.evaluate
          or: from src.evaluate import main; main()
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

from src.environment import build_env
from src.agents import build_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGENT_TYPES       = ["ddpg", "ppo", "a2c"]
ENCODER_MODE      = "hybrid"
TRADING_DAYS_YEAR = 252


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


# ── Rollout ────────────────────────────────────────────────────────────────────

def run_rollout(agent, env):
    """
    Greedy (explore=False) rollout over a full episode on the given env.
    Returns: dict(dates, values, daily_returns) all aligned, length = n_steps
    """
    obs, info = env.reset()
    done    = False
    dates   = [info["date"]]
    values  = [info["portfolio_value"]]

    while not done:
        action = agent.select_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dates.append(info["date"])
        values.append(info["portfolio_value"])

    values = np.array(values, dtype=np.float64)
    daily_returns = np.diff(values) / values[:-1]
    return {"dates": dates, "values": values, "daily_returns": daily_returns}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(values: np.ndarray, daily_returns: np.ndarray, risk_free: float = 0.0):
    """
    Computes standard portfolio performance metrics from a value series.
    """
    n_days = len(daily_returns)
    if n_days == 0 or values[0] <= 0:
        return {k: float("nan") for k in
                ["cumulative_return", "annualized_return", "annualized_volatility",
                 "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate"]}

    cumulative_return = (values[-1] / values[0]) - 1.0

    mean_daily = daily_returns.mean()
    std_daily  = daily_returns.std(ddof=0)
    annualized_return     = (1.0 + mean_daily) ** TRADING_DAYS_YEAR - 1.0
    annualized_volatility = std_daily * np.sqrt(TRADING_DAYS_YEAR)

    excess = daily_returns - (risk_free / TRADING_DAYS_YEAR)
    sharpe_ratio = (excess.mean() / (excess.std(ddof=0) + 1e-12)) * np.sqrt(TRADING_DAYS_YEAR)

    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0) if len(downside) > 0 else 1e-12
    sortino_ratio = (excess.mean() / (downside_std + 1e-12)) * np.sqrt(TRADING_DAYS_YEAR)

    running_peak = np.maximum.accumulate(values)
    drawdowns    = (running_peak - values) / running_peak
    max_drawdown = float(drawdowns.max())

    calmar_ratio = annualized_return / (max_drawdown + 1e-12)

    win_rate = float((daily_returns > 0).mean())

    return {
        "cumulative_return"      : float(cumulative_return),
        "annualized_return"      : float(annualized_return),
        "annualized_volatility"  : float(annualized_volatility),
        "sharpe_ratio"           : float(sharpe_ratio),
        "sortino_ratio"          : float(sortino_ratio),
        "max_drawdown"           : float(max_drawdown),
        "calmar_ratio"           : float(calmar_ratio),
        "win_rate"               : float(win_rate),
    }


# ── Evaluate one agent ─────────────────────────────────────────────────────────

def evaluate_agent(atype: str, cfg: dict, checkpoint_dir: str):
    checkpoint_path = os.path.join(checkpoint_dir, f"{atype}_best.pt")
    if not os.path.exists(checkpoint_path):
        log.warning("[%s] checkpoint not found at %s — skipping", atype.upper(), checkpoint_path)
        return None

    env   = build_env(cfg, split="test")
    agent = build_agent(atype, cfg, encoder_mode=ENCODER_MODE)
    agent.load(checkpoint_path)

    rollout = run_rollout(agent, env)
    metrics = compute_metrics(rollout["values"], rollout["daily_returns"])
    metrics["agent"]        = atype
    metrics["final_value"]  = float(rollout["values"][-1])

    log.info("[%s] cum_return=%.4f | sharpe=%.4f | max_dd=%.4f | final_value=%.2f",
             atype.upper(), metrics["cumulative_return"], metrics["sharpe_ratio"],
             metrics["max_drawdown"], metrics["final_value"])

    return {"metrics": metrics, "rollout": rollout}


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_equity_curves(results: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    for atype, res in results.items():
        if res is None:
            continue
        rollout = res["rollout"]
        plt.plot(rollout["dates"], rollout["values"], label=atype.upper())

    plt.title("Equity Curves — Test Split")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Saved equity curve plot -> %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: dict = None):
    if cfg is None:
        cfg = load_config()

    checkpoint_dir = _resolve("results/checkpoints")
    table_dir      = _resolve("results/tables")
    figure_dir     = _resolve("results/figures")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    results = {}
    rows    = []
    for atype in AGENT_TYPES:
        res = evaluate_agent(atype, cfg, checkpoint_dir)
        results[atype] = res
        if res is not None:
            rows.append(res["metrics"])

    if not rows:
        log.error("No checkpoints found — nothing to evaluate. Run train.py first.")
        return None

    summary_df = pd.DataFrame(rows).set_index("agent")
    col_order  = ["cumulative_return", "annualized_return", "annualized_volatility",
                  "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
                  "win_rate", "final_value"]
    summary_df = summary_df[col_order]

    table_path = os.path.join(table_dir, "evaluation_summary.csv")
    summary_df.to_csv(table_path)
    log.info("Saved evaluation summary -> %s", table_path)

    figure_path = os.path.join(figure_dir, "equity_curves.png")
    plot_equity_curves(results, figure_path)

    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY — TEST SPLIT")
    print("=" * 100)
    print(summary_df.round(4).to_string())
    print("=" * 100)

    return summary_df


if __name__ == "__main__":
    main()
