"""
================================================================================
 File   : src/train.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Trains PPO and A2C agents sequentially on the PSX portfolio
          environment using the shared temporal encoder. Saves only the best
          checkpoint per agent (highest episode-end portfolio value) and logs
          per-episode training history to results/tables/.

 Usage  : python -m src.train
          or: from src.train import main; main()

 Config keys used:
   training.episodes, training.seed
   environment.* (via build_env)
   agents.* (via build_agent)
================================================================================
"""

import os
import logging

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.environment import build_env
from src.agents import build_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGENT_TYPES   = ["ppo", "a2c"]
ENCODER_MODE  = "hybrid"
EPISODES_OVERRIDE = 20   # force 20 episodes regardless of config.yaml


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


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(agent, env, atype: str, explore: bool = True):
    """
    Runs one full episode on the given env with the given agent.
    Handles the differing update() signatures:
      - ppo/a2c : on-policy, agent.update(last_obs, last_done) called once
                  at episode end (rollout accumulates across the episode)

    Returns: total_reward (float), final_portfolio_value (float), losses (dict or None)
    """
    obs, info = env.reset()
    done           = False
    total_reward   = 0.0
    final_value    = info["portfolio_value"]
    last_losses    = None

    while not done:
        action = agent.select_action(obs, explore=explore)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        total_reward += reward
        final_value   = info["portfolio_value"]

        obs = next_obs

    if atype in ("ppo", "a2c"):
        last_losses = agent.update(last_obs=obs, last_done=done)

    return total_reward, final_value, last_losses


# ── Train one agent type ──────────────────────────────────────────────────────

def train_agent(atype: str, cfg: dict, checkpoint_dir: str, log_dir: str):
    log.info("=" * 70)
    log.info("Training agent: %s", atype.upper())
    log.info("=" * 70)

    np.random.seed(cfg["training"]["seed"])

    env   = build_env(cfg, split="train")
    agent = build_agent(atype, cfg, encoder_mode=ENCODER_MODE)

    episodes      = EPISODES_OVERRIDE
    best_value    = -np.inf
    history       = []

    checkpoint_path = os.path.join(checkpoint_dir, f"{atype}_best.pt")

    pbar = tqdm(
        range(1, episodes + 1),
        desc=f"{atype.upper():<6}",
        unit="ep",
        dynamic_ncols=True,
    )

    for ep in pbar:
        total_reward, final_value, losses = run_episode(agent, env, atype, explore=True)

        history.append({
            "episode"      : ep,
            "total_reward" : total_reward,
            "final_value"  : final_value,
        })

        is_best = final_value > best_value
        if is_best:
            best_value = final_value
            agent.save(checkpoint_path)
            log.info("[%s] ep=%03d | reward=%.4f | value=%.2f | NEW BEST -> saved",
                     atype.upper(), ep, total_reward, final_value)
        else:
            log.info("[%s] ep=%03d | reward=%.4f | value=%.2f",
                     atype.upper(), ep, total_reward, final_value)

        pbar.set_postfix({
            "reward": f"{total_reward:.2f}",
            "value": f"{final_value:.2f}",
            "best": f"{best_value:.2f}",
            "saved": "✓" if is_best else "",
        })

    pbar.close()

    log_path = os.path.join(log_dir, f"{atype}_train_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    log.info("[%s] training complete | best_value=%.2f | log -> %s",
             atype.upper(), best_value, log_path)

    return {"agent": atype, "best_value": best_value, "checkpoint": checkpoint_path}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: dict = None):
    if cfg is None:
        cfg = load_config()

    checkpoint_dir = _resolve("results/checkpoints")
    log_dir        = _resolve("results/tables")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary = []
    for atype in tqdm(AGENT_TYPES, desc="ALL AGENTS", unit="agent"):
        result = train_agent(atype, cfg, checkpoint_dir, log_dir)
        summary.append(result)

    log.info("=" * 70)
    log.info("ALL AGENTS TRAINED")
    for s in summary:
        log.info("  %-6s | best_value=%.2f | checkpoint=%s",
                 s["agent"].upper(), s["best_value"], s["checkpoint"])
    log.info("=" * 70)

    return summary


if __name__ == "__main__":
    main()
