
"""
================================================================================
 File   : src/agents.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: DRL agents (DDPG, PPO, A2C) for PSX portfolio optimization.

 All agents share the same interface:
   agent.select_action(obs)            -> action np.ndarray (n_stocks+1,)
   agent.store(obs, action, reward, next_obs, done)
   agent.update()                      -> dict of loss scalars (or None)
   agent.save(path) / agent.load(path)

 Encoder (PortfolioEncoder) is embedded inside each agent.
 Observation input : np.ndarray (n_stocks, window, n_features)
 Action output     : np.ndarray (n_stocks+1,)  raw logits (softmax in env)

 Config keys used:
   data.n_stocks
   encoder.*
   training.batch_size, lr_actor, lr_critic, gamma, seed
   agents.ddpg.tau, agents.ddpg.noise
   agents.ppo.clip_epsilon, agents.ppo.epochs
   agents.a2c.entropy_coef, agents.a2c.value_coef
================================================================================
"""

import os
import copy
import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from src.encoder import build_encoder

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Device ─────────────────────────────────────────────────────────────────────

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer  (DDPG — off-policy)
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.
    Stores flat numpy arrays; returns torch tensors on sample().
    """

    def __init__(self, capacity: int, obs_shape: tuple,
                 action_dim: int, device: torch.device):
        self.capacity   = capacity
        self.device     = device
        self.ptr        = 0
        self.size       = 0

        self.obs      = np.zeros((capacity, *obs_shape),  dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape),  dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),           dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),           dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer  (PPO / A2C — on-policy)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    On-policy rollout storage. Cleared after every update.
    """

    def __init__(self):
        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.log_probs = []
        self.values   = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ══════════════════════════════════════════════════════════════════════════════
# Ornstein-Uhlenbeck Noise  (DDPG exploration)
# ══════════════════════════════════════════════════════════════════════════════

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Suitable for continuous action spaces.
    """

    def __init__(self, size: int, sigma: float = 0.1,
                 mu: float = 0.0, theta: float = 0.15, dt: float = 1e-2):
        self.size  = size
        self.sigma = sigma
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.dt    = dt
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = (self.theta * (self.mu - self.state) * self.dt
              + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size))
        self.state = self.state + dx
        return self.state.copy()


# ══════════════════════════════════════════════════════════════════════════════
# Shared network building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _mlp(dims: list[int], activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    """Build a fully-connected MLP from a list of layer widths."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# DDPG Networks
# ══════════════════════════════════════════════════════════════════════════════

class DDPGActor(nn.Module):
    """
    Deterministic actor: encoded_obs -> action logits.
    Input  : (batch, encoder_out_dim)
    Output : (batch, action_dim)   — raw logits, softmax applied in env.step()
    """

    def __init__(self, encoder_out_dim: int, action_dim: int,
                 hidden: int = 256):
        super().__init__()
        self.net = _mlp([encoder_out_dim, hidden, hidden, action_dim])

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        return self.net(enc)


class DDPGCritic(nn.Module):
    """
    Q-network: (encoded_obs, action) -> Q-value scalar.
    Input  : enc (batch, encoder_out_dim)  +  action (batch, action_dim)
    Output : (batch, 1)
    """

    def __init__(self, encoder_out_dim: int, action_dim: int,
                 hidden: int = 256):
        super().__init__()
        self.net = _mlp([encoder_out_dim + action_dim, hidden, hidden, 1])

    def forward(self, enc: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([enc, action], dim=-1))


# ══════════════════════════════════════════════════════════════════════════════
# PPO / A2C Networks
# ══════════════════════════════════════════════════════════════════════════════

class ActorCriticNet(nn.Module):
    """
    Shared MLP backbone with separate actor (policy) and critic (value) heads.
    Used by both PPO and A2C.

    Actor head  : outputs mean logits for a Gaussian policy in action space.
    Critic head : outputs a scalar state value V(s).
    """

    def __init__(self, encoder_out_dim: int, action_dim: int,
                 hidden: int = 256):
        super().__init__()
        self.backbone = _mlp([encoder_out_dim, hidden, hidden], activation=nn.Tanh)

        # Actor: mean logits (log_std learned as a parameter)
        self.actor_mean    = nn.Linear(hidden, action_dim)
        self.log_std       = nn.Parameter(torch.zeros(action_dim))

        # Critic: scalar value
        self.critic_value  = nn.Linear(hidden, 1)

        # Initialize actor output near-zero for stable early training
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)

    def forward(self, enc: torch.Tensor):
        """Returns (mean, std, value) for given encoded observation."""
        h     = self.backbone(enc)
        mean  = self.actor_mean(h)
        std   = self.log_std.exp().expand_as(mean)
        value = self.critic_value(h)
        return mean, std, value

    def get_action_and_value(self, enc: torch.Tensor, action=None):
        """
        Sample action from Gaussian policy and compute log_prob + entropy.
        If action provided (during PPO update), compute log_prob for that action.
        Returns: action, log_prob, entropy, value
        """
        mean, std, value = self.forward(enc)
        dist             = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy  = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# Base Agent
# ══════════════════════════════════════════════════════════════════════════════

class BaseAgent:
    """
    Common interface for all agents.
    Subclasses must implement: select_action, store, update.
    """

    def __init__(self, cfg: dict, encoder_mode: str = "hybrid"):
        self.cfg        = cfg
        self.device     = _get_device()
        self.n_stocks   = cfg["data"]["n_stocks"]
        self.action_dim = self.n_stocks + 1          # cash + stocks
        self.obs_shape  = (self.n_stocks,
                           cfg["encoder"]["window"],
                           21)                       # N_FEATURES = 21

        # Seed
        seed = cfg["training"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Shared encoder
        self.encoder = build_encoder(cfg, input_dim=21, mode=encoder_mode)
        self.encoder.to(self.device)
        self.encoder_out_dim = self.n_stocks * self.encoder.output_dim

        log.info(
            "%s | device=%s | obs=%s | action_dim=%d | enc_out=%d",
            self.__class__.__name__, self.device,
            self.obs_shape, self.action_dim, self.encoder_out_dim,
        )

    def _encode(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        obs_t : (batch, n_stocks, window, n_features)
        return: (batch, n_stocks * encoder_out_dim)
        """
        enc = self.encoder(obs_t)               # (batch, n_stocks, output_dim)
        return enc.reshape(enc.size(0), -1)     # (batch, n_stocks * output_dim)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """np (n_stocks, window, n_features) -> torch (1, n_stocks, window, n_features)"""
        return torch.FloatTensor(obs).unsqueeze(0).to(self.device)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def store(self, obs, action, reward, next_obs, done):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# DDPG Agent
# ══════════════════════════════════════════════════════════════════════════════

class DDPGAgent(BaseAgent):
    """
    Deep Deterministic Policy Gradient.

    - Off-policy, continuous action space.
    - Encoder + Actor share one forward pass; separate Critic network.
    - Target networks for both encoder+actor and critic (soft updates).
    - Ornstein-Uhlenbeck noise for exploration.
    - Replay buffer of 100k transitions.

    Config used:
      training: batch_size, lr_actor, lr_critic, gamma
      agents.ddpg: tau, noise
    """

    BUFFER_SIZE = 100_000

    def __init__(self, cfg: dict, encoder_mode: str = "hybrid"):
        super().__init__(cfg, encoder_mode)

        ddpg_cfg   = cfg["agents"]["ddpg"]
        train_cfg  = cfg["training"]

        self.tau        = ddpg_cfg["tau"]
        self.noise_std  = ddpg_cfg["noise"]
        self.gamma      = train_cfg["gamma"]
        self.batch_size = train_cfg["batch_size"]
        self.lr_actor   = train_cfg["lr_actor"]
        self.lr_critic  = train_cfg["lr_critic"]

        # ── Online networks ──────────────────────────────────────────────────
        self.actor  = DDPGActor(self.encoder_out_dim, self.action_dim).to(self.device)
        self.critic = DDPGCritic(self.encoder_out_dim, self.action_dim).to(self.device)

        # ── Target networks (frozen copies, soft-updated) ────────────────────
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_actor   = copy.deepcopy(self.actor)
        self.target_critic  = copy.deepcopy(self.critic)
        for net in [self.target_encoder, self.target_actor, self.target_critic]:
            for p in net.parameters():
                p.requires_grad = False

        # ── Optimizers ───────────────────────────────────────────────────────
        actor_params  = list(self.encoder.parameters()) + list(self.actor.parameters())
        self.opt_actor  = optim.Adam(actor_params,    lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # ── Replay buffer ────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            self.BUFFER_SIZE, self.obs_shape, self.action_dim, self.device
        )

        # ── Exploration noise ─────────────────────────────────────────────────
        self.ou_noise = OUNoise(self.action_dim, sigma=self.noise_std)

        log.info("DDPGAgent | tau=%.4f | noise=%.3f | buffer=%d",
                 self.tau, self.noise_std, self.BUFFER_SIZE)

    # ── Soft update helper ────────────────────────────────────────────────────

    def _soft_update(self, online: nn.Module, target: nn.Module):
        for op, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    # ── Interface ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        obs : (n_stocks, window, n_features)
        Returns action logits (n_stocks+1,) with OU noise if explore=True.
        """
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            obs_t  = self._obs_to_tensor(obs)        # (1, N, T, F)
            enc    = self._encode(obs_t)             # (1, enc_dim)
            action = self.actor(enc).cpu().numpy()[0]  # (action_dim,)
        self.encoder.train()
        self.actor.train()

        if explore:
            action = action + self.ou_noise.sample()
        return action.astype(np.float32)

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self):
        """
        One gradient step on actor and critic.
        Returns dict of losses, or None if buffer not ready.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(self.batch_size)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_enc    = self._encode_target(next_obs_b)
            next_action = self.target_actor(next_enc)
            next_q      = self.target_critic(next_enc, next_action)
            target_q    = rew_b + self.gamma * (1.0 - done_b) * next_q

        enc    = self._encode(obs_b)
        curr_q = self.critic(enc.detach(), act_b)
        critic_loss = F.mse_loss(curr_q, target_q)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # ── Actor update ──────────────────────────────────────────────────────
        enc_fresh   = self._encode(obs_b)
        actor_loss  = -self.critic(enc_fresh.detach(), self.actor(enc_fresh)).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()), 1.0
        )
        self.opt_actor.step()

        # ── Soft target updates ────────────────────────────────────────────────
        self._soft_update(self.encoder, self.target_encoder)
        self._soft_update(self.actor,   self.target_actor)
        self._soft_update(self.critic,  self.target_critic)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss" : actor_loss.item(),
        }

    def _encode_target(self, obs_t: torch.Tensor) -> torch.Tensor:
        enc = self.target_encoder(obs_t)
        return enc.reshape(enc.size(0), -1)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder"       : self.encoder.state_dict(),
            "actor"         : self.actor.state_dict(),
            "critic"        : self.critic.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_actor"  : self.target_actor.state_dict(),
            "target_critic" : self.target_critic.state_dict(),
        }, path)
        log.info("DDPGAgent saved -> %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        log.info("DDPGAgent loaded <- %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# PPO Agent
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (clipped objective).

    - On-policy; rollout buffer cleared after each update.
    - Gaussian policy with learned log_std.
    - Encoder shared between actor and critic via ActorCriticNet.
    - Generalized Advantage Estimation (GAE, lambda=0.95).

    Config used:
      training: batch_size, lr_actor (used as unified lr), gamma
      agents.ppo: clip_epsilon, epochs
    """

    GAE_LAMBDA = 0.95

    def __init__(self, cfg: dict, encoder_mode: str = "hybrid"):
        super().__init__(cfg, encoder_mode)

        ppo_cfg   = cfg["agents"]["ppo"]
        train_cfg = cfg["training"]

        self.clip_epsilon = ppo_cfg["clip_epsilon"]
        self.ppo_epochs   = ppo_cfg["epochs"]
        self.gamma        = train_cfg["gamma"]
        self.batch_size   = train_cfg["batch_size"]
        lr                = train_cfg["lr_actor"]

        self.ac_net = ActorCriticNet(self.encoder_out_dim, self.action_dim).to(self.device)

        all_params = (list(self.encoder.parameters())
                      + list(self.ac_net.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer()

        log.info("PPOAgent | clip=%.2f | epochs=%d | gae_lambda=%.2f",
                 self.clip_epsilon, self.ppo_epochs, self.GAE_LAMBDA)

    # ── GAE computation ────────────────────────────────────────────────────────

    def _compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimates and discounted returns.
        rewards, values, dones : lists of scalars / 0-d tensors
        last_value             : bootstrap value for non-terminal last step
        Returns: advantages (T,), returns (T,) as tensors
        """
        T           = len(rewards)
        advantages  = torch.zeros(T, device=self.device)
        gae         = 0.0

        values_t    = torch.stack(values).squeeze(-1).detach()   # (T,)
        rewards_t   = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t     = torch.tensor(dones,   dtype=torch.float32, device=self.device)
        next_values = torch.cat([values_t[1:], last_value.unsqueeze(0)])

        deltas = rewards_t + self.gamma * next_values * (1 - dones_t) - values_t
        for t in reversed(range(T)):
            gae            = deltas[t] + self.gamma * self.GAE_LAMBDA * (1 - dones_t[t]) * gae
            advantages[t]  = gae

        returns = advantages + values_t
        return advantages, returns

    # ── Interface ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, explore: bool = True):
        """
        Returns action (n_stocks+1,) and stores log_prob + value internally.
        explore=False -> use mean (greedy).
        """
        self.encoder.eval()
        self.ac_net.eval()
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            enc                        = self._encode(obs_t)
            action, log_prob, _, value = self.ac_net.get_action_and_value(enc)

        self.encoder.train()
        self.ac_net.train()

        self._last_log_prob = log_prob.squeeze(0)   # stored for buffer.add
        self._last_value    = value.squeeze(0)

        if not explore:
            mean, _, _ = self.ac_net(enc)
            return mean.squeeze(0).cpu().detach().numpy().astype(np.float32)

        return action.squeeze(0).cpu().detach().numpy().astype(np.float32)

    def store(self, obs, action, reward, next_obs, done):
        """Store one transition into the rollout buffer."""
        self.buffer.add(
            obs, action, reward, done,
            self._last_log_prob.detach(),
            self._last_value.detach(),
        )

    def update(self, last_obs: np.ndarray = None, last_done: bool = False):
        """
        PPO update over self.ppo_epochs on the current rollout.
        last_obs  : observation after last step (for bootstrap value).
        last_done : whether last step was terminal.
        Returns dict of mean losses over epochs, or None if buffer empty.
        """
        if len(self.buffer) == 0:
            return None

        # Bootstrap value
        if last_obs is not None and not last_done:
            with torch.no_grad():
                enc_last     = self._encode(self._obs_to_tensor(last_obs))
                _, _, last_v = self.ac_net(enc_last)
                last_value   = last_v.squeeze()
        else:
            last_value = torch.zeros(1, device=self.device).squeeze()

        advantages, returns = self._compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            last_value,
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack buffer tensors
        obs_arr   = np.array(self.buffer.obs)
        act_arr   = np.array(self.buffer.actions)
        old_lps   = torch.stack(self.buffer.log_probs).squeeze(-1).detach()  # (T,)

        T         = len(obs_arr)
        losses    = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.ppo_epochs):
            idx   = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                b     = idx[start : start + self.batch_size]
                if len(b) < 2:
                    continue

                obs_b  = torch.FloatTensor(obs_arr[b.cpu()]).to(self.device)
                act_b  = torch.FloatTensor(act_arr[b.cpu()]).to(self.device)
                adv_b  = advantages[b]
                ret_b  = returns[b]
                olp_b  = old_lps[b]

                enc_b  = self._encode(obs_b)
                _, lp_b, ent_b, val_b = self.ac_net.get_action_and_value(enc_b, act_b)
                lp_b   = lp_b.squeeze(-1)
                ent_b  = ent_b.squeeze(-1)
                val_b  = val_b.squeeze(-1)

                ratio  = (lp_b - olp_b).exp()
                surr1  = ratio * adv_b
                surr2  = ratio.clamp(1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * adv_b

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(val_b, ret_b)
                entropy     = ent_b.mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.ac_net.parameters()), 0.5
                )
                self.optimizer.step()

                losses["policy_loss"].append(policy_loss.item())
                losses["value_loss"].append(value_loss.item())
                losses["entropy"].append(entropy.item())

        self.buffer.clear()

        return {k: float(np.mean(v)) for k, v in losses.items()}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "ac_net" : self.ac_net.state_dict(),
        }, path)
        log.info("PPOAgent saved -> %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.ac_net.load_state_dict(ckpt["ac_net"])
        log.info("PPOAgent loaded <- %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# A2C Agent
# ══════════════════════════════════════════════════════════════════════════════

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (synchronous, on-policy).

    - No clipping; single update per rollout.
    - Shared encoder + ActorCriticNet (same architecture as PPO).
    - Entropy bonus to encourage exploration.
    - n-step returns (no GAE; simpler and faster than PPO).

    Config used:
      training: batch_size, lr_actor (as actor lr), lr_critic, gamma
      agents.a2c: entropy_coef, value_coef
    """

    def __init__(self, cfg: dict, encoder_mode: str = "hybrid"):
        super().__init__(cfg, encoder_mode)

        a2c_cfg   = cfg["agents"]["a2c"]
        train_cfg = cfg["training"]

        self.entropy_coef = a2c_cfg["entropy_coef"]
        self.value_coef   = a2c_cfg["value_coef"]
        self.gamma        = train_cfg["gamma"]
        self.batch_size   = train_cfg["batch_size"]
        lr_actor          = train_cfg["lr_actor"]
        lr_critic         = train_cfg["lr_critic"]

        self.ac_net  = ActorCriticNet(self.encoder_out_dim, self.action_dim).to(self.device)

        # Separate learning rates for actor vs critic parameters
        self.opt_actor  = optim.Adam(
            list(self.encoder.parameters()) + list(self.ac_net.parameters()),
            lr=lr_actor
        )
        self.opt_critic = optim.Adam(
            list(self.ac_net.critic_value.parameters()),
            lr=lr_critic
        )

        self.buffer = RolloutBuffer()

        log.info("A2CAgent | entropy_coef=%.3f | value_coef=%.3f",
                 self.entropy_coef, self.value_coef)

    # ── n-step returns ────────────────────────────────────────────────────────

    def _compute_returns(self, rewards, dones, last_value):
        """
        Simple discounted n-step returns (no GAE).
        Returns: torch.Tensor (T,)
        """
        T       = len(rewards)
        returns = torch.zeros(T, device=self.device)
        R       = last_value

        for t in reversed(range(T)):
            R          = rewards[t] + self.gamma * R * (1.0 - dones[t])
            returns[t] = R
        return returns

    # ── Interface ─────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, explore: bool = True):
        self.encoder.eval()
        self.ac_net.eval()
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            enc                        = self._encode(obs_t)
            action, log_prob, _, value = self.ac_net.get_action_and_value(enc)

        self.encoder.train()
        self.ac_net.train()

        self._last_log_prob = log_prob.squeeze(0)
        self._last_value    = value.squeeze(0)

        if not explore:
            mean, _, _ = self.ac_net(enc)
            return mean.squeeze(0).cpu().detach().numpy().astype(np.float32)

        return action.squeeze(0).cpu().detach().numpy().astype(np.float32)

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(
            obs, action, reward, done,
            self._last_log_prob.detach(),
            self._last_value.detach(),
        )

    def update(self, last_obs: np.ndarray = None, last_done: bool = False):
        """
        Single A2C update on current rollout.
        Returns dict of losses, or None if buffer empty.
        """
        if len(self.buffer) == 0:
            return None

        # Bootstrap
        if last_obs is not None and not last_done:
            with torch.no_grad():
                enc_last     = self._encode(self._obs_to_tensor(last_obs))
                _, _, last_v = self.ac_net(enc_last)
                last_value   = last_v.squeeze().item()
        else:
            last_value = 0.0

        rewards = self.buffer.rewards
        dones   = self.buffer.dones
        returns = self._compute_returns(rewards, dones, last_value)

        obs_arr = np.array(self.buffer.obs)
        act_arr = np.array(self.buffer.actions)
        old_lps = torch.stack(self.buffer.log_probs).squeeze(-1)
        old_vs  = torch.stack(self.buffer.values).squeeze(-1)

        obs_t   = torch.FloatTensor(obs_arr).to(self.device)
        act_t   = torch.FloatTensor(act_arr).to(self.device)

        enc_t   = self._encode(obs_t)
        _, lp_t, ent_t, val_t = self.ac_net.get_action_and_value(enc_t, act_t)
        lp_t    = lp_t.squeeze(-1)
        ent_t   = ent_t.squeeze(-1)
        val_t   = val_t.squeeze(-1)

        advantages   = (returns - old_vs.detach())
        advantages   = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss   = -(lp_t * advantages.detach()).mean()
        value_loss   = F.mse_loss(val_t, returns.detach())
        entropy_loss = -ent_t.mean()

        loss = (actor_loss
                + self.value_coef   * value_loss
                + self.entropy_coef * entropy_loss)

        self.opt_actor.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.ac_net.parameters()), 0.5
        )
        self.opt_actor.step()

        self.buffer.clear()

        return {
            "actor_loss" : actor_loss.item(),
            "value_loss" : value_loss.item(),
            "entropy"    : -entropy_loss.item(),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "ac_net" : self.ac_net.state_dict(),
        }, path)
        log.info("A2CAgent saved -> %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.ac_net.load_state_dict(ckpt["ac_net"])
        log.info("A2CAgent loaded <- %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_agent(agent_type: str, cfg: dict = None,
                encoder_mode: str = "hybrid") -> BaseAgent:
    """
    Convenience factory.

    Parameters
    ----------
    agent_type   : 'ddpg' | 'ppo' | 'a2c'
    cfg          : loaded config dict (loads from config.yaml if None)
    encoder_mode : 'lstm' | 'transformer' | 'hybrid'
    """
    if cfg is None:
        cfg = load_config()

    agent_type = agent_type.lower()
    if agent_type == "ddpg":
        return DDPGAgent(cfg, encoder_mode)
    elif agent_type == "ppo":
        return PPOAgent(cfg, encoder_mode)
    elif agent_type == "a2c":
        return A2CAgent(cfg, encoder_mode)
    else:
        raise ValueError(f"agent_type must be 'ddpg'|'ppo'|'a2c', got {agent_type!r}")


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    cfg       = load_config()
    n_stocks  = cfg["data"]["n_stocks"]
    window    = cfg["encoder"]["window"]
    n_feat    = 21

    dummy_obs  = np.random.randn(n_stocks, window, n_feat).astype(np.float32)
    dummy_next = np.random.randn(n_stocks, window, n_feat).astype(np.float32)

    for atype in ["ddpg", "ppo", "a2c"]:
        print(f"\n{'='*60}")
        print(f"  Agent: {atype.upper()}")
        print('='*60)
        agent  = build_agent(atype, cfg, encoder_mode="hybrid")
        action = agent.select_action(dummy_obs)
        print(f"  select_action -> shape={action.shape}  min={action.min():.3f}  max={action.max():.3f}")

        # Fill buffer and test update
        for _ in range(cfg["training"]["batch_size"] + 5):
            a = agent.select_action(dummy_obs)
            agent.store(dummy_obs, a, 0.01, dummy_next, False)

        if atype == "ddpg":
            losses = agent.update()
        else:
            losses = agent.update(last_obs=dummy_next, last_done=False)

        print(f"  update()  -> {losses}")
        print(f"  ✅ {atype.upper()} OK")
