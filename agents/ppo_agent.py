"""
agents/ppo_agent.py
===================
Independent PPO (IPPO) agent for one company (Section 3).

Each company has its own PPOAgent instance. They share no parameters
and do not communicate (independent learning).

Training loop:
    1. Collect ROLLOUT_LEN transitions in a RolloutBuffer.
    2. Compute returns and GAE advantages.
    3. Run PPO_EPOCHS gradient steps over mini-batches.
    4. Clear the buffer and repeat.

Reference: Schulman et al. 2017 — "Proximal Policy Optimization Algorithms"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Optional, Tuple

from config import (
    OBS_DIM, ACTION_DIM,
    LR_ACTOR, LR_CRITIC,
    GAMMA, LAMBDA_GAE,
    CLIP_EPS, ENTROPY_COEF, VF_COEF,
    MAX_GRAD_NORM,
    PPO_EPOCHS, MINI_BATCH_SIZE, ROLLOUT_LEN,
)
from agents.networks import Actor, Critic


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores transitions collected during one rollout.

    Holds ROLLOUT_LEN steps of (obs, action, log_prob, reward, value, done).
    After collection, computes returns and advantages in-place.
    """

    def __init__(self, rollout_len: int = ROLLOUT_LEN):
        self.rollout_len = rollout_len
        self._ptr        = 0
        self._full       = False

        self.obs      = np.zeros((rollout_len, OBS_DIM),    dtype=np.float32)
        self.actions  = np.zeros((rollout_len, ACTION_DIM), dtype=np.float32)
        self.log_probs= np.zeros(rollout_len,               dtype=np.float32)
        self.rewards  = np.zeros(rollout_len,               dtype=np.float32)
        self.values   = np.zeros(rollout_len,               dtype=np.float32)
        self.dones    = np.zeros(rollout_len,               dtype=np.float32)

        # Filled in by compute_returns_and_advantages()
        self.returns    = np.zeros(rollout_len, dtype=np.float32)
        self.advantages = np.zeros(rollout_len, dtype=np.float32)

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        """Store one transition."""
        i = self._ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = float(done)

        self._ptr  = (self._ptr + 1) % self.rollout_len
        self._full = self._full or (self._ptr == 0)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = GAMMA,
        lam: float = LAMBDA_GAE
    ) -> None:
        """
        GAE (Generalized Advantage Estimation).

        Advantage at step t:
            δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)·A_{t+1}

        Return = advantage + value (used as critic target).
        """
        last_adv = 0.0
        n        = self.rollout_len

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value        = last_value if t == n - 1 else self.values[t + 1]

            delta   = (self.rewards[t]
                       + gamma * next_value * next_non_terminal
                       - self.values[t])
            last_adv = delta + gamma * lam * next_non_terminal * last_adv

            self.advantages[t] = last_adv

        self.returns = self.advantages + self.values

        # Normalize advantages for training stability
        adv_mean = self.advantages.mean()
        adv_std  = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_batches(self, batch_size: int = MINI_BATCH_SIZE):
        """
        Yield random mini-batches as torch tensors.
        Used in the PPO update loop.
        """
        n       = self.rollout_len
        indices = np.random.permutation(n)

        obs      = torch.FloatTensor(self.obs)
        actions  = torch.FloatTensor(self.actions)
        log_probs= torch.FloatTensor(self.log_probs)
        returns  = torch.FloatTensor(self.returns)
        advantages = torch.FloatTensor(self.advantages)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (obs[idx], actions[idx], log_probs[idx],
                   returns[idx], advantages[idx])

    def is_ready(self) -> bool:
        """True when the buffer has been filled at least once."""
        return self._full or (self._ptr == self.rollout_len)

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False


# ── PPO Agent ──────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Independent PPO agent for one company.

    Parameters
    ----------
    company_id : int    0 or 1, used only for logging
    device     : str    "cpu" or "cuda"
    """

    def __init__(self, company_id: int, device: str = "cpu"):
        self.company_id = company_id
        self.device     = torch.device(device)

        self.actor  = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor_opt  = Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_opt = Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer     = RolloutBuffer()
        self._step      = 0        # total environment steps seen

    # ── Interaction ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray,
            deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Choose an action given an observation.

        Parameters
        ----------
        obs          : shape (OBS_DIM,)
        deterministic: Use mean action (no exploration). Used for evaluation.

        Returns
        -------
        action   : np.ndarray shape (ACTION_DIM,) in (-1, 1)
        log_prob : float
        value    : float   — critic's V(s) estimate
        """
        obs_t    = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action_t, log_prob_t = self.actor.act(obs_t, deterministic)
        value_t  = self.critic(obs_t)

        action   = action_t.squeeze(0).cpu().numpy()
        log_prob = log_prob_t.item()
        value    = value_t.squeeze().item()
        return action, log_prob, value

    def store(self, obs, action, log_prob, reward, value, done) -> None:
        """Store one transition in the rollout buffer."""
        self.buffer.add(obs, action, log_prob, reward, value, done)
        self._step += 1

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self, last_obs: np.ndarray) -> dict:
        """
        Run the PPO update using the current rollout buffer.

        Called when buffer is full (every ROLLOUT_LEN steps).

        Parameters
        ----------
        last_obs : observation AFTER the last stored step, used to bootstrap
                   the value for GAE computation.

        Returns
        -------
        dict with training losses (for logging).
        """
        # Bootstrap value for the last state
        with torch.no_grad():
            obs_t      = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_value = self.critic(obs_t).item()

        self.buffer.compute_returns_and_advantages(last_value)

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(PPO_EPOCHS):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in \
                    self.buffer.get_batches():

                obs_b   = obs_b.to(self.device)
                act_b   = act_b.to(self.device)
                old_lp_b= old_lp_b.to(self.device)
                ret_b   = ret_b.to(self.device)
                adv_b   = adv_b.to(self.device)

                # ── Actor loss (clipped surrogate) ────────────────────────────
                new_lp, entropy = self.actor.evaluate(obs_b, act_b)
                ratio            = torch.exp(new_lp - old_lp_b)

                # PPO clipping: prevent too-large policy updates
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= ENTROPY_COEF * entropy.mean()   # entropy bonus

                # ── Critic loss (MSE on value predictions) ────────────────────
                value_pred   = self.critic(obs_b).squeeze(-1)
                critic_loss  = VF_COEF * nn.functional.mse_loss(value_pred, ret_b)

                # ── Gradient step ─────────────────────────────────────────────
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_opt.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        self.buffer.reset()

        return {
            "actor_loss":  total_actor_loss  / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy":     total_entropy     / max(n_updates, 1),
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "actor_opt":   self.actor_opt.state_dict(),
            "critic_opt":  self.critic_opt.state_dict(),
            "step":        self._step,
        }, path)
        print(f"[PPOAgent {self.company_id}] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self._step = ckpt.get("step", 0)
        print(f"[PPOAgent {self.company_id}] Loaded checkpoint ← {path}")
