"""
agents/networks.py
==================
Actor and Critic neural networks for IPPO (Section 3).

Actor
-----
Input  : observation vector, shape (OBS_DIM,) = (162,)
Output : mean and log_std for each action dimension (151,)
         The output is NOT squashed here — PPO uses the Gaussian log-prob
         directly. The environment's _decode_action() applies tanh + rescaling.

Critic
------
Input  : observation vector, shape (OBS_DIM,)
Output : scalar value estimate V(s)

Both networks use the same 2-layer MLP backbone with LayerNorm for stability,
which is important given the wide range of observation magnitudes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

from config import OBS_DIM, ACTION_DIM, HIDDEN_DIM

# Minimum standard deviation to prevent collapse to deterministic policy
LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    Two-hidden-layer MLP with LayerNorm and Tanh activations.
    LayerNorm is preferred over BatchNorm for RL (works with batch size 1).
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class Actor(nn.Module):
    """
    Gaussian policy network.

    Outputs a Normal distribution over the action space.
    The mean is passed through tanh so it stays in (-1, 1),
    matching the environment's expected input range.

    Parameters
    ----------
    obs_dim    : int   Input dimension (default OBS_DIM = 162)
    action_dim : int   Output dimension (default ACTION_DIM = 151)
    hidden_dim : int   Hidden layer width
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.backbone  = _mlp(obs_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd    = nn.Parameter(torch.zeros(action_dim))  # learnable

        # Initialize output layer with small weights → near-uniform initial policy
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> Normal:
        """
        Parameters
        ----------
        obs : Tensor shape (..., OBS_DIM)

        Returns
        -------
        dist : Normal distribution over actions in (-1, 1)
        """
        features = self.backbone(obs)
        mean     = torch.tanh(self.mean_head(features))
        log_std  = self.logstd.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std      = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and return it with its log-probability.

        Parameters
        ----------
        obs          : Tensor shape (OBS_DIM,) or (B, OBS_DIM)
        deterministic: If True, return the mean (no sampling). Used at eval.

        Returns
        -------
        action   : Tensor shape (..., ACTION_DIM) in (-1, 1)
        log_prob : Tensor shape (...,)  — sum of log-probs across action dims
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()    # rsample allows gradients to flow
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate stored actions (used inside the PPO update loop).

        Returns
        -------
        log_prob : Tensor shape (B,)
        entropy  : Tensor shape (B,)   — mean entropy across action dims
        """
        dist     = self.forward(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value network V(s).

    Estimates the expected cumulative discounted reward from state s.
    Used to compute advantages in PPO.

    Parameters
    ----------
    obs_dim    : int
    hidden_dim : int
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, 1)

        # Initialize last layer with small weights for stable initial values
        last_layer = self.net[-1]
        nn.init.orthogonal_(last_layer.weight, gain=1.0)
        nn.init.zeros_(last_layer.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor shape (..., OBS_DIM)

        Returns
        -------
        value : Tensor shape (..., 1)
        """
        return self.net(obs)
