"""
evaluate.py
===========
Load trained agents and visualize the decision differences between
CompanyA and CompanyB.

Produces three plots:
  1. Zone prices per company (mean HV price multiplier over time)
  2. Cumulative reward comparison
  3. Market share over time

Usage:
    python3 evaluate.py                              # run 1 eval episode
    python3 evaluate.py --ckpt-a checkpoints/CompanyA_final.pt \
                        --ckpt-b checkpoints/CompanyB_final.pt
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    N_ZONES, N_COMPANIES, COMPANY_NAMES, PLANNING_HORIZON,
    SEED, CHECKPOINT_DIR,
)
from data_loader import make_mock_demand, load_demand
from env.ride_hailing_env import RideHailingEnv
from env.traffic_interface import TrafficInterface
from agents.ppo_agent import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a",       type=str, default=None,
                   help="Checkpoint for CompanyA (optional)")
    p.add_argument("--ckpt-b",       type=str, default=None,
                   help="Checkpoint for CompanyB (optional)")
    p.add_argument("--episodes",     type=int, default=1)
    p.add_argument("--seed",         type=int, default=None,
                   help="Random seed (defaults to training seed if available)")
    p.add_argument("--save-fig",     type=str, default="results.png",
                   help="Save figure to this path")
    # Env settings — auto-loaded from training_config.json when available;
    # explicit flags here always take precedence.
    p.add_argument("--reward-mode",  type=str, default=None,
                   choices=["revenue", "decomposed"],
                   help="Override reward mode (default: match training config)")
    p.add_argument("--use-sumo",     action="store_true", default=None,
                   help="Force real SUMO (default: match training config)")
    p.add_argument("--trips",        type=str, default=None,
                   help="Path to demand Parquet (default: match training config)")
    p.add_argument("--zone-edges",   type=str, default=None)
    return p.parse_args()


def _load_training_config(ckpt_a, ckpt_b):
    """
    Look for training_config.json in the same directory as either checkpoint.
    Returns a dict of training settings, or {} if not found.
    """
    for ckpt in (ckpt_a, ckpt_b):
        if ckpt:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                                    "training_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                print(f"[evaluate] Loaded training config from {cfg_path}")
                return cfg
    return {}


def run_episode(env, agents, deterministic=True):
    """
    Run one evaluation episode, recording decisions each epoch.

    Returns
    -------
    history : dict with keys
        'prices_hv': list (n_epochs) of [prices_A, prices_B] (each N_ZONES)
        'rewards'  : list (n_epochs) of [r_A, r_B]
        'completed': list (n_epochs) of [n_A, n_B]
        'dropped'  : list (n_epochs) of [n_A, n_B]
    """
    obs_list = env.reset()
    history  = {k: [] for k in
                ["prices_hv", "rewards", "completed", "dropped"]}

    for _ in range(PLANNING_HORIZON):
        actions = [agents[c].act(obs_list[c], deterministic=deterministic)[0]
                   for c in range(N_COMPANIES)]

        obs_list, rewards, done, info = env.step(actions)

        history["prices_hv"].append(info["prices_hv"])
        history["rewards"].append(rewards)
        history["completed"].append(info["completed"])
        history["dropped"].append(info["dropped"])

        if done:
            break

    return history


def plot_decisions(history, save_path="results.png"):
    """
    Three-panel figure showing decision differences between the two companies.
    """
    n_epochs = len(history["rewards"])
    epochs   = np.arange(n_epochs)

    # Use HV prices for the pricing panel (representative)
    prices_A = np.array([h[0] for h in history["prices_hv"]])   # (T, N_ZONES)
    prices_B = np.array([h[1] for h in history["prices_hv"]])
    rewards  = np.array(history["rewards"])                  # (T, 2)
    completed= np.array(history["completed"])                # (T, 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CompanyA vs CompanyB — Decision Differences", fontsize=14,
                 fontweight="bold")

    colors = ["#2196F3", "#F44336"]   # blue for A, red for B

    # ── Panel 1: Zone prices over time ────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, prices_A.mean(axis=1), color=colors[0],
             label="CompanyA mean price", linewidth=2)
    ax1.plot(epochs, prices_B.mean(axis=1), color=colors[1],
             label="CompanyB mean price", linewidth=2)
    ax1.fill_between(epochs,
                     prices_A.min(axis=1), prices_A.max(axis=1),
                     alpha=0.15, color=colors[0])
    ax1.fill_between(epochs,
                     prices_B.min(axis=1), prices_B.max(axis=1),
                     alpha=0.15, color=colors[1])
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8,
                label="neutral price (1.0)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Price Multiplier")
    ax1.set_title("Zone Pricing Strategy")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Cumulative reward ────────────────────────────────────────────
    ax2 = axes[1]
    cum_A = np.cumsum(rewards[:, 0])
    cum_B = np.cumsum(rewards[:, 1])
    ax2.plot(epochs, cum_A, color=colors[0], label="CompanyA", linewidth=2)
    ax2.plot(epochs, cum_B, color=colors[1], label="CompanyB", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Reward Comparison")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Market share (rolling window) ────────────────────────────────
    ax3 = axes[2]
    total    = completed.sum(axis=1).clip(min=1)
    share_A  = completed[:, 0] / total
    share_B  = completed[:, 1] / total
    window   = max(1, n_epochs // 20)
    share_A_smooth = np.convolve(share_A, np.ones(window)/window, "same")
    share_B_smooth = np.convolve(share_B, np.ones(window)/window, "same")
    ax3.plot(epochs, share_A_smooth, color=colors[0],
             label="CompanyA market share", linewidth=2)
    ax3.plot(epochs, share_B_smooth, color=colors[1],
             label="CompanyB market share", linewidth=2)
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Market Share")
    ax3.set_title("Market Share Over Time")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] Figure saved → {save_path}")
    plt.show()


def main():
    args = parse_args()

    # ── Resolve env settings: training config < CLI overrides ─────────────────
    train_cfg = _load_training_config(args.ckpt_a, args.ckpt_b)

    reward_mode = args.reward_mode or train_cfg.get("reward_mode", "revenue")
    use_sumo    = args.use_sumo    or train_cfg.get("use_sumo",    False)
    trips       = args.trips       or train_cfg.get("trips",       None)
    zone_edges  = args.zone_edges  or train_cfg.get("zone_edges",  None)
    seed        = args.seed        if args.seed is not None \
                                   else train_cfg.get("seed", SEED)

    print(f"[evaluate] reward_mode={reward_mode}  use_sumo={use_sumo}  "
          f"trips={'<mock>' if trips is None else trips}")

    # ── Demand data ───────────────────────────────────────────────────────────
    if trips:
        demand_data = load_demand(trips, zone_edges)
    else:
        demand_data = make_mock_demand(
            n_epochs=PLANNING_HORIZON,
            trips_per_epoch=20.0,
            rng=np.random.default_rng(seed),
        )

    # ── Environment ───────────────────────────────────────────────────────────
    env = RideHailingEnv(
        demand_data=demand_data,
        traffic=TrafficInterface(mock=not use_sumo, seed=seed),
        reward_mode=reward_mode,
        seed=seed,
    )

    agents = [PPOAgent(company_id=c) for c in range(N_COMPANIES)]
    np.random.seed(seed)

    # Load checkpoints if provided
    ckpts = [args.ckpt_a, args.ckpt_b]
    for c, ckpt in enumerate(ckpts):
        if ckpt and os.path.exists(ckpt):
            agents[c].load(ckpt)
        else:
            print(f"[evaluate] {COMPANY_NAMES[c]}: no checkpoint, "
                  f"using random policy")

    # Run evaluation episode(s) and average
    all_histories = [run_episode(env, agents) for _ in range(args.episodes)]

    # Average across episodes
    def avg_field(key):
        return [
            [np.mean([ep[key][t][c] for ep in all_histories], axis=0)
             for c in range(N_COMPANIES)]
            for t in range(PLANNING_HORIZON)
        ]

    avg_history = {
        "prices_hv": avg_field("prices_hv"),
        "rewards":   avg_field("rewards"),
        "completed": avg_field("completed"),
        "dropped":   avg_field("dropped"),
    }

    # Print summary
    total_rewards = np.array(avg_history["rewards"]).sum(axis=0)
    print(f"\n{'='*50}")
    print(f"Evaluation summary ({args.episodes} episode(s)):")
    for c in range(N_COMPANIES):
        print(f"  {COMPANY_NAMES[c]}: total reward = {total_rewards[c]:.3f}")

    mean_hv = [np.mean([h[c] for h in avg_history["prices_hv"]]) for c in range(N_COMPANIES)]
    print(f"\n  Mean HV price multiplier:")
    for c in range(N_COMPANIES):
        print(f"    {COMPANY_NAMES[c]}: {mean_hv[c]:.3f}")

    plot_decisions(avg_history, save_path=args.save_fig)


if __name__ == "__main__":
    main()
