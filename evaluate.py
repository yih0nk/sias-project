"""
evaluate.py
===========
Load trained agents and visualize the decision differences between
CompanyA and CompanyB.

Produces four plots:
  1. Zone prices per company (heatmap over 9 zones × time)
  2. Theta_av over time (routing greediness)
  3. Cumulative reward comparison
  4. Market share over time

Usage:
    python3 evaluate.py                              # run 1 eval episode
    python3 evaluate.py --ckpt-a checkpoints/CompanyA_final.pt \
                        --ckpt-b checkpoints/CompanyB_final.pt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (
    N_ZONES, N_COMPANIES, COMPANY_NAMES, PLANNING_HORIZON,
    SEED, CHECKPOINT_DIR,
)
from data_loader import make_mock_demand
from env.ride_hailing_env import RideHailingEnv
from env.traffic_interface import TrafficInterface
from agents.ppo_agent import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a",    type=str, default=None,
                   help="Checkpoint for CompanyA (optional)")
    p.add_argument("--ckpt-b",    type=str, default=None,
                   help="Checkpoint for CompanyB (optional)")
    p.add_argument("--episodes",  type=int, default=1)
    p.add_argument("--seed",      type=int, default=SEED)
    p.add_argument("--save-fig",  type=str, default="results.png",
                   help="Save figure to this path")
    return p.parse_args()


def run_episode(env, agents, deterministic=True):
    """
    Run one evaluation episode, recording decisions each epoch.

    Returns
    -------
    history : dict with keys
        'prices'   : list (n_epochs) of [prices_A, prices_B] (each N_ZONES)
        'thetas'   : list (n_epochs) of [theta_A, theta_B]
        'rewards'  : list (n_epochs) of [r_A, r_B]
        'completed': list (n_epochs) of [n_A, n_B]
        'dropped'  : list (n_epochs) of [n_A, n_B]
    """
    obs_list = env.reset()
    history  = {k: [] for k in
                ["prices", "thetas", "rewards", "completed", "dropped"]}

    for _ in range(PLANNING_HORIZON):
        actions = [agents[c].act(obs_list[c], deterministic=deterministic)[0]
                   for c in range(N_COMPANIES)]

        obs_list, rewards, done, info = env.step(actions)

        history["prices"].append(info["prices"])
        history["thetas"].append(info["thetas"])
        history["rewards"].append(rewards)
        history["completed"].append(info["completed"])
        history["dropped"].append(info["dropped"])

        if done:
            break

    return history


def plot_decisions(history, save_path="results.png"):
    """
    Four-panel figure showing decision differences between the two companies.
    """
    n_epochs = len(history["rewards"])
    epochs   = np.arange(n_epochs)

    prices_A = np.array([h[0] for h in history["prices"]])   # (T, N_ZONES)
    prices_B = np.array([h[1] for h in history["prices"]])
    thetas   = np.array(history["thetas"])                   # (T, 2)
    rewards  = np.array(history["rewards"])                  # (T, 2)
    completed= np.array(history["completed"])                # (T, 2)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CompanyA vs CompanyB — Decision Differences", fontsize=14,
                 fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    colors = ["#2196F3", "#F44336"]   # blue for A, red for B

    # ── Panel 1: Zone prices over time ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
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

    # ── Panel 2: Theta_av (routing greediness) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, thetas[:, 0], color=colors[0],
             label="CompanyA θ_av", linewidth=2)
    ax2.plot(epochs, thetas[:, 1], color=colors[1],
             label="CompanyB θ_av", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("θ_av (higher = more greedy)")
    ax2.set_title("AV Routing Greediness (θ_av)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative reward ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cum_A = np.cumsum(rewards[:, 0])
    cum_B = np.cumsum(rewards[:, 1])
    ax3.plot(epochs, cum_A, color=colors[0], label="CompanyA", linewidth=2)
    ax3.plot(epochs, cum_B, color=colors[1], label="CompanyB", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Cumulative Reward")
    ax3.set_title("Cumulative Reward Comparison")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Market share (rolling window) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    total    = completed.sum(axis=1).clip(min=1)
    share_A  = completed[:, 0] / total
    share_B  = completed[:, 1] / total
    window   = max(1, n_epochs // 20)   # ~5% of horizon rolling window
    share_A_smooth = np.convolve(share_A, np.ones(window)/window, "same")
    share_B_smooth = np.convolve(share_B, np.ones(window)/window, "same")
    ax4.plot(epochs, share_A_smooth, color=colors[0],
             label="CompanyA market share", linewidth=2)
    ax4.plot(epochs, share_B_smooth, color=colors[1],
             label="CompanyB market share", linewidth=2)
    ax4.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Market Share")
    ax4.set_title("Market Share Over Time")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] Figure saved → {save_path}")
    plt.show()


def main():
    args = parse_args()

    demand_data = make_mock_demand(
        n_epochs=PLANNING_HORIZON,
        trips_per_epoch=20.0,
        rng=np.random.default_rng(args.seed),
    )
    env = RideHailingEnv(
        demand_data=demand_data,
        traffic=TrafficInterface(mock=True, seed=args.seed),
        reward_mode="revenue",
        seed=args.seed,
    )

    agents = [PPOAgent(company_id=c) for c in range(N_COMPANIES)]

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
        "prices":    avg_field("prices"),
        "thetas":    avg_field("thetas"),
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

    mean_prices = [np.mean([h[c] for h in avg_history["prices"]]) for c in range(N_COMPANIES)]
    mean_thetas = [np.mean([h[c] for h in avg_history["thetas"]]) for c in range(N_COMPANIES)]
    print(f"\n  Mean price multiplier:")
    for c in range(N_COMPANIES):
        print(f"    {COMPANY_NAMES[c]}: {mean_prices[c]:.3f}")
    print(f"\n  Mean theta_av:")
    for c in range(N_COMPANIES):
        print(f"    {COMPANY_NAMES[c]}: {mean_thetas[c]:.2f}")

    plot_decisions(avg_history, save_path=args.save_fig)


if __name__ == "__main__":
    main()
