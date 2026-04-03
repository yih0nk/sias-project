"""
benchmark.py
============
Train IPPO agents for 200 episodes, then compare them against a
fixed-price-multiplier=1.0 baseline over the same evaluation episodes.

Usage:
    python3 benchmark.py                     # train + eval (fresh start)
    python3 benchmark.py --skip-train        # load existing checkpoints, only eval
    python3 benchmark.py --eval-episodes 5   # more eval episodes for stability

Output: printed comparison table + benchmark_results.png
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    N_ZONES, N_COMPANIES, COMPANY_NAMES, PLANNING_HORIZON,
    ROLLOUT_LEN, CHECKPOINT_DIR, LOG_DIR, SEED,
    M_MIN, M_MAX, ACTION_DIM,
)
from data_loader import make_mock_demand
from env.ride_hailing_env import RideHailingEnv
from env.traffic_interface import TrafficInterface
from agents.ppo_agent import PPOAgent


# ── Fixed-price baseline agent ────────────────────────────────────────────────

class FixedPriceAgent:
    """
    Dummy agent that always outputs actions mapping exactly to
    zone price multiplier = fixed_price for all zones.

    The action is in [-1, 1] (tanh space), back-calculated from the
    _decode_action formula used in RideHailingEnv.
    """

    def __init__(self, fixed_price: float = 1.0):
        # Invert _decode_action:  raw = 2*t - 1,  t = (val - min) / (max - min)
        t_price   = (fixed_price - M_MIN) / (M_MAX - M_MIN)
        raw_price = float(np.clip(2.0 * t_price - 1.0, -1.0, 1.0))
        self._action = np.array([raw_price] * N_ZONES, dtype=np.float32)

    def act(self, obs, deterministic=True):
        return self._action, 0.0, 0.0


# ── Training (mirrors train.py logic) ─────────────────────────────────────────

def run_training(episodes: int, seed: int, checkpoint_dir: str,
                 save_every: int = 50) -> list:
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    demand_data = make_mock_demand(
        n_epochs=PLANNING_HORIZON,
        trips_per_epoch=20.0,
        rng=np.random.default_rng(seed),
    )
    traffic = TrafficInterface(mock=True, seed=seed)
    env = RideHailingEnv(demand_data=demand_data, traffic=traffic,
                         reward_mode="revenue", seed=seed)
    agents = [PPOAgent(company_id=c, device="cpu") for c in range(N_COMPANIES)]

    for episode in range(1, episodes + 1):
        obs_list  = env.reset()
        ep_rewards = [0.0] * N_COMPANIES

        for epoch in range(PLANNING_HORIZON):
            actions, log_probs, values = [], [], []
            for c in range(N_COMPANIES):
                a, lp, v = agents[c].act(obs_list[c])
                actions.append(a); log_probs.append(lp); values.append(v)

            next_obs, rewards, done, info = env.step(actions)

            for c in range(N_COMPANIES):
                agents[c].store(obs_list[c], actions[c], log_probs[c],
                                rewards[c], values[c], float(done))
                ep_rewards[c] += rewards[c]

            obs_list = next_obs

            if (epoch + 1) % ROLLOUT_LEN == 0:
                for c in range(N_COMPANIES):
                    agents[c].update(last_obs=obs_list[c])

            if done:
                break

        print(f"[train] ep {episode:3d}/{episodes}  "
              f"A={ep_rewards[0]:.3f}  B={ep_rewards[1]:.3f}  "
              f"completed={info['completed']}  dropped={info['dropped']}")

        if episode % save_every == 0:
            for c, agent in enumerate(agents):
                agent.save(os.path.join(checkpoint_dir,
                                        f"{COMPANY_NAMES[c]}_ep{episode}.pt"))

    for c, agent in enumerate(agents):
        agent.save(os.path.join(checkpoint_dir, f"{COMPANY_NAMES[c]}_final.pt"))

    with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
        json.dump({"reward_mode": "revenue", "use_sumo": False,
                   "trips": None, "zone_edges": None, "seed": seed}, f, indent=2)

    env.traffic.close()
    return agents


# ── Single evaluation episode ─────────────────────────────────────────────────

def run_eval_episode(env, agents):
    """Returns aggregated stats for one episode."""
    obs_list = env.reset()
    totals = {
        "revenue": [0.0] * N_COMPANIES,
        "completed": [0] * N_COMPANIES,
        "dropped": [0] * N_COMPANIES,
        "requests": [0] * N_COMPANIES,
        "price_history": [[] for _ in range(N_COMPANIES)],
    }

    for _ in range(PLANNING_HORIZON):
        actions = [agents[c].act(obs_list[c], deterministic=True)[0]
                   for c in range(N_COMPANIES)]
        obs_list, rewards, done, info = env.step(actions)

        for c in range(N_COMPANIES):
            totals["revenue"][c]   += info["revenue"][c]
            totals["completed"][c] += info["completed"][c]
            totals["dropped"][c]   += info["dropped"][c]
            totals["requests"][c]  += (info["completed"][c] + info["dropped"][c])
            totals["price_history"][c].append(np.mean(info["prices"][c]))

        if done:
            break

    return totals


def avg_episodes(all_eps, key, c):
    return np.mean([ep[key][c] for ep in all_eps])


# ── Comparison report ─────────────────────────────────────────────────────────

def print_report(ppo_eps, base_eps):
    print("\n" + "=" * 65)
    print(f"{'BENCHMARK RESULTS':^65}")
    print("=" * 65)

    metrics = ["revenue", "completed", "dropped"]
    labels  = ["Total Revenue ($)", "Rides Completed", "Requests Dropped"]

    for label, key in zip(labels, metrics):
        print(f"\n  {label}:")
        print(f"    {'Company':<12} {'PPO Agent':>12} {'Fixed 1.0':>12} {'Delta':>10}")
        print(f"    {'-'*48}")
        for c in range(N_COMPANIES):
            ppo_val  = avg_episodes(ppo_eps, key, c)
            base_val = avg_episodes(base_eps, key, c)
            delta    = ppo_val - base_val
            pct      = (delta / max(base_val, 1e-9)) * 100
            sign     = "+" if delta >= 0 else ""
            print(f"    {COMPANY_NAMES[c]:<12} {ppo_val:>12.2f} "
                  f"{base_val:>12.2f} {sign}{pct:>8.1f}%")

    # Drop rate
    print(f"\n  Drop Rate (dropped / total requests):")
    print(f"    {'Company':<12} {'PPO Agent':>12} {'Fixed 1.0':>12} {'Delta':>10}")
    print(f"    {'-'*48}")
    for c in range(N_COMPANIES):
        ppo_comp  = avg_episodes(ppo_eps, "completed", c)
        ppo_drop  = avg_episodes(ppo_eps, "dropped",   c)
        base_comp = avg_episodes(base_eps, "completed", c)
        base_drop = avg_episodes(base_eps, "dropped",   c)
        ppo_rate  = ppo_drop  / max(ppo_comp  + ppo_drop,  1)
        base_rate = base_drop / max(base_comp + base_drop, 1)
        delta     = ppo_rate - base_rate
        sign      = "+" if delta >= 0 else ""
        print(f"    {COMPANY_NAMES[c]:<12} {ppo_rate:>11.1%} "
              f"{base_rate:>11.1%} {sign}{delta*100:>7.1f}pp")

    print("\n" + "=" * 65)

    # One-liner summary (for bullet-point use)
    combined_ppo_rev  = sum(avg_episodes(ppo_eps,  "revenue", c)
                            for c in range(N_COMPANIES))
    combined_base_rev = sum(avg_episodes(base_eps, "revenue", c)
                            for c in range(N_COMPANIES))
    rev_uplift = (combined_ppo_rev - combined_base_rev) / max(combined_base_rev, 1e-9) * 100

    combined_ppo_drop  = sum(avg_episodes(ppo_eps,  "dropped", c)
                             for c in range(N_COMPANIES))
    combined_base_drop = sum(avg_episodes(base_eps, "dropped", c)
                             for c in range(N_COMPANIES))
    drop_reduction = (combined_base_drop - combined_ppo_drop) / max(combined_base_drop, 1e-9) * 100

    print(f"\n  SUMMARY (both companies combined):")
    print(f"    Revenue uplift:      {rev_uplift:+.1f}%  "
          f"(PPO ${combined_ppo_rev:.0f} vs baseline ${combined_base_rev:.0f})")
    print(f"    Drop-rate reduction: {drop_reduction:+.1f}%  "
          f"(PPO {combined_ppo_drop:.0f} vs baseline {combined_base_drop:.0f} drops)")
    print("=" * 65 + "\n")

    return combined_ppo_rev, combined_base_rev, combined_ppo_drop, combined_base_drop


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(ppo_eps, base_eps, save_path="benchmark_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PPO Agents vs Fixed-Price Baseline (price=1.0)",
                 fontsize=13, fontweight="bold")

    colors_ppo  = ["#2196F3", "#1565C0"]
    colors_base = ["#EF9A9A", "#E53935"]

    # ── 1. Revenue per company ────────────────────────────────────────────────
    ax = axes[0]
    x  = np.arange(N_COMPANIES)
    w  = 0.35
    ppo_rev  = [avg_episodes(ppo_eps,  "revenue", c) for c in range(N_COMPANIES)]
    base_rev = [avg_episodes(base_eps, "revenue", c) for c in range(N_COMPANIES)]
    ax.bar(x - w/2, ppo_rev,  w, label="PPO agents",    color=colors_ppo)
    ax.bar(x + w/2, base_rev, w, label="Fixed price=1.0", color=colors_base)
    ax.set_xticks(x); ax.set_xticklabels(COMPANY_NAMES)
    ax.set_ylabel("Total Revenue ($)")
    ax.set_title("Revenue per Company")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── 2. Drop counts per company ────────────────────────────────────────────
    ax = axes[1]
    ppo_drop  = [avg_episodes(ppo_eps,  "dropped", c) for c in range(N_COMPANIES)]
    base_drop = [avg_episodes(base_eps, "dropped", c) for c in range(N_COMPANIES)]
    ax.bar(x - w/2, ppo_drop,  w, label="PPO agents",    color=colors_ppo)
    ax.bar(x + w/2, base_drop, w, label="Fixed price=1.0", color=colors_base)
    ax.set_xticks(x); ax.set_xticklabels(COMPANY_NAMES)
    ax.set_ylabel("Requests Dropped")
    ax.set_title("Dropped Requests per Company")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # ── 3. Mean price trajectory (company A only, first episode) ─────────────
    ax = axes[2]
    ep = ppo_eps[0]   # first eval episode
    t  = np.arange(len(ep["price_history"][0]))
    ax.plot(t, ep["price_history"][0], color=colors_ppo[0],
            label=f"PPO {COMPANY_NAMES[0]}", linewidth=1.5)
    ax.plot(t, ep["price_history"][1], color=colors_ppo[1],
            label=f"PPO {COMPANY_NAMES[1]}", linewidth=1.5, linestyle="--")
    ax.axhline(1.0, color="#E53935", linestyle=":", linewidth=1.5,
               label="Baseline price=1.0")
    ax.set_xlabel("Epoch (15-min interval)")
    ax.set_ylabel("Mean Price Multiplier")
    ax.set_title("PPO Pricing Strategy vs Baseline")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[benchmark] Figure saved → {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",      type=int, default=200,
                   help="Training episodes (default 200)")
    p.add_argument("--eval-episodes", type=int, default=3,
                   help="Evaluation episodes to average over (default 3)")
    p.add_argument("--skip-train",    action="store_true",
                   help="Skip training; load existing checkpoints")
    p.add_argument("--checkpoint",    type=str, default=CHECKPOINT_DIR)
    p.add_argument("--seed",          type=int, default=SEED)
    p.add_argument("--save-fig",      type=str, default="benchmark_results.png")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Train or load ──────────────────────────────────────────────────────
    if args.skip_train:
        print("[benchmark] Loading existing checkpoints …")
        ppo_agents = [PPOAgent(company_id=c) for c in range(N_COMPANIES)]
        for c in range(N_COMPANIES):
            ckpt = os.path.join(args.checkpoint, f"{COMPANY_NAMES[c]}_final.pt")
            if os.path.exists(ckpt):
                ppo_agents[c].load(ckpt)
            else:
                print(f"  WARNING: {ckpt} not found — using random policy for {COMPANY_NAMES[c]}")
    else:
        print(f"[benchmark] Training for {args.episodes} episodes …")
        ppo_agents = run_training(
            episodes=args.episodes, seed=args.seed,
            checkpoint_dir=args.checkpoint,
        )

    # ── 2. Build shared eval environment ─────────────────────────────────────
    eval_seed = args.seed + 1000   # different seed from training
    demand_data = make_mock_demand(
        n_epochs=PLANNING_HORIZON,
        trips_per_epoch=20.0,
        rng=np.random.default_rng(eval_seed),
    )

    def make_env():
        return RideHailingEnv(
            demand_data=demand_data,
            traffic=TrafficInterface(mock=True, seed=eval_seed),
            reward_mode="revenue",
            seed=eval_seed,
        )

    # ── 3. Evaluate PPO agents ────────────────────────────────────────────────
    print(f"\n[benchmark] Evaluating PPO agents ({args.eval_episodes} episodes) …")
    ppo_env = make_env()
    ppo_episodes = [run_eval_episode(ppo_env, ppo_agents)
                    for _ in range(args.eval_episodes)]
    ppo_env.traffic.close()

    # ── 4. Evaluate fixed-price baseline ─────────────────────────────────────
    print(f"[benchmark] Evaluating fixed-price baseline ({args.eval_episodes} episodes) …")
    base_agents = [FixedPriceAgent(fixed_price=1.0) for _ in range(N_COMPANIES)]
    base_env = make_env()
    base_episodes = [run_eval_episode(base_env, base_agents)
                     for _ in range(args.eval_episodes)]
    base_env.traffic.close()

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print_report(ppo_episodes, base_episodes)
    plot_comparison(ppo_episodes, base_episodes, save_path=args.save_fig)


if __name__ == "__main__":
    main()
