"""
train.py
========
Main training script. Run this to start the IPPO training loop.

Quick start (no data files needed):
    python3 train.py

Full options:
    python3 train.py --episodes 200 --reward-mode decomposed --device cpu
    python3 train.py --trips data/fhv.parquet --zone-edges data/zones.csv  # real data
"""

import os
import json
import argparse
import numpy as np
import torch

from config import (
    N_COMPANIES, COMPANY_NAMES, PLANNING_HORIZON,
    ROLLOUT_LEN, CHECKPOINT_DIR, LOG_DIR, SEED,
)
from data_loader import make_mock_demand, load_demand
from env.ride_hailing_env import RideHailingEnv
from env.traffic_interface import TrafficInterface
from agents.ppo_agent import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trips",        type=str,   default=None,
                   help="Path to TLC FHV Parquet (real data mode)")
    p.add_argument("--zone-edges",   type=str,   default=None)
    p.add_argument("--episodes",     type=int,   default=200)
    p.add_argument("--reward-mode",  type=str,   default="revenue",
                   choices=["revenue", "decomposed"])
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--gui",          action="store_true",
                   help="Launch SUMO with visual GUI (slow)")
    p.add_argument("--use-sumo",     action="store_true",
                   help="Connect to real SUMO (default: mock mode)")
    p.add_argument("--seed",         type=int,   default=SEED)
    p.add_argument("--checkpoint",   type=str,   default=CHECKPOINT_DIR)
    p.add_argument("--save-every",   type=int,   default=50,
                   help="Save checkpoints every N episodes")
    return p.parse_args()


def train(args):
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. Demand data ────────────────────────────────────────────────────────
    if args.trips:
        demand_data = load_demand(args.trips, args.zone_edges)
    else:
        print("[train] Using mock demand (9-zone grid, random trips)")
        demand_data = make_mock_demand(
            n_epochs=PLANNING_HORIZON,
            trips_per_epoch=20.0,
            rng=np.random.default_rng(args.seed),
        )

    # ── 2. Traffic interface ──────────────────────────────────────────────────
    use_mock = not args.use_sumo
    traffic  = TrafficInterface(mock=use_mock, use_gui=args.gui, seed=args.seed)

    # ── 3. Environment ────────────────────────────────────────────────────────
    env = RideHailingEnv(
        demand_data=demand_data,
        traffic=traffic,
        reward_mode=args.reward_mode,
        seed=args.seed,
    )

    # ── 4. Agents ─────────────────────────────────────────────────────────────
    agents = [PPOAgent(company_id=c, device=args.device)
              for c in range(N_COMPANIES)]

    # ── 5. Training loop ──────────────────────────────────────────────────────
    episode_reward_history = [[] for _ in range(N_COMPANIES)]

    for episode in range(1, args.episodes + 1):
        obs_list  = env.reset()
        ep_rewards = [0.0] * N_COMPANIES

        print(f"\nEpisode {episode}/{args.episodes}")

        for epoch in range(PLANNING_HORIZON):
            actions, log_probs, values = [], [], []
            for c in range(N_COMPANIES):
                a, lp, v = agents[c].act(obs_list[c])
                actions.append(a)
                log_probs.append(lp)
                values.append(v)

            next_obs, rewards, done, info = env.step(actions)

            for c in range(N_COMPANIES):
                agents[c].store(obs_list[c], actions[c], log_probs[c],
                                rewards[c], values[c], float(done))
                ep_rewards[c] += rewards[c]

            obs_list = next_obs

            # PPO update when buffer is full
            if (epoch + 1) % ROLLOUT_LEN == 0:
                for c in range(N_COMPANIES):
                    losses = agents[c].update(last_obs=obs_list[c])
                    print(
                        f"  [{COMPANY_NAMES[c]}] epoch={epoch+1:3d} "
                        f"actor={losses['actor_loss']:+.4f} "
                        f"critic={losses['critic_loss']:.4f} "
                        f"entropy={losses['entropy']:.3f}"
                    )

            if done:
                break

        # Episode summary
        for c in range(N_COMPANIES):
            episode_reward_history[c].append(ep_rewards[c])

        print(f"  Episode {episode} rewards: "
              f"{COMPANY_NAMES[0]}={ep_rewards[0]:.3f}  "
              f"{COMPANY_NAMES[1]}={ep_rewards[1]:.3f}")
        print(f"  Last epoch — "
              f"completed: {info['completed']}  "
              f"dropped: {info['dropped']}  "
              f"revenue: {[f'${r:.2f}' for r in info['revenue']]}")
        print(f"  Last prices HV/AV (mean): "
              f"{COMPANY_NAMES[0]}="
              f"{np.mean(info['prices_hv'][0]):.3f}/{np.mean(info['prices_av'][0]):.3f}  "
              f"{COMPANY_NAMES[1]}="
              f"{np.mean(info['prices_hv'][1]):.3f}/{np.mean(info['prices_av'][1]):.3f}")

        # Save checkpoints
        if episode % args.save_every == 0:
            for c, agent in enumerate(agents):
                path = os.path.join(args.checkpoint,
                                    f"{COMPANY_NAMES[c]}_ep{episode}.pt")
                agent.save(path)

    # Save final checkpoints
    for c, agent in enumerate(agents):
        agent.save(os.path.join(args.checkpoint, f"{COMPANY_NAMES[c]}_final.pt"))

    # Save training config so evaluate.py can reconstruct the same env
    config_path = os.path.join(args.checkpoint, "training_config.json")
    training_cfg = {
        "reward_mode": args.reward_mode,
        "use_sumo":    args.use_sumo,
        "trips":       args.trips,
        "zone_edges":  getattr(args, "zone_edges", None),
        "seed":        args.seed,
    }
    with open(config_path, "w") as f:
        json.dump(training_cfg, f, indent=2)
    print(f"[train] Training config saved → {config_path}")

    print("\nTraining complete.")
    env.traffic.close()

    return agents, episode_reward_history


if __name__ == "__main__":
    args = parse_args()
    train(args)
