# Multi-Company Robotaxi Control in Mixed-Autonomy Networks

A multi-agent reinforcement learning simulation where two competing ride-hailing companies (CompanyA, CompanyB) learn pricing and routing strategies via Independent PPO (IPPO). Each company operates a mixed fleet of human-driven vehicles (HV) and autonomous vehicles (AV) on a shared road network.

## Overview

The simulation is closed-loop: customer demand drives vehicle dispatch, vehicle routing affects congestion, and congestion feeds back into the next decision. Two independent RL agents compete for the same customer pool — observing each other's prices and adapting their strategies over time.

The core research question: **do the two companies converge to the same strategy, or do they specialize into different competitive roles?**

## Architecture

```
sias-project/
├── config.py              All hyperparameters (single source of truth)
├── data_loader.py         Demand generation: mock (sinusoidal daily pattern) + real TLC data
├── train.py               Training entry point
├── evaluate.py            Loads checkpoints, plots decision comparison between companies
├── grid.net.xml           4×4 SUMO grid road network (300m edges, 48 roads)
├── grid.sumocfg           SUMO simulation configuration
├── grid.rou.xml           Vehicle type definitions (HV and AV)
├── tools/
│   └── zone_map.py        Maps 9 zones to SUMO edge IDs on the grid
├── env/
│   ├── vehicle.py         Vehicle state machine: IDLE → TO_PICKUP → OCCUPIED → IDLE
│   ├── customer_model.py  Logit discrete-choice demand model (Eq. 8–9)
│   ├── fleet_manager.py   Per-epoch dispatch, nearest-vehicle matching, route sampling
│   ├── traffic_interface.py  SUMO TraCI wrapper (real + mock mode)
│   └── ride_hailing_env.py   Gym-style MARL environment
└── agents/
    ├── networks.py        Actor-Critic MLP (LayerNorm + Tanh)
    └── ppo_agent.py       PPO agent: rollout buffer, GAE, clipped surrogate loss
```

## Model

### Entities

| Entity | Details |
|--------|---------|
| Companies | CompanyA, CompanyB |
| Vehicle types | HV = 10, AV = 10 per company |
| Zones | 9 zones (3×3 grid layout) |
| Network | 4×4 SUMO grid, 300m per edge |
| Epoch | 15 minutes |
| Planning horizon | 96 epochs (1 simulated day) |

### Action Space (per company, dim = 10)

Each company controls:
- `zone_price[z]` — price multiplier per zone ∈ [0.5, 2.0], applied to all vehicles
- `theta_av` — AV routing logit dispersion ∈ [0.1, 10.0]

High `theta_av` → AV always takes the shortest route (greedy).
Low `theta_av` → AV explores alternative routes (random).

### Observation Space (per company, dim = 29)

| Slice | Meaning |
|-------|---------|
| [0:2] | Time of day (sin/cos encoding) |
| [2:4] | Network congestion (mean travel time, mean occupancy) |
| [4] | Total demand this epoch |
| [5:14] | Zone outflow — trips departing each zone |
| [14:23] | Zone inflow — trips arriving at each zone |
| [23:27] | Own fleet state (idle / en-route / occupied / total) |
| [27:29] | Competitor's mean price and theta_av (from last epoch) |

### Customer Choice (Logit Model)

Each customer computes a utility for each (company, vehicle_type) option:

```
U(c, vt) = -β_price × price  -  β_wait × wait  -  β_tt × travel_time
price     = (base_fare + per_min_rate × travel_time) × multiplier[c][zone]
```

| Parameter | Value |
|-----------|-------|
| β_price | 1.0 |
| β_wait | 0.2 |
| β_tt | 0.05 |
| Base fare | $2.50 |
| Per-minute rate | $0.50/min |

The customer samples from a softmax over all 4 alternatives {A_hv, A_av, B_hv, B_av}.

### Reward (per company, per epoch)

```
reward = (revenue  -  drop_penalty × n_dropped  -  pend_penalty × n_pending) / n_completed
```

| Parameter | Value |
|-----------|-------|
| drop_penalty | 0.2 per dropped request |
| pend_penalty | 0.1 per waiting request |

### IPPO Agent

Each company has an independent PPO agent — no shared parameters, no communication. They interact only through the environment (prices visible to competitor, shared customer pool).

- **Actor**: 2-layer MLP (128 hidden, LayerNorm, Tanh) → Gaussian policy
- **Critic**: 2-layer MLP → scalar value V(s)
- **GAE** with γ=0.99, λ=0.95
- **PPO clip** ε=0.2, 10 gradient epochs per rollout

## Quickstart

### Requirements

```bash
pip install torch numpy matplotlib
```

SUMO v1.26+ must be installed and `SUMO_HOME` set (for real SUMO mode). Mock mode works without SUMO.

### Train

```bash
# Mock mode — no SUMO required, trains immediately
python3 train.py --episodes 200

# Real SUMO mode
python3 train.py --use-sumo --episodes 200

# Decomposed reward (price × market_share + service − congestion)
python3 train.py --episodes 200 --reward-mode decomposed
```

### Evaluate

```bash
python3 evaluate.py \
    --ckpt-a checkpoints/CompanyA_final.pt \
    --ckpt-b checkpoints/CompanyB_final.pt
```

Produces a 4-panel figure (`results.png`) showing:
1. Zone pricing strategy over time (A vs B)
2. AV routing greediness (θ_av) over time
3. Cumulative reward comparison
4. Market share over time

## Extending to Manhattan

The codebase supports a full-scale Manhattan experiment:

1. Download the Manhattan SUMO network (OSM → netconvert)
2. Download NYC TLC FHV trip records (July 2025) from nyc.gov/tlc
3. Build a zone→edge mapping CSV (`tools/map_zones_to_edges.py`)
4. Update `config.py`: set `N_ZONES = 75`, `N_HV_PER_COMPANY = 200`, `N_AV_PER_COMPANY = 200`
5. Run: `python3 train.py --use-sumo --trips data/fhv.parquet --zone-edges data/zones.csv`
