"""
config.py
=========
Single source of truth for every hyperparameter in the simulation.
All other modules import from here — never hard-code magic numbers.

Simplified setup (mentor feedback)
-----------------------------------
- N_ZONES = 9  (3×3 grid, manageable for experiments)
- Single price multiplier per zone (not separate HV/AV)
- ACTION_DIM = N_ZONES + 1  (zone prices + theta_av)
- Simple synthetic SUMO grid network (no real Manhattan required)
- Random demand generation (no real TLC data required)
"""

# ── Simulation structure ──────────────────────────────────────────────────────
N_ZONES          = 9           # 3×3 zone grid (simple network)
N_COMPANIES      = 2           # CompanyA (0), CompanyB (1)
COMPANY_NAMES    = ["CompanyA", "CompanyB"]
VEHICLE_TYPES    = ["hv", "av"]   # human-driven, autonomous

# Vehicles per (company, type) — reduced for simple network
N_HV_PER_COMPANY = 10
N_AV_PER_COMPANY = 10

# Time
EPOCH_SEC        = 900         # 15 min = 900 seconds per decision epoch
SUMO_STEP_SEC    = 1           # SUMO advances 1 second per step
STEPS_PER_EPOCH  = EPOCH_SEC // SUMO_STEP_SEC   # 900 steps/epoch
PLANNING_HORIZON = 96          # 1 simulated day (96 × 15min epochs)

# ── Vehicle state IDs ─────────────────────────────────────────────────────────
STATE_IDLE       = 0
STATE_TO_PICKUP  = 1
STATE_OCCUPIED   = 2

# ── Routing ───────────────────────────────────────────────────────────────────
K_SHORTEST_PATHS = 3           # candidate routes for logit sampling
MAX_WAIT_SEC     = 900         # request dropped after 15 min without service

# ── Action space ──────────────────────────────────────────────────────────────
# Each company controls:
#   zone_price[z]  : one price multiplier per zone (applies to all vehicle types)
#   theta_av       : AV routing logit dispersion
M_MIN            = 0.5         # minimum price multiplier
M_MAX            = 2.0         # maximum price multiplier
THETA_AV_MIN     = 0.1         # near-random AV routing
THETA_AV_MAX     = 10.0        # near-greedy AV routing

# Action dimension per company: N_ZONES prices + 1 theta_av
ACTION_DIM       = N_ZONES + 1   # 10

# ── Observation space ─────────────────────────────────────────────────────────
# time-of-day sin/cos                    : 2
# network congestion (mean_tt, n_edges)  : 2
# total demand + zone inflow + outflow   : 1 + 2*N_ZONES = 19
# own fleet state (idle/pickup/occ/total): 4
# competitor last price (mean) + theta   : 2
# ─────────────────────────────────────────
# Total = 11 + 2*N_ZONES = 29
OBS_DIM          = 11 + 2 * N_ZONES   # 29

# ── Reward (Eq. 1 — revenue-based) ───────────────────────────────────────────
DROP_PENALTY     = 0.2         # per dropped request
PEND_PENALTY     = 0.1         # per request still waiting in queue

# ── Reward (Eq. 2 — decomposed alternative) ──────────────────────────────────
W_PRICE          = 1.0
W_SERVICE        = 0.5
W_CONG           = 0.3

# ── Customer logit model ──────────────────────────────────────────────────────
BETA_PRICE       = 1.0
BETA_WAIT        = 0.2
BETA_TT          = 0.05
BASE_FARE        = 2.50        # dollars
PER_MIN_RATE     = 0.50        # dollars per minute

# ── IPPO hyperparameters ──────────────────────────────────────────────────────
LR_ACTOR         = 3e-4
LR_CRITIC        = 1e-3
GAMMA            = 0.99        # discount factor
LAMBDA_GAE       = 0.95        # GAE lambda
CLIP_EPS         = 0.2         # PPO clipping epsilon
ENTROPY_COEF     = 0.01
VF_COEF          = 0.5
MAX_GRAD_NORM    = 0.5
PPO_EPOCHS       = 10          # gradient steps per update
MINI_BATCH_SIZE  = 32
ROLLOUT_LEN      = 96          # one full day = one rollout

# ── Neural network architecture ───────────────────────────────────────────────
HIDDEN_DIM       = 128         # smaller net for smaller obs/action dims

# ── SUMO ──────────────────────────────────────────────────────────────────────
SUMO_BINARY      = "sumo"      # use "sumo-gui" for visual debugging
SUMO_CFG         = "grid.sumocfg"    # simple grid network config

# Manhattan config (for future full-scale experiments)
MANHATTAN_SUMO_CFG = "sim.sumocfg"

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED             = 42
LOG_DIR          = "logs"
CHECKPOINT_DIR   = "checkpoints"
