"""
config.py
=========
Single source of truth for every hyperparameter in the simulation.
All other modules import from here — never hard-code magic numbers.

Parameters match Section 1 / Table 1 of the paper.
N_ZONES=70 (paper uses 75; 5 non-Manhattan zones dropped).
Real mode requires the Manhattan SUMO network and TLC zone→edge mapping.
"""

# ── Simulation structure ──────────────────────────────────────────────────────
N_ZONES          = 70          # 70 Manhattan TLC zones with valid SUMO edges (75 in paper; 5 non-Manhattan dropped)
N_COMPANIES      = 2           # CompanyA (0), CompanyB (1)
COMPANY_NAMES    = ["CompanyA", "CompanyB"]
VEHICLE_TYPES    = ["hv", "av"]   # human-driven, autonomous

# Vehicles per (company, type) — Table 1
N_HV_PER_COMPANY = 200
N_AV_PER_COMPANY = 200

# Time
EPOCH_SEC        = 900         # 15 min = 900 seconds per decision epoch
SUMO_STEP_SEC    = 1           # SUMO advances 1 second per step
STEPS_PER_EPOCH  = EPOCH_SEC // SUMO_STEP_SEC   # 900 steps/epoch
PLANNING_HORIZON = 96          # 1 simulated day (96 × 15min epochs); scale to 2976 with real data

# ── Vehicle state IDs ─────────────────────────────────────────────────────────
STATE_IDLE       = 0
STATE_TO_PICKUP  = 1
STATE_OCCUPIED   = 2

# ── Routing ───────────────────────────────────────────────────────────────────
K_SHORTEST_PATHS = 3           # candidate routes for logit sampling
MAX_WAIT_SEC     = 900         # request dropped after 15 min without service

# ── Action space ──────────────────────────────────────────────────────────────
# Each company controls separate HV and AV price multipliers per zone (Table 2):
#   zone_price_hv[z] : HV price multiplier for zone z   in [M_MIN, M_MAX]
#   zone_price_av[z] : AV price multiplier for zone z   in [M_MIN, M_MAX]
M_MIN            = 0.1         # minimum price multiplier (Eq. 3)
M_MAX            = 2.0         # maximum price multiplier (Eq. 3)

# Action dimension: N_ZONES HV prices + N_ZONES AV prices (Table 2, no theta)
ACTION_DIM       = 2 * N_ZONES   # 140 (2 × 70 zones)

# ── Observation space (Section 3.1, no congestion, no theta) ─────────────────
# time-of-day sin/cos                       : 2
# total demand + zone inflow + outflow       : 1 + 2*N_ZONES
# own fleet state (idle/pickup/occ/total)    : 4
# competitor last HV mean price + AV mean price : 2
# ──────────────────────────────────────────
# Total = 9 + 2*N_ZONES
OBS_DIM          = 9 + 2 * N_ZONES   # 149 (9 + 2×70)

# ── Reward (Eq. 1 — revenue-based) ───────────────────────────────────────────
DROP_PENALTY     = 0.2         # per dropped request
PEND_PENALTY     = 0.1         # per request still waiting in queue

# ── Reward (Eq. 2 — decomposed alternative) ──────────────────────────────────
W_PRICE          = 1.0
W_SERVICE        = 0.5

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
SUMO_CFG         = "sim.sumocfg"     # Manhattan network config

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED             = 42
LOG_DIR          = "logs"
CHECKPOINT_DIR   = "checkpoints"
