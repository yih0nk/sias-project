"""
env/ride_hailing_env.py
=======================
Main multi-agent Gym-style environment (Sections 2–3).

Simplified action space (mentor feedback)
-----------------------------------------
Each company controls:
  - zone_price[z]  : one price multiplier per zone (applies to all vehicles)
  - theta_av       : AV routing logit dispersion
  ACTION_DIM = N_ZONES + 1

Observation per company (dim = 11 + 2*N_ZONES = 29 for N_ZONES=9):
  [0:2]                time-of-day sin/cos
  [2:4]                network congestion (mean_tt, mean_occ)
  [4]                  total demand (normalized)
  [5:5+N_ZONES]        zone outflow (demand leaving each zone)
  [5+N_ZONES:5+2*N_ZONES]  zone inflow (demand arriving each zone)
  [5+2*N_ZONES:9+2*N_ZONES]  own fleet state (idle/pickup/occ/total)
  [9+2*N_ZONES:11+2*N_ZONES]  competitor's mean price + theta_av
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import (
    N_ZONES, N_COMPANIES, COMPANY_NAMES, VEHICLE_TYPES,
    N_HV_PER_COMPANY, N_AV_PER_COMPANY,
    M_MIN, M_MAX, THETA_AV_MIN, THETA_AV_MAX,
    ACTION_DIM, OBS_DIM,
    EPOCH_SEC, PLANNING_HORIZON,
    DROP_PENALTY, PEND_PENALTY,
    W_PRICE, W_SERVICE, W_CONG,
    SEED,
)
from env.vehicle import Vehicle
from env.customer_model import CustomerModel, Request
from env.fleet_manager import FleetManager
from env.traffic_interface import TrafficInterface


class RideHailingEnv:
    """
    Multi-agent ride-hailing environment.

    Parameters
    ----------
    demand_data  : list[dict]   Pre-loaded trip records (from data_loader).
    traffic      : TrafficInterface or None (None → mock mode).
    reward_mode  : "revenue" (Eq.1) or "decomposed" (Eq.2).
    seed         : int
    """

    def __init__(
        self,
        demand_data: List[dict],
        traffic:     Optional[TrafficInterface] = None,
        reward_mode: str = "revenue",
        seed:        int = SEED,
    ):
        self.reward_mode = reward_mode
        self.rng         = np.random.default_rng(seed)

        self.traffic = traffic or TrafficInterface(mock=True, seed=seed)

        self.customer_model = CustomerModel(demand_data, self.rng)

        # One Vehicle object per physical vehicle — reused across episodes
        self.vehicles: List[List[Vehicle]] = [[], []]
        for c in range(N_COMPANIES):
            for i in range(N_HV_PER_COMPANY):
                self.vehicles[c].append(
                    Vehicle(f"{COMPANY_NAMES[c]}_hv_{i}", c, "hv",
                            self.traffic.depot_edge)
                )
            for i in range(N_AV_PER_COMPANY):
                self.vehicles[c].append(
                    Vehicle(f"{COMPANY_NAMES[c]}_av_{i}", c, "av",
                            self.traffic.depot_edge)
                )

        self.fleet_managers: List[FleetManager] = [
            FleetManager(c, self.vehicles[c], self.traffic, self.rng)
            for c in range(N_COMPANIES)
        ]

        self.epoch         = 0
        self.done          = False
        # Store last decoded actions for the competitor observation
        self._last_prices  = np.ones((N_COMPANIES, N_ZONES))   # price per zone
        self._last_thetas  = np.ones(N_COMPANIES) * 5.0        # theta_av

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self) -> List[np.ndarray]:
        """Start a new episode. Returns initial observations [obs_A, obs_B]."""
        self.epoch = 0
        self.done  = False
        self._last_prices[:] = 1.0   # neutral starting price
        self._last_thetas[:] = 5.0   # mid-range theta

        for fm in self.fleet_managers:
            fm.reset()
        self.traffic.reset()

        # Register vehicles at random starting positions spread across all zones.
        # Vehicles are NOT inserted into SUMO yet — they enter on-demand when
        # dispatched. This avoids lane-blocking from all vehicles sharing one edge.
        from tools.zone_map import ZONE_REP_EDGE
        zone_edges = list(ZONE_REP_EDGE.values())
        all_vehicles = [v for fleet in self.vehicles for v in fleet]
        for i, v in enumerate(all_vehicles):
            v.current_edge = zone_edges[i % len(zone_edges)]
            self.traffic.register_vehicle(v.vid, v.vtype, v.current_edge)

        return self._build_observations(0, np.zeros(N_ZONES), np.zeros(N_ZONES))

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Execute one 15-minute epoch.

        Parameters
        ----------
        actions : [action_A, action_B]
            Each action is shape (ACTION_DIM,) = (N_ZONES+1,) in [-1, 1].

        Returns
        -------
        observations  : [obs_A, obs_B]  each shape (OBS_DIM,)
        rewards       : [reward_A, reward_B]  floats
        done          : bool
        info          : dict with detailed metrics
        """
        assert not self.done, "Episode finished — call reset() first."

        # ── 1. Decode actions ─────────────────────────────────────────────────
        prices  = []   # prices[c] = array shape (N_ZONES,)
        thetas  = []   # thetas[c] = float
        for c in range(N_COMPANIES):
            p, th = self._decode_action(actions[c])
            prices.append(p)
            thetas.append(th)
            self._last_prices[c] = p
            self._last_thetas[c] = th

        # ── 2. Get demand for this epoch ──────────────────────────────────────
        edge_tt  = self.traffic.get_edge_travel_times()
        requests = self.customer_model.get_requests_for_epoch(self.epoch, edge_tt)
        demand_total, zone_inflow, zone_outflow = \
            self.customer_model.compute_zone_demand(requests)

        # ── 3. Customer choice ────────────────────────────────────────────────
        for req in requests:
            # Price multiplier matrix: shape (N_COMPANIES, 2)
            # Both HV and AV in a company get the same zone price
            mults = np.array([
                [prices[c][req.pickup_zone], prices[c][req.pickup_zone]]
                for c in range(N_COMPANIES)
            ])
            # Approximate wait: proportional to pending queue length
            wait_times = np.array([
                [len(self.fleet_managers[c]._pending) * 30.0] * 2
                for c in range(N_COMPANIES)
            ], dtype=np.float32)

            chosen_c, chosen_vt = self.customer_model.customer_choice(
                req, mults, wait_times
            )
            req.chosen_company = chosen_c
            req.chosen_vtype   = chosen_vt
            self.fleet_managers[chosen_c].add_request(req)

        # ── 4. Reset per-epoch counters ───────────────────────────────────────
        for fm in self.fleet_managers:
            fm.reset_epoch_stats()

        # ── 5. Dispatch vehicles (mock or real SUMO) ──────────────────────────
        for c, fm in enumerate(self.fleet_managers):
            fm.dispatch_epoch(
                price_multipliers=np.array([prices[c].mean(), prices[c].mean()]),
                theta_av=thetas[c],
            )

        def dispatch_cb(event, vid, step):
            if event == "teleport" and vid:
                for fm in self.fleet_managers:
                    fm.on_vehicle_teleported(vid, self.traffic.depot_edge)
            elif event == "dropoff" and vid:
                for fm in self.fleet_managers:
                    v = fm._get_vehicle(vid)
                    if v:
                        dropoff_edge = v.current_edge   # updated by vehicle state
                        fm.on_dropoff_reached(vid, dropoff_edge, fare=v.planned_fare)
                        # Remove from SUMO; vehicle is now virtually idle again
                        self.traffic.release_vehicle(vid, dropoff_edge)

        edge_metrics = self.traffic.step_epoch(dispatch_cb)

        # ── 6. Compute rewards ────────────────────────────────────────────────
        rewards = [
            self._compute_reward(c, self.fleet_managers[c],
                                 prices[c], thetas[c], edge_metrics)
            for c in range(N_COMPANIES)
        ]

        # ── 7. Advance epoch ──────────────────────────────────────────────────
        self.epoch += 1
        self.done   = (self.epoch >= PLANNING_HORIZON)

        obs  = self._build_observations(demand_total, zone_inflow, zone_outflow)
        info = {
            "epoch":      self.epoch,
            "rewards":    rewards,
            "completed":  [fm.n_completed   for fm in self.fleet_managers],
            "dropped":    [fm.n_dropped     for fm in self.fleet_managers],
            "pending":    [len(fm._pending) for fm in self.fleet_managers],
            "revenue":    [fm.total_revenue for fm in self.fleet_managers],
            "prices":     [prices[c].tolist()  for c in range(N_COMPANIES)],
            "thetas":     thetas,
        }
        return obs, rewards, self.done, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observations(
        self,
        demand_total: int,
        zone_inflow:  np.ndarray,
        zone_outflow: np.ndarray,
    ) -> List[np.ndarray]:
        """Build the OBS_DIM-vector for each company."""
        # Time of day (sin/cos for circular encoding)
        t_frac = (self.epoch % 96) / 96.0
        tod = np.array([
            math.sin(2 * math.pi * t_frac),
            math.cos(2 * math.pi * t_frac),
        ], dtype=np.float32)

        # Network congestion
        if self.traffic.edge_metrics:
            em   = self.traffic.edge_metrics
            cong = np.array([em.mean_travel_time / 300.0,
                             em.mean_occupancy], dtype=np.float32)
        else:
            cong = np.zeros(2, dtype=np.float32)

        demand_feat = np.array([demand_total / 100.0], dtype=np.float32)

        obs_list = []
        for c in range(N_COMPANIES):
            comp = 1 - c   # the other company
            fleet = self.fleet_managers[c].get_fleet_state()
            comp_feat = np.array([
                self._last_prices[comp].mean(),   # competitor mean price
                self._last_thetas[comp] / THETA_AV_MAX,  # competitor theta (normed)
            ], dtype=np.float32)

            obs = np.concatenate([
                tod,
                cong,
                demand_feat,
                zone_outflow.astype(np.float32),
                zone_inflow.astype(np.float32),
                fleet,
                comp_feat,
            ])
            assert obs.shape == (OBS_DIM,), \
                f"OBS shape mismatch: got {obs.shape}, expected ({OBS_DIM},)"
            obs_list.append(obs)

        return obs_list

    # ── Reward computation ────────────────────────────────────────────────────

    def _compute_reward(
        self,
        company:     int,
        fm:          FleetManager,
        prices:      np.ndarray,
        theta:       float,
        edge_metrics,
    ) -> float:
        n_drop   = fm.n_dropped
        n_pend   = len(fm._pending)
        n_comp   = fm.n_completed
        revenue  = fm.total_revenue
        total    = sum(m.n_completed for m in self.fleet_managers)

        if self.reward_mode == "revenue":
            # Eq. 1
            denom  = max(n_comp, 1)
            reward = (revenue
                      - DROP_PENALTY * n_drop
                      - PEND_PENALTY * n_pend) / denom
        else:
            # Eq. 2
            mean_mult    = float(prices.mean())
            market_share = n_comp / max(total, 1)
            serve_frac   = n_comp / max(n_comp + n_drop, 1)
            mean_occ     = edge_metrics.mean_occupancy if edge_metrics else 0.0
            reward = (W_PRICE   * mean_mult * market_share
                    + W_SERVICE * serve_frac
                    - W_CONG    * mean_occ)

        return float(reward)

    # ── Action decoding ───────────────────────────────────────────────────────

    def _decode_action(
        self, raw: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map tanh output [-1, 1] → real action ranges.

        raw[:N_ZONES]  → zone prices  in [M_MIN, M_MAX]
        raw[-1]        → theta_av     in [THETA_AV_MIN, THETA_AV_MAX]
        """
        raw = np.clip(raw, -1.0, 1.0)
        t   = (raw + 1.0) / 2.0   # shift to [0, 1]

        prices   = M_MIN + t[:N_ZONES] * (M_MAX - M_MIN)
        theta_av = THETA_AV_MIN + t[-1] * (THETA_AV_MAX - THETA_AV_MIN)
        return prices, float(theta_av)
