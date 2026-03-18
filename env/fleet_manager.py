"""
env/fleet_manager.py
====================
Handles vehicle dispatch and routing for one company (Section 2.1.1).

Every epoch, FleetManager does the following:
    1. Pull all pending requests assigned to this company/vtype.
    2. For each request, find the nearest IDLE vehicle (min travel time).
    3. Compute a passenger route (pickup → dropoff):
         - HV: deterministic shortest path
         - AV: logit sampling over k-shortest paths (Eq. 7)
    4. Issue SUMO TraCI commands to move vehicles.
    5. Age pending requests; drop any that exceed MAX_WAIT_SEC.

FleetManager is stateless between epochs except for the vehicle list and
pending queue — the TrafficInterface owns the SUMO connection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from config import (
    STATE_IDLE, STATE_TO_PICKUP, STATE_OCCUPIED,
    K_SHORTEST_PATHS, MAX_WAIT_SEC, EPOCH_SEC,
    VEHICLE_TYPES,
)
from env.vehicle import Vehicle
from env.customer_model import Request


class FleetManager:
    """
    Manages the fleet for ONE company.

    Parameters
    ----------
    company : int           0 or 1
    vehicles : List[Vehicle]
        All vehicles belonging to this company (both HV and AV).
    traffic_interface : TrafficInterface
        Reference to the SUMO wrapper — used to query travel times and
        issue route commands.
    rng : np.random.Generator
    """

    def __init__(self, company: int, vehicles: List[Vehicle],
                 traffic_interface, rng: np.random.Generator):
        self.company           = company
        self.vehicles          = vehicles
        self.traffic           = traffic_interface
        self.rng               = rng

        # Pending queue: requests that have been assigned to this company
        # but not yet served.  Maps request_id → (Request, wait_time_so_far)
        self._pending: Dict[int, Tuple[Request, float]] = {}

        # Track completed / dropped counts for reward computation
        self.n_completed  = 0
        self.n_dropped    = 0
        self.total_revenue = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Add a newly assigned request to the pending queue."""
        self._pending[request.request_id] = (request, 0.0)

    def dispatch_epoch(self, price_multipliers: np.ndarray,
                       theta_av: float) -> Dict[str, int]:
        """
        Core dispatch loop called once per epoch (Section 2.1.1).

        Parameters
        ----------
        price_multipliers : np.ndarray shape (2,)
            [hv_mult, av_mult] for this company's current action.
        theta_av : float
            AV routing logit dispersion (action component θ_av).

        Returns
        -------
        stats : dict with keys 'completed', 'dropped', 'pending'
        """
        idle_vehicles = self._get_idle_vehicles()

        # Sort pending by arrival time (FIFO)
        pending_list = sorted(
            self._pending.values(), key=lambda x: x[0].arrival_epoch
        )

        newly_dispatched = []

        for request, wait_time in pending_list:
            if not idle_vehicles:
                break   # no more vehicles available this epoch

            # Find nearest idle vehicle by travel time to pickup
            best_vehicle, tt_to_pickup = self._find_nearest_idle(
                request.pickup_edge, idle_vehicles
            )
            if best_vehicle is None:
                continue

            # Compute passenger route (pickup → dropoff)
            route = self._compute_passenger_route(
                request.pickup_edge,
                request.dropoff_edge,
                best_vehicle.vtype,
                theta_av,
            )
            if route is None:
                continue   # no route found (disconnected graph)

            # Issue SUMO commands
            self.traffic.route_vehicle_to_pickup(
                best_vehicle.vid, request.pickup_edge, route
            )

            # Estimate fare using request travel time and zone price multiplier
            from config import BASE_FARE, PER_MIN_RATE
            tt_min = request.travel_time_est / 60.0
            mult   = price_multipliers[0] if best_vehicle.vtype == "hv" \
                     else price_multipliers[1]
            best_vehicle.planned_fare = (BASE_FARE + PER_MIN_RATE * tt_min) * mult

            # Update vehicle state
            best_vehicle.assign(request.request_id, request.pickup_edge)
            request.assigned_vid = best_vehicle.vid

            idle_vehicles.remove(best_vehicle)
            newly_dispatched.append(request.request_id)

        # Remove dispatched requests from pending
        for rid in newly_dispatched:
            del self._pending[rid]

        # In mock mode (no real SUMO), immediately complete all dispatched
        # trips so the reward signal is non-zero during training/testing.
        if self.traffic.mock:
            for rid in newly_dispatched:
                # Find the vehicle that was assigned this request
                for v in self.vehicles:
                    if v.request_id == rid:
                        # Simulate pickup + dropoff instantly
                        v.pickup()
                        fare = float(self.rng.uniform(5, 20))
                        v.dropoff(self.traffic.depot_edge)
                        self.n_completed   += 1
                        self.total_revenue += fare
                        break

        # Age remaining pending requests and drop expired ones
        self._age_and_drop_pending()

        return {
            "completed": self.n_completed,
            "dropped":   self.n_dropped,
            "pending":   len(self._pending),
        }

    def on_pickup_reached(self, vid: str) -> None:
        """Called by TrafficInterface when a vehicle arrives at pickup stop."""
        vehicle = self._get_vehicle(vid)
        if vehicle and vehicle.state == STATE_TO_PICKUP:
            vehicle.pickup()

    def on_dropoff_reached(self, vid: str, dropoff_edge: str,
                           fare: float) -> None:
        """
        Called by TrafficInterface when a vehicle completes a trip.

        Accepts TO_PICKUP or OCCUPIED state — in SUMO mode we don't fire
        a separate pickup event, so the vehicle may still be in TO_PICKUP
        when it arrives at the dropoff.
        """
        vehicle = self._get_vehicle(vid)
        if vehicle and vehicle.state in (STATE_TO_PICKUP, STATE_OCCUPIED):
            vehicle.state        = STATE_IDLE
            vehicle.request_id   = None
            vehicle.current_edge = dropoff_edge
            self.n_completed    += 1
            self.total_revenue  += fare

    def on_vehicle_teleported(self, vid: str, depot_edge: str) -> None:
        """
        Called when SUMO teleports a vehicle out of the network due to
        congestion. Reset it to IDLE at the depot (Section 2.1.1).
        """
        vehicle = self._get_vehicle(vid)
        if vehicle:
            # If it was serving a request, re-queue the request
            if vehicle.request_id is not None:
                # Request goes back to pending with its current wait time
                # (simplified: re-add with 0 extra wait — could improve later)
                pass
            vehicle.reset_to_idle(depot_edge)

    # ── Fleet state summary (used for observations) ───────────────────────────

    def get_fleet_state(self) -> np.ndarray:
        """
        Returns a 4-element array: [n_idle, n_to_pickup, n_occupied, n_total]
        normalized by total fleet size.

        Used as part of the observation vector (Section 3.1).
        """
        n_idle      = sum(1 for v in self.vehicles if v.state == STATE_IDLE)
        n_pickup    = sum(1 for v in self.vehicles if v.state == STATE_TO_PICKUP)
        n_occupied  = sum(1 for v in self.vehicles if v.state == STATE_OCCUPIED)
        n_total     = len(self.vehicles)
        return np.array([n_idle, n_pickup, n_occupied, n_total],
                        dtype=np.float32) / max(n_total, 1)

    def reset_epoch_stats(self) -> None:
        """Reset per-epoch counters. Call at the start of each epoch."""
        self.n_completed   = 0
        self.n_dropped     = 0
        self.total_revenue = 0.0

    def reset(self) -> None:
        """Full reset for a new episode."""
        self._pending.clear()
        self.reset_epoch_stats()
        for v in self.vehicles:
            v.state      = STATE_IDLE
            v.request_id = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_idle_vehicles(self) -> List[Vehicle]:
        return [v for v in self.vehicles if v.is_idle]

    def _get_vehicle(self, vid: str) -> Optional[Vehicle]:
        for v in self.vehicles:
            if v.vid == vid:
                return v
        return None

    def _find_nearest_idle(
        self,
        pickup_edge: str,
        idle_vehicles: List[Vehicle],
    ) -> Tuple[Optional[Vehicle], float]:
        """
        Find the idle vehicle with the minimum travel time to pickup_edge.

        Uses TrafficInterface to query travel time between two edges.
        Falls back to a large constant if no route exists.
        """
        best_vehicle = None
        best_tt      = float("inf")

        for vehicle in idle_vehicles:
            tt = self.traffic.get_travel_time_between(
                vehicle.current_edge, pickup_edge
            )
            if tt < best_tt:
                best_tt      = tt
                best_vehicle = vehicle

        return best_vehicle, best_tt

    def _compute_passenger_route(
        self,
        pickup_edge: str,
        dropoff_edge: str,
        vtype: str,
        theta_av: float,
    ) -> Optional[List[str]]:
        """
        Compute the route a vehicle takes while carrying a passenger.

        HV: always takes the single shortest path.
        AV: samples from k-shortest paths using logit probabilities (Eq. 7).

        Returns
        -------
        list of SUMO edge IDs, or None if no route found.
        """
        routes = self.traffic.get_k_shortest_routes(
            pickup_edge, dropoff_edge, k=K_SHORTEST_PATHS
        )
        if not routes:
            return None

        if vtype == "hv" or len(routes) == 1:
            # Deterministic: always pick the shortest (first) route
            return routes[0]["edges"]

        # AV logit sampling (Eq. 7)
        travel_times = np.array([r["travel_time"] for r in routes],
                                dtype=np.float64)
        tt_min = travel_times.min()

        # P(route_i) = exp(-θ_av * (tt_i - tt_min)) / Σ exp(...)
        log_probs = -theta_av * (travel_times - tt_min)
        probs     = np.exp(log_probs - log_probs.max())
        probs    /= probs.sum()

        chosen_idx = self.rng.choice(len(routes), p=probs)
        return routes[chosen_idx]["edges"]

    def _age_and_drop_pending(self) -> None:
        """
        Increment wait time for all pending requests by one epoch.
        Drop any that have waited longer than MAX_WAIT_SEC.
        """
        to_drop = []
        updated = {}

        for rid, (request, wait) in self._pending.items():
            new_wait = wait + EPOCH_SEC
            if new_wait >= MAX_WAIT_SEC:
                to_drop.append(rid)
                self.n_dropped += 1
            else:
                updated[rid] = (request, new_wait)

        self._pending = updated
