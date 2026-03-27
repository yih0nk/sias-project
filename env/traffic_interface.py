"""
env/traffic_interface.py
========================
Wraps the SUMO TraCI connection (Section 2.3).

Vehicle lifecycle (SUMO mode)
------------------------------
Vehicles do NOT pre-exist in SUMO when idle.  Only when a vehicle is
dispatched to a request does it enter SUMO — with its full route already
set (depot→pickup→dropoff).  After dropoff, the vehicle is removed from
SUMO and its stored edge is updated to the dropoff location.

This avoids the lane-blocking problem that occurs when 40 vehicles all
try to park at the same depot edge.

Two modes
---------
SUMO mode  : connects to a real SUMO process (requires SUMO installed)
Mock mode  : returns fake metrics — used for RL training without SUMO
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import SUMO_BINARY, SUMO_CFG, STEPS_PER_EPOCH, K_SHORTEST_PATHS

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[TrafficInterface] WARNING: traci not found — running in mock mode.")


class EdgeMetrics:
    """Snapshot of traffic conditions after one epoch."""
    def __init__(self, travel_times: Dict[str, float],
                 occupancies: Dict[str, float]):
        self.travel_times = travel_times
        self.occupancies  = occupancies
        all_tt  = list(travel_times.values())
        all_occ = list(occupancies.values())
        self.mean_travel_time = float(np.mean(all_tt))  if all_tt  else 0.0
        self.mean_occupancy   = float(np.mean(all_occ)) if all_occ else 0.0
        self.n_edges          = len(travel_times)


class TrafficInterface:
    """
    Manages the SUMO simulation process and exposes a clean API.

    Parameters
    ----------
    sumo_cfg : str       Path to .sumocfg file
    use_gui  : bool      Launch sumo-gui for visual debugging (slow)
    seed     : int       SUMO random seed
    mock     : bool      If True, skip SUMO entirely (unit testing)
    """

    def __init__(self, sumo_cfg: str = SUMO_CFG,
                 use_gui: bool = False, seed: int = 42, mock: bool = False):
        self.sumo_cfg  = sumo_cfg
        self.use_gui   = use_gui
        self.seed      = seed
        self.mock      = mock or not TRACI_AVAILABLE
        self._started  = False
        self._edge_metrics: Optional[EdgeMetrics] = None

        # Maps vid → (pickup_edge, dropoff_edge) for active SUMO trips
        self._active_routes: Dict[str, Tuple[str, str]] = {}

        # Maps vid → (vtype, current_edge) for ALL vehicles (idle or active)
        # This is our Python-side record; SUMO only knows about active vehicles.
        self._vehicle_registry: Dict[str, Tuple[str, str]] = {}

        # Route ID counter — ensures unique SUMO route IDs
        self._route_counter = 0

        # Tracks vehicles whose pickup stop has already been reported
        self._pickup_notified: set = set()

        from tools.zone_map import DEPOT_EDGE
        self.depot_edge = DEPOT_EDGE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.mock:
            self._started = True
            return

        binary = "sumo-gui" if self.use_gui else SUMO_BINARY
        sumo_path = os.path.join(os.environ.get("SUMO_HOME", ""), "bin", binary)
        if not os.path.exists(sumo_path):
            sumo_path = binary

        cmd = [
            sumo_path,
            "-c", self.sumo_cfg,
            "--seed", str(self.seed),
            "--no-warnings",
            "--no-step-log",
            "--collision.action", "remove",
            "--time-to-teleport", "300",
        ]
        traci.start(cmd)
        self._started = True

        edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        if self.depot_edge not in edges and edges:
            self.depot_edge = edges[0]

    def close(self) -> None:
        if TRACI_AVAILABLE and self._started and not self.mock:
            traci.close()
        self._started = False

    def reset(self) -> None:
        self.close()
        self._edge_metrics     = None
        self._active_routes.clear()
        self._vehicle_registry.clear()
        self._route_counter    = 0
        self._pickup_notified.clear()
        self.start()

    def warmup(self, target_vehicle_count: int = 0, max_steps: int = 500) -> None:
        """No-op in the new design — vehicles enter SUMO only at dispatch time."""
        pass   # kept for API compatibility; no pre-spawning needed

    # ── Simulation stepping ───────────────────────────────────────────────────

    def step_epoch(self, dispatch_callback) -> EdgeMetrics:
        """Advance SUMO by one full epoch (STEPS_PER_EPOCH seconds)."""
        if self.mock:
            return self._mock_edge_metrics()

        teleported_seen = set()

        for step in range(STEPS_PER_EPOCH):
            traci.simulationStep()

            for vid in traci.simulation.getStartingTeleportIDList():
                if vid not in teleported_seen:
                    dispatch_callback(event="teleport", vid=vid, step=step)
                    teleported_seen.add(vid)

            for vid in list(self._active_routes.keys()):
                self._check_stop_events(vid, dispatch_callback)

            dispatch_callback(event="step", vid=None, step=step)

        self._edge_metrics = self._snapshot_edges()
        return self._edge_metrics

    # ── Routing queries ───────────────────────────────────────────────────────

    def get_k_shortest_routes(
        self, from_edge: str, to_edge: str, k: int = K_SHORTEST_PATHS
    ) -> List[Dict]:
        if self.mock:
            return self._mock_routes(from_edge, to_edge, k)
        if from_edge == to_edge:
            return [{"edges": [from_edge], "travel_time": 0.0}]
        routes = []
        try:
            stage = traci.simulation.findRoute(from_edge, to_edge)
            if not stage.edges:
                return []
            routes.append({"edges": list(stage.edges),
                            "travel_time": stage.travelTime})
            for _ in range(k - 1):
                noise = {}
                for e in stage.edges:
                    orig  = traci.edge.getTraveltime(e)
                    noisy = max(orig * (1 + 0.15 * np.random.randn()), 0.1)
                    traci.edge.adaptTraveltime(e, noisy)
                    noise[e] = orig
                alt = traci.simulation.findRoute(from_edge, to_edge)
                if alt.edges and list(alt.edges) not in [r["edges"] for r in routes]:
                    routes.append({"edges": list(alt.edges),
                                   "travel_time": alt.travelTime})
                for e, orig in noise.items():
                    traci.edge.adaptTraveltime(e, orig)
        except Exception:
            pass
        return routes

    def get_travel_time_between(self, from_edge: str, to_edge: str) -> float:
        if self.mock:
            return float(np.random.uniform(60, 600))
        try:
            stage = traci.simulation.findRoute(from_edge, to_edge)
            return stage.travelTime if stage.edges else float("inf")
        except Exception:
            return float("inf")

    def get_edge_travel_times(self) -> Dict[str, float]:
        if self.mock:
            return {}
        return {e: traci.edge.getTraveltime(e)
                for e in traci.edge.getIDList()
                if not e.startswith(":")}

    # ── Vehicle commands ──────────────────────────────────────────────────────

    def register_vehicle(self, vid: str, vtype: str, edge: str) -> None:
        """
        Register a vehicle in our Python registry (no SUMO insertion).

        Vehicles only enter SUMO when dispatched via route_vehicle_to_pickup().
        This replaces the old add_vehicle() approach.
        """
        self._vehicle_registry[vid] = (vtype, edge)

    def add_vehicle(self, vid: str, vtype: str, edge: str) -> None:
        """Alias for register_vehicle() — kept for backwards compatibility."""
        self.register_vehicle(vid, vtype, edge)

    def route_vehicle_to_pickup(
        self, vid: str, pickup_edge: str, passenger_route: List[str]
    ) -> None:
        """
        Dispatch a vehicle by inserting it into SUMO with its full route.

        Full route = current_edge → pickup_edge → dropoff_edge.
        The vehicle is inserted fresh each trip (not pre-existing in SUMO).
        """
        if self.mock:
            return

        if vid not in self._vehicle_registry:
            print(f"[TrafficInterface] {vid} not registered — call register_vehicle() first")
            return

        vtype, current_edge = self._vehicle_registry[vid]

        try:
            # Build full route: current_edge → pickup_edge (→ dropoff via passenger_route)
            if current_edge == pickup_edge:
                full_route = passenger_route
            else:
                to_pickup = traci.simulation.findRoute(current_edge, pickup_edge)
                if not to_pickup.edges:
                    print(f"[TrafficInterface] No route from {current_edge} to {pickup_edge} for {vid}")
                    return
                # passenger_route goes pickup→dropoff; avoid duplicating pickup_edge
                suffix = (passenger_route[1:]
                          if passenger_route and passenger_route[0] == pickup_edge
                          else passenger_route)
                full_route = list(to_pickup.edges) + suffix

            if len(full_route) < 1:
                return

            # Remove old SUMO vehicle if it somehow still exists
            if vid in traci.vehicle.getIDList():
                traci.vehicle.remove(vid)

            # Insert vehicle with a unique route ID
            self._route_counter += 1
            route_id = f"rt_{self._route_counter}"
            traci.route.add(route_id, full_route)
            traci.vehicle.add(vid, routeID=route_id, typeID=vtype,
                              departLane="best", departSpeed="max")

            # Stop at pickup edge to simulate boarding
            traci.vehicle.setStop(vid, pickup_edge, duration=2)

            self._active_routes[vid] = (pickup_edge, full_route[-1])

        except Exception as e:
            print(f"[TrafficInterface] route_vehicle_to_pickup({vid}) error: {e}")

    def release_vehicle(self, vid: str, new_edge: str) -> None:
        """
        Remove a vehicle from SUMO after dropoff and update its registry edge.
        The vehicle returns to 'virtual idle' — no SUMO presence until next dispatch.
        """
        if self.mock:
            return
        try:
            if vid in traci.vehicle.getIDList():
                traci.vehicle.remove(vid)
        except Exception:
            pass
        if vid in self._active_routes:
            del self._active_routes[vid]
        self._pickup_notified.discard(vid)
        if vid in self._vehicle_registry:
            vtype, _ = self._vehicle_registry[vid]
            self._vehicle_registry[vid] = (vtype, new_edge)

    # ── Edge metrics ──────────────────────────────────────────────────────────

    @property
    def edge_metrics(self) -> Optional[EdgeMetrics]:
        return self._edge_metrics

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _snapshot_edges(self) -> EdgeMetrics:
        travel_times, occupancies = {}, {}
        for e in traci.edge.getIDList():
            if e.startswith(":"):
                continue
            travel_times[e] = traci.edge.getTraveltime(e)
            occupancies[e]  = traci.edge.getLastStepOccupancy(e)
        return EdgeMetrics(travel_times, occupancies)

    def _check_stop_events(self, vid: str, callback) -> None:
        if vid not in self._active_routes:
            return
        try:
            active_ids = traci.vehicle.getIDList()

            if vid not in active_ids:
                # Vehicle finished its route and was removed by SUMO ("arrived").
                # This is the normal trip completion signal.
                callback(event="dropoff", vid=vid, step=None)
                return

            # Check stop transitions for pickup and dropoff
            pickup_edge, dropoff_edge = self._active_routes[vid]
            current = traci.vehicle.getRoadID(vid)
            stops   = traci.vehicle.getStops(vid)

            if not stops:
                # Pickup stop has been served — fire once per trip
                if vid not in self._pickup_notified:
                    self._pickup_notified.add(vid)
                    callback(event="pickup", vid=vid, step=None)

                # Vehicle has reached dropoff edge with no remaining stops
                if current == dropoff_edge:
                    callback(event="dropoff", vid=vid, step=None)

        except Exception:
            pass

    def _mock_edge_metrics(self) -> EdgeMetrics:
        from tools.zone_map import ALL_ZONE_EDGES
        rng = np.random.default_rng()
        return EdgeMetrics(
            travel_times={e: float(rng.uniform(10, 120)) for e in ALL_ZONE_EDGES},
            occupancies={e: float(rng.uniform(0, 0.4))  for e in ALL_ZONE_EDGES},
        )

    def _mock_routes(self, from_edge: str, to_edge: str, k: int) -> List[Dict]:
        return [{"edges": [from_edge, to_edge],
                 "travel_time": float(np.random.uniform(60, 400))}
                for _ in range(min(k, 2))]
