"""
env/traffic_interface.py
========================
Wraps the SUMO TraCI connection (Section 2.3).

Responsibilities:
    - Start / stop the SUMO process
    - Advance the simulation one epoch at a time (900 × 1-second steps)
    - Provide routing queries (k-shortest paths, travel times)
    - Snapshot edge-level metrics after each epoch
    - Detect teleported vehicles and reset them

All other modules call methods here — nothing else imports traci directly.

Two modes
---------
SUMO mode  : connects to a real SUMO process (requires SUMO installed)
Mock mode  : returns fake metrics for unit testing without SUMO
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
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[TrafficInterface] WARNING: traci not found — running in mock mode.")


class EdgeMetrics:
    """
    Snapshot of traffic conditions on all edges after one epoch.
    Fed into the next observation vector.
    """
    def __init__(self, travel_times: Dict[str, float],
                 occupancies: Dict[str, float]):
        self.travel_times = travel_times    # edge_id → seconds
        self.occupancies  = occupancies     # edge_id → [0, 1]
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
        self._vehicle_routes: Dict[str, Tuple[str, str]] = {}

        # Set after start() — the depot is where vehicles are spawned/reset
        from tools.zone_map import DEPOT_EDGE
        self.depot_edge = DEPOT_EDGE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch SUMO and open the TraCI connection."""
        if self.mock:
            self._started = True
            return

        binary = "sumo-gui" if self.use_gui else SUMO_BINARY
        sumo_path = os.path.join(os.environ.get("SUMO_HOME", ""), "bin", binary)
        if not os.path.exists(sumo_path):
            sumo_path = binary   # fall back to PATH

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

        # Verify depot edge exists in this network
        edges = traci.edge.getIDList()
        if self.depot_edge not in edges and edges:
            self.depot_edge = [e for e in edges if not e.startswith(":")][0]

    def close(self) -> None:
        if TRACI_AVAILABLE and self._started and not self.mock:
            traci.close()
        self._started = False

    def reset(self) -> None:
        self.close()
        self._edge_metrics = None
        self._vehicle_routes.clear()
        self.start()

    # ── Simulation stepping ───────────────────────────────────────────────────

    def step_epoch(self, dispatch_callback) -> EdgeMetrics:
        """
        Advance SUMO by one full epoch (STEPS_PER_EPOCH seconds).

        dispatch_callback(event, vid, step) is called each second so the
        FleetManagers can react to teleports and stop events.
        """
        if self.mock:
            return self._mock_edge_metrics()

        teleported_seen = set()

        for step in range(STEPS_PER_EPOCH):
            traci.simulationStep()

            # Detect newly teleported vehicles
            for vid in traci.simulation.getStartingTeleportIDList():
                if vid not in teleported_seen:
                    dispatch_callback(event="teleport", vid=vid, step=step)
                    teleported_seen.add(vid)

            # Check stop arrival events
            for vid in list(self._vehicle_routes.keys()):
                self._check_stop_events(vid, dispatch_callback)

            # General per-step callback
            dispatch_callback(event="step", vid=None, step=step)

        self._edge_metrics = self._snapshot_edges()
        return self._edge_metrics

    # ── Routing queries ───────────────────────────────────────────────────────

    def get_k_shortest_routes(
        self, from_edge: str, to_edge: str, k: int = K_SHORTEST_PATHS
    ) -> List[Dict]:
        """
        Return up to k routes between two edges.
        Each route: {'edges': [str, ...], 'travel_time': float}
        """
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
            # Approximate k-shortest by adding noise to edge weights
            for _ in range(k - 1):
                noise = {}
                for e in stage.edges:
                    orig = traci.edge.getTraveltime(e)
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
        """Current estimated travel time (seconds) between two edges."""
        if self.mock:
            return float(np.random.uniform(60, 600))
        try:
            stage = traci.simulation.findRoute(from_edge, to_edge)
            return stage.travelTime if stage.edges else float("inf")
        except Exception:
            return float("inf")

    def get_edge_travel_times(self) -> Dict[str, float]:
        """Current travel time (seconds) per edge."""
        if self.mock:
            return {}
        return {e: traci.edge.getTraveltime(e)
                for e in traci.edge.getIDList()
                if not e.startswith(":")}

    # ── Vehicle commands ──────────────────────────────────────────────────────

    def add_vehicle(self, vid: str, vtype: str, edge: str) -> None:
        """Spawn a vehicle at a given edge."""
        if self.mock:
            return
        try:
            route_id = f"route_{vid}"
            traci.route.add(route_id, [edge])
            traci.vehicle.add(vid, routeID=route_id, typeID=vtype,
                              departLane="best", departSpeed="0")
        except Exception as e:
            print(f"[TrafficInterface] add_vehicle({vid}) error: {e}")

    def route_vehicle_to_pickup(
        self, vid: str, pickup_edge: str, passenger_route: List[str]
    ) -> None:
        """
        Tell SUMO vehicle `vid` to go to pickup_edge (stop there 1s),
        then follow passenger_route to the dropoff.
        """
        if self.mock:
            return
        try:
            full_route = passenger_route
            traci.vehicle.setRoute(vid, full_route)
            traci.vehicle.setStop(vid, pickup_edge, duration=1)
            dropoff_edge = passenger_route[-1]
            self._vehicle_routes[vid] = (pickup_edge, dropoff_edge)
        except Exception as e:
            print(f"[TrafficInterface] route_vehicle_to_pickup({vid}) error: {e}")

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
        if vid not in self._vehicle_routes:
            return
        try:
            pickup_edge, dropoff_edge = self._vehicle_routes[vid]
            current = traci.vehicle.getRoadID(vid)
            stops   = traci.vehicle.getStops(vid)
            if not stops and current == dropoff_edge:
                callback(event="dropoff", vid=vid, step=None)
                del self._vehicle_routes[vid]
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
