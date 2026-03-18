"""
env/vehicle.py
==============
Represents a single vehicle in the simulation.

State machine (Section 2.1.1):
    IDLE → TO_PICKUP → OCCUPIED → IDLE

Each vehicle belongs to one (company, vehicle_type) pair and lives on a
specific SUMO edge. The FleetManager moves vehicles between states by
issuing SUMO TraCI commands.
"""

from config import STATE_IDLE, STATE_TO_PICKUP, STATE_OCCUPIED


class Vehicle:
    """
    Attributes
    ----------
    vid : str
        Unique SUMO vehicle ID, e.g. "A_hv_0", "B_av_12"
    company : int
        0 = CompanyA, 1 = CompanyB
    vtype : str
        "hv" or "av"
    state : int
        One of STATE_IDLE / STATE_TO_PICKUP / STATE_OCCUPIED
    current_edge : str
        The SUMO edge ID where the vehicle currently is (updated each epoch).
    request_id : int or None
        ID of the request being served (None when IDLE).
    """

    def __init__(self, vid: str, company: int, vtype: str, start_edge: str):
        self.vid          = vid
        self.company      = company
        self.vtype        = vtype
        self.state        = STATE_IDLE
        self.current_edge = start_edge
        self.request_id   = None
        self.planned_fare = 0.0   # fare estimated at dispatch time, used at dropoff

    # ── State transitions ─────────────────────────────────────────────────────

    def assign(self, request_id: int, pickup_edge: str):
        """Called by FleetManager when this vehicle is dispatched to a request."""
        assert self.state == STATE_IDLE, f"{self.vid} is not IDLE"
        self.state      = STATE_TO_PICKUP
        self.request_id = request_id
        # current_edge stays the same until SUMO confirms movement

    def pickup(self):
        """Called when the vehicle reaches the pickup stop."""
        assert self.state == STATE_TO_PICKUP
        self.state = STATE_OCCUPIED

    def dropoff(self, new_edge: str):
        """Called when the vehicle reaches the dropoff stop. Returns to IDLE."""
        assert self.state == STATE_OCCUPIED
        self.state        = STATE_IDLE
        self.request_id   = None
        self.current_edge = new_edge

    def reset_to_idle(self, depot_edge: str):
        """
        Emergency reset: used when SUMO teleports the vehicle out of the
        network due to congestion (Section 2.1.1 — respawn at depot).
        """
        self.state        = STATE_IDLE
        self.request_id   = None
        self.current_edge = depot_edge

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        return self.state == STATE_IDLE

    def __repr__(self):
        state_names = {0: "IDLE", 1: "TO_PICKUP", 2: "OCCUPIED"}
        return (f"Vehicle({self.vid}, {state_names[self.state]}, "
                f"edge={self.current_edge})")
