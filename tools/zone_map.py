"""
tools/zone_map.py
=================
Defines the 9-zone layout for the 4×4 grid network.

The grid has nodes A0–D3 laid out like this:

    A0 — A1 — A2 — A3
    |    |    |    |
    B0 — B1 — B2 — B3
    |    |    |    |
    C0 — C1 — C2 — C3
    |    |    |    |
    D0 — D1 — D2 — D3

We group the 48 edges into 9 zones (3×3 layout):

    Zone 0 (top-left)     Zone 1 (top-mid)     Zone 2 (top-right)
    Zone 3 (mid-left)     Zone 4 (center)      Zone 5 (mid-right)
    Zone 6 (bot-left)     Zone 7 (bot-mid)     Zone 8 (bot-right)

Each zone has a "representative edge" used for vehicle spawning and
demand assignment.

This module is imported by data_loader.make_mock_demand() and
TrafficInterface to place vehicles at startup.
"""

# Zone index → list of SUMO edge IDs in that zone
ZONE_EDGES = {
    0: ["A0A1", "A0B0"],
    1: ["A1A2", "A1B1"],
    2: ["A2A3", "A2B2"],
    3: ["B0B1", "B0C0"],
    4: ["B1B2", "B1C1"],
    5: ["B2B3", "B2C2"],
    6: ["C0C1", "C0D0"],
    7: ["C1C2", "C1D1"],
    8: ["C2C3", "C2D2"],
}

# Representative edge for each zone (used as the single pickup/dropoff point)
ZONE_REP_EDGE = {z: edges[0] for z, edges in ZONE_EDGES.items()}

# All unique edges that appear in any zone
ALL_ZONE_EDGES = sorted({e for edges in ZONE_EDGES.values() for e in edges})

# Depot edge: where vehicles are spawned/reset
DEPOT_EDGE = "B1B2"


def zone_of_edge(edge_id: str) -> int:
    """Return the zone index for a given edge, or -1 if not found."""
    for z, edges in ZONE_EDGES.items():
        if edge_id in edges:
            return z
    return -1
