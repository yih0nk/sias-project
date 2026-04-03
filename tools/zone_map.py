"""
tools/zone_map.py
=================
Zone layout for the simulation.

Mock mode (default): 75 zones with generated representative edge names.
Real mode: replace ZONE_EDGES / ZONE_REP_EDGE with actual SUMO edge IDs
           derived from the Manhattan OSM network and TLC zone shapefiles.

Zone indices 0–74 correspond to MANHATTAN_ZONE_IDS in data_loader.py
(the 75 TLC taxi zones used in the full-scale experiment).
"""

from config import N_ZONES

# ── Mock zone layout ──────────────────────────────────────────────────────────
# Each zone gets a synthetic edge name "zone{i}_e0" / "zone{i}_e1".
# In real mode, replace with SUMO edge IDs from the Manhattan network.

ZONE_EDGES: dict = {
    i: [f"zone{i}_e0", f"zone{i}_e1"]
    for i in range(N_ZONES)
}

# Representative edge for each zone (used for vehicle spawning / demand)
ZONE_REP_EDGE: dict = {i: f"zone{i}_e0" for i in range(N_ZONES)}

# All unique mock edges
ALL_ZONE_EDGES = sorted({e for edges in ZONE_EDGES.values() for e in edges})

# Depot edge: where vehicles are reset after teleport
DEPOT_EDGE = "zone0_e0"


def zone_of_edge(edge_id: str) -> int:
    """Return the zone index for a given edge, or -1 if not found."""
    for z, edges in ZONE_EDGES.items():
        if edge_id in edges:
            return z
    return -1
