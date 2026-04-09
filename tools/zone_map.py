"""
tools/zone_map.py
=================
Maps zone indices (0..N_ZONES-1) to representative SUMO edge IDs.

Real mode: loads data/zones.csv produced by tools/build_zone_edge_map.py.
Mock mode: falls back to synthetic edge names (zone{i}_e0) when zones.csv
           is not present.

Zone index i corresponds to MANHATTAN_ZONE_IDS[i] in data_loader.py.
"""

import os
from config import N_ZONES

_ZONES_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "zones.csv"
)


def _load() -> tuple:
    """Returns (ZONE_EDGES, ZONE_REP_EDGE, DEPOT_EDGE)."""
    if os.path.exists(_ZONES_CSV):
        # Real mode — use SUMO edges from the Manhattan network
        import csv
        from data_loader import MANHATTAN_ZONE_IDS
        tlc_to_edge = {}
        with open(_ZONES_CSV) as f:
            for row in csv.DictReader(f):
                tlc_to_edge[int(row["zone_id"])] = row["edge_id"]

        zone_rep = {}
        for idx, tlc_id in enumerate(MANHATTAN_ZONE_IDS):
            if tlc_id in tlc_to_edge:
                zone_rep[idx] = tlc_to_edge[tlc_id]

        zone_edges = {z: [e] for z, e in zone_rep.items()}
        depot = zone_rep.get(0, list(zone_rep.values())[0])
        return zone_edges, zone_rep, depot
    else:
        # Mock mode — synthetic edge names
        zone_edges = {i: [f"zone{i}_e0", f"zone{i}_e1"] for i in range(N_ZONES)}
        zone_rep   = {i: f"zone{i}_e0" for i in range(N_ZONES)}
        depot      = "zone0_e0"
        return zone_edges, zone_rep, depot


ZONE_EDGES, ZONE_REP_EDGE, DEPOT_EDGE = _load()

ALL_ZONE_EDGES = sorted({e for edges in ZONE_EDGES.values() for e in edges})


def zone_of_edge(edge_id: str) -> int:
    """Return the zone index for a given edge, or -1 if not found."""
    for z, edges in ZONE_EDGES.items():
        if edge_id in edges:
            return z
    return -1
