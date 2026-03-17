"""
data_loader.py
==============
Provides demand data for the simulation in two modes:

  1. MOCK mode  — randomly generated trips, no real data files needed.
                  Uses the 9-zone grid layout from tools/zone_map.py.
                  This is what you use for experiments.

  2. REAL mode  — loads TLC FHV Parquet files (NYC July 2025 data).
                  Requires real data files + a zone→edge CSV mapping.
                  For future full-scale experiments on the Manhattan network.

Output format (both modes) — list of dicts:
    {
        'epoch':           int,    # which 15-min epoch this trip starts in
        'pickup_zone':     int,    # zone index 0..N_ZONES-1
        'dropoff_zone':    int,    # zone index 0..N_ZONES-1
        'pickup_edge':     str,    # SUMO edge at pickup location
        'dropoff_edge':    str,    # SUMO edge at dropoff location
        'travel_time_ff':  float,  # free-flow travel time estimate (seconds)
    }
"""

import os
import datetime
import numpy as np
from typing import List

from config import N_ZONES, EPOCH_SEC, PLANNING_HORIZON

# ── Mock demand ───────────────────────────────────────────────────────────────

def make_mock_demand(
    n_epochs:        int   = PLANNING_HORIZON,
    trips_per_epoch: float = 20.0,
    rng:             np.random.Generator = None,
) -> List[dict]:
    """
    Generate random demand for the 9-zone grid network.

    Demand follows a simple sinusoidal daily pattern — more trips during
    "rush hour" epochs (morning and evening) and fewer overnight.

    Parameters
    ----------
    n_epochs        : How many epochs to generate trips for.
    trips_per_epoch : Average trips per epoch (Poisson-sampled).
    rng             : numpy random generator. If None, uses seed 42.

    Returns
    -------
    List of trip dicts ready for CustomerModel.
    """
    from tools.zone_map import ZONE_REP_EDGE

    if rng is None:
        rng = np.random.default_rng(42)

    demand_data = []

    for epoch in range(n_epochs):
        # Sinusoidal demand pattern: peak at epoch 32 (8am) and 68 (5pm)
        # within a 96-epoch day
        epoch_in_day = epoch % 96
        demand_mult  = 1.0 + 0.8 * (
            0.5 * np.sin(2 * np.pi * (epoch_in_day - 16) / 96) +
            0.5 * np.sin(2 * np.pi * (epoch_in_day - 68) / 96)
        )
        demand_mult = max(0.2, demand_mult)
        n_trips = rng.poisson(trips_per_epoch * demand_mult)

        for _ in range(n_trips):
            pu_zone = int(rng.integers(0, N_ZONES))
            # Dropoff zone: biased toward adjacent zones (not pure random)
            do_zone = int(rng.integers(0, N_ZONES))

            pu_edge = ZONE_REP_EDGE[pu_zone]
            do_edge = ZONE_REP_EDGE[do_zone]

            # Travel time estimate: proportional to zone distance
            zone_dist = abs(pu_zone // 3 - do_zone // 3) + \
                        abs(pu_zone  % 3 - do_zone  % 3)
            tt_ff = max(60.0, float(rng.normal(120 * (zone_dist + 1), 30)))

            demand_data.append({
                "epoch":          epoch,
                "pickup_zone":    pu_zone,
                "dropoff_zone":   do_zone,
                "pickup_edge":    pu_edge,
                "dropoff_edge":   do_edge,
                "travel_time_ff": tt_ff,
            })

    total = len(demand_data)
    print(f"[data_loader] Generated {total:,} mock trips "
          f"over {n_epochs} epochs "
          f"(avg {total/n_epochs:.1f} trips/epoch)")
    return demand_data


# ── Real TLC demand (future use) ──────────────────────────────────────────────

# Simulation start time (matches Section 1 of the paper)
_SIM_START = datetime.datetime(2025, 7, 1, 0, 0, 0)

# Manhattan TLC zone IDs (75 zones used in the full-scale version)
MANHATTAN_ZONE_IDS = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50,
    68, 74, 75, 79, 87, 88, 90, 100, 103, 104,
    105, 107, 113, 114, 116, 120, 125, 127, 128, 137,
    140, 141, 142, 143, 144, 148, 151, 152, 153, 158,
    161, 162, 163, 164, 166, 170, 186, 194, 202, 209,
    211, 224, 229, 230, 231, 232, 233, 234, 236, 237,
    238, 239, 243, 244, 246, 249, 261, 262, 263, 8,
    9, 10, 11, 14, 15,
][:75]

_TLC_TO_IDX = {z: i for i, z in enumerate(MANHATTAN_ZONE_IDS)}


def load_demand(
    trips_path:        str,
    zone_edge_map_path: str,
    max_trips:         int = None,
) -> List[dict]:
    """
    Load TLC FHV trip data and convert to simulation format.
    Used for the full-scale Manhattan experiments (not needed for simple grid).

    Parameters
    ----------
    trips_path         : Path to TLC Parquet or CSV file.
    zone_edge_map_path : Path to zone→SUMO-edge mapping CSV.
    max_trips          : If set, only load this many trips (for quick testing).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas pyarrow")

    if not os.path.exists(trips_path):
        raise FileNotFoundError(
            f"Trip data not found: {trips_path}\n"
            "Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
        )

    df = (pd.read_parquet(trips_path) if trips_path.endswith(".parquet")
          else pd.read_csv(trips_path, parse_dates=["pickup_datetime"]))

    if max_trips:
        df = df.head(max_trips)

    print(f"[data_loader] Loaded {len(df):,} raw trips from {trips_path}")

    zone_edges = _load_zone_edge_map(zone_edge_map_path)
    manhattan_ids = set(MANHATTAN_ZONE_IDS)
    df = df[
        df["PULocationID"].isin(manhattan_ids) &
        df["DOLocationID"].isin(manhattan_ids)
    ].copy()

    df["pickup_datetime"]     = pd.to_datetime(df["pickup_datetime"])
    df["seconds_from_start"]  = (df["pickup_datetime"] - _SIM_START).dt.total_seconds()
    df = df[(df["seconds_from_start"] >= 0) &
            (df["seconds_from_start"] < PLANNING_HORIZON * EPOCH_SEC)].copy()
    df["epoch"] = (df["seconds_from_start"] // EPOCH_SEC).astype(int)

    if "trip_time" in df.columns:
        df["travel_time_ff"] = df["trip_time"].clip(60, 7200)
    else:
        df["travel_time_ff"] = 600.0

    demand_data = []
    for _, row in df.iterrows():
        pu_idx = _TLC_TO_IDX.get(int(row["PULocationID"]))
        do_idx = _TLC_TO_IDX.get(int(row["DOLocationID"]))
        pu_edge = zone_edges.get(int(row["PULocationID"]))
        do_edge = zone_edges.get(int(row["DOLocationID"]))
        if None in (pu_idx, do_idx, pu_edge, do_edge):
            continue
        demand_data.append({
            "epoch":          int(row["epoch"]),
            "pickup_zone":    pu_idx,
            "dropoff_zone":   do_idx,
            "pickup_edge":    pu_edge,
            "dropoff_edge":   do_edge,
            "travel_time_ff": float(row["travel_time_ff"]),
        })

    print(f"[data_loader] {len(demand_data):,} usable Manhattan trips")
    return demand_data


def _load_zone_edge_map(path: str):
    if not os.path.exists(path):
        print(f"[data_loader] WARNING: Zone-edge map not found: {path}")
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return dict(zip(df["zone_id"].astype(int), df["edge_id"].astype(str)))
    except Exception as e:
        print(f"[data_loader] Failed to load zone-edge map: {e}")
        return {}
