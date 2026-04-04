"""
tools/build_zone_edge_map.py
============================
One-time script: maps each of the 75 Manhattan TLC taxi zones to its
nearest driveable SUMO edge in manhattan.net.xml.

Output: data/zones.csv  (columns: zone_id, edge_id)

Usage:
    python3 tools/build_zone_edge_map.py \
        --net manhattan.net.xml \
        --shp data/taxi_zones/taxi_zones.shp \
        --out data/zones.csv
"""

import argparse
import sys
import pandas as pd

# The 75 Manhattan TLC zone IDs (from data_loader.py)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="manhattan.net.xml")
    p.add_argument("--shp", default="data/taxi_zones/taxi_zones.shp")
    p.add_argument("--out", default="data/zones.csv")
    p.add_argument("--radius", type=float, default=500.0,
                   help="Search radius in metres for nearest edge (default 500)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("Missing: pip install geopandas")

    try:
        import sumolib
    except ImportError:
        sys.exit("Missing: pip install sumolib  (or add $SUMO_HOME/tools to PYTHONPATH)")

    print(f"[zone_map] Loading SUMO network from {args.net} …")
    net = sumolib.net.readNet(args.net, withInternal=False)

    print(f"[zone_map] Loading TLC shapefile from {args.shp} …")
    zones = gpd.read_file(args.shp)

    # Reproject to WGS-84 so centroids are in lon/lat
    zones = zones.to_crs("EPSG:4326")

    manhattan = zones[zones["LocationID"].astype(int).isin(MANHATTAN_ZONE_IDS)].copy()
    print(f"[zone_map] Found {len(manhattan)} Manhattan zones in shapefile")

    rows = []
    missing = []

    for _, row in manhattan.iterrows():
        zone_id  = int(row["LocationID"])
        centroid = row.geometry.centroid
        lon, lat = centroid.x, centroid.y

        # sumolib uses (lon, lat) → (x, y) via net projection
        x, y = net.convertLonLat2XY(lon, lat)

        edges = net.getNeighboringEdges(x, y, r=args.radius)
        # Filter to edges that allow passenger vehicles
        edges = [(e, d) for e, d in edges
                 if e.allows("passenger") and not e.getID().startswith(":")]

        if not edges:
            print(f"  WARNING: no edge found within {args.radius}m for zone {zone_id} "
                  f"({lon:.4f}, {lat:.4f}) — widening search")
            edges = net.getNeighboringEdges(x, y, r=args.radius * 3)
            edges = [(e, d) for e, d in edges
                     if e.allows("passenger") and not e.getID().startswith(":")]

        if edges:
            best_edge = sorted(edges, key=lambda e: e[1])[0][0]
            rows.append({"zone_id": zone_id, "edge_id": best_edge.getID()})
            print(f"  zone {zone_id:3d} → edge {best_edge.getID()}")
        else:
            missing.append(zone_id)
            print(f"  zone {zone_id:3d} → NOT FOUND")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[zone_map] Saved {len(df)} zone→edge mappings to {args.out}")

    if missing:
        print(f"[zone_map] WARNING: {len(missing)} zones have no edge: {missing}")
        print("           Try increasing --radius or check the shapefile CRS.")


if __name__ == "__main__":
    main()
