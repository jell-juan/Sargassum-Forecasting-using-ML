"""
landmask_chunked.py

Same behavior as your original landmask.py (drop land points via a
Natural Earth land shapefile, then sectorize into a GRID_ROWS x GRID_COLS
grid), restructured into two chunked passes so the full CSV never has to
be loaded into memory at once.

PASS 1 (get_ocean_lookup): scans the CSV in chunks, reading ONLY the
  latitude/longitude columns, to build the small set of unique
  coordinates and the overall bounding box. Runs the land mask once on
  that small unique-coordinate table (not the full file).

PASS 2 (filter_and_sectorize): scans the CSV again in chunks, this time
  with all columns, keeps only rows whose coordinates are in the ocean
  lookup from Pass 1, vectorizes the sector assignment (no more
  row-by-row .apply), and appends each chunk straight to disk.

IMPORTANT: OUTPUT_FILE must be different from CSV_FILE. Streaming
output while still streaming input from the same path will corrupt data.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

# ============================================================
# FILES / CONFIG
# ============================================================

CSV_FILE = r"E:\MLprojectbackup\noaa_aoml_atlantic_oceanwatch_AFAI_7D_7326_c8f5_d5ec.csv"
SHP_FILE = r"C:\Users\harsh.PC\Downloads\ne_110m_land\ne_110m_land.shp"

# Must NOT be the same path as CSV_FILE.
OUTPUT_FILE = r"C:\Users\harsh.PC\Downloads\noaa_aoml_atlantic_oceanwatch_AFAI_7D_ocean_sectorized.csv"

GRID_ROWS = 10
GRID_COLS = 20  # 2:1 ratio to roughly match lat/lon aspect ratio

CHUNK_SIZE = 500_000  # rows per chunk; lower this (e.g. 100_000) if memory is still tight

# If your ERDDAP export has a units row under the header, keep this. If
# it doesn't (check row 2 of the file), set to None.
SKIPROWS = [1]

assert OUTPUT_FILE != CSV_FILE, "OUTPUT_FILE must differ from CSV_FILE for chunked processing."

# ============================================================
# PASS 1: unique coordinates, bounding box, ocean mask
# ============================================================


def get_ocean_lookup():
    """
    Scans the file (lat/lon columns only) in chunks, deduplicating
    INCREMENTALLY against what's already been found so far -- this file
    is ordered time-major (the whole grid repeats every timestep), so
    duplicates mostly only show up ACROSS chunks, not within one. Waiting
    until the end to dedupe means holding the full un-deduped file in
    memory first, which is what crashed last time (2.1+ billion rows).
    """
    dtype = {"latitude": "float32", "longitude": "float32"}
    coordinates = pd.DataFrame(columns=["latitude", "longitude"]).astype(dtype)

    for i, chunk in enumerate(pd.read_csv(CSV_FILE, skiprows=SKIPROWS, usecols=["latitude", "longitude"], dtype=dtype, chunksize=CHUNK_SIZE)):
        chunk_unique = chunk.drop_duplicates()
        # keep only rows from this chunk not already in `coordinates`
        check = chunk_unique.merge(coordinates, on=["latitude", "longitude"], how="left", indicator=True)
        new_rows = check.loc[check["_merge"] == "left_only", ["latitude", "longitude"]]
        if len(new_rows):
            coordinates = pd.concat([coordinates, new_rows], ignore_index=True)
        if i % 20 == 0:
            print(f"  ...scanned {(i + 1) * CHUNK_SIZE:,} rows, {len(coordinates):,} unique coordinates so far")

    coordinates = coordinates.reset_index(drop=True)
    print("Unique coordinates:", len(coordinates))

    min_lat, max_lat = coordinates["latitude"].min(), coordinates["latitude"].max()
    min_lon, max_lon = coordinates["longitude"].min(), coordinates["longitude"].max()
    print(f"Data extent: Lat [{min_lat}, {max_lat}], Lon [{min_lon}, {max_lon}]")

    land = gpd.read_file(SHP_FILE).to_crs("EPSG:4326")
    land_geometry = land.geometry.union_all()
    prepared_land = prep(land_geometry)  # speeds up repeated point-in-polygon tests a lot at this scale

    points = [Point(lon, lat) for lat, lon in zip(coordinates["latitude"], coordinates["longitude"])]
    is_land = np.array([prepared_land.contains(pt) for pt in points])
    coordinates["ocean_mask"] = ~is_land

    ocean_coordinates = coordinates.loc[coordinates["ocean_mask"], ["latitude", "longitude"]].reset_index(drop=True)
    print("Ocean coordinates:", len(ocean_coordinates))

    return ocean_coordinates, min_lat, max_lat, min_lon, max_lon


# ============================================================
# PASS 2: filter + vectorized sectorize + streamed write
# ============================================================


def filter_and_sectorize(ocean_coordinates, min_lat, max_lat, min_lon, max_lon):
    lat_step = (max_lat - min_lat) / GRID_ROWS
    lon_step = (max_lon - min_lon) / GRID_COLS

    total_in, total_out = 0, 0
    first_chunk = True

    # latitude/longitude must be read with the SAME dtype (float32) used in
    # get_ocean_lookup(), otherwise the merge below can silently fail to
    # match rows due to float32 vs float64 rounding differences.
    dtype = {"latitude": "float32", "longitude": "float32"}
    for chunk in pd.read_csv(CSV_FILE, skiprows=SKIPROWS, dtype=dtype, chunksize=CHUNK_SIZE):
        total_in += len(chunk)

        ocean_chunk = chunk.merge(ocean_coordinates, on=["latitude", "longitude"], how="inner")

        if len(ocean_chunk):
            lat_c = ocean_chunk["latitude"].clip(min_lat, max_lat)
            lon_c = ocean_chunk["longitude"].clip(min_lon, max_lon)
            row_idx = np.floor((lat_c - min_lat) / lat_step).astype(int).clip(upper=GRID_ROWS - 1)
            col_idx = np.floor((lon_c - min_lon) / lon_step).astype(int).clip(upper=GRID_COLS - 1)
            ocean_chunk["sector"] = row_idx * GRID_COLS + col_idx

        total_out += len(ocean_chunk)
        ocean_chunk.to_csv(OUTPUT_FILE, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

    return total_in, total_out


def main():
    ocean_coordinates, min_lat, max_lat, min_lon, max_lon = get_ocean_lookup()
    total_in, total_out = filter_and_sectorize(ocean_coordinates, min_lat, max_lat, min_lon, max_lon)

    print("\nFinished!")
    print("Original rows:", total_in)
    print("Ocean-only rows:", total_out)
    print("Removed rows:", total_in - total_out)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()