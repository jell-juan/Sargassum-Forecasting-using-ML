"""
build_lagged_dataset.py

Merges the combined environmental NetCDF (all variables it contains) with
the AFAI ERDDAP CSV export, engineers t/t-1/t-2/t-3/t-4 lagged features
for AFAI AND every environmental variable, plus a "1 month ahead" AFAI
target, and writes the result to a single output CSV -- reading the huge
AFAI CSV in batches rather than all at once.

WHY THIS ISN'T JUST "ADD chunksize="
  Lag features need a location's OWN PAST rows; the target needs a
  location's FUTURE row (~1 month ahead). Neither is available if you only
  ever hold one small chunk in memory. Since the AFAI export is ordered
  time-major (confirmed by how big it turned out to be when we tried to
  fully load it earlier -- the same grid repeats every timestep), this
  script instead buffers a SLIDING WINDOW of just enough consecutive
  timesteps to cover 4 steps of lookback and ~1 month of lookahead. Once a
  given timestep's window is complete, its rows are finalized and written,
  and older buffered timesteps are dropped. Memory scales with (a handful
  of timesteps x rows-per-timestep), not with the size of the whole file.

  Spatial matching (nearest environmental grid cell per AFAI point, via
  KD-tree) works exactly as before, just applied per chunk instead of once
  on the full file.

CAVEAT: if a single timestep's row count is itself huge (millions of ocean
cells), the buffer can still be sizeable (roughly 10 timesteps' worth held
at once). If that's still too much for your machine, run the ocean-masking
script (landmask_chunked.py) first and point CSV_PATH at its smaller,
ocean-only output instead of the raw AFAI export.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ============================== CONFIG ======================================
NC_PATH = r"C:\Users\harsh.PC\Downloads\ML project\combined_output.nc"
CSV_PATH = r"C:\Users\harsh.PC\Downloads\noaa_aoml_atlantic_oceanwatch_AFAI_7D_ocean_sectorized.csv"
OUTPUT_CSV = r"C:\Users\harsh.PC\Downloads\ML project\lagged_dataset.csv"

CSV_CHUNK_SIZE = 500_000  # AFAI rows read per batch; lower this if memory is still tight
SKIPROWS = [1]            # ERDDAP's units row right under the header; set to None if your file doesn't have one

# Coordinate-name standardization only. Variable names are picked up
# automatically from whatever combined_output.nc actually contains.
COORD_RENAME = {"lat": "latitude", "lon": "longitude"}
CSV_RENAME = {"afai": "AFAI", "AFAI": "AFAI"}

LAG_STEPS = [1, 2, 3, 4]  # every variable (AFAI + every env variable) gets these, plus lag 0 ("_t")

MERGE_TOLERANCE = pd.Timedelta("4D")    # max gap allowed when attaching env data to an AFAI timestamp
TARGET_OFFSET = pd.DateOffset(months=1)
TARGET_TOLERANCE = pd.Timedelta("10D")  # max gap allowed when looking up the "~1 month ahead" AFAI value
LOOKBACK_RETENTION = max(LAG_STEPS)     # how many already-finalized timesteps to keep buffered for future lookback
SAFETY_MAX_PENDING = 500                # abort with a diagnostic instead of silently growing the buffer forever

# =============================================================================


def load_environment():
    """
    Opens the NetCDF WITHOUT materializing it into a DataFrame. Since it's
    a regular lat/lon/time grid, we only need the 1D coordinate arrays
    (tiny) to build the spatial lookup; the actual variable VALUES are
    read point-by-point later, per AFAI chunk, so the full multi-variable
    cube is never held in memory at once.
    """
    ds = xr.open_dataset(NC_PATH)
    ds = ds.rename({k: v for k, v in COORD_RENAME.items() if k in ds.dims or k in ds.coords})
    if not {"time", "latitude", "longitude"}.issubset(ds.coords):
        raise ValueError(f"combined_output.nc is missing time/latitude/longitude after renaming -- got {list(ds.coords)}")

    # Leftover singleton dims (e.g. "depth" from the currents product, once
    # only the surface layer was kept) would otherwise silently produce
    # (N, 1)-shaped arrays later instead of (N,) -- collapse them now.
    singleton_dims = [d for d in ds.dims if d not in ("time", "latitude", "longitude") and ds.sizes[d] == 1]
    if singleton_dims:
        print(f"[nc] dropping leftover singleton dimension(s): {singleton_dims}")
        ds = ds.squeeze(singleton_dims, drop=True)

    # The merge + to_netcdf round-trip in combine_netcdf.py can lose the
    # time coordinate's units/calendar metadata, leaving it as raw numbers
    # instead of real timestamps when the file is reopened. Detect and fix.
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        print(f"[nc] WARNING: 'time' wasn't auto-decoded (dtype={ds['time'].dtype}, attrs={ds['time'].attrs}). Decoding manually.")
        decoded = xr.decode_cf(ds[["time"]])
        if not np.issubdtype(decoded["time"].dtype, np.datetime64):
            raise ValueError(
                "Could not decode combined_output.nc's time coordinate into real dates -- its units/calendar "
                "metadata appears to be missing entirely (likely lost during the earlier xr.merge/to_netcdf "
                "step). Regenerate combined_output.nc with the updated combine_netcdf.py, which now sets "
                "explicit time encoding on write."
            )
        ds = ds.assign_coords(time=decoded["time"])
        print(f"[nc] decoded time range: {ds['time'].values.min()} to {ds['time'].values.max()} -- "
              f"check this looks right before trusting the results.")

    env_vars = list(ds.data_vars)
    print(f"[nc] {NC_PATH}: dims={dict(ds.sizes)}, detected environmental variables: {env_vars}")

    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    env_points = pd.DataFrame({"latitude": lat_grid.ravel(), "longitude": lon_grid.ravel()})
    tree = cKDTree(env_points[["latitude", "longitude"]].to_numpy())

    env_times = np.sort(ds["time"].values)
    return ds, env_points, tree, env_times, env_vars


def attach_environment(chunk, ds, env_points, tree, env_times, env_vars):
    """
    For each row in this chunk: nearest environmental grid cell (exact
    match against the NetCDF's own coordinate values, via KD-tree) and
    nearest available timestamp (within MERGE_TOLERANCE, found manually
    so we can do one exact vectorized xarray lookup instead of relying on
    .sel's tolerance, which doesn't mix cleanly across a datetime dim and
    float dims in a single vectorized call). Only the exact points needed
    are read from disk -- the full cube is never expanded.
    """
    chunk = chunk.copy()

    _, idx = tree.query(chunk[["latitude", "longitude"]].to_numpy(), k=1)
    matched_lat = env_points.loc[idx, "latitude"].to_numpy()
    matched_lon = env_points.loc[idx, "longitude"].to_numpy()

    chunk_times = chunk["time"].to_numpy()
    pos = np.searchsorted(env_times, chunk_times)
    pos = np.clip(pos, 1, len(env_times) - 1)
    left, right = env_times[pos - 1], env_times[pos]
    use_right = np.abs(right - chunk_times) < np.abs(chunk_times - left)
    matched_time = np.where(use_right, right, left)
    within_tolerance = np.abs(matched_time - chunk_times) <= np.timedelta64(MERGE_TOLERANCE)

    selected = ds[env_vars].sel(
        time=xr.DataArray(matched_time, dims="points"),
        latitude=xr.DataArray(matched_lat, dims="points"),
        longitude=xr.DataArray(matched_lon, dims="points"),
    )
    for var in env_vars:
        values = selected[var].values.astype("float64")
        values[~within_tolerance] = np.nan
        chunk[var] = values

    return chunk


def add_lag_columns(df, base_vars):
    """t/t-1/.../t-4 for every variable in base_vars, grouped per (latitude, longitude)."""
    df = df.sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)
    grouped = df.groupby(["latitude", "longitude"])
    for var in base_vars:
        df[f"{var}_t"] = df[var]
        for lag in LAG_STEPS:
            df[f"{var}_t-{lag}"] = grouped[var].shift(lag)
    return df


def finalize_timestep(anchor_time, buffer, base_vars, out_cols):
    """Computes lags + target for one timestep's rows, using the current buffer window."""
    lagged = add_lag_columns(buffer, base_vars)

    target_lookup = buffer[["time", "latitude", "longitude", "AFAI"]].copy()
    target_lookup["time"] -= TARGET_OFFSET
    target_lookup = target_lookup.rename(columns={"AFAI": "AFAI_next_month"}).sort_values("time")

    lagged = lagged.sort_values("time")
    with_target = pd.merge_asof(
        lagged, target_lookup,
        on="time", by=["latitude", "longitude"],
        direction="nearest", tolerance=TARGET_TOLERANCE,
    )

    anchor_rows = with_target[with_target["time"] == anchor_time]
    return anchor_rows[out_cols].dropna().reset_index(drop=True)


def main():
    env_ds, env_points, tree, env_times, env_vars = load_environment()
    base_vars = ["AFAI"] + env_vars

    out_cols = ["time", "latitude", "longitude"]
    for var in base_vars:
        out_cols.append(f"{var}_t")
        for lag in LAG_STEPS:
            out_cols.append(f"{var}_t-{lag}")
    out_cols.append("AFAI_next_month")
    print(f"\nFinal columns ({len(out_cols)}):")
    for c in out_cols:
        print(f"  {c}")
    print()

    buffer = pd.DataFrame()
    pending_times, finalized_times = [], []
    first_write = True
    total_rows, rows_written = 0, 0
    afai_seen, afai_present = 0, 0

    reader = pd.read_csv(CSV_PATH, skiprows=SKIPROWS, chunksize=CSV_CHUNK_SIZE)
    for chunk_num, chunk in enumerate(reader):
        chunk = chunk.rename(columns=CSV_RENAME)
        chunk["time"] = pd.to_datetime(chunk["time"])
        if chunk["time"].dt.tz is not None:
            # ERDDAP timestamps end in "Z", which makes pandas parse them as
            # timezone-aware. A tz-aware column's .to_numpy() silently
            # returns boxed Timestamp objects instead of real datetime64,
            # which is exactly what broke searchsorted() against the
            # NetCDF's plain (naive) datetime64 array. Strip the tz label
            # without shifting the values (they're already UTC).
            chunk["time"] = chunk["time"].dt.tz_localize(None)
        total_rows += len(chunk)

        # Data-completeness metric: how much of the raw AFAI signal itself is present
        # (independent of environmental matching or lag/target availability).
        afai_seen += len(chunk)
        afai_present += int(chunk["AFAI"].notna().sum())

        chunk = attach_environment(chunk, env_ds, env_points, tree, env_times, env_vars)
        buffer = pd.concat([buffer, chunk], ignore_index=True)

        new_times = set(chunk["time"].unique()) - set(pending_times) - set(finalized_times)
        pending_times = sorted(set(pending_times) | new_times)

        # Safety net: if AFAI timestamps aren't perfectly identical across every
        # grid cell in a "snapshot" (e.g. sub-second/day drift per pixel), every
        # slightly-different value becomes its own never-resolving pending
        # anchor and the buffer grows without bound instead of ever trimming.
        # Fail loudly with real numbers instead of silently OOMing.
        if len(pending_times) > SAFETY_MAX_PENDING:
            oldest, newest = pending_times[0], pending_times[-1]
            raise RuntimeError(
                f"pending_times grew to {len(pending_times)} distinct timestamps (buffer has {len(buffer):,} rows) "
                f"without ever finalizing -- oldest pending: {oldest}, newest: {newest}, buffer max time: {buffer['time'].max()}. "
                f"This usually means timestamps aren't exactly identical across all grid cells in a nominal "
                f"composite period (check a few raw values with chunk['time'].value_counts() on a sample). "
                f"If so, round 'time' to the nearest expected period (e.g. .dt.floor('D')) right after parsing, "
                f"before it's used for grouping."
            )

        # finalize every leading pending timestep whose lookahead window is now fully buffered
        while pending_times:
            anchor = pd.Timestamp(pending_times[0])
            buffer_max = pd.Timestamp(buffer["time"].max())
            needed_until = anchor + TARGET_OFFSET + TARGET_TOLERANCE
            if buffer_max < needed_until:
                break
            result = finalize_timestep(anchor, buffer, base_vars, out_cols)
            result.to_csv(OUTPUT_CSV, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            rows_written += len(result)
            finalized_times.append(anchor)
            pending_times.pop(0)

        # trim buffer: keep only the most recent LOOKBACK_RETENTION finalized timesteps + all pending ones
        if len(finalized_times) > LOOKBACK_RETENTION:
            drop_times = set(finalized_times[:-LOOKBACK_RETENTION])
            buffer = buffer[~buffer["time"].isin(drop_times)]
            finalized_times = finalized_times[-LOOKBACK_RETENTION:]

        if chunk_num % 20 == 0:
            oldest_pending = pd.Timestamp(pending_times[0]) if pending_times else None
            buffer_max = pd.Timestamp(buffer["time"].max()) if len(buffer) else None
            gap = (oldest_pending + TARGET_OFFSET + TARGET_TOLERANCE - buffer_max) if oldest_pending is not None else None
            print(f"  ...processed {total_rows:,} rows | buffer: {len(buffer):,} rows, {len(pending_times)} distinct pending timesteps "
                  f"| oldest pending: {oldest_pending}, needs data until {oldest_pending + TARGET_OFFSET + TARGET_TOLERANCE if oldest_pending is not None else None}, "
                  f"buffer currently at {buffer_max} (gap remaining: {gap}) | {rows_written:,} rows finalized so far")

    # end of stream: finalize whatever's left (tail rows simply won't have a target -> dropped by dropna)
    for anchor in pending_times:
        result = finalize_timestep(anchor, buffer, base_vars, out_cols)
        result.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        rows_written += len(result)

    pct_present = 100 * afai_present / afai_seen if afai_seen else 0.0
    print(f"\nAFAI (t=0) completeness: {afai_present:,}/{afai_seen:,} rows present ({pct_present:.2f}%), "
          f"{100 - pct_present:.2f}% missing")
    print(f"Total input rows: {total_rows:,}")
    print(f"Final usable (lagged + targeted) rows: {rows_written:,}")
    print(f"[done] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()