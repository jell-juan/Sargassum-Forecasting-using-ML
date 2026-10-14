"""
combine_netcdf.py

Combines 4 NetCDF files into one xarray Dataset, removes the "mean wave
direction" variable (detected by name/attributes so it still gets caught
even if that file names it something other than "mean_wave_direction"),
and prints summary statistics for whatever variables remain.
"""

import xarray as xr

# ============================== CONFIG ======================================
# Edit these to your actual file paths. Order doesn't matter.
FILE_PATHS = [
    r"C:\Users\harsh.PC\Downloads\2e7e8fa032fdfbd843d154123c7e91b9\data_stream-wave_stepType-instant.nc",
    r"C:\Users\harsh.PC\Downloads\2e7e8fa032fdfbd843d154123c7e91b9\data_stream-oper_stepType-accum.nc",
    r"C:\Users\harsh.PC\Downloads\2e7e8fa032fdfbd843d154123c7e91b9\data_stream-oper_stepType-instant.nc",
    r"C:\Users\harsh.PC\Downloads\ML project\cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_85.00W-38.00W_0.00N-22.00N_0.49m_2018-01-01-2025-12-31.nc",
]
# =============================================================================

# Common short names/aliases seen for mean wave direction across sources
# (ERA5 calls it "mwd", some wave models use "VMDR", etc.). Matched
# case-insensitively as a whole variable name.
KNOWN_WAVE_DIRECTION_ALIASES = {"mwd", "vmdr", "meanwavedir", "mean_wave_dir", "wave_dir", "wavedir"}


def _is_wave_direction_var(name, var):
    """
    True if this variable looks like "mean wave direction", whether that's
    from its short name matching a known alias, or from "wave" AND
    "direction" both appearing in its name/long_name/standard_name.
    """
    if name.lower() in KNOWN_WAVE_DIRECTION_ALIASES:
        return True
    haystack = " ".join([
        name,
        str(var.attrs.get("long_name", "")),
        str(var.attrs.get("standard_name", "")),
    ]).lower()
    return "wave" in haystack and "direct" in haystack


def _standardize_dims(ds, label):
    """
    ERA5 files downloaded through the newer Copernicus CDS API name their
    time dimension "valid_time" instead of "time", and often carry extra
    "expver"/"number" coordinates. xr.merge aligns purely by dimension
    NAME, so a "time" vs "valid_time" mismatch between files doesn't
    error -- it silently creates two separate axes and merge blows the
    result up to their full cross product (this is what caused the
    multi-TB allocation error). Standardize every file's dims BEFORE
    merging so this can't happen.
    """
    rename_map = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "lat" in ds.dims:
        rename_map["lat"] = "latitude"
    if "lon" in ds.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    if "expver" in ds.dims:
        if ds.sizes["expver"] == 1:
            ds = ds.squeeze("expver", drop=True)
        else:
            # Mix of validated ERA5 (expver=1) and preliminary ERA5T (expver=5):
            # prefer the validated value, fall back to ERA5T where it's missing.
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    for extra in ("expver", "number"):
        if extra in ds.coords:
            ds = ds.drop_vars(extra)

    print(f"  [{label}] dims after standardizing: {dict(ds.sizes)}")
    return ds


def combine_netcdf_files(file_paths=FILE_PATHS):
    """
    Opens and merges all NetCDF files in file_paths into a single Dataset,
    then drops whichever variable(s) represent mean wave direction.

    Note: xr.merge aligns files on shared coordinate names (time,
    latitude, longitude, ...). If your 4 files are on different grids or
    resolutions, merging will still work but will produce NaNs wherever
    coordinates don't overlap exactly -- regrid to a common grid first if
    that's not what you want.
    """
    print("Opening and standardizing dimension names for each file:")
    datasets = [_standardize_dims(xr.open_dataset(path), path) for path in file_paths]
    combined = xr.merge(datasets, compat="no_conflicts", join="outer")
    print(f"\nMerged dataset dims: {dict(combined.sizes)}")

    wave_dir_vars = [name for name, var in combined.data_vars.items() if _is_wave_direction_var(name, var)]
    if wave_dir_vars:
        print(f"Dropping mean wave direction variable(s): {wave_dir_vars}")
        combined = combined.drop_vars(wave_dir_vars)
    else:
        print("No mean wave direction variable found to drop.")

    return combined


def print_summary_statistics(ds):
    """Prints count/mean/std/min/max for every remaining data variable."""
    for name, var in ds.data_vars.items():
        print(f"\n--- {name} ---")
        print(f"  count : {int(var.count())}")
        print(f"  mean  : {float(var.mean(skipna=True))}")
        print(f"  std   : {float(var.std(skipna=True))}")
        print(f"  min   : {float(var.min(skipna=True))}")
        print(f"  max   : {float(var.max(skipna=True))}")


if __name__ == "__main__":
    merged = combine_netcdf_files()
    print_summary_statistics(merged)

    merged.to_netcdf("combined_output.nc")
    print("\n[done] combined_output.nc")