"""
sort_csv_by_location_time.py

Sorts a CSV by (latitude, longitude, time) -- latitude and longitude as
one compound primary key, time as the secondary key within each pair --
e.g. all of (0,0)'s timesteps in order, then all of (0,1)'s, etc.

Uses an EXTERNAL MERGE SORT so it works even when the file is far bigger
than available RAM:

  PASS 1 (create_sorted_runs): read the CSV in chunks, sort each chunk
    fully in memory (always small enough to fit), write each sorted
    chunk out as its own temporary "run" file.

  PASS 2 (merge_sorted_runs): stream all the sorted run files at once,
    one row at a time each, and merge them into the final output with a
    heap-based merge (heapq.merge). This produces one fully sorted
    stream without ever loading more than a handful of rows into memory,
    no matter how large the whole file is. If there are more runs than
    is safe to hold open at once, they're merged in batches (a second,
    intermediate merge level) rather than opening thousands of files
    simultaneously.

DATA SAFETY: every column is read and written as raw text (dtype=str) --
latitude/longitude/time are only ever parsed into a SEPARATE, temporary
sort key, never written back out. This guarantees the output has the
exact same values as the input, just reordered -- no reformatting, no
precision changes, nothing lost.

DISK SPACE: this needs roughly 2x the input file's size free on disk
(the sorted run files, plus the final output) while it runs. Temp runs
are deleted as soon as they're no longer needed.
"""

import csv
import glob
import heapq
import os
import tempfile
from datetime import datetime

import pandas as pd

# ============================== CONFIG ======================================
INPUT_CSV = r"C:\Users\harsh.PC\Downloads\noaa_aoml_atlantic_oceanwatch_AFAI_7D_ocean_sectorized.csv"        # <-- set this to your actual file
OUTPUT_CSV = r"C:\Users\harsh.PC\Downloads\ML project\finalsort.csv"  # <-- this is what you'll "download"/keep

CHUNK_SIZE = 500_000   # rows sorted in memory per run; lower this if memory is tight
SKIPROWS = [1]         # ERDDAP units row -- set to None if your file doesn't have one
TEMP_DIR = None        # None = system default temp folder

# If Pass 1 already completed in an earlier run (e.g. it only crashed
# during the final write) and left sorted run files behind, point this at
# a glob pattern matching them to skip re-sorting from scratch entirely.
# Leave as None to run Pass 1 normally. Find your leftover files with:
#   python -c "import glob, tempfile, os; print('\n'.join(glob.glob(os.path.join(tempfile.gettempdir(), '*_run*.csv'))))"
EXISTING_RUN_PATTERN = r"C:\Users\harsh.PC\AppData\Local\Temp\*_L0_*.csv"  # e.g. r"C:\Users\harsh.PC\AppData\Local\Temp\*_run*.csv"

MAX_MERGE_FANIN = 100  # max run files merged simultaneously, to stay well under OS open-file limits

SORT_KEYS = ["latitude", "longitude", "time"]  # compound sort priority, exactly as requested
# =============================================================================


def create_sorted_runs(input_csv, skiprows, chunk_size, temp_dir):
    """Pass 1: read in chunks, sort each chunk (by a separate typed key), write each as its own sorted temp file."""
    run_paths = []
    reader = pd.read_csv(input_csv, skiprows=skiprows, chunksize=chunk_size, dtype=str)
    for i, chunk in enumerate(reader):
        sort_key = pd.DataFrame({
            "lat": chunk["latitude"].astype(float),
            "lon": chunk["longitude"].astype(float),
            "time": pd.to_datetime(chunk["time"]),
        })
        order = sort_key.sort_values(["lat", "lon", "time"]).index
        # reindex the ORIGINAL, untouched string columns -- values are never reformatted
        chunk = chunk.loc[order]

        fd, path = tempfile.mkstemp(suffix=f"_run{i}.csv", dir=temp_dir)
        os.close(fd)
        chunk.to_csv(path, index=False)
        run_paths.append(path)
        print(f"  [run {i}] {len(chunk):,} rows sorted -> {os.path.basename(path)}")
    return run_paths


def _row_stream(path):
    """Yields (sort_key, row_dict) for one sorted run file -- row_dict keeps the original raw strings."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                float(row["latitude"]),
                float(row["longitude"]),
                datetime.fromisoformat(row["time"].replace("Z", "+00:00")),
            )
            yield key, row


def merge_batch(run_paths, output_path, fieldnames):
    """Merges a batch of already-sorted run files into one file, streaming row by row."""
    streams = [_row_stream(p) for p in run_paths]
    merged = heapq.merge(*streams, key=lambda item: item[0])

    total = 0
    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in merged:
            writer.writerow(row)
            total += 1
            if total % 1_000_000 == 0:
                print(f"    ...written {total:,} rows")
    return total


def merge_sorted_runs(run_paths, output_csv, temp_dir):
    """Pass 2: merges all sorted runs into the final output, batching if there are too many to open at once."""
    with open(run_paths[0], newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames

    current_runs = run_paths
    level = 0
    while len(current_runs) > MAX_MERGE_FANIN:
        print(f"  {len(current_runs)} runs exceeds MAX_MERGE_FANIN ({MAX_MERGE_FANIN}) -- merging in batches (level {level})")
        next_runs = []
        for i in range(0, len(current_runs), MAX_MERGE_FANIN):
            batch = current_runs[i:i + MAX_MERGE_FANIN]
            fd, path = tempfile.mkstemp(suffix=f"_L{level}_{i}.csv", dir=temp_dir)
            os.close(fd)
            merge_batch(batch, path, fieldnames)
            next_runs.append(path)
        for p in current_runs:
            os.remove(p)
        current_runs = next_runs
        level += 1

    print(f"  Final merge of {len(current_runs)} run(s) -> {output_csv}")
    total = merge_batch(current_runs, output_csv, fieldnames)
    for p in current_runs:
        os.remove(p)
    return total


def main():
    output_dir = os.path.dirname(OUTPUT_CSV) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"OUTPUT_CSV's folder doesn't exist: {output_dir}\n"
            f"Fix OUTPUT_CSV in the CONFIG section before running -- this check runs first specifically "
            f"so a bad path fails immediately instead of after the whole sort completes."
        )

    print("Pass 1: splitting into sorted runs...")
    if EXISTING_RUN_PATTERN:
        run_paths = sorted(glob.glob(EXISTING_RUN_PATTERN))
        if not run_paths:
            raise FileNotFoundError(f"EXISTING_RUN_PATTERN matched no files: {EXISTING_RUN_PATTERN}")
        print(f"  Reusing {len(run_paths)} existing sorted run file(s), skipping re-sort.")
    else:
        run_paths = create_sorted_runs(INPUT_CSV, SKIPROWS, CHUNK_SIZE, TEMP_DIR)

    print(f"\nPass 2: merging {len(run_paths)} sorted run(s)...")
    total = merge_sorted_runs(run_paths, OUTPUT_CSV, TEMP_DIR)

    print(f"\n[done] {total:,} rows written to {OUTPUT_CSV}")
    print("Sort order: latitude -> longitude -> time (ascending)")


if __name__ == "__main__":
    main()