"""
ml_training_pipeline_fast.py

Fast, time-budgeted training pipeline (~25 min execution).
Retains all temporal purge (40-day), CV embargo (40-day), and spatial mitigation logic.
"""

import gc
import json
import time as _time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

# ============================== CONFIG ======================================
PARQUET_PATH = r"C:\Users\harsh.PC\Downloads\ML project\lagged_dataset.parquet"
OUTPUT_DIR = r"C:\Users\harsh.PC\Downloads\ML project"
TARGET = "AFAI_next_month"

# Total execution time target in minutes (Default: 25 minutes)
TOTAL_BUDGET_MINUTES = 25  
TOTAL_BUDGET_SECONDS = TOTAL_BUDGET_MINUTES * 60

TARGET_OFFSET_DAYS = 30
TARGET_TOLERANCE_DAYS = 10
MAX_TARGET_LEAD_DAYS = TARGET_OFFSET_DAYS + TARGET_TOLERANCE_DAYS

SAMPLE_FRACTION = 0.02
RECENCY_HALF_LIFE_DAYS = 730

EXCLUDE_FROM_FEATURES = ["time", "sector", "latitude", "longitude", TARGET]
COLUMNS_TO_SKIP_READING = ["sector"]

CORR_THRESHOLD = 0.95
N_SPLITS = 3
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

# Budget distribution
XGB_INITIAL_BUDGET_SHARE = 0.40  # 40% time to XGBoost, remainder to RF
SAFETY_MARGIN = 0.80
MIN_ITER, MAX_ITER_CAP = 3, 10   # Reduced cap for fast search
COMPLEXITY_MULTIPLIER = 2.5       # Accounts for deeper trees vs probe
# =============================================================================


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_split_threshold(path, train_fraction):
    """
    Memory-efficient pass to find the exact chronological boundary date.
    Extracts unique dates per row group to prevent RAM overflow on massive datasets.
    """
    pf = pq.ParquetFile(path)
    unique_dates = set()
    
    for i in range(pf.metadata.num_row_groups):
        # 1. Read only the time column for this chunk
        t_col = pf.read_row_group(i, columns=["time"]).column("time").to_pandas()
        
        # 2. Extract unique values locally and add to our master set
        unique_dates.update(t_col.dropna().unique())
        
        # Optional: Print progress on massive files so you know it's not hanging
        if (i + 1) % 10 == 0 or (i + 1) == pf.metadata.num_row_groups:
            print(f"    ... scanned time metadata in chunk {i + 1}/{pf.metadata.num_row_groups}", flush=True)
            
    # 3. Sort the final small list of unique dates
    unique_times = np.sort(list(unique_dates))
    split_idx = int(len(unique_times) * train_fraction)
    
    return pd.Timestamp(unique_times[split_idx])


def load_and_sample_eager(path, sample_frac, half_life_days, train_feature_cutoff):
    pf = pq.ParquetFile(path)
    columns_to_read = [c for c in pf.schema_arrow.names if c not in COLUMNS_TO_SKIP_READING]
    n_row_groups = pf.metadata.num_row_groups
    
    log(f"[loader] Beginning eager load & sample across {n_row_groups} row groups...")
    sampled_dfs = []
    
    for i in range(n_row_groups):
        df_batch = pf.read_row_group(i, columns=columns_to_read).to_pandas()
        
        # Fast datetime check/conversion
        if not pd.api.types.is_datetime64_any_dtype(df_batch["time"]):
            df_batch["time"] = pd.to_datetime(df_batch["time"])
        
        if sample_frac < 1.0:
            is_train = df_batch["time"] <= train_feature_cutoff
            weights = pd.Series(1.0, index=df_batch.index)
            
            if is_train.any():
                # Fast vector time difference in days
                age_days = (train_feature_cutoff - df_batch.loc[is_train, "time"]).dt.total_seconds() / 86400.0
                weights.loc[is_train] = 0.5 ** (age_days / half_life_days)
            
            df_batch = df_batch.sample(frac=sample_frac, weights=weights, random_state=RANDOM_STATE + i)
        
        if not df_batch.empty:
            # Downcast float64 -> float32 to reduce memory footprint by 50%
            float_cols = df_batch.select_dtypes(include=["float64"]).columns
            if len(float_cols) > 0:
                df_batch[float_cols] = df_batch[float_cols].astype(np.float32)
            sampled_dfs.append(df_batch)
        
        # Print progress every 5 chunks so you know exactly where it is
        if (i + 1) % 5 == 0 or (i + 1) == n_row_groups:
            log(f"    ... processed & sampled row group {i + 1}/{n_row_groups}")
            gc.collect()  # Throttle GC calls

    final_df = pd.concat(sampled_dfs, ignore_index=True)
    del sampled_dfs
    gc.collect()
    
    return final_df


def create_grouped_time_splits(df_train, n_splits, gap_days=40):
    unique_times = np.sort(df_train["time"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    custom_cv_splits = []
    
    for train_g_idx, test_g_idx in tscv.split(unique_times):
        val_start_time = unique_times[test_g_idx[0]]
        embargo_threshold = val_start_time - pd.Timedelta(days=gap_days)
        
        valid_train_times = unique_times[train_g_idx][unique_times[train_g_idx] <= embargo_threshold]
        test_times = unique_times[test_g_idx]
        
        train_idx = np.where(df_train["time"].isin(valid_train_times))[0]
        test_idx = np.where(df_train["time"].isin(test_times))[0]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            custom_cv_splits.append((train_idx, test_idx))
            
    return custom_cv_splits


def prune_redundant_features(X_train, threshold):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    return to_drop


def evaluate_split(name, split, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log(f"  {name} [{split}] MAE={mae:.5f}  RMSE={rmse:.5f}  R2={r2:.5f}")
    return {"Model": name, "Split": split, "MAE": mae, "RMSE": rmse, "R2": r2}


def adaptive_randomized_search(base_kwargs, model_class, param_distributions, X_train, y_train,
                               custom_cv_splits, time_budget_seconds, model_name):
    # Select median complexity parameter for probe to improve time estimation
    probe_params = {k: v[len(v)//2] for k, v in param_distributions.items()}
    probe_model = model_class(**base_kwargs, **probe_params)
    
    t0 = _time.time()
    probe_model.fit(X_train, y_train)
    probe_fit_time = _time.time() - t0
    del probe_model
    gc.collect()

    n_splits = len(custom_cv_splits)
    # Apply complexity safety factor to account for larger grid candidates
    cost_per_candidate = max(probe_fit_time * n_splits * COMPLEXITY_MULTIPLIER, 1e-4)
    
    n_iter = int((time_budget_seconds * SAFETY_MARGIN) // cost_per_candidate)
    n_iter = max(MIN_ITER, min(n_iter, MAX_ITER_CAP))
    
    log(f"  [{model_name}] Budget: {time_budget_seconds/60:.1f}m | Probe: {probe_fit_time:.2f}s | Running {n_iter} iterations...")

    search = RandomizedSearchCV(
        model_class(**base_kwargs), param_distributions, n_iter=n_iter, cv=custom_cv_splits,
        scoring="neg_mean_absolute_error", n_jobs=1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def main():
    budget_start = _time.time()
    log(f"[setup] Starting fast training run. Target budget: {TOTAL_BUDGET_MINUTES} minutes.")

    nominal_cutoff_time = get_split_threshold(PARQUET_PATH, TRAIN_FRACTION)
    train_feature_cutoff = nominal_cutoff_time - pd.Timedelta(days=MAX_TARGET_LEAD_DAYS)

    df = load_and_sample_eager(PARQUET_PATH, SAMPLE_FRACTION, RECENCY_HALF_LIFE_DAYS, train_feature_cutoff)
    df = df.dropna(subset=[TARGET]).sort_values("time").reset_index(drop=True)

    train_mask = df["time"] <= train_feature_cutoff
    test_mask = df["time"] >= nominal_cutoff_time
    
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    custom_cv_splits = create_grouped_time_splits(df_train, N_SPLITS, gap_days=MAX_TARGET_LEAD_DAYS)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    X_train, y_train = df_train[feature_cols].copy(), df_train[TARGET].copy()
    X_test, y_test = df_test[feature_cols].copy(), df_test[TARGET].copy()
    
    persistence_train_feat = df_train["AFAI_t"] if "AFAI_t" in df_train else df_train["AFAI"]
    persistence_test_feat = df_test["AFAI_t"] if "AFAI_t" in df_test else df_test["AFAI"]
    
    del df, df_train, df_test
    gc.collect()

    cols_to_drop = prune_redundant_features(X_train, CORR_THRESHOLD)
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

    results = []
    best_params_dict = {}

    # PERSISTENCE BASELINE
    log("=" * 50 + "\nDIAGNOSTIC: Persistence Baseline\n" + "=" * 50)
    results.append(evaluate_split("Persistence", "train", y_train, persistence_train_feat))
    results.append(evaluate_split("Persistence", "test", y_test, persistence_test_feat))

    # 1. XGBOOST (Fast Grid)
    log("=" * 50 + "\n1/3: XGBoost\n" + "=" * 50)
    remaining = TOTAL_BUDGET_SECONDS - (_time.time() - budget_start)
    xgb_budget = remaining * XGB_INITIAL_BUDGET_SHARE
    xgb_base = dict(random_state=RANDOM_STATE, tree_method="hist", device="cuda", n_jobs=1)
    xgb_params = {
        "n_estimators": [100, 150],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_search = adaptive_randomized_search(xgb_base, XGBRegressor, xgb_params, X_train, y_train,
                                            custom_cv_splits, xgb_budget, "XGBoost")
    best_params_dict["XGBoost"] = xgb_search.best_params_
    results.append(evaluate_split("XGBoost", "train", y_train, xgb_search.predict(X_train)))
    results.append(evaluate_split("XGBoost", "test", y_test, xgb_search.predict(X_test)))
    joblib.dump(xgb_search.best_estimator_, f"{OUTPUT_DIR}\\xgboost_best_model.joblib")

    # 2. LINEAR REGRESSION (Fast)
    log("=" * 50 + "\n2/3: Linear Regression\n" + "=" * 50)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate_split("Linear Regression", "train", y_train, lr.predict(X_train)))
    results.append(evaluate_split("Linear Regression", "test", y_test, lr.predict(X_test)))
    joblib.dump(lr, f"{OUTPUT_DIR}\\linear_regression_model.joblib")

    # 3. RANDOM FOREST (Fast Bounded Grid)
    log("=" * 50 + "\n3/3: Random Forest\n" + "=" * 50)
    remaining = max(60, TOTAL_BUDGET_SECONDS - (_time.time() - budget_start))
    rf_base = dict(random_state=RANDOM_STATE, n_jobs=-1)  
    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15],  # Bounded depth avoids CPU thrashing
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt", 0.7],
        "max_samples": [0.7],
    }
    rf_search = adaptive_randomized_search(rf_base, RandomForestRegressor, rf_params, X_train, y_train,
                                            custom_cv_splits, remaining, "Random Forest")
    best_params_dict["RandomForest"] = rf_search.best_params_
    
    results.append(evaluate_split("Random Forest", "train", y_train, rf_search.predict(X_train)))
    results.append(evaluate_split("Random Forest", "test", y_test, rf_search.predict(X_test)))
    joblib.dump(rf_search.best_estimator_, f"{OUTPUT_DIR}\\random_forest_best_model.joblib")

    # EXPORT RESULTS
    elapsed = (_time.time() - budget_start) / 60.0
    log(f"\nPipeline finished in {elapsed:.2f} minutes.")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}\\model_comparison_results.csv", index=False)
    with open(f"{OUTPUT_DIR}\\best_model_params.json", "w") as f:
        json.dump(best_params_dict, f, indent=4, default=str)


if __name__ == "__main__":
    main()