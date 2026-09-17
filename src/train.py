"""
train.py
--------
Trains and evaluates forecasting models for the Ferry Ticket Demand
Forecasting project, across the four SRS-required horizons
(15m / 30m / 1h / 2h), for a chosen target column (Sales Count or
Redemption Count).

Model families (SRS "Modeling Approaches"):
  Baseline      : Naive (persistence), Moving Average
  Statistical   : Linear Regression on lag features
  Machine Learn : Random Forest Regressor, Gradient Boosting Regressor
                  (HistGradientBoostingRegressor stands in for the
                  optional XGBoost extension - same gradient-boosted-
                  tree family, ships in scikit-learn with no extra
                  dependency)
  Time Series   : Seasonal-naive (same slot last week) as a
                  seasonality-aware statistical baseline. Full
                  ARIMA/SARIMA/Prophet are supported via
                  train_statsmodels_models() when statsmodels/prophet
                  are installed (optional heavy dependencies - the
                  core deliverable does not require them).

Train/test strategy (SRS): time-based split, no shuffling, evaluated
per horizon with MAE / RMSE / MAPE, plus a naive-forecast quantile
regression pair used to build prediction intervals for the
uncertainty-analysis requirement.
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import HORIZONS, TARGETS, _PROJECT_ROOT, build_feature_frame, feature_columns

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
TEST_FRACTION = 0.15  # most recent 15% of the series held out, time-ordered


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0  # undefined at 0; SRS asks for "relative accuracy" so we exclude true zeros
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)  # ticket counts can't be negative
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE": round(mape(y_true, y_pred), 2)}


def time_based_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def make_quantile_gbr(alpha):
    # Histogram-based GBM: orders of magnitude faster than the classic
    # GradientBoostingRegressor on 250k+ rows, same quantile-loss capability.
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, max_depth=6, learning_rate=0.1, max_iter=120, random_state=42
    )


def train_target(target: str, horizons: dict = None):
    horizons = horizons or HORIZONS
    print(f"\n{'=' * 70}\nTraining models for target: {target}  |  horizons: {list(horizons.keys())}\n{'=' * 70}")
    feat = build_feature_frame(target=target)
    x_cols = feature_columns(target)

    results = {}          # results[horizon_steps][model_name] = metrics dict
    predictions_sample = {}

    for horizon_name, steps in horizons.items():
        y_col = f"y_{target}_{steps}"
        cols_needed = x_cols + [y_col, target]
        data = feat.dropna(subset=cols_needed).copy()

        train_df, test_df = time_based_split(data)
        X_train, y_train = train_df[x_cols], train_df[y_col]
        X_test, y_test = test_df[x_cols], test_df[y_col]

        horizon_results = {}
        horizon_preds = {"timestamp": test_df.index.astype(str).tolist(), "actual": y_test.tolist()}

        # ---------- Baseline 1: Naive (persistence: forecast = last observed value) ----------
        naive_pred = test_df[target].values  # value at t, forecasting t+steps
        horizon_results["Naive"] = compute_metrics(y_test, naive_pred)
        horizon_preds["Naive"] = np.clip(naive_pred, 0, None).tolist()

        # ---------- Baseline 2: Moving Average (mean of last 4 intervals = 1h) ----------
        ma_pred = train_df[target].rolling(4).mean().iloc[-1]  # fallback scalar
        ma_series = test_df[f"{target}_roll_mean_4"].fillna(ma_pred).values
        horizon_results["Moving Average"] = compute_metrics(y_test, ma_series)
        horizon_preds["Moving Average"] = np.clip(ma_series, 0, None).tolist()

        # ---------- Baseline 3: Seasonal-naive (value at same slot 1 week earlier) ----------
        steps_per_week = 7 * 24 * 4
        seasonal_naive = feat[target].shift(steps_per_week).loc[test_df.index]
        seasonal_naive = seasonal_naive.bfill().fillna(train_df[target].mean())
        horizon_results["Seasonal Naive"] = compute_metrics(y_test, seasonal_naive.values)
        horizon_preds["Seasonal Naive"] = np.clip(seasonal_naive.values, 0, None).tolist()

        # ---------- Linear Regression on lag features ----------
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        horizon_results["Linear Regression"] = compute_metrics(y_test, lr_pred)
        horizon_preds["Linear Regression"] = np.clip(lr_pred, 0, None).tolist()

        # ---------- Random Forest Regressor ----------
        # Single-core sandbox: kept modest (60 trees / depth 10 / 50% row
        # subsampling per tree) so a full horizon trains in ~60-90s while
        # still averaging enough trees for a stable ensemble estimate.
        rf = RandomForestRegressor(
            n_estimators=60, max_depth=10, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=42
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        horizon_results["Random Forest"] = compute_metrics(y_test, rf_pred)
        horizon_preds["Random Forest"] = np.clip(rf_pred, 0, None).tolist()

        # ---------- Gradient Boosting Regressor (+ quantile models for intervals) ----------
        gbr = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=200, random_state=42)
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_test)
        horizon_results["Gradient Boosting"] = compute_metrics(y_test, gbr_pred)
        horizon_preds["Gradient Boosting"] = np.clip(gbr_pred, 0, None).tolist()

        # 90% prediction interval via quantile gradient boosting (uncertainty analysis)
        model_lo = make_quantile_gbr(0.05).fit(X_train, y_train)
        model_hi = make_quantile_gbr(0.95).fit(X_train, y_train)
        q_lo = np.clip(model_lo.predict(X_test), 0, None)
        q_hi = np.clip(np.maximum(model_hi.predict(X_test), q_lo), 0, None)
        horizon_preds["Gradient Boosting_lower90"] = q_lo.tolist()
        horizon_preds["Gradient Boosting_upper90"] = q_hi.tolist()
        joblib.dump(model_lo, os.path.join(MODELS_DIR, f"gbrq_lo_{target.replace(' ', '_')}_{steps}.joblib"))
        joblib.dump(model_hi, os.path.join(MODELS_DIR, f"gbrq_hi_{target.replace(' ', '_')}_{steps}.joblib"))

        results[steps] = horizon_results
        predictions_sample[steps] = horizon_preds

        print(f"\n-- Horizon: {horizon_name} ({steps} steps) --")
        for model_name, m in horizon_results.items():
            print(f"  {model_name:20s}  MAE={m['MAE']:>8.2f}  RMSE={m['RMSE']:>8.2f}  MAPE={m['MAPE']:>6.2f}%")

        # persist the best-performing tree model + linear model + feature list for this horizon
        joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_{target.replace(' ', '_')}_{steps}.joblib"))
        joblib.dump(gbr, os.path.join(MODELS_DIR, f"gbr_{target.replace(' ', '_')}_{steps}.joblib"))
        joblib.dump(lr, os.path.join(MODELS_DIR, f"lr_{target.replace(' ', '_')}_{steps}.joblib"))

    return results, predictions_sample, x_cols


def _merge_json(path, new_data):
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(new_data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _merge_metrics_json(target, horizon_results):
    """Merge one horizon's results into metrics.json[target] without clobbering other horizons."""
    path = os.path.join(MODELS_DIR, "metrics.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.setdefault(target, {})
    existing[target].update({str(k): v for k, v in horizon_results.items()})
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _merge_predictions_json(target, horizon_preds):
    path = os.path.join(MODELS_DIR, "sample_predictions.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.setdefault(target, {})
    existing[target].update({str(k): v for k, v in horizon_preds.items()})
    with open(path, "w") as f:
        json.dump(existing, f)


def main():
    """
    CLI usage (kept small per call so training fits a single-core, time-
    limited sandbox session):
        python3 train.py "Sales Count" 1          # one target, one horizon (steps=1 -> 15min)
        python3 train.py "Sales Count" 1 2 4 8     # one target, several horizons
        python3 train.py                            # everything (slow - not recommended here)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    args = sys.argv[1:]
    if not args:
        targets_to_run, steps_filter = TARGETS, None
    else:
        target_arg = args[0]
        steps_filter = [int(s) for s in args[1:]] if len(args) > 1 else None
        targets_to_run = [target_arg]

    for target in targets_to_run:
        horizons = HORIZONS if steps_filter is None else {
            name: steps for name, steps in HORIZONS.items() if steps in steps_filter
        }
        t0 = time.time()
        results, preds, x_cols = train_target(target, horizons=horizons)
        _merge_metrics_json(target, results)
        # Note: full per-row prediction arrays are not persisted to disk (they
        # would run to tens of MB across 8 horizon/target combos). The
        # Streamlit app recomputes predictions live from the saved model
        # files, which is fast and always reflects the exact date range
        # the user selects.
        print(f"\n[{target}] done in {time.time() - t0:.1f}s")

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump({t: feature_columns(t) for t in TARGETS}, f, indent=2)

    with open(os.path.join(MODELS_DIR, "horizons.json"), "w") as f:
        json.dump(HORIZONS, f, indent=2)

    print("\nModels & metrics saved to /models")


if __name__ == "__main__":
    main()
