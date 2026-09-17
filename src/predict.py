"""
predict.py
----------
Loads trained models and computes forecasts / backtests for the
Streamlit dashboard. Kept separate from app.py so the prediction
logic is easy to unit-test or reuse outside Streamlit.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

MODEL_LABELS = {
    "Naive": "Naive (last observed value)",
    "Moving Average": "Moving Average (1h window)",
    "Seasonal Naive": "Seasonal Naive (same slot last week)",
    "Linear Regression": "Linear Regression",
    "Random Forest": "Random Forest Regressor",
    "Gradient Boosting": "Gradient Boosting Regressor",
}

TREE_MODEL_FILE_PREFIX = {"Random Forest": "rf", "Gradient Boosting": "gbr", "Linear Regression": "lr"}


def load_metrics() -> dict:
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        return json.load(f)


def load_horizons() -> dict:
    with open(os.path.join(MODELS_DIR, "horizons.json")) as f:
        return json.load(f)


def load_feature_columns() -> dict:
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        return json.load(f)


def _slug(target: str) -> str:
    return target.replace(" ", "_")


def load_ml_model(target: str, steps: int, model_name: str):
    prefix = TREE_MODEL_FILE_PREFIX[model_name]
    path = os.path.join(MODELS_DIR, f"{prefix}_{_slug(target)}_{steps}.joblib")
    return joblib.load(path)


def load_quantile_models(target: str, steps: int):
    lo = joblib.load(os.path.join(MODELS_DIR, f"gbrq_lo_{_slug(target)}_{steps}.joblib"))
    hi = joblib.load(os.path.join(MODELS_DIR, f"gbrq_hi_{_slug(target)}_{steps}.joblib"))
    return lo, hi


def baseline_predict(model_name: str, feat: pd.DataFrame, target: str, at_index, steps: int):
    """Compute a single-point baseline forecast made *as of* at_index, for `steps` ahead."""
    if model_name == "Naive":
        return float(feat.loc[at_index, target])
    if model_name == "Moving Average":
        val = feat.loc[at_index, f"{target}_roll_mean_4"]
        return float(val) if pd.notna(val) else float(feat.loc[:at_index, target].tail(4).mean())
    if model_name == "Seasonal Naive":
        target_time = at_index - pd.Timedelta(minutes=15 * steps) + pd.Timedelta(days=7)
        if target_time in feat.index:
            return float(feat.loc[target_time, target])
        return float(feat.loc[:at_index, target].mean())
    raise ValueError(model_name)


def ml_predict(model_name: str, feat: pd.DataFrame, x_cols: list, target: str, at_index, steps: int):
    row = feat.loc[[at_index], x_cols]
    model = load_ml_model(target, steps, model_name)
    pred = model.predict(row)[0]
    return float(max(pred, 0))


def point_forecast(model_name: str, feat: pd.DataFrame, x_cols: list, target: str, at_index, steps: int):
    if model_name in ("Naive", "Moving Average", "Seasonal Naive"):
        return baseline_predict(model_name, feat, target, at_index, steps)
    return ml_predict(model_name, feat, x_cols, target, at_index, steps)


def confidence_interval(feat: pd.DataFrame, x_cols: list, target: str, at_index, steps: int):
    """90% prediction interval from the quantile Gradient Boosting models."""
    lo_model, hi_model = load_quantile_models(target, steps)
    row = feat.loc[[at_index], x_cols]
    lo = max(float(lo_model.predict(row)[0]), 0)
    hi = max(float(hi_model.predict(row)[0]), lo)
    return lo, hi


def backtest_series(model_name: str, feat: pd.DataFrame, x_cols: list, target: str, steps: int,
                     start, end) -> pd.DataFrame:
    """
    Predicted-vs-actual over a date range, at the given horizon.
    Uses direct multi-horizon models trained on the held-out (later,
    time-ordered) part of the series, so this is a genuine backtest,
    not an in-sample fit.
    """
    y_col = f"y_{target}_{steps}"
    window = feat.loc[start:end].dropna(subset=x_cols + [y_col])
    if window.empty:
        return pd.DataFrame(columns=["timestamp", "actual", "predicted"])

    if model_name in ("Naive", "Moving Average", "Seasonal Naive"):
        if model_name == "Naive":
            preds = window[target].values
        elif model_name == "Moving Average":
            preds = window[f"{target}_roll_mean_4"].fillna(window[target].mean()).values
        else:
            steps_per_week = 7 * 24 * 4
            seasonal = feat[target].shift(steps_per_week).loc[window.index]
            preds = seasonal.bfill().fillna(window[target].mean()).values
    else:
        model = load_ml_model(target, steps, model_name)
        preds = model.predict(window[x_cols])

    preds = np.clip(preds, 0, None)
    forecast_time = window.index + pd.Timedelta(minutes=15 * steps)
    return pd.DataFrame({
        "timestamp": forecast_time,
        "actual": window[y_col].values,
        "predicted": preds,
    })


def kpis_from_backtest(bt: pd.DataFrame, peak_quantile: float = 0.9) -> dict:
    """
    Compute the SRS Key Performance Indicators from a predicted-vs-actual
    backtest slice:
      - Forecast Accuracy (%): 100 - WAPE (see note below)
      - Error Drift: RMSE trend proxy = RMSE(second half) - RMSE(first half)
      - Peak Miss Rate: share of true demand spikes (top decile) the
        model under-predicted by more than 25%
      - Confidence Band Width: mean(upper90 - lower90) if present

    Note on the accuracy metric: at 15-minute resolution, ~32% of intervals
    are true zeros and the non-zero intervals are mostly single-digit ticket
    counts (median redemption count in an active slot is small). Classic
    point-wise MAPE = mean(|actual-pred|/actual) explodes past 100% on series
    like this (being off by 2 tickets on an actual of 1 is a 200% error), so
    `100 - MAPE` is negative almost everywhere and the KPI floors at 0 for
    every model/window - it stops being able to tell models apart, which
    defeats the purpose of the KPI.

    WAPE (a.k.a. MAD/Mean ratio) = sum(|actual-pred|) / sum(actual) fixes
    this: it weights errors by volume instead of dividing point-by-point, so
    a handful of low-count intervals can't dominate the score. It's the
    standard substitute for MAPE on intermittent/low-count demand series and
    is bounded the same way (0% = perfect, 100% = forecasting all zeros).
    """
    if bt.empty:
        return {"Forecast Accuracy (%)": np.nan, "Error Drift": np.nan, "Peak Miss Rate (%)": np.nan}

    actual, pred = bt["actual"].values, bt["predicted"].values
    total_actual = actual.sum()
    wape = float(np.sum(np.abs(actual - pred)) / total_actual * 100) if total_actual > 0 else np.nan
    accuracy = max(0.0, 100 - wape) if not np.isnan(wape) else np.nan

    half = len(bt) // 2
    if half >= 5:
        rmse1 = np.sqrt(np.mean((actual[:half] - pred[:half]) ** 2))
        rmse2 = np.sqrt(np.mean((actual[half:] - pred[half:]) ** 2))
        drift = rmse2 - rmse1
    else:
        drift = np.nan

    threshold = np.quantile(actual, peak_quantile) if len(actual) else np.nan
    peaks = actual >= threshold if not np.isnan(threshold) and threshold > 0 else np.zeros_like(actual, dtype=bool)
    if peaks.sum() > 0:
        missed = np.abs(pred[peaks] - actual[peaks]) > 0.25 * actual[peaks]
        peak_miss_rate = float(missed.mean() * 100)
    else:
        peak_miss_rate = np.nan

    return {
        "Forecast Accuracy (%)": round(accuracy, 1) if not np.isnan(accuracy) else None,
        "Error Drift": round(float(drift), 2) if not np.isnan(drift) else None,
        "Peak Miss Rate (%)": round(peak_miss_rate, 1) if not np.isnan(peak_miss_rate) else None,
    }
