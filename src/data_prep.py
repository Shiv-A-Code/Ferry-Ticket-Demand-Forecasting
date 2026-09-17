"""
data_prep.py
------------
Time-series preparation & feature engineering for the
Short-Term Ferry Ticket Demand Forecasting project.

Implements the SRS "Forecasting Methodology" steps:
  1. Time-Series Preparation
     - datetime index, strict chronological ordering
     - handling of missing 15-minute intervals
  2. Feature Engineering for Forecasting
     - lag features (t-1, t-2, t-4, t-8)
     - rolling statistics (mean, std, max)
     - temporal encodings (hour, day of week, month, weekend)
"""

import os

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
RAW_PATH = os.path.join(_PROJECT_ROOT, "data", "Toronto_Island_Ferry_Tickets.csv")
FREQ = "15min"

LAGS = [1, 2, 4, 8]              # 15m, 30m, 1h, 2h back
ROLL_WINDOWS = [4, 8, 16]        # 1h, 2h, 4h rolling windows

# Forecast horizons required by the SRS, expressed in units of the
# 15-minute base frequency.
HORIZONS = {
    "15 minutes": 1,
    "30 minutes": 2,
    "1 hour": 4,
    "2 hours": 8,
}

TARGETS = ["Sales Count", "Redemption Count"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw ticket export and enforce strict chronological order."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.drop_duplicates(subset="Timestamp")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df = df.set_index("Timestamp")
    return df[["Sales Count", "Redemption Count"]]


def build_regular_grid(df: pd.DataFrame, max_gap_for_interpolation: str = "60min") -> pd.DataFrame:
    """
    Re-index the series onto a strictly regular 15-minute grid.

    Ferry service is seasonal / does not run 24 hours a day, so most
    "missing" 15-minute slots represent genuine non-operating periods
    (no tickets sold or redeemed), not sensor/logging failures.

    Strategy (per SRS: "Handle missing 15-minute intervals via
    interpolation or masking"):
      - Small gaps (<= max_gap_for_interpolation, i.e. a slot or two
        dropped *inside* an otherwise active window) are linearly
        interpolated -> these are treated as logging gaps.
      - Larger gaps (overnight / off-season closures) are masked to 0,
        since no service was running and therefore no tickets moved.
    An `is_interpolated` flag is kept so downstream users can audit
    which rows were filled.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    reindexed = df.reindex(full_index)
    reindexed.index.name = "Timestamp"

    is_missing = reindexed["Sales Count"].isna()

    # Identify contiguous missing blocks and their length
    missing_block = is_missing.ne(is_missing.shift()).cumsum()
    block_sizes = is_missing.groupby(missing_block).transform("sum")
    max_gap_steps = int(pd.Timedelta(max_gap_for_interpolation) / pd.Timedelta(FREQ))

    small_gap_mask = is_missing & (block_sizes <= max_gap_steps)
    large_gap_mask = is_missing & (block_sizes > max_gap_steps)

    reindexed["is_interpolated"] = small_gap_mask
    reindexed["is_non_operating"] = large_gap_mask

    for col in ["Sales Count", "Redemption Count"]:
        reindexed.loc[small_gap_mask, col] = np.nan  # ensure interpolation applies
        reindexed[col] = reindexed[col].interpolate(method="linear", limit=max_gap_steps)
        reindexed.loc[large_gap_mask, col] = 0.0

    return reindexed


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    for window in ROLL_WINDOWS:
        shifted = df[target].shift(1)  # avoid leakage: only past values
        df[f"{target}_roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"{target}_roll_std_{window}"] = shifted.rolling(window).std()
        df[f"{target}_roll_max_{window}"] = shifted.rolling(window).max()
    return df


def add_horizon_targets(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Create the future-value columns we are trying to predict for each horizon."""
    df = df.copy()
    for name, steps in HORIZONS.items():
        df[f"y_{target}_{steps}"] = df[target].shift(-steps)
    return df


def build_feature_frame(target: str = "Sales Count", max_gap_for_interpolation: str = "60min") -> pd.DataFrame:
    """Full pipeline: raw -> regular grid -> temporal + lag/rolling features -> horizon targets."""
    raw = load_raw()
    grid = build_regular_grid(raw, max_gap_for_interpolation=max_gap_for_interpolation)
    feat = add_temporal_features(grid)
    feat = add_lag_and_rolling_features(feat, target)
    feat = add_horizon_targets(feat, target)
    return feat


def feature_columns(target: str) -> list:
    cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
    cols += [f"{target}_lag_{lag}" for lag in LAGS]
    for window in ROLL_WINDOWS:
        cols += [f"{target}_roll_mean_{window}", f"{target}_roll_std_{window}", f"{target}_roll_max_{window}"]
    return cols


if __name__ == "__main__":
    feat = build_feature_frame()
    print(feat.shape)
    print(feat.head(10))
    print("\nMissing after feature build (expected at series start due to lags):")
    print(feat.isna().sum().sort_values(ascending=False).head(10))
