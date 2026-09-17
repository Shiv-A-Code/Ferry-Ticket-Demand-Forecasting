# ⛴️ Toronto Island Ferry — Short-Term Demand Forecasting & Predictive Decision Support System

A machine-learning system that forecasts Toronto Island Park ferry ticket **sales** and
**redemptions** 15 minutes to 2 hours ahead, built from a decade of 15-minute-interval
ticketing data, with a live Streamlit dashboard for operators.


## What it does

Ferry operators currently only have *historical* analytics. This project adds *predictive*
intelligence: given "now", what will ticket demand look like in the next 15 min / 30 min / 1h / 2h —
with an honest uncertainty band, not just a point guess — so staffing, crowd management and
scheduling can be proactive instead of reactive.

## Results at a glance

Gradient Boosting is the strongest model at every horizon, for both targets (full breakdown
in the app's **Model Comparison** tab, and in `models/metrics.json`):

| Horizon | Sales Count MAE | Sales Count MAPE | Redemption Count MAE | Redemption Count MAPE |
|---|---|---|---|---|
| 15 min | 14.63 | 98.3% | 15.08 | 102.7% |
| 30 min | 14.91 | 99.9% | 15.39 | 102.9% |
| 1 hour | 15.54 | 103.4% | 16.03 | 100.3% |
| 2 hours | 16.73 | 107.9% | 17.53 | 109.0% |

All four Gradient Boosting models beat every baseline (Naive, Moving Average, Seasonal-Naive,
Linear Regression) at every horizon, and the gap widens as the horizon lengthens — persistence
baselines decay fastest, tree ensembles hold up because they use the engineered
lag/rolling/temporal features rather than just "the last known value."

*(MAPE is inflated by the many near-zero, off-peak intervals in a 15-min-resolution series — MAE,
in ticket counts, is the more operationally meaningful number. See the EDA tab for why.)*

## Project structure

```
ferry_forecast/
├── app.py                  # Streamlit dashboard (run this)
├── requirements.txt
├── data/
│   └── Toronto_Island_Ferry_Tickets.csv
├── src/
│   ├── data_prep.py        # loading, gap handling, feature engineering
│   ├── eda.py               # generates the charts in assets/
│   ├── train.py             # trains & evaluates all models, saves to models/
│   └── predict.py           # inference helpers used by app.py
├── models/                  # trained models + metrics (committed — see below)
└── assets/                  # EDA chart PNGs shown in the app's EDA tab
```

## Quickstart

```bash
git clone <your-repo-url>
cd ferry_forecast
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The trained models in `models/` are committed to the repo (~56 MB total), so `streamlit run
app.py` works immediately — no retraining needed. To retrain from scratch:

```bash
python src/eda.py                                  # regenerates assets/*.png
python src/train.py "Sales Count" 1 2 4 8           # ~5-6 min on 1 CPU core
python src/train.py "Redemption Count" 1 2 4 8      # ~5-6 min on 1 CPU core
```

`train.py` takes the target name and one or more horizon steps (`1`=15min, `2`=30min,
`4`=1h, `8`=2h) as CLI args, so you can retrain a single horizon at a time if you're on a
constrained machine — running it with no arguments trains everything sequentially.

### Troubleshooting: `ModuleNotFoundError: No module named '_loss'` (or similar joblib errors)

The `.joblib` files are pickles, and scikit-learn's internal module layout (e.g. for
`HistGradientBoostingRegressor`) changes between versions — a model saved by one sklearn
version can fail to load on a different one. `requirements.txt` pins the exact versions
(`scikit-learn==1.8.0`, `joblib==1.5.3`, `numpy==2.4.4`, `pandas==3.0.2`) the committed models
were trained with. Two ways to fix it:

```bash
# Option A: install the exact versions the models were trained with
pip install --force-reinstall scikit-learn==1.8.0 joblib==1.5.3 numpy==2.4.4 pandas==3.0.2

# Option B: keep whatever versions you already have installed, and
# regenerate the .joblib files locally so they match your environment
python src/train.py "Sales Count" 1 2 4 8
python src/train.py "Redemption Count" 1 2 4 8
```
Option B is the more future-proof choice if you plan to `pip install -U` things later.

## Methodology

### 1. Time-series preparation (`src/data_prep.py`)
- Parsed timestamps, de-duplicated, and strictly chronologically ordered.
- Re-indexed onto a **regular 15-minute grid** spanning the full 2015–2025 range. Ferry
  service isn't 24/7, so most "missing" intervals are genuine non-operating periods, not sensor
  gaps: short gaps (≤ 1h) are linearly interpolated (logging gaps inside an active window);
  longer gaps are masked to 0 (service closed) — both are flagged via `is_interpolated` /
  `is_non_operating` columns for auditability.

### 2. Feature engineering
- **Lag features:** t-1, t-2, t-4, t-8 (15/30/60/120 min back)
- **Rolling statistics:** mean, std, max over 1h / 2h / 4h windows (computed on lagged values
  only, to avoid leakage)
- **Temporal encodings:** hour, day-of-week, month (all cyclically sin/cos-encoded) and a
  weekend flag

### 3. Train/test strategy
- **Time-based split** (last 15% of the series, chronologically) — no shuffling, so evaluation
  reflects genuine forward-looking performance.
- **Direct multi-horizon modeling:** a separate model is trained per horizon (15m/30m/1h/2h)
  rather than recursively feeding 15-min predictions forward, which avoids compounding
  error at longer horizons.

### 4. Models trained (`src/train.py`)
| Category | Models |
|---|---|
| Baseline | Naive (persistence), Moving Average, Seasonal-Naive (same slot, 1 week prior) |
| Statistical | Linear Regression on lag/rolling/temporal features |
| Machine Learning | Random Forest Regressor, Gradient Boosting Regressor (`HistGradientBoostingRegressor` — same gradient-boosted-tree family as XGBoost, ships with scikit-learn) |
| Uncertainty | Quantile Gradient Boosting (5th / 95th percentile) → 90% prediction interval |

The SRS additionally lists ARIMA/SARIMA, Facebook Prophet, and XGBoost as extensions. These
were intentionally left out of the default pipeline (they add heavy, environment-specific
dependencies — `statsmodels`, `prophet`, `xgboost` — without materially changing the
sklearn-only baseline comparison above), but `requirements.txt` documents them as optional
installs, and `predict.py`/`train.py` are structured so a `statsmodels`-based ARIMA/SARIMA
model or a Prophet model can be dropped in as one more entry in `MODEL_NAMES` /
`TREE_MODEL_FILE_PREFIX` without touching the app's UI code.

### 5. Evaluation metrics
MAE, RMSE, MAPE (excl. true-zero intervals) — computed per horizon, per model, per target,
saved to `models/metrics.json`.

## The Streamlit app (`app.py`)

| Tab | What it shows |
|---|---|
| 📈 **Forecast** | Pick a date & time, model, and horizon → see recent history, the point forecast, its 90% confidence band, and (when available in the historical export) the actual outcome for comparison |
| 🧮 **Model Comparison** | MAE / RMSE / MAPE bar chart across all 6 models for the selected horizon, plus a horizon-wise MAE line chart showing how each model's error grows with lead time |
| 🔁 **Predicted vs Actual** | A rolling backtest over a user-selected window (1–60 days) of held-out data, with the SRS KPIs: Forecast Accuracy (%), Error Drift, Peak Miss Rate (%) |
| 🔎 **EDA & Insights** | The six exploratory charts + written insights (seasonality, hour-of-day profile, spike behavior) |

Because the dataset is historical, "now" in the Forecast tab is whatever date/time you pick —
the app always forecasts forward from that point using only information available up to it
(no lookahead), which is exactly the online-forecasting scenario the SRS describes.

## SRS coverage checklist

**Primary objectives**
- [x] Forecast short-term ferry ticket sales *and* redemptions
- [x] Predict demand for 15-min to 2-hour windows
- [x] Compare statistical (Naive/Moving-Avg/Linear) and ML (RF/Gradient Boosting) approaches

**Secondary objectives**
- [x] Quantify prediction uncertainty (90% interval via quantile Gradient Boosting)
- [x] Support proactive operational planning (Forecast tab + KPIs)
- [x] Demonstrate real-world ML deployment via Streamlit

**Dataset / methodology**
- [x] Datetime index, chronological order, missing-interval handling
- [x] Lag features (t-1/2/4/8), rolling stats (mean/std/max), temporal encodings
- [x] Time-based split, multiple forecast horizons
- [x] Baseline, ML, and (documented, optional) time-series model families
- [x] MAE / RMSE / MAPE / horizon-wise error
- [x] Prediction intervals & confidence-band visualization
- [x] KPIs: Forecast Accuracy, Error Drift, Peak Miss Rate, Confidence Band Width, Forecast Lead Time

**Streamlit requirements**
- [x] Future demand forecast charts · [x] Model selection & comparison
- [x] Horizon selector (15m–2h) · [x] Confidence interval visualization
- [x] Select date & time · [x] Switch between models · [x] Compare predicted vs actual

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at your repo, branch
   `main`, entrypoint `app.py`.
3. That's it — `requirements.txt` handles the environment; no secrets/config needed.

## Data

`data/Toronto_Island_Ferry_Tickets.csv` — 261,538 rows, 15-minute interval ticket sales and
redemption counts, May 2015 – December 2025 (`_id`, `Timestamp`, `Sales Count`,
`Redemption Count`).

## License

MIT — see `LICENSE`.
