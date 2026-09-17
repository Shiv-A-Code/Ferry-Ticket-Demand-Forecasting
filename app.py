"""
app.py
------
Short-Term Ferry Ticket Demand Forecasting & Predictive Decision
Support System — Streamlit dashboard.

Implements the SRS "Streamlit Web Application Requirements":
  Core modules   : future demand forecast charts, model selection &
                   comparison, horizon selector (15m-2h), confidence
                   interval visualization
  User abilities : select date & time, switch between models,
                   compare predicted vs actual values

Run with:  streamlit run app.py   (from the project root)
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_prep import HORIZONS, build_feature_frame, feature_columns  # noqa: E402
from predict import (  # noqa: E402
    MODEL_LABELS,
    backtest_series,
    confidence_interval,
    kpis_from_backtest,
    load_metrics,
    point_forecast,
)

st.set_page_config(
    page_title="Toronto Island Ferry — Demand Forecast",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------ theme system --
# Two full color palettes (light / dark). Everything downstream — CSS,
# Plotly figures, metric cards — reads from `T` so the whole app switches
# together instead of only re-skinning Streamlit's own chrome.
THEMES = {
    "Light": dict(
        bg="#FBFCFC", card_bg="#F4F7F7", panel_bg="#FFFFFF", sidebar_bg="#F0F4F4",
        text="#16333A", subtext="#4B5B60", border="#DCE6E6",
        accent="#1F6F78", accent_light="#5FA8AE", warn="#D97757", good="#2E7D32",
        plot_template="plotly_white", plot_bg="#FFFFFF", paper_bg="#FFFFFF", grid="#E6EDED",
    ),
    "Dark": dict(
        bg="#0E1418", card_bg="#182226", panel_bg="#141C20", sidebar_bg="#10171B",
        text="#E8F1F1", subtext="#A9BEC0", border="#26343A",
        accent="#5FD1D8", accent_light="#8FE0E3", warn="#F2A65A", good="#6FDD86",
        plot_template="plotly_dark", plot_bg="#141C20", paper_bg="#141C20", grid="#243136",
    ),
}

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

with st.sidebar:
    st.session_state["dark_mode"] = st.toggle(
        "🌙 Dark mode", value=st.session_state["dark_mode"],
        help="Switch the dashboard (charts included) between light and dark palettes.",
    )

DARK = st.session_state["dark_mode"]
T = THEMES["Dark" if DARK else "Light"]
ACCENT, ACCENT_LIGHT, WARN, GOOD = T["accent"], T["accent_light"], T["warn"], T["good"]

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {T['bg']}; color: {T['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {T['sidebar_bg']}; }}
    section[data-testid="stSidebar"] * {{ color: {T['text']}; }}

    div[data-testid="stMetric"] {{
        background-color: {T['card_bg']};
        border: 1px solid {T['border']};
        border-left: 4px solid {ACCENT};
        border-radius: 6px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {T['subtext']}; }}
    div[data-testid="stMetricValue"] {{ color: {T['text']}; }}

    h1, h2, h3, h4 {{ color: {T['text']}; }}
    p, li, span, label {{ color: {T['text']}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {T['subtext']} !important; }}

    div[data-baseweb="tab-list"] {{ border-bottom: 1px solid {T['border']}; }}
    button[data-baseweb="tab"] {{ color: {T['subtext']}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {ACCENT}; }}

    div[data-testid="stExpander"] {{
        background-color: {T['panel_bg']}; border: 1px solid {T['border']}; border-radius: 6px;
    }}
    hr {{ border-color: {T['border']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, *, hover_unified: bool = True) -> go.Figure:
    """Apply the active theme (template, colors, gridlines) to a Plotly figure
    and turn on a unified hover tooltip so comparing traces at a given
    x-position is a single glance instead of hunting across lines."""
    fig.update_layout(
        template=T["plot_template"],
        plot_bgcolor=T["plot_bg"],
        paper_bgcolor=T["paper_bg"],
        font=dict(color=T["text"]),
        hovermode="x unified" if hover_unified else "closest",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    return fig


# ------------------------------------------------------------- data cache --
@st.cache_data(show_spinner="Loading and preparing ferry ticket data...")
def get_feature_frame(target: str) -> pd.DataFrame:
    return build_feature_frame(target=target)


@st.cache_data(show_spinner=False)
def get_metrics():
    return load_metrics()


TARGETS = ["Sales Count", "Redemption Count"]
HORIZON_NAMES = list(HORIZONS.keys())
MODEL_NAMES = ["Naive", "Moving Average", "Seasonal Naive", "Linear Regression", "Random Forest", "Gradient Boosting"]
MODEL_COLORWAY = [WARN, "#8E7CC3", "#D4A373", ACCENT_LIGHT, "#6FA8DC", ACCENT]
MODEL_COLOR = dict(zip(MODEL_NAMES, MODEL_COLORWAY))

# ------------------------------------------------------------------ header --
st.title("⛴️ Toronto Island Ferry — Short-Term Demand Forecasting")
st.caption(
    "Predictive decision support for ferry operations: 15-minute to 2-hour ahead ticket "
    "sales & redemption forecasts, with model comparison and uncertainty bands."
)

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Forecast controls")

    target = st.selectbox("Metric to forecast", TARGETS, index=0)
    horizon_name = st.selectbox("Forecast horizon", HORIZON_NAMES, index=2)
    steps = HORIZONS[horizon_name]
    model_name = st.selectbox(
        "Model", MODEL_NAMES, index=len(MODEL_NAMES) - 1, format_func=lambda m: MODEL_LABELS[m]
    )

    st.divider()
    st.subheader("Select date & time")

feat = get_feature_frame(target)
x_cols = feature_columns(target)
metrics = get_metrics()

min_date, max_date = feat.index.min().date(), feat.index.max().date()

with st.sidebar:
    sel_date = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)
    day_slots = feat.loc[str(sel_date)]
    if day_slots.empty:
        st.warning("No service recorded on this date — pick another.")
        st.stop()
    time_options = [t.strftime("%H:%M") for t in day_slots.index]
    default_idx = len(time_options) // 2
    sel_time = st.selectbox("Time (forecast made as of this moment)", time_options, index=default_idx)
    at_index = pd.Timestamp(f"{sel_date} {sel_time}")

    if st.button("🌊 Jump to a historic demand spike", use_container_width=True):
        spike_time = feat[target].sort_values(ascending=False).index[3]
        st.session_state["_spike"] = spike_time

    st.divider()
    st.caption(
        "Data: Toronto Island Park ferry ticket exports, 15-minute intervals, "
        f"{min_date} to {max_date}."
    )

if "_spike" in st.session_state:
    at_index = st.session_state["_spike"]

tab_forecast, tab_compare, tab_backtest, tab_eda = st.tabs(
    ["📈 Forecast", "🧮 Model Comparison", "🔁 Predicted vs Actual", "🔎 EDA & Insights"]
)

# =============================================================== TAB 1 ====
with tab_forecast:
    st.subheader(f"Forecast as of {at_index}")

    c_hist, c_overlay = st.columns([1, 2])
    with c_hist:
        hist_hours = st.slider("History shown (hours)", min_value=1, max_value=48, value=6)
    with c_overlay:
        overlay_models = st.multiselect(
            "Also compare against",
            [m for m in MODEL_NAMES if m != model_name],
            format_func=lambda m: MODEL_LABELS[m],
            help="Overlay other models' point forecasts for the same date/time/horizon.",
        )

    history_window = feat.loc[at_index - pd.Timedelta(hours=hist_hours): at_index, target]
    forecast_time = at_index + pd.Timedelta(minutes=15 * steps)

    pred = point_forecast(model_name, feat, x_cols, target, at_index, steps)
    lo, hi = confidence_interval(feat, x_cols, target, at_index, steps)

    actual_future = None
    if forecast_time in feat.index:
        actual_future = float(feat.loc[forecast_time, target])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Forecast @ {horizon_name}", f"{round(pred):,d} tickets")
    c2.metric("90% interval", f"{lo:,.0f} – {hi:,.0f}")
    if actual_future is not None:
        err = pred - actual_future
        c3.metric("Actual (known, held out)", f"{actual_future:,.0f}", delta=f"{err:+.1f} error")
    else:
        c3.metric("Actual", "not yet observed")
    c4.metric("Last observed value", f"{feat.loc[at_index, target]:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_window.index, y=history_window.values, mode="lines",
        name="Observed history", line=dict(color=ACCENT, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[at_index, forecast_time], y=[feat.loc[at_index, target], pred], mode="lines+markers",
        name=f"{MODEL_LABELS[model_name]} forecast", line=dict(color=WARN, width=2, dash="dash"),
        marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=[forecast_time, forecast_time], y=[lo, hi], mode="lines",
        name="90% confidence band", line=dict(color=ACCENT_LIGHT, width=8), opacity=0.4,
    ))
    for m in overlay_models:
        m_pred = point_forecast(m, feat, x_cols, target, at_index, steps)
        fig.add_trace(go.Scatter(
            x=[at_index, forecast_time], y=[feat.loc[at_index, target], m_pred], mode="lines+markers",
            name=f"{MODEL_LABELS[m]} forecast", line=dict(color=MODEL_COLOR.get(m, "#999999"), width=1.5, dash="dot"),
            marker=dict(size=6), opacity=0.85,
        ))
    if actual_future is not None:
        fig.add_trace(go.Scatter(
            x=[forecast_time], y=[actual_future], mode="markers", name="Actual outcome",
            marker=dict(size=11, color=GOOD, symbol="star"),
        ))
    fig.add_vline(x=at_index, line_dash="dot", line_color=T["subtext"])
    fig.update_layout(
        height=460, xaxis_title="Time", yaxis_title=f"{target}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Dashed line: point forecast for the selected model & horizon. Shaded band: 90% "
        "prediction interval from a quantile Gradient Boosting model (uncertainty analysis, "
        "per SRS). Dotted lines: any models added under \"Also compare against\". Star: the "
        "true value, shown only when it already exists in the historical export so you can "
        "sanity-check accuracy."
    )

# =============================================================== TAB 2 ====
with tab_compare:
    st.subheader(f"Model comparison — {target}, {horizon_name} horizon")
    m = metrics[target][str(steps)]
    comp_df = pd.DataFrame(m).T.reset_index().rename(columns={"index": "Model_key"})
    comp_df["Model"] = comp_df["Model_key"].map(MODEL_LABELS)

    metric_choice = st.radio("Metric", ["MAE", "RMSE", "MAPE"], horizontal=True, key="bar_metric")
    fig2 = go.Figure(go.Bar(
        x=comp_df["Model"], y=comp_df[metric_choice],
        marker_color=[ACCENT if v == comp_df[metric_choice].min() else ACCENT_LIGHT for v in comp_df[metric_choice]],
        text=comp_df[metric_choice], texttemplate="%{text:.2f}", textposition="outside",
    ))
    fig2.update_layout(
        height=420, yaxis_title=metric_choice + (" (%)" if metric_choice == "MAPE" else " (tickets)"),
    )
    style_fig(fig2, hover_unified=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Horizon-wise error drift** — does accuracy degrade for longer lead times?")
    horizon_metric_choice = st.radio(
        "Horizon chart metric", ["MAE", "RMSE", "MAPE"], horizontal=True, key="horizon_metric"
    )
    rows = []
    for h_name, h_steps in HORIZONS.items():
        hm = metrics[target][str(h_steps)]
        for mdl, vals in hm.items():
            rows.append({"Horizon": h_name, "steps": h_steps, "Model": MODEL_LABELS[mdl], **vals})
    horizon_df = pd.DataFrame(rows)
    fig3 = go.Figure()
    for mdl_key, mdl_label in MODEL_LABELS.items():
        sub = horizon_df[horizon_df["Model"] == mdl_label].sort_values("steps")
        fig3.add_trace(go.Scatter(
            x=sub["Horizon"], y=sub[horizon_metric_choice], mode="lines+markers", name=mdl_label,
            line=dict(color=MODEL_COLOR.get(mdl_key, "#999999")),
        ))
    fig3.update_layout(
        height=380,
        yaxis_title=horizon_metric_choice + (" (%)" if horizon_metric_choice == "MAPE" else " (tickets)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_fig(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    table_style = comp_df.set_index("Model")[["MAE", "RMSE", "MAPE"]].style.highlight_min(
        color=("#2A3F3D" if DARK else "#D7ECE9"), axis=0
    )
    st.dataframe(table_style, use_container_width=True)

# =============================================================== TAB 3 ====
with tab_backtest:
    st.subheader("Predicted vs. actual — backtest window")
    bt_dates = st.slider(
        "Backtest window (last N days of held-out data)",
        min_value=1, max_value=60, value=14,
    )
    bt_end = feat.index.max()
    bt_start = bt_end - pd.Timedelta(days=bt_dates)
    bt = backtest_series(model_name, feat, x_cols, target, steps, bt_start, bt_end)

    kpis = kpis_from_backtest(bt)
    k1, k2, k3 = st.columns(3)
    k1.metric("Forecast Accuracy (%)", kpis["Forecast Accuracy (%)"])
    k2.metric("Error Drift (RMSE, 2nd half − 1st half)", kpis["Error Drift"])
    k3.metric("Peak Miss Rate (%)", kpis["Peak Miss Rate (%)"])

    show_error_band = st.checkbox("Shade the error between actual and predicted", value=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=bt["timestamp"], y=bt["actual"], name="Actual", line=dict(color=ACCENT)))
    fig4.add_trace(go.Scatter(x=bt["timestamp"], y=bt["predicted"], name="Predicted", line=dict(color=WARN, dash="dash")))
    if show_error_band:
        fig4.add_trace(go.Scatter(
            x=pd.concat([bt["timestamp"], bt["timestamp"][::-1]]),
            y=pd.concat([bt["actual"], bt["predicted"][::-1]]),
            fill="toself", fillcolor="rgba(217,119,87,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Error", showlegend=True, hoverinfo="skip",
        ))
    fig4.update_layout(height=440, yaxis_title=target,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    style_fig(fig4)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "KPIs follow the SRS definitions: Forecast Accuracy = 100 − WAPE (volume-weighted "
        "error — the standard MAPE substitute for low-count/intermittent series like this one, "
        "where point-wise MAPE explodes past 100% and can't distinguish models); "
        "Error Drift compares RMSE across the first vs. second half of the window (stability "
        "across horizons/time); Peak Miss Rate is the share of top-decile demand spikes missed "
        "by more than 25%."
    )
    st.download_button(
        "⬇️ Download this backtest as CSV", bt.to_csv(index=False),
        file_name=f"backtest_{target.replace(' ', '_')}_{model_name.replace(' ', '_')}_{steps}steps.csv",
    )

# =============================================================== TAB 4 ====
with tab_eda:
    st.subheader("Exploratory Data Analysis & Insights")
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    charts = [
        ("01_weekly_history.png", "Ten years of ferry demand, weekly totals — long-run growth & seasonality."),
        ("02_hourly_profile.png", "Average activity by hour of day — clear mid-day peak."),
        ("03_dow_profile.png", "Average activity by day of week — weekend lift."),
        ("04_monthly_profile.png", "Average activity by month — strong summer seasonality."),
        ("05_sample_week.png", "One representative week at full 15-minute resolution."),
        ("06_sales_distribution.png", "Distribution of non-zero sales intervals (log scale) — heavy right tail."),
    ]
    col1, col2 = st.columns(2)
    for i, (fname, caption) in enumerate(charts):
        target_col = col1 if i % 2 == 0 else col2
        path = os.path.join(assets_dir, fname)
        if os.path.exists(path):
            with target_col.container(border=True):
                st.image(path, caption=caption, use_container_width=True)

    with st.expander("**Key insights**", expanded=True):
        st.markdown(
            """
            - Demand is strongly time-of-day driven (near-zero overnight, peak ~11:00–15:00) and
              seasonal (summer months dominate volume) — both are encoded as model features.
            - A small number of extreme spikes (single 15-min intervals with 3,000-7,000+ tickets)
              sit far outside the typical range and are the hardest events for every model to catch,
              motivating the Peak Miss Rate KPI and the prediction-interval band.
            - Tree-based models (Random Forest, Gradient Boosting) consistently beat the Naive,
              Moving Average and Linear Regression baselines across all four horizons, and their
              advantage widens at longer horizons (1-2h) where persistence-based baselines decay
              fastest — see the Model Comparison tab.
            """
        )

st.divider()
st.caption(
    "Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System · "
    "Built with scikit-learn + Streamlit."
)
