import streamlit as st
import pandas as pd
from model_utils import prepare_data, naive_forecast, random_forest_forecast, prophet_forecast
from evaluation import kpi_summary
from config import DATA_PATH

st.title("Short-Term Ferry Ticket Demand Forecasting Dashboard")

# Sidebar controls
model_choice = st.sidebar.selectbox("Select Model", ["Naive", "Random Forest", "Prophet"])
horizon = st.sidebar.slider("Forecast Horizon (minutes)", 15, 120, 30)

# Load data
df = prepare_data(DATA_PATH)

# Apply model
if model_choice == "Naive":
    forecast_df = naive_forecast(df, horizon=horizon//15)
    st.line_chart(forecast_df[['Sales Count','predicted']])
    summary = kpi_summary(forecast_df['Sales Count'].dropna(), forecast_df['predicted'].dropna(), horizon=horizon//15)
    st.write("### Metrics & KPIs")
    st.table(summary)

elif model_choice == "Random Forest":
    forecast_df = random_forest_forecast(df, horizon=horizon//15)
    st.line_chart(forecast_df[['Sales Count','predicted']])
    summary = kpi_summary(forecast_df['Sales Count'], forecast_df['predicted'], horizon=horizon//15)
    st.write("###Metrics & KPIs")
    st.table(summary)

elif model_choice == "Prophet":
    forecast_df = prophet_forecast(df, horizon=horizon)
    st.line_chart(forecast_df[['ds','yhat']].set_index('ds'))
    st.area_chart(forecast_df[['ds','yhat_lower','yhat_upper']].set_index('ds'))
    summary = kpi_summary(df['Sales Count'], forecast_df['yhat'], forecast_df, horizon=horizon//15)
    st.write("###Metrics & KPIs")
    st.table(summary)
    st.download_button("Download KPI Summary", summary.to_csv().encode('utf-8'), "kpi_summary.csv", "text/csv")

