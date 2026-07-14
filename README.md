# Ferry Ticket Demand Forecasting

This project forecasts short-term ferry ticket demand (15m–2h horizons) using machine learning and time-series models.  
It includes an interactive **Streamlit dashboard** for visualization and model comparison.

---

## Features
- Baseline models (Naïve, Moving Average, Linear Regression)
- Machine Learning models (Random Forest, Gradient Boosting, XGBoost)
- Time-Series models (ARIMA, Prophet)
- Forecast uncertainty visualization
- KPI tracking:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - Peak Miss Rate
  - Confidence Band Width
  - Forecast Lead Time

---

## Project Structure
ferry_forecasting/
│__ README.md  
│__ requirements.txt
│__ data/
│   └── Toronto Island Ferry Tickets.csv
│__ app.py
│__ model_utils.py
│__ evaluation.py
│__ config.py

## Install dependencies:
pip install -r requirements.txt

## Run the Streamlit app:
streamlit run app.py

## Outputs
1. Interactive forecast charts
2. Model comparison (Naïve, Random Forest, Prophet)
3. KPI summary table
4. Option to download KPI results as CSV

## Goal
This project demonstrates how predictive intelligence can support real-world ferry operations by enabling proactive scheduling, staff readiness, crowd management, and safety planning.
