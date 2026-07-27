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
  - Horizon-wise error
  - Confidence Band Width
  - Forecast Lead Time
  - Forecast Accuracy (%)
  - Error Drift	Stability across horizons




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


# Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System

This project transforms historical ferry ticket data into **predictive intelligence** for Toronto Island Park operations.  
It leverages machine learning and time-series forecasting to anticipate demand in upcoming 15-minute to 2-hour windows.

---

## 🎯 Primary Objectives
- Forecast short-term ferry ticket **sales and redemptions**
- Predict demand for upcoming **15-minute to 2-hour horizons**
- Compare **statistical vs machine-learning forecasting approaches**

## 🔍 Secondary Objectives
- Quantify **prediction uncertainty**
- Support **proactive operational planning**
- Demonstrate **real-world ML deployment via Streamlit**

---

## 📊 Dataset
| Column            | Description                          |
|-------------------|--------------------------------------|
|       id          |  Unique row identifier               |
|    Timestamp      |  15-minute interval end time         |
|    Sales Count    |  Tickets sold in interval            |
|  Redemption Count |  Tickets redeemed in interval        |

---

## ⚙️ Methodology
1. **Time-Series Preparation**  
   - Convert timestamps to datetime index  
   - Handle missing intervals via interpolation/masking  

2. **Feature Engineering**  
   - Lag features (t-1, t-2, t-4, t-8)  
   - Rolling statistics (mean, std, max)  
   - Temporal encodings: hour, day of week, weekend indicator  

3. **Models Implemented**  
   - Baseline: Naïve forecast, Moving average  
   - ML: Random Forest, Gradient Boosting, XGBoost  
   - Time-Series: ARIMA/SARIMA, Prophet  

---

## 📈 Evaluation Metrics
- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Squared Error)  
- **MAPE** (Mean Absolute Percentage Error)  
- Horizon-wise error comparison  

---

## 🚀 Streamlit Dashboard Features
- Interactive **future demand forecast charts**  
- **Model selection & comparison**  
- Horizon selector (15m–2h)  
- Confidence interval visualization  
- Compare predicted vs actual values  

---

## 📦 Deliverables
- Research paper (EDA, insights, recommendations)  
- Streamlit dashboard (live analytics)  
- Executive summary for stakeholders  

---

##  How to Run
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

## Outputs
1. Interactive forecast charts
2. Model comparison (Naïve, Random Forest, Prophet)
3. KPI summary table
4. Option to download KPI results as CSV

## Goal
This project demonstrates how predictive intelligence can support real-world ferry operations by enabling proactive scheduling, staff readiness, crowd management, and safety planning.
