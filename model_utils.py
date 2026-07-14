import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()

    # Resample to 15-min intervals
    df = df.resample('15T').sum().fillna(0)

    # Feature engineering
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    return df

def naive_forecast(df, horizon=1):
    """Naïve forecast: predict next value = last observed"""
    df['predicted'] = df['Sales Count'].shift(horizon)
    return df

def random_forest_forecast(df, horizon=1):
    """Random Forest forecast using time features"""
    X = df[['hour','day_of_week','is_weekend']].values
    y = df['Sales Count'].values
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X[:-horizon], y[:-horizon])
    df['predicted'] = model.predict(X)
    return df

def prophet_forecast(df, horizon=60):
    """Prophet forecast with confidence intervals"""
    prophet_df = df.reset_index()[['Timestamp','Sales Count']]
    prophet_df.columns = ['ds','y']
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon//15, freq='15min')
    forecast = model.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

