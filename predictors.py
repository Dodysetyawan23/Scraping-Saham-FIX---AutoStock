from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
from datetime import timedelta

def determine_forecast_days(period):
    """Menentukan jumlah hari forecast berdasarkan periode data"""
    if period == '1mo':
        return 7  # 1 minggu ke depan
    elif period == '3mo':
        return 21  # 1 bulan ke depan
    elif period == '6mo':
        return 30  # 1.5 bulan ke depan
    elif period == '1y':
        return 60  # 2 bulan ke depan
    elif period == '2y':
        return 90  # 3 bulan ke depan
    elif period == '5y':
        return 180  # 6 bulan ke depan
    elif period == '10y':
        return 365  # 1 tahun ke depan
    elif period == 'ytd':
        return 90  # 3 bulan ke depan
    else:
        return 30  # default


def determine_seasonal_period(period):
    """Menentukan seasonal period berdasarkan timeframe data"""
    if period in ['1mo', '3mo']:
        return 5  # Weekly seasonality
    elif period in ['6mo', '1y']:
        return 21  # Monthly seasonality
    else:
        return 63  # Quarterly seasonality
    
def holtwinters_forecast(train_data, test_data, future_days, period):
    """Implementasi forecasting menggunakan Holt-Winters"""
    seasonal_period = determine_seasonal_period(period)
    
    model = ExponentialSmoothing(
        train_data['Close'],
        seasonal_periods=seasonal_period,
        trend='add',
        seasonal='add',
        initialization_method='estimated'
    )
    fitted_model = model.fit()
    
    # Predictions
    test_predictions = fitted_model.forecast(len(test_data))
    future_predictions = fitted_model.forecast(future_days)
    
    return test_predictions, future_predictions, fitted_model

def prophet_forecast(train_data, test_data, future_days):
    """Implementasi forecasting menggunakan Prophet"""
    # Prepare data for Prophet: Ensure it only contains two columns
    train_prophet = train_data[['Date', 'Close']].copy()
    train_prophet.columns = ['ds', 'y']
    
    # Create the Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(train_prophet)
    
    # Create future dataframe
    future_dates = pd.date_range(
        start=train_data['Date'].iloc[-1] + timedelta(days=1),  # Use the last date in the training data
        periods=len(test_data) + future_days,
        freq='D'
    )
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Make predictions
    forecast = model.predict(future_df)
    test_predictions = forecast['yhat'][:len(test_data)].values
    future_predictions = forecast['yhat'][len(test_data):].values
    
    return test_predictions, future_predictions, model


def evaluate_model(test_data, predictions):
    """Menghitung metrik evaluasi untuk model"""
    mape = mean_absolute_percentage_error(test_data['Close'], predictions)
    rmse = np.sqrt(mean_squared_error(test_data['Close'], predictions))
    return {'mape': mape, 'rmse': rmse}