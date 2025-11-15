import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
    
    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
         
        data_clean = data.ffill().bfill()
        return data_clean
    
    def detect_outliers_iqr(self, data, column):
        """Detect outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    def remove_outliers(self, data, column):
        """Remove outliers from specified column"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return filtered_data
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for time series analysis"""
        df = data.copy()
        
       
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
       
        df['PriceChange'] = df['Close'].diff()
        df['Gain'] = df['PriceChange'].apply(lambda x: x if x > 0 else 0)
        df['Loss'] = df['PriceChange'].apply(lambda x: -x if x < 0 else 0)
        
        avg_gain = df['Gain'].rolling(window=14).mean()
        avg_loss = df['Loss'].rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
       
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
         
        df['Daily_Return'] = df['Close'].pct_change()
        
        return df
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        from sklearn.preprocessing import MinMaxScaler
        
        close_prices = data[['Close']].values
        
      
        self.scalers['LSTM'] = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scalers['LSTM'].fit_transform(close_prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        

        return np.array(X), np.array(y)
