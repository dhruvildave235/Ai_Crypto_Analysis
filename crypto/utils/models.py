import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import os
import pickle
import json

class TimeSeriesModels:
    def __init__(self):
        self.models = {}
        self.models_dir = "models/saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def arima_forecast(self, data, symbol, order=(5,1,0), forecast_days=30):
        """ARIMA model for forecasting with model saving"""
        try:
            if len(data) < 30:
                st.warning(f"ARIMA requires at least 30 data points, but only {len(data)} are available.")
                return None, None
            
          
            model_path = os.path.join(self.models_dir, f"arima_{symbol}.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        fitted_model = pickle.load(f)
                    st.info("✅ Loaded saved ARIMA model")
                except:
                    fitted_model = None
            else:
                fitted_model = None
            
            if fitted_model is None:
                model = ARIMA(data, order=order)
                fitted_model = model.fit()
          
                with open(model_path, 'wb') as f:
                    pickle.dump(fitted_model, f)
                st.info("💾 ARIMA model saved")
            
            forecast = fitted_model.forecast(steps=forecast_days)
            return fitted_model, forecast
            
        except Exception as e:
            st.error(f"ARIMA model error: {str(e)}")
            return None, None
    
    def prophet_forecast(self, data, symbol, forecast_days=30):
        """Facebook Prophet model for forecasting with model saving"""
        try:
         
            df_prophet = data.reset_index()[['Date', 'Close']].rename(
                columns={'Date': 'ds', 'Close': 'y'}
            )
            
     
            df_prophet = df_prophet.dropna()
            
            if len(df_prophet) < 10:
                st.warning(f"Prophet requires at least 10 data points, but only {len(df_prophet)} are available.")
                return None, None
            
          
            model_path = os.path.join(self.models_dir, f"prophet_{symbol}.json")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'r') as f:
                        model_json = json.load(f)
                    model = Prophet(**model_json)
                   
                    st.info("✅ Loaded Prophet configuration")
                except:
                    model = None
            else:
                model = None
            
            if model is None:
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
            
            model.fit(df_prophet)
            
       
            with open(model_path, 'w') as f:
                json.dump({
                    'daily_seasonality': True,
                    'weekly_seasonality': True,
                    'yearly_seasonality': True
                }, f)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            return model, forecast
            
        except Exception as e:
            st.error(f"Prophet model error: {str(e)}")
            return None, None
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def lstm_forecast(self, X_train, y_train, symbol, lookback=60, forecast_days=30):
        """LSTM model for forecasting with model saving"""
        try:
          
            model_path = os.path.join(self.models_dir, f"lstm_{symbol}.h5")
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    st.info("✅ Loaded saved LSTM model")
                except:
                    model = None
            else:
                model = None
            
         
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            if model is None:
              
                model = self.build_lstm_model((X_train.shape[1], 1))
                
                history = model.fit(
                    X_train_reshaped, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
              
                model.save(model_path)
                st.info("💾 LSTM model saved")
            else:
                history = None
            
            
            self.models['LSTM'] = model
            
          
            last_sequence = X_train[-1]
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_days):
                current_sequence_reshaped = current_sequence.reshape(1, lookback, 1)
                next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
                predictions.append(next_pred)
                
              
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            return model, history, np.array(predictions)
            
        except Exception as e:
            st.error(f"LSTM model error: {str(e)}")
            return None, None, None
    
    def calculate_metrics(self, actual, predicted):
        """Calculate model performance metrics"""
        if len(actual) != len(predicted):
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def get_saved_models(self):
        """Get list of saved models"""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                models.append(file)
        return models
    
    def delete_model(self, model_name):
        """Delete a saved model"""
        try:
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                os.remove(model_path)
                return True
        except:
            pass

        return False
