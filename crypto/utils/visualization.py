import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class Visualizer:
    def __init__(self):
        self.set_style()
    
    def set_style(self):
        """Set visualization style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_trend(self, data, symbol):
        """Plot cryptocurrency price trend"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['Date'], 
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00D4AA', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} Price Trend',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def plot_technical_indicators(self, data):
        """Plot technical indicators"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Moving Averages', 'RSI', 'Volatility', 'Daily Returns'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
      
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        if 'MA_7' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['MA_7'], name='7-day MA', line=dict(color='orange')),
                row=1, col=1
            )
        if 'MA_30' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['MA_30'], name='30-day MA', line=dict(color='red')),
                row=1, col=1
            )
        
      
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=1, col=2
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
       
        if 'Volatility' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['Volatility'], name='Volatility', line=dict(color='brown')),
                row=2, col=1
            )
  
        if 'Daily_Return' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['Daily_Return'], name='Daily Returns', line=dict(color='green')),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, template='plotly_dark')
        return fig
    
    def plot_forecast_comparison(self, actual_dates, actual_prices, forecasts, models):
        """Compare forecasts from different models"""
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=actual_dates, y=actual_prices,
            mode='lines', name='Actual Prices',
            line=dict(color='white', width=3)
        ))
        
      
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for i, (model_name, forecast_data) in enumerate(zip(models, forecasts)):
            fig.add_trace(go.Scatter(
                x=forecast_data['dates'], y=forecast_data['values'],
                mode='lines', name=f'{model_name} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Price Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def plot_outlier_detection(self, data, column):
        """Visualize outlier detection using box plot and scatter plot"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Box Plot - {column}', f'Scatter Plot - {column}')
        )
        
        
        fig.add_trace(go.Box(y=data[column], name=column), row=1, col=1)
 
        fig.add_trace(go.Scatter(
            x=data['Date'], y=data[column],
            mode='markers', name=column,
            marker=dict(size=6, color='lightblue')
        ), row=1, col=2)
        
        fig.update_layout(height=400, template='plotly_dark')
        return fig
    
    def plot_correlation_heatmap(self, data):
        """Plot correlation heatmap for numeric features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
 
        relevant_cols = [col for col in numeric_cols if col in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'MA_30', 'RSI', 'Volatility', 'Daily_Return']]
        
        if len(relevant_cols) < 2:
           
            relevant_cols = [col for col in numeric_cols if col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        if len(relevant_cols) >= 2:
            correlation_matrix = data[relevant_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            return fig
        else:

            return None
