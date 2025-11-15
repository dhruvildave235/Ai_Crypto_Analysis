import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

 
from utils.data_loader import CryptoDataLoader
from utils.preprocessing import DataPreprocessor
from utils.models import TimeSeriesModels
from utils.visualization import Visualizer

 
st.set_page_config(
    page_title="Cryptocurrency Time Series Analysis  ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# st.markdown('<h1 class="main-header">📈 Cryptocurrency Time Series Analysis</h1>', unsafe_allow_html=True)
# st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-top: -20px;">by Dhruvil</p>', unsafe_allow_html=True)
# st.markdown("---")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #00D4AA;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 15px 0px;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        margin: 15px 0px;
        color: #0c5460;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 15px 0px;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

class CryptoAnalysisApp:
    def __init__(self):
        self.data_loader = CryptoDataLoader()
        self.preprocessor = DataPreprocessor()
        self.models = TimeSeriesModels()
        self.visualizer = Visualizer()
        self.data = None
        self.processed_data = None
        self.selected_symbol = None
        
    def run(self):
        
        st.markdown('<h1 class="main-header">📈 Cryptocurrency Time Series Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.5rem; color: #666; margin-top: -20px;">by Dhruvil Dave</p>', unsafe_allow_html=True)
        st.markdown("---")
        
       
        st.markdown("""
        <div class="info-box">
        💡 <strong>Welcome!</strong> This app analyzes cryptocurrency price trends using advanced time series forecasting models including ARIMA, Prophet, and LSTM neural networks.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Note:</strong> Due to API rate limits, the app may use realistic sample data for demonstration. All analysis features work identically with both real and sample data.
        </div>
        """, unsafe_allow_html=True)
         
        self.sidebar_controls()
         
        if self.data is not None and not self.data.empty:
            self.show_data_overview()
            self.show_eda()
            self.show_forecasting()
            self.show_technical_analysis()
            self.show_model_management()
        else:
            st.info("👈 Select a cryptocurrency and time period from the sidebar, then click 'Load Data' to begin analysis.")
    
    def sidebar_controls(self):
        """Sidebar controls for user input"""
        st.sidebar.title("🔧 Configuration")
        
        
        st.sidebar.subheader("Cryptocurrency Selection")
        self.selected_symbol = st.sidebar.selectbox(
            "Select Cryptocurrency:",
            list(self.data_loader.crypto_symbols.keys()),
            format_func=lambda x: f"{x} - {self.data_loader.crypto_symbols[x]}"
        )
      
        st.sidebar.subheader("Time Period")
        period = st.sidebar.selectbox(
            "Select Time Period:",
            ['1mo', '3mo', '6mo', '1y', '2y'],
            index=3
        )
       
        forecast_days = st.sidebar.slider(
            "Forecast Days:",
            min_value=7,
            max_value=90,
            value=30
        )
        
       
        if st.sidebar.button("🚀 Load Data", use_container_width=True):
            with st.spinner(f"Loading {self.selected_symbol} data..."):
                self.data = self.data_loader.get_crypto_data(self.selected_symbol, period)
                if self.data is not None and not self.data.empty:
                    try:
                        self.processed_data = self.preprocessor.calculate_technical_indicators(
                            self.preprocessor.handle_missing_values(self.data)
                        )
                        st.sidebar.success(f"✅ Data loaded successfully! ({len(self.data)} records)")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error processing data: {str(e)}")
                else:
                    st.sidebar.error("❌ Failed to load data. Please try again.")
       
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Saved Data")
        saved_datasets = self.data_loader.get_saved_datasets()
        if saved_datasets:
            st.sidebar.write(f"**Cached datasets:** {len(saved_datasets)}")
            with st.sidebar.expander("View cached files"):
                for dataset in saved_datasets:
                    st.write(f"📄 {dataset}")
        else:
            st.sidebar.write("No cached datasets found")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **🔍 Analysis Features:**
        - Price Trend Analysis
        - Outlier Detection (IQR Method)
        - Technical Indicators (RSI, Moving Averages)
        - Time Series Forecasting
        - Volatility Analysis
        """)
        
        st.sidebar.markdown("""
        **🤖 Models Used:**
        - ARIMA (Statistical Modeling)
        - Prophet (Facebook's Forecast Tool)
        - LSTM (Deep Learning)
        """)
    
    def show_data_overview(self):
        """Display data overview"""
        st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        try:
            
            if self.data.empty or len(self.data) == 0:
                st.error("No data available. Please load data first.")
                return
            
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = self.data['Close'].iloc[-1]
                if len(self.data) > 1:
                    price_change = self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]
                    price_change_pct = (price_change / self.data['Close'].iloc[-2]) * 100
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:,.2f}",
                        delta=f"{price_change_pct:.2f}%"
                    )
                else:
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:,.2f}",
                        delta="N/A"
                    )
            
            with col2:
                avg_volume = self.data['Volume'].mean()
                st.metric(
                    label="Average Volume",
                    value=f"{avg_volume:,.0f}"
                )
            
            with col3:
                volatility = self.data['Close'].std()
                st.metric(
                    label="Volatility (Std Dev)",
                    value=f"${volatility:,.2f}"
                )
            
            with col4:
                total_return = ((self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0]) * 100
                st.metric(
                    label="Total Return",
                    value=f"{total_return:.2f}%"
                )
            
           
            with st.expander("📋 View Raw Data (Last 10 Records)"):
                st.dataframe(self.data.tail(10))
                
               
                csv = self.data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full CSV",
                    data=csv,
                    file_name=f"crypto_data_{self.selected_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error displaying data overview: {str(e)}")
    
    def show_eda(self):
        """Display Exploratory Data Analysis"""
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        try: 
            st.subheader("Price Trend")
            fig_trend = self.visualizer.plot_price_trend(self.data, self.selected_symbol)
            st.plotly_chart(fig_trend, use_container_width=True)
            
          
            st.subheader("📊 Outlier Detection using IQR Method")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                outlier_column = st.selectbox("Select column for outlier detection:", 
                                            ['Close', 'Volume', 'High', 'Low'])
                if st.button("🔍 Detect Outliers", use_container_width=True):
                    outliers, lower_bound, upper_bound = self.preprocessor.detect_outliers_iqr(self.data, outlier_column)
                    
                    st.write(f"**Outlier Detection Results for {outlier_column}:**")
                    st.write(f"**Lower Bound:** {lower_bound:,.2f}")
                    st.write(f"**Upper Bound:** {upper_bound:,.2f}")
                    st.write(f"**Number of Outliers:** {len(outliers)}")
                    
                    if len(outliers) > 0:
                        with st.expander("View Outlier Data"):
                            st.dataframe(outliers[['Date', outlier_column]])
            
            with col2:
                fig_outliers = self.visualizer.plot_outlier_detection(self.data, outlier_column)
                st.plotly_chart(fig_outliers, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in EDA: {str(e)}")
    
    def show_forecasting(self):
        """Display forecasting section"""
        st.markdown('<h2 class="section-header">🔮 Price Forecasting</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        💡 The forecasting models will predict future price movements using different algorithms:
        - **ARIMA**: Traditional statistical time series model
        - **Prophet**: Facebook's forecasting tool optimized for business metrics
        - **LSTM**: Deep learning model that captures complex patterns
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Run All Forecasting Models", use_container_width=True):
            with st.spinner("Training models... This may take a few minutes depending on data size"):
                self.run_forecasting_models()
    
    def run_forecasting_models(self):
        """Run all forecasting models and display results"""
        try:
            if self.processed_data is None or self.processed_data.empty:
                st.error("No processed data available. Please load data first.")
                return
            
            close_prices = self.processed_data['Close']
            
            if len(close_prices) < 30:
                st.warning(f"Insufficient data for forecasting. Need at least 30 data points, but only {len(close_prices)} are available.")
                return
            
          
            with st.expander("📈 ARIMA Model", expanded=True):
                st.write("**ARIMA (AutoRegressive Integrated Moving Average) Results**")
                arima_model, arima_forecast = self.models.arima_forecast(close_prices, self.selected_symbol)
                
                if arima_forecast is not None:
                     
                    last_date = self.processed_data['Date'].iloc[-1]
                    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
                    
                     
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'ARIMA_Forecast': arima_forecast
                    })
                    st.dataframe(forecast_df.style.format({'ARIMA_Forecast': '{:,.2f}'}))
                    
                    
                    fig_arima = self.visualizer.plot_forecast_comparison(
                        self.processed_data['Date'].tolist()[-60:],
                        close_prices.tolist()[-60:],
                        [{'dates': forecast_dates, 'values': arima_forecast}],
                        ['ARIMA']
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)
                else:
                    st.warning("ARIMA model could not be trained with the available data.")
          
            with st.expander("📊 Prophet Model", expanded=True):
                st.write("**Facebook Prophet Model Results**")
                prophet_model, prophet_forecast = self.models.prophet_forecast(self.processed_data, self.selected_symbol)
                
                if prophet_forecast is not None:
                    # Display forecast
                    prophet_results = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
                    st.dataframe(prophet_results.rename(columns={'ds': 'Date', 'yhat': 'Forecast'}).style.format({
                        'Forecast': '{:,.2f}', 'yhat_lower': '{:,.2f}', 'yhat_upper': '{:,.2f}'
                    }))
                else:
                    st.warning("Prophet model could not be trained with the available data.")
            
            
            with st.expander("🧠 LSTM Model", expanded=True):
                st.write("**LSTM (Long Short-Term Memory) Neural Network Results**")
                
                lookback = min(60, len(self.processed_data) - 10)
                if lookback < 10:
                    st.warning(f"Insufficient data for LSTM. Need at least {10 + lookback} data points.")
                    return
                
                X, y = self.preprocessor.prepare_lstm_data(self.processed_data, lookback)
                
                if len(X) > 0:
                    
                    split_idx = int(0.8 * len(X))
                    X_train, y_train = X[:split_idx], y[:split_idx]
                    
                    lstm_model, history, lstm_forecast = self.models.lstm_forecast(
                        X_train, y_train, self.selected_symbol, lookback
                    )
                    
                    if lstm_forecast is not None:
                        
                        lstm_forecast_inverse = self.preprocessor.scalers['LSTM'].inverse_transform(
                            lstm_forecast.reshape(-1, 1)
                        ).flatten()
                        
                        
                        last_date = self.processed_data['Date'].iloc[-1]
                        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
                        
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'LSTM_Forecast': lstm_forecast_inverse
                        })
                        st.dataframe(forecast_df.style.format({'LSTM_Forecast': '{:,.2f}'}))
                    else:
                        st.warning("LSTM model could not generate forecasts.")
                else:
                    st.warning("Insufficient data for LSTM training.")
                
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")
    
    def show_technical_analysis(self):
        """Display technical analysis"""
        st.markdown('<h2 class="section-header">📈 Technical Analysis</h2>', unsafe_allow_html=True)
        
        try:
            if self.processed_data is not None and not self.processed_data.empty:
               
                st.subheader("Technical Indicators")
                fig_technical = self.visualizer.plot_technical_indicators(self.processed_data)
                st.plotly_chart(fig_technical, use_container_width=True)
                
                 
                st.subheader("Feature Correlation Heatmap")
                fig_corr = self.visualizer.plot_correlation_heatmap(self.processed_data)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough numeric features for correlation analysis.")
                
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
    
    def show_model_management(self):
        """Show model management section"""
        st.markdown('<h2 class="section-header">💾 Model Management</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Saved Models")
            saved_models = self.models.get_saved_models()
            if saved_models:
                st.write(f"**Total saved models:** {len(saved_models)}")
                for model in saved_models:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"🤖 {model}")
                    with col_b:
                        if st.button("🗑️", key=f"delete_{model}"):
                            if self.models.delete_model(model):
                                st.success(f"Deleted {model}")
                                st.rerun()
            else:
                st.info("No models saved yet. Run forecasting to save models.")
        
        with col2:
            st.subheader("Data Management")
            saved_datasets = self.data_loader.get_saved_datasets()
            if saved_datasets:
                st.write(f"**Cached datasets:** {len(saved_datasets)}")
                for dataset in saved_datasets:
                    st.write(f"📄 {dataset}")
            else:
                st.info("No cached datasets found")

 
if __name__ == "__main__":
    app = CryptoAnalysisApp()

    app.run()
