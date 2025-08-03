import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.graph_objects as go

# Set page title and layout
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# App title and description
st.title("Stock Price Prediction App")
st.markdown("This app uses an LSTM model to predict the next day's closing price based on the last 100 days of data.")

# Load the saved model
@st.cache_resource
def load_lstm_model():
    return load_model('lstm_stock.h5')

model = load_lstm_model()

# Create main area for inputs
st.header("Input Parameters")

# Date range selection
today = dt.datetime.now().date()
default_start_date = today - dt.timedelta(days=365)

# Create columns for better layout
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT)", "AAPL")

with col2:
    start_date = st.date_input("Start Date", default_start_date)

with col3:
    end_date = st.date_input("End Date", today)

# Prediction button
center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
with center_col2:
    predict_button = st.button("Predict", use_container_width=True)

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to prepare data for prediction
def prepare_data(data):
    # Extract close price
    try:
        # Handle both single-level and multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-level columns (common with yfinance)
            close_col = None
            for col in data.columns:
                if 'Close' in str(col):
                    close_col = col
                    break
            if close_col is None:
                st.error("Could not find 'Close' column in the data.")
                return None, None, None
            df = pd.DataFrame(data[close_col])
            df.columns = ['Close']
        else:
            # Single-level columns
            if 'Close' not in data.columns:
                st.error("Could not find 'Close' column in the data.")
                return None, None, None
            df = pd.DataFrame(data['Close'])
    except Exception as e:
        st.error(f"Error extracting closing prices: {e}")
        return None, None, None
    
    # Prepare data for LSTM prediction
    if len(df) < 100:
        st.error("Insufficient data. Need at least 100 days of historical data for prediction.")
        return None, None, None
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    # Prepare the last 100 days for prediction
    X_test = []
    X_test.append(scaled_data[-100:, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_test, scaler, df

# Create sidebar for model info
st.sidebar.subheader("About the Model")
st.sidebar.write("""
This app uses an LSTM model to predict the next day's closing price based on the last 100 days of data.
""")
st.sidebar.markdown("---")

# Main app logic
if predict_button:
    # Show loading spinner
    with st.spinner('Fetching stock data and making prediction...'):
        # Fetch the stock data
        data = fetch_stock_data(stock_ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            # Prepare data for prediction
            X_test, scaler, df = prepare_data(data)
            
            if X_test is not None:
                # Make prediction
                predicted_scaled = model.predict(X_test)
                
                # Inverse transform to get the actual price
                predicted_price = scaler.inverse_transform(predicted_scaled)
                
                # Get the last closing price
                last_price = df['Close'].iloc[-1]
                
                # Calculate percentage change
                change_percent = ((predicted_price[0][0] - last_price) / last_price) * 100
                
                # Display prediction results
                st.subheader("Prediction Results")
                
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Last Closing Price", f"${last_price:.2f}")
                with col2:
                    st.metric("Predicted Next Day Price", f"${predicted_price[0][0]:.2f}", 
                             f"{change_percent:.2f}%")
                with col3:
                    if change_percent > 0:
                        st.metric("Prediction", "Bullish 📈")
                    else:
                        st.metric("Prediction", "Bearish 📉")
                with col4:
                    st.metric("Confidence", "LSTM Model")
                
                # Display data summary
                st.subheader(f"Data Summary for {stock_ticker}")
                st.dataframe(data.tail(5))
                
                # Plot the results
                st.subheader("Price Trend and Prediction")
                
                # Create a dataframe for plotting
                plot_df = df[-30:].copy()
                
                # Add a row for the prediction
                next_day = pd.DataFrame(index=[plot_df.index[-1] + pd.Timedelta(days=1)], 
                                       data={'Close': predicted_price[0][0]})
                
                plot_df = pd.concat([plot_df, next_day])
                
                # Create a plotly figure
                fig = go.Figure()
                
                # Add historical prices
                fig.add_trace(go.Scatter(
                    x=plot_df.index[:-1],
                    y=plot_df['Close'][:-1],
                    mode='lines',
                    name='Historical Prices',
                    line=dict(color='blue')
                ))
                
                # Add predicted price
                fig.add_trace(go.Scatter(
                    x=plot_df.index[-2:],
                    y=plot_df['Close'][-2:],
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', dash='dash'),
                    marker=dict(size=[0, 10])
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{stock_ticker} Stock Price Prediction (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    legend=dict(y=0.99, x=0.01),
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add history and volume graphs
                st.subheader("Historical Price and Volume Analysis")
                
                # Display date range info
                date_range_info = f"({start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')})"
                st.caption(f"Showing data from {date_range_info}")
                
                # Create columns for the charts
                hist_col1, hist_col2 = st.columns(2)
                
                with hist_col1:
                    # Historical Price Chart
                    st.subheader("Historical Price Trend")
                    price_fig = go.Figure()
                    
                    price_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='green', width=2)
                    ))
                    
                    price_fig.update_layout(
                        title=f"{stock_ticker} Historical Price {date_range_info}",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(price_fig, use_container_width=True)
                
                with hist_col2:
                    # Volume Chart
                    st.subheader("Trading Volume")
                    volume_fig = go.Figure()
                    
                    # Handle column structure for Volume
                    volume_data = None
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            for col in data.columns:
                                if 'Volume' in str(col):
                                    volume_data = data[col]
                                    break
                            if volume_data is None:
                                volume_data = data.iloc[:, data.columns.get_level_values(0) == 'Volume'][0]
                        else:
                            volume_data = data['Volume']
                    except:
                        volume_data = data.iloc[:, 4] if len(data.columns) > 4 else data.iloc[:, -1]
                    
                    volume_fig.add_trace(go.Bar(
                        x=data.index,
                        y=volume_data,
                        name='Volume',
                        marker_color='orange'
                    ))
                    
                    volume_fig.update_layout(
                        title=f"{stock_ticker} Trading Volume {date_range_info}",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                # Add 1-year historical vs predicted prices analysis
                st.subheader("1-Year Historical vs Predicted Closing Price Analysis")
                
                # Generate predictions for the entire dataset
                def generate_lstm_predictions(data_df):
                    """Generate LSTM predictions for the entire dataset"""
                    try:
                        # Extract close price
                        if isinstance(data_df.columns, pd.MultiIndex):
                            close_col = None
                            for col in data_df.columns:
                                if 'Close' in str(col):
                                    close_col = col
                                    break
                            if close_col is None:
                                return None, None
                            df = pd.DataFrame(data_df[close_col])
                            df.columns = ['Close']
                        else:
                            if 'Close' not in data_df.columns:
                                return None, None
                            df = pd.DataFrame(data_df['Close'])
                        
                        # Scale the data
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
                        
                        # Prepare sequences for prediction
                        lookback = 100
                        X = []
                        dates = []
                        
                        for i in range(lookback, len(scaled_data)):
                            X.append(scaled_data[i-lookback:i, 0])
                            dates.append(df.index[i])
                        
                        if len(X) == 0:
                            return None, None
                        
                        X = np.array(X)
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                        
                        # Generate predictions
                        predictions_scaled = model.predict(X)
                        predictions = scaler.inverse_transform(predictions_scaled)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Date': dates,
                            'Actual': df['Close'][lookback:].values,
                            'Predicted': predictions.flatten()
                        })
                        results_df.set_index('Date', inplace=True)
                        
                        return results_df, scaler
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
                        return None, None
                
                # Generate 1-year predictions
                with st.spinner('Generating 1-year historical vs predicted analysis...'):
                    predictions_df, _ = generate_lstm_predictions(data)
                    
                    if predictions_df is not None:
                        # Filter for the actual date range
                        plot_data = predictions_df[
                            (predictions_df.index >= pd.Timestamp(start_date)) & 
                            (predictions_df.index <= pd.Timestamp(end_date))
                        ]
                        
                        # Create comparison chart
                        comparison_fig = go.Figure()
                        
                        # Add actual prices
                        comparison_fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['Actual'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add predicted prices
                        comparison_fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=plot_data['Predicted'],
                            mode='lines',
                            name='Predicted Price',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Add confidence bands
                        mae = np.mean(np.abs(plot_data['Actual'] - plot_data['Predicted']))
                        upper_bound = plot_data['Predicted'] + mae
                        lower_bound = plot_data['Predicted'] - mae
                        
                        comparison_fig.add_trace(go.Scatter(
                            x=plot_data.index.tolist() + plot_data.index.tolist()[::-1],
                            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,0,0,0)'),
                            name='Confidence Band',
                            showlegend=True
                        ))
                        
                        # Calculate performance metrics
                        mae_value = np.mean(np.abs(plot_data['Actual'] - plot_data['Predicted']))
                        rmse_value = np.sqrt(np.mean((plot_data['Actual'] - plot_data['Predicted'])**2))
                        accuracy = 100 - (mae_value / np.mean(plot_data['Actual']) * 100)
                        
                        comparison_fig.update_layout(
                            title=f"{stock_ticker} Historical vs Predicted Closing Prices {date_range_info}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            hovermode="x unified",
                            template="plotly_white",
                            height=500
                        )
                        
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # Display performance metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Error", f"${mae_value:.2f}")
                        with col2:
                            st.metric("RMSE", f"${rmse_value:.2f}")
                        with col3:
                            st.metric("Model Accuracy", f"{accuracy:.1f}%")
                        
                    else:
                        st.warning("Could not generate 1-year predictions. Please check the data availability.")
                
                # Add confidence disclaimer
                st.sidebar.info("⚠️ Disclaimer: This prediction is based on historical data and may not accurately reflect future market movements. Always do your own research before making investment decisions.")
                
                # Add insightful stock data section
                st.markdown("---")
                st.header("📊 Stock Insights & Analysis")
                
                # Calculate key metrics
                try:
                    # Extract close price data
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = None
                        for col in data.columns:
                            if 'Close' in str(col):
                                close_col = col
                                break
                        if close_col is None:
                            close_data = data.iloc[:, 3]  # Assume 4th column is Close
                        else:
                            close_data = data[close_col]
                    else:
                        close_data = data['Close']
                    
                    # Calculate metrics
                    current_price = close_data.iloc[-1]
                    previous_price = close_data.iloc[-2]
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100
                    
                    # Volume analysis
                    if isinstance(data.columns, pd.MultiIndex):
                        volume_col = None
                        for col in data.columns:
                            if 'Volume' in str(col):
                                volume_col = col
                                break
                        if volume_col is None:
                            volume_data = data.iloc[:, -1]
                        else:
                            volume_data = data[volume_col]
                    else:
                        volume_data = data['Volume']
                    
                    avg_volume = volume_data.mean()
                    current_volume = volume_data.iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    
                    # Price range analysis
                    max_52w = close_data.max()
                    min_52w = close_data.min()
                    price_range_pct = ((current_price - min_52w) / (max_52w - min_52w)) * 100
                    
                    # Volatility analysis
                    daily_returns = close_data.pct_change().dropna()
                    volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    
                    # Moving averages
                    ma_20 = close_data.rolling(window=20).mean().iloc[-1]
                    ma_50 = close_data.rolling(window=50).mean().iloc[-1]
                    
                    # RSI calculation (14-period)
                    delta = close_data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    # Display metrics in columns
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Current Price", f"${current_price:.2f}", 
                                 f"{price_change_pct:+.2f}%")
                        st.metric("52-Week Range", f"${min_52w:.2f} - ${max_52w:.2f}")
                        
                    with metric_col2:
                        st.metric("RSI (14)", f"{rsi:.1f}")
                        st.metric("Volatility", f"{volatility:.1f}%")
                        
                    with metric_col3:
                        st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
                        st.metric("Price vs 52W Range", f"{price_range_pct:.1f}%")
                    
                    # Technical indicators
                    st.subheader("🔍 Technical Indicators")
                    
                    # Trend analysis
                    trend_col1, trend_col2 = st.columns(2)
                    
                    with trend_col1:
                        # Moving average signals
                        if current_price > ma_20 > ma_50:
                            st.success("📈 Strong Bullish Trend")
                            st.write("Price above both 20-day and 50-day MA")
                        elif current_price > ma_20:
                            st.info("📊 Moderate Bullish Trend")
                            st.write("Price above 20-day MA but below 50-day MA")
                        elif current_price < ma_20 < ma_50:
                            st.error("📉 Strong Bearish Trend")
                            st.write("Price below both 20-day and 50-day MA")
                        else:
                            st.warning("⚖️ Mixed Signals")
                            st.write("Price between moving averages")
                    
                    with trend_col2:
                        # RSI interpretation
                        if rsi > 70:
                            st.error("🔴 Overbought (RSI > 70)")
                            st.write("Consider potential pullback")
                        elif rsi < 30:
                            st.success("🟢 Oversold (RSI < 30)")
                            st.write("Potential buying opportunity")
                        else:
                            st.info("🟡 Neutral Zone")
                            st.write("RSI in healthy range")
                    
                    # Volume insights
                    st.subheader("📈 Volume Analysis")
                    
                    if volume_ratio > 1.5:
                        st.success(f"🔥 High Volume Alert: {volume_ratio:.1f}x average")
                        st.write("Unusual trading activity detected")
                    elif volume_ratio < 0.5:
                        st.warning(f"💤 Low Volume: {volume_ratio:.1f}x average")
                        st.write("Below normal trading activity")
                    else:
                        st.info(f"📊 Normal Volume: {volume_ratio:.1f}x average")
                        st.write("Standard trading activity")
                    
                    # Support and Resistance levels
                    st.subheader("🎯 Key Levels")
                    
                    # Calculate support and resistance
                    recent_data = close_data.tail(20)
                    resistance = recent_data.max()
                    support = recent_data.min()
                    
                    level_col1, level_col2 = st.columns(2)
                    
                    with level_col1:
                        st.metric("Immediate Resistance", f"${resistance:.2f}")
                        st.metric("Distance to Resistance", f"${resistance - current_price:.2f}")
                        
                    with level_col2:
                        st.metric("Immediate Support", f"${support:.2f}")
                        st.metric("Distance to Support", f"${current_price - support:.2f}")
                    
                    # Market sentiment summary
                    st.subheader("🎯 Market Sentiment Summary")
                    
                    sentiment_score = 0
                    if current_price > ma_20: sentiment_score += 2
                    if current_price > ma_50: sentiment_score += 2
                    if rsi < 50: sentiment_score += 1
                    if rsi > 50: sentiment_score -= 1
                    if volume_ratio > 1.2: sentiment_score += 1
                    
                    if sentiment_score >= 4:
                        st.success("🟢 **BULLISH** - Strong upward momentum")
                    elif sentiment_score >= 2:
                        st.info("🟡 **NEUTRAL-BULLISH** - Moderate upward bias")
                    elif sentiment_score <= -2:
                        st.error("🔴 **BEARISH** - Downward pressure")
                    else:
                        st.warning("⚪ **NEUTRAL** - Mixed signals")
                        
                except Exception as e:
                    st.error(f"Could not calculate insights: {str(e)}")
                    st.info("Please ensure sufficient data is available")
                
                # Add simple footer
                st.markdown("---")
                st.markdown("""
                <div style='text-align: center; padding: 10px; color: #666;'>
                    <p><em>Disclaimer: This tool is for educational purposes. Always consult with financial advisors before making investment decisions.</em></p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.sidebar.error(f"Could not fetch data for {stock_ticker}. Please check the ticker symbol and try again.")