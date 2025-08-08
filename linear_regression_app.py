import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import altair as alt
import joblib

# Set page title and layout
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# App title and description
st.title("Stock Price Prediction App (Linear Regression)")
st.markdown("This app uses a Linear Regression model to predict the next day's closing price based on historical data.")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('linear_regression_stock_model.joblib')

model = load_model()

# Fetch stock data
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Feature calculations
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Prepare features for model
def prepare_data(data):
    if 'Close' not in data.columns:
        st.error("Could not find 'Close' column in the data.")
        return None, None

    df = data.copy()
    df['SMA_100'] = calculate_sma(df, 100)
    df['SMA_50'] = calculate_sma(df, 50)
    df['EMA_100'] = calculate_ema(df, 100)
    df['EMA_50'] = calculate_ema(df, 50)

    if len(df) < 100:
        st.error("Insufficient data for feature calculation. Need at least 100 days of data.")
        return None, None

    df_features_only = df.dropna()

    if df_features_only.empty:
        st.error("Insufficient data after calculating features.")
        return None, df

    X_test = df_features_only[['SMA_100', 'SMA_50', 'EMA_100', 'EMA_50']].iloc[-1].values.reshape(1, -1)
    return X_test, df_features_only

# Input section
st.header("Input Parameters")
today = dt.datetime.now().date()
default_start_date = today - dt.timedelta(days=365)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL, MSFT)", "AAPL")
with col2:
    start_date = st.date_input("Start Date", default_start_date, max_value=today)
with col3:
    end_date = st.date_input("End Date", today, max_value=today)

center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
with center_col2:
    predict_button = st.button("Predict", use_container_width=True)

if predict_button:
    if stock_ticker and start_date and end_date:
        data = fetch_stock_data(stock_ticker, start_date, end_date)

        if data is not None and not data.empty:
            X_test, df_features = prepare_data(data)

            if X_test is not None and df_features is not None and not df_features.empty:
                df_train = df_features.dropna()
                X_train = df_train[['SMA_100', 'SMA_50', 'EMA_100', 'EMA_50']]
                y_train_predicted = model.predict(X_train)

                df_features['Predicted_Close'] = np.nan
                df_features.loc[df_train.index, 'Predicted_Close'] = y_train_predicted.flatten()

                predicted_price = model.predict(X_test).item()
                last_price = df_features['Close'].iloc[-1].item()
                change_percent = ((predicted_price - last_price) / last_price) * 100

                # Display prediction results
                st.subheader("Prediction Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Last Closing Price", f"${last_price:.2f}")
                with col2:
                    st.metric("Predicted Next Day Price", f"${predicted_price:.2f}", f"{change_percent:.2f}%")
                with col3:
                    trend = "Bullish 📈" if change_percent > 0 else "Bearish 📉"
                    st.metric("Prediction", trend)
                with col4:
                    st.metric("Confidence", "Linear Regression Model")

                st.subheader(f"Data Summary for {stock_ticker}")
                st.dataframe(data.tail(5))

                st.subheader("Price History and Predictions (Last 30 Days)")

                # Prepare data for Altair chart
                chart_data = df_features.tail(30).reset_index()
                chart_data['Date'] = pd.to_datetime(chart_data['Date'])

                # Create a DataFrame for the predicted next day's price
                predicted_next_day_date = chart_data['Date'].iloc[-1] + pd.Timedelta(days=1)
                predicted_next_day_df = pd.DataFrame({
                    'Date': [predicted_next_day_date],
                    'Close': [predicted_price],
                    'Type': ['Predicted']
                })

                # Add the last actual closing price to the predicted_next_day_df for the connecting line
                last_actual_price_df = pd.DataFrame({
                    'Date': [chart_data['Date'].iloc[-1]],
                    'Close': [chart_data['Close'].iloc[-1]],
                    'Type': ['Actual']
                })

                # Combine actual and predicted data for the connecting line
                connecting_line_data = pd.concat([last_actual_price_df, predicted_next_day_df])

                # Base chart for actual closing prices
                base = alt.Chart(chart_data).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Close:Q', title='Closing Price (USD)', scale=alt.Scale(zero=False))
                ).properties(
                    title=f'{stock_ticker} Stock Price Prediction'
                )

                # Line for actual closing prices
                line = base.mark_line(color='blue').encode(
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Close:Q']
                )

                # Points for actual closing prices
                points = base.mark_circle(size=60, color='blue').encode(
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Close:Q']
                )

                # Connecting line to predicted price
                connecting_line = alt.Chart(connecting_line_data).mark_line(color='red', strokeDash=[5, 5]).encode(
                    x='Date:T',
                    y='Close:Q'
                )

                # Predicted point
                predicted_point = alt.Chart(predicted_next_day_df).mark_circle(size=100, color='red').encode(
                    x='Date:T',
                    y='Close:Q',
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Close:Q', title='Predicted Price')]
                )

                # Combine all layers
                chart = (line + points + connecting_line + predicted_point).interactive()

                st.altair_chart(chart, use_container_width=True)

                st.subheader("Historical Price (Last Year) and Volume Analysis")

                # Fetch data for the last year for historical analysis
                one_year_ago = today - dt.timedelta(days=365)
                historical_data = fetch_stock_data(stock_ticker, one_year_ago, today)

                if historical_data is not None and not historical_data.empty:
                    historical_data = historical_data.reset_index()
                    historical_data['Date'] = pd.to_datetime(historical_data['Date'])

                    # Historical Price Chart
                    price_chart = alt.Chart(historical_data).mark_line(color='darkblue').encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Close:Q', title='Closing Price', scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Close:Q']
                    ).properties(
                        title=f'{stock_ticker} Historical Closing Price (Last Year)'
                    )
                    col_price, col_volume = st.columns(2)
                    with col_price:
                        st.altair_chart(price_chart, use_container_width=True)

                    with col_volume:
                        # Volume Analysis Chart
                        volume_chart = alt.Chart(historical_data).mark_bar(color='orange').encode(
                            x=alt.X('Date:T', title='Date'),
                            y=alt.Y('Volume:Q', title='Volume'),
                            tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), 'Volume:Q']
                        ).properties(
                            title=f'{stock_ticker} Volume Analysis (Last Year)'
                        )
                        st.altair_chart(volume_chart, use_container_width=True)
                else:
                    st.warning("Could not fetch historical data for the last year.")
            else:
                st.warning("Could not prepare data for prediction. Please ensure sufficient historical data is available.")
        else:
            st.warning("No data fetched for the given ticker and date range.")
    else:
        st.warning("Please enter a stock ticker and select a date range.")

# Sidebar
st.sidebar.subheader("About the Model")
st.sidebar.write("""
This app uses a Linear Regression model to predict the next day's closing price based on:
- SMA (Simple Moving Averages): 50 & 100
- EMA (Exponential Moving Averages): 50 & 100
""")
st.sidebar.write("""
Note: This model is a basic linear regression and does not account for complex market factors.
For more accurate predictions, consider using more advanced models and including additional features.
""")
st.sidebar.markdown("---")
