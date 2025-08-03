# Stock Price Prediction App

This application uses an LSTM (Long Short-Term Memory) neural network model to predict stock prices based on historical data. The model has been trained on historical stock data and can predict the next day's closing price based on the last 100 days of closing prices.

## Features

- Input a stock ticker symbol (e.g., "AAPL", "GOOGL")
- Select a date range for historical data
- View historical stock data
- Get a prediction for the next day's closing price
- Visualize the historical prices and prediction in an interactive chart

## Requirements

All required packages are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Make sure you have all the required packages installed
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. The app will open in your default web browser

## Model Information

The LSTM model used in this application:

- Was trained on historical stock data
- Uses the MinMaxScaler from scikit-learn to normalize the data
- Has multiple LSTM layers with dropout for regularization
- Was trained to minimize mean squared error
- Takes the last 100 days of closing prices as input
- Outputs a prediction for the next day's closing price

## Disclaimer

This application is for educational and demonstration purposes only. The predictions made by this model should not be used as financial advice. Always do your own research before making investment decisions.