import streamlit as st
import yfinance as yf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the trained model
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("📈 Stock Market Trend Prediction (Local)")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOG)", "AAPL").upper()
days = st.slider("Select Days of Data", 30, 365, 90)

# Fetch stock data
end_date = datetime.today()
start_date = end_date - timedelta(days=days)
stock_data = yf.download(ticker, start=start_date, end=end_date)

if not stock_data.empty:
    st.subheader(f"📊 {ticker} Stock Price Trends")
    st.line_chart(stock_data["Close"])

    # Predict Future Prices using Loaded Model
    future_days = np.array([[len(stock_data) + i] for i in range(1, 6)])
    predicted_prices = model.predict(future_days)
    
    st.subheader("📉 Predicted Prices for Next 5 Days")
    for i, price in enumerate(predicted_prices):
        st.write(f"Day {i+1}: ${price:.2f}")

else:
    st.error("❌ Invalid Ticker! Please enter a valid stock symbol.")