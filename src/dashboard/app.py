import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
import time
import random

# Import new clients
from src.core_ingestion.yahoo import fetch_yahoo_data
from src.core_ingestion.binance_ws import BinanceWSClient

st.set_page_config(page_title="Advanced Market Forecaster", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = "data/market_data.csv"
SEQ_LENGTH = 24

st.title("📈 Advanced Market Forecaster")
st.markdown("Visual dashboard for predicting the next time step's closing price with real-time verification.")

# --- Sidebar ---
st.sidebar.header("Settings & Data Source")
data_source = st.sidebar.selectbox("Select Data Source", ["Local Synthetic CSV", "Yahoo Finance", "Live Binance WS"])

mode = st.sidebar.radio("Operating Mode", ["Manual", "Step-by-Step", "Auto"])

try:
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        st.sidebar.success("API Status: Online")
    else:
        st.sidebar.error(f"API Status: Error {health.status_code}")
except requests.exceptions.RequestException:
    st.sidebar.error("API Status: Offline")

# --- Load Data Based on Source ---
df = None

if data_source == "Local Synthetic CSV":
    @st.cache_data
    def load_local_data():
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        return None
    df = load_local_data()

elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    if st.sidebar.button("Fetch Data"):
        st.cache_data.clear() # Clear cache to force re-fetch
    
    @st.cache_data(ttl=600)
    def load_yahoo_data(sym):
        try:
            return fetch_yahoo_data(sym)
        except Exception as e:
            st.error(f"Error fetching Yahoo Data: {e}")
            return None
    
    with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
        df = load_yahoo_data(ticker)

elif data_source == "Live Binance WS":
    ticker = st.sidebar.text_input("Binance Symbol", value="btcusdt")
    
    # Store the ws client in session state so we don't recreate it on every streamlit rerun
    if "ws_client" not in st.session_state or st.session_state.ws_client.symbol != ticker.lower():
        st.session_state.ws_client = BinanceWSClient(symbol=ticker.lower(), interval="1m", max_buffer=300)
        st.session_state.ws_client.start()
        
    df = st.session_state.ws_client.get_dataframe()
    if len(df) < SEQ_LENGTH + 1:
        st.warning(f"Buffering live data for {ticker.upper()}... Waiting for at least {SEQ_LENGTH + 1} points. Currently have: {len(df)}")
        time.sleep(2)
        st.rerun()

# --- Main Logic ---
if df is not None and len(df) >= SEQ_LENGTH + 1:
    # Ensure we have at least 1 point in the future for truth verification
    max_idx = len(df) - SEQ_LENGTH - 1 
    
    # Initialize session state variables for Step-by-Step
    if "step_idx" not in st.session_state or st.session_state.step_idx > max_idx:
        st.session_state.step_idx = max_idx
        
    def plot_sequence(start_idx, show_prediction=False):
        end_idx = start_idx + SEQ_LENGTH
        seq_df = df.iloc[start_idx:end_idx]
        
        # Ground truth next step
        truth_step = df.iloc[end_idx:end_idx+1]
        
        fig = go.Figure()
        # Historical trace
        fig.add_trace(go.Scatter(x=seq_df.index, y=seq_df['close'], mode='lines+markers', name='Historical Close', line=dict(color='#2E86C1', width=2)))
        
        if show_prediction:
            with st.spinner("Calling API for prediction..."):
                sequence_data = seq_df[['close', 'volume']].values.tolist()
                try:
                    response = requests.post(f"{API_URL}/predict", json={"sequence": sequence_data})
                    if response.status_code == 200:
                        pred_val = response.json().get("predicted_close")
                        truth_val = truth_step['close'].values[0]
                        
                        # Plot Prediction (Red Cross)
                        fig.add_trace(go.Scatter(x=truth_step.index, y=[pred_val], mode='markers', marker=dict(size=14, color='#E74C3C', symbol='x', line=dict(width=2, color='white')), name='Prediction (Model)'))
                        
                        # Plot Actual Truth (Green Circle)
                        fig.add_trace(go.Scatter(x=truth_step.index, y=[truth_val], mode='markers', marker=dict(size=12, color='#27AE60', symbol='circle'), name='Actual Truth (Data)'))
                        
                        # Compute Error
                        err = abs(pred_val - truth_val)
                        err_pct = (err / truth_val) * 100 if truth_val != 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Predicted Price", f"{pred_val:.4f}")
                        col2.metric("Actual Price", f"{truth_val:.4f}")
                        col3.metric("Absolute Error", f"{err:.4f}", f"{err_pct:.2f}%", delta_color="inverse")
                        
                    else:
                        st.error(f"Prediction failed. Status Code: {response.status_code}")
                except Exception as e:
                    st.error(f"API Request Failed: {e}")
                    
        fig.update_layout(
            title=f"24-Step Sequence ({data_source})", 
            xaxis_title="Time", 
            yaxis_title="Price",
            hovermode="x unified",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    if mode == "Manual":
        st.subheader("🔧 Manual Mode")
        st.markdown("Select an index manually and verify the prediction against actual future data.")
        # Reverse the logic to allow end-of-list by default
        start_idx = st.slider("Select Sequence Start Index", 0, max_idx, max_idx)
        
        if st.button("🔮 Predict & Verify"):
            plot_sequence(start_idx, show_prediction=True)
        else:
            plot_sequence(start_idx, show_prediction=False)
            
        if data_source == "Live Binance WS":
            if st.button("🔄 Refresh Data"):
                st.rerun()

    elif mode == "Step-by-Step":
        st.subheader("👣 Step-by-Step Mode")
        st.markdown("Advance through time step by step to see how the model adapts.")
        
        jump_col1, jump_col2 = st.columns([3, 1])
        with jump_col1:
            jump_idx = st.number_input("Jump to Index", min_value=0, max_value=max_idx, value=st.session_state.step_idx)
        with jump_col2:
            st.write("")
            st.write("")
            if st.button("Go"):
                st.session_state.step_idx = jump_idx
                st.rerun()

        if st.button("🔮 Predict & Advance 1 Step", type="primary"):
            plot_sequence(st.session_state.step_idx, show_prediction=True)
            if st.session_state.step_idx < max_idx:
                st.session_state.step_idx += 1
        else:
            plot_sequence(st.session_state.step_idx, show_prediction=False)

    elif mode == "Auto":
        st.subheader("🤖 Auto Mode")
        if data_source == "Live Binance WS":
            st.markdown("Monitoring the live websocket stream. Automatically evaluating the latest window...")
            auto_placeholder = st.empty()
            with auto_placeholder.container():
                st.markdown(f"**Evaluating latest buffered batch...**")
                plot_sequence(max_idx, show_prediction=True)
                
            progress_text = "Waiting 5 seconds for new live data..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05) 
                my_bar.progress(percent_complete + 1, text=progress_text)
            st.rerun()
            
        else:
            st.markdown("Simulating continuous live monitoring. A new random batch is evaluated every 10 seconds.")
            auto_placeholder = st.empty()
            
            with auto_placeholder.container():
                random_idx = random.randint(0, max_idx)
                st.markdown(f"**Loading random batch from index {random_idx}...**")
                plot_sequence(random_idx, show_prediction=True)
                
            progress_text = "Waiting 10 seconds for next batch..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.1) 
                my_bar.progress(percent_complete + 1, text=progress_text)
            st.rerun()

elif df is not None and len(df) < SEQ_LENGTH + 1 and data_source != "Live Binance WS":
    st.warning("Not enough data to form a sequence (needs 24+1 points).")
