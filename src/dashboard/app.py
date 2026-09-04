import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
import time
import random

st.set_page_config(page_title="Advanced Market Forecaster", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = "data/market_data.csv"
SEQ_LENGTH = 24

st.title("📈 Advanced Market Forecaster")
st.markdown("Visual dashboard for predicting the next time step's closing price with real-time verification.")

# --- Sidebar ---
st.sidebar.header("Settings & Modes")
mode = st.sidebar.radio("Operating Mode", ["Manual", "Step-by-Step", "Auto"])

try:
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        st.sidebar.success("API Status: Online")
    else:
        st.sidebar.error(f"API Status: Error {health.status_code}")
except requests.exceptions.RequestException:
    st.sidebar.error("API Status: Offline")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return None

df = load_data()

if df is not None:
    # Ensure we have at least 1 point in the future for truth verification
    max_idx = len(df) - SEQ_LENGTH - 1 
    
    # Initialize session state variables for Step-by-Step
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = max_idx - 100
        
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
                        err_pct = (err / truth_val) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Predicted Price", f"{pred_val:.4f}")
                        col2.metric("Actual Price", f"{truth_val:.4f}")
                        col3.metric("Absolute Error", f"{err:.4f}", f"{err_pct:.2f}%", delta_color="inverse")
                        
                    else:
                        st.error(f"Prediction failed. Status Code: {response.status_code}")
                except Exception as e:
                    st.error(f"API Request Failed: {e}")
                    
        fig.update_layout(
            title=f"24-Step Sequence (Start Index: {start_idx})", 
            xaxis_title="Time", 
            yaxis_title="Price",
            hovermode="x unified",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    if mode == "Manual":
        st.subheader("🔧 Manual Mode")
        st.markdown("Select an index manually and verify the prediction against actual future data.")
        start_idx = st.slider("Select Sequence Start Index", 0, max_idx, max_idx - 100)
        
        if st.button("🔮 Predict & Verify"):
            plot_sequence(start_idx, show_prediction=True)
        else:
            plot_sequence(start_idx, show_prediction=False)

    elif mode == "Step-by-Step":
        st.subheader("👣 Step-by-Step Mode")
        st.markdown("Advance through time step by step to see how the model adapts.")
        
        # Give user a chance to jump to a specific index
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
            st.session_state.step_idx += 1
        else:
            plot_sequence(st.session_state.step_idx, show_prediction=False)

    elif mode == "Auto":
        st.subheader("🤖 Auto Mode")
        st.markdown("Simulating continuous live monitoring. A new random batch is evaluated every 10 seconds.")
        
        auto_placeholder = st.empty()
        
        with auto_placeholder.container():
            random_idx = random.randint(0, max_idx)
            st.markdown(f"**Loading random batch from index {random_idx}...**")
            plot_sequence(random_idx, show_prediction=True)
            
        progress_text = "Waiting 10 seconds for next batch..."
        my_bar = st.progress(0, text=progress_text)
        
        # 10 second countdown via progress bar
        for percent_complete in range(100):
            time.sleep(0.1) 
            my_bar.progress(percent_complete + 1, text=progress_text)
            
        st.rerun()

else:
    st.warning(f"Data not found at `{DATA_PATH}`.")
