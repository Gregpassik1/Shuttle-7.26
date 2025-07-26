
# Save this file as: app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Shuttle Tech Platform",
    page_icon="🚐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state init
if 'data' not in st.session_state:
    st.session_state.data = None

# Generate sample data
def generate_sample_data():
    np.random.seed(42)
    pickup_locations = ["Hilton Downtown", "Marriott Downtown", "Sheraton Downtown", "ANC Airport"]
    time_blocks = [f"{h:02d}:{m:02d}" for h in range(24) for m in [0, 30]]
    start_date = datetime.now() - timedelta(days=365)
    data = []
    for day in range(365):
        date = (start_date + timedelta(days=day)).date()
        for tb in time_blocks:
            for loc in pickup_locations:
                pc = np.random.poisson(2 if "Airport" in loc else 5)
                if np.random.rand() > 0.7:
                    data.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "time_block": tb,
                        "pickup_location": loc,
                        "passenger_count": pc
                    })
    return pd.DataFrame(data)

# Upload or generate data
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.data = df
elif st.sidebar.button("Generate Sample Data"):
    df = generate_sample_data()
    st.session_state.data = df

# If data is available
if st.session_state.data is not None:
    df = st.session_state.data
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()

    st.markdown('<div class="main-header"><h1>🚐 Shuttle Tech Platform</h1></div>', unsafe_allow_html=True)

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Total Passengers", f"{df['passenger_count'].sum():,}")
    with col2:
        avg_pass = df.groupby('date')['passenger_count'].sum().mean()
        st.metric("Avg Passengers per Day", f"{avg_pass:.1f}")

    # Plots
    fig = px.density_heatmap(df, x="day_of_week", y="time_block", z="passenger_count", histfunc="avg", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(df.groupby("pickup_location")["passenger_count"].sum().reset_index(),
                  x="pickup_location", y="passenger_count", title="Total Passengers by Pickup Location")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Upload a CSV file or generate sample data to get started.")
