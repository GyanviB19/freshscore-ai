
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from streamlit_lottie import st_lottie
import requests

# --- App Configuration ---
st.set_page_config(page_title="FreshScore Predictor", layout="wide")

# --- Lottie Animation Loader ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- Load Animation ---
lottie_json = load_lottieurl("https://lottie.host/f00c9014-fef2-4a44-8ccf-93e5d7d0cc6e/7TBG3gzdcO.json")

# --- Sidebar Theme Switch ---
theme = st.sidebar.radio("Choose Theme", ("🌞 Light", "🌙 Dark"))

if theme == "🌙 Dark":
    bg_color = "#0e1117"
    text_color = "#FAFAFA"
    meter_color = "lightgreen"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    meter_color = "green"

# --- Header Section ---
st.markdown(f"<h1 style='color:{text_color}; text-align:center;'>🍓 FreshScore Predictor – ColdChain AI</h1>", unsafe_allow_html=True)
if lottie_json:
    st_lottie(lottie_json, height=200, key="coldchain")
else:
    st.warning("⚠️ Animation failed to load. Please check your internet or try again later.")

# --- Layout Columns ---
left, right = st.columns([1, 1.2])

with left:
    with st.form("freshscore_form"):
        st.subheader("📋 Enter Item Details")
        category = st.selectbox("Perishable Category", [
            "strawberries", "flowers", "frozen_food", "milk", "vaccines", "cheese",
            "meat", "leafy_greens", "ice_cream", "seafood", "juice", "eggs", "yogurt", "berries", "herbs"
        ])
        temperature = st.slider("Internal Temperature (°C)", 0.0, 25.0, 6.0)
        humidity = st.slider("Humidity (%)", 30, 100, 75)
        duration_hours = st.slider("Transit Duration (hours)", 0.0, 48.0, 8.0)
        distance_km = st.slider("Distance Traveled (km)", 0.0, 2000.0, 250.0)
        door_openings = st.slider("Door Openings", 0, 10, 2)
        submitted = st.form_submit_button("🚀 Predict FreshScore")

if submitted:
    # One-hot encode input
    input_data = {
        "temperature": temperature,
        "humidity": humidity,
        "duration_hours": duration_hours,
        "distance_km": distance_km,
        "door_openings": door_openings
    }
    for cat in [
        "berries", "cheese", "eggs", "flowers", "frozen_food", "herbs", "ice_cream",
        "juice", "leafy_greens", "meat", "milk", "seafood", "strawberries", "vaccines", "yogurt"
    ]:
        input_data[f"category_{cat}"] = 1 if category == cat else 0

    input_df = pd.DataFrame([input_data])

    # Train dummy model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_dummy, y_dummy = make_regression(n_samples=1000, n_features=len(input_df.columns), noise=0.1)
    model.fit(X_dummy, y_dummy)

    # Predict
    score = model.predict(input_df)[0]
    score = float(np.clip(score, 0, 100))

    with right:
        st.subheader("🌡️ Freshness Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "FreshScore (0–100)", 'font': {'size': 24}},
            delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': meter_color},
                'steps': [
                    {'range': [0, 60], 'color': 'red'},
                    {'range': [60, 80], 'color': 'yellow'},
                    {'range': [80, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        st.plotly_chart(fig)

        # Status feedback
        if score > 80:
            st.success("✅ Excellent freshness maintained!")
            st.balloons()
        elif score > 60:
            st.info("🟡 Acceptable freshness level.")
        else:
            st.error("🚨 Spoilage risk! Immediate action recommended.")

        # Explanation block
        st.markdown("""
        ---
        ### 🧾 What Does the FreshScore Mean?

        The **FreshScore** is a machine learning–generated indicator (0–100) representing the likely freshness and quality of a perishable product at the point of evaluation.

        - **FreshScore 80–100:** ✅ **Healthy & Fresh**  
          Optimal temperature, low door openings, short duration. Products are likely to retain full quality, safety, and shelf life.

        - **FreshScore 60–79:** ⚠️ **Acceptable but Monitor Closely**  
          Slight deviation in temperature or transit duration. While still usable, quality may begin to degrade.

        - **FreshScore below 60:** 🚨 **Spoilage Risk**  
          High probability of reduced quality, contamination, or spoilage. Action needed — reroute, inspect, or dispose.

        This score helps cold chain operators, retailers, and quality controllers make **real-time decisions** based on AI-powered predictions.
        """)

