# ============================================================
# 🚖 UBER FARE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import folium
from streamlit_folium import st_folium

# ============================================================
# 1️⃣ Custom Functions Definition
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    return distance_km

def convert_to_datetime(df):
    df_copy = df.copy()
    df_copy['pickup_datetime'] = pd.to_datetime(df_copy['pickup_datetime'])
    return df_copy

def create_features(df):
    df_copy = df.copy()
    df_copy['trip_distance_km'] = df_copy.apply(
        lambda row: haversine(
            row['pickup_latitude'], row['pickup_longitude'],
            row['dropoff_latitude'], row['dropoff_longitude']
        ), axis=1
    )
    df_copy['year'] = df_copy['pickup_datetime'].dt.year
    df_copy['month'] = df_copy['pickup_datetime'].dt.month
    df_copy['day'] = df_copy['pickup_datetime'].dt.day
    df_copy['dayofweek'] = df_copy['pickup_datetime'].dt.dayofweek
    df_copy['hour'] = df_copy['pickup_datetime'].dt.hour
    
    df_copy = df_copy.drop(columns=['pickup_datetime', 'key', 'Unnamed: 0', 
                                    'fare_per_km', 'fare_per_passenger', 
                                    'log_fare_amount', 'is_weekend'], errors='ignore')
    return df_copy

# ============================================================
# 2️⃣ Load Best Model
# ============================================================
@st.cache_resource
def load_model():
    model_path = "random_forest_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found.")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ============================================================
# 3️⃣ App Configuration & State Management
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Select your pickup and drop-off points on the map to predict your Uber fare.")

if 'pickup' not in st.session_state:
    st.session_state.pickup = None
if 'dropoff' not in st.session_state:
    st.session_state.dropoff = None

# ============================================================
# 4️⃣ Interactive Map
# ============================================================
m = folium.Map(location=[40.7, -74.0], zoom_start=11)

if st.session_state.pickup:
    folium.Marker(
        st.session_state.pickup,
        popup="Pickup Location",
        icon=folium.Icon(color="green", icon="fa-car", prefix='fa')
    ).add_to(m)
if st.session_state.dropoff:
    folium.Marker(
        st.session_state.dropoff,
        popup="Dropoff Location",
        icon=folium.Icon(color="red", icon="fa-flag-checkered", prefix='fa')
    ).add_to(m)

map_data = st_folium(m, height=400, width=1000, returned_objects=["last_clicked"])

# Logic to store clicked coordinates and display messages
if map_data and map_data.get("last_clicked"):
    coords = map_data["last_clicked"]
    new_coords = (coords['lat'], coords['lng'])

    if not st.session_state.pickup:
        st.session_state.pickup = new_coords
        st.rerun()
    elif not st.session_state.dropoff:
        st.session_state.dropoff = new_coords
        st.rerun()
    else:
        st.session_state.pickup = None
        st.session_state.dropoff = None
        st.rerun()

# Display messages based on state
if not st.session_state.pickup:
    st.info("👋 Please **click on the map** to select your pickup point.")
elif st.session_state.pickup and not st.session_state.dropoff:
    st.info("👉 Pickup point selected. Please **click again on the map** to choose your drop-off point.")
else:
    st.success("✅ Pickup and drop-off points have been selected. Please fill in other details to predict the fare.")
    st.markdown("🔄 **Click on the map again** to choose new pickup and drop-off points.")


# ============================================================
# 5️⃣ Other Form Inputs & Prediction
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 More Trip Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Pickup Point:** {st.session_state.pickup}")
    with col2:
        st.info(f"**Drop-off Point:** {st.session_state.dropoff}")
    
    pickup_date = st.date_input("Pickup Date", value=datetime.now())
    pickup_time = st.time_input("Pickup Time", value=datetime.now())
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    submitted = st.form_submit_button("🔮 Predict Fare")

    if submitted:
        if st.session_state.pickup and st.session_state.dropoff:
            try:
                pickup_datetime_full = datetime.combine(pickup_date, pickup_time)
                
                input_data = pd.DataFrame([{
                    'key': 'dummy_key',
                    'pickup_longitude': st.session_state.pickup[1],
                    'pickup_latitude': st.session_state.pickup[0],
                    'dropoff_longitude': st.session_state.dropoff[1],
                    'dropoff_latitude': st.session_state.dropoff[0],
                    'passenger_count': passenger_count,
                    'pickup_datetime': pickup_datetime_full
                }])
                
                prediction = model.predict(input_data)
                
                st.subheader("✅ Prediction Successful!")
                st.success(f"Predicted Uber fare: ${prediction[0]:,.2f}")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
        else:
            st.error("⚠️ Please select both pickup and drop-off points on the map.")
