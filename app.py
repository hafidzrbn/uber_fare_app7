# ============================================================
# 🚖 UBER FARE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ============================================================
# 1️⃣ Definisi Fungsi Kustom
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
        st.error(f"❌ File model '{model_path}' tidak ditemukan.")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ============================================================
# 3️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Masukkan detail perjalanan Anda untuk memprediksi tarif Uber")

# ============================================================
# 4️⃣ Input Form
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 Trip Details Input")
    
    # Menggunakan datetime_input untuk input yang lebih intuitif
    pickup_date = st.date_input("Pickup Date", value=datetime.now())
    pickup_time = st.time_input("Pickup Time", value=datetime.now())

    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.number_input("Pickup Latitude", value=40.738354, format="%.6f")
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.999817, format="%.6f")
    
    with col2:
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.723217, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.999512, format="%.6f")

    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    submitted = st.form_submit_button("🔮 Predict Fare")

    if submitted:
        try:
            # Menggabungkan date dan time menjadi satu objek datetime
            pickup_datetime_full = datetime.combine(pickup_date, pickup_time)
            
            # Buat DataFrame dari input mentah pengguna
            input_data = pd.DataFrame([{
                'key': 'dummy_key',
                'pickup_longitude': pickup_longitude,
                'pickup_latitude': pickup_latitude,
                'dropoff_longitude': dropoff_longitude,
                'dropoff_latitude': dropoff_latitude,
                'passenger_count': passenger_count,
                'pickup_datetime': pickup_datetime_full
            }])
            
            # Prediksi menggunakan pipeline
            prediction = model.predict(input_data)
            
            st.subheader("✅ Prediction Successful!")
            st.success(f"Predicted Uber fare: ${prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")