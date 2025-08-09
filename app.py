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
# 1️⃣ Definisi Fungsi Kustom (Sama seperti sebelumnya)
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
# 2️⃣ Load Best Model (Sama seperti sebelumnya)
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
# 3️⃣ App Configuration & State Management
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Select your pickup and drop-off points on the map to predict your Uber fare.")

# Inisialisasi state untuk menyimpan koordinat
if 'pickup' not in st.session_state:
    st.session_state.pickup = None
if 'dropoff' not in st.session_state:
    st.session_state.dropoff = None

# ============================================================
# 4️⃣ Peta Interaktif
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

# Logic untuk menyimpan koordinat klik
if map_data and map_data.get("last_clicked"):
    coords = map_data["last_clicked"]
    new_coords = (coords['lat'], coords['lng'])

    if not st.session_state.pickup:
        st.session_state.pickup = new_coords
        st.warning("Titik penjemputan terpilih. Silakan pilih titik pengantaran.")
        # Mengganti st.experimental_rerun() dengan st.rerun()
        st.rerun()
    elif not st.session_state.dropoff:
        st.session_state.dropoff = new_coords
        st.success("Titik pengantaran terpilih. Anda dapat melanjutkan ke prediksi.")
        # Mengganti st.experimental_rerun() dengan st.rerun()
        st.rerun()
    else:
        st.session_state.pickup = None
        st.session_state.dropoff = None
        # Mengganti st.experimental_rerun() dengan st.rerun()
        st.rerun()

# ============================================================
# 5️⃣ Input Form Lainnya & Prediksi
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 More Trip Details")

    # Menampilkan koordinat yang dipilih
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
                # Menggabungkan date dan time menjadi satu objek datetime
                pickup_datetime_full = datetime.combine(pickup_date, pickup_time)
                
                # Buat DataFrame dari input mentah
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
            st.error("⚠️ Mohon pilih titik penjemputan dan pengantaran pada peta.")

