import streamlit as st
import numpy as np
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Prediksi Potensi Wisata Pantai",
    page_icon="🌊",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('model_xgboost_wisata.pkl')
    le_target = joblib.load('label_encoder_target.pkl')
    le_provinsi = joblib.load('le_provinsi.pkl')
    return model, le_target, le_provinsi

model, le_target, le_provinsi = load_artifacts()

st.title("🌊 Prediksi Potensi Wisata Pantai Indonesia")
st.write("Aplikasi berbasis *Machine Learning* (XGBoost) untuk memprediksi tingkat potensi wisata pantai berdasarkan data geospasial dan rating.")
st.markdown("---")

# Layout menjadi 2 kolom: Kiri untuk Input, Kanan untuk Peta
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 Parameter Input")
    daftar_provinsi = list(le_provinsi.classes_)
    provinsi_pilihan = st.selectbox("Pilih Provinsi", daftar_provinsi, index=daftar_provinsi.index("Lampung") if "Lampung" in daftar_provinsi else 0)
    
    rating_angka = st.slider("Rating Angka Pantai", 1.0, 5.0, 4.5, 0.1)
    
    # Titik awal peta disetel ke area Bandar Lampung
    latitude = st.number_input("Latitude", value=-5.4254, format="%.4f")
    longitude = st.number_input("Longitude", value=105.2580, format="%.4f")

with col2:
    st.subheader("🗺️ Peta Lokasi Pantai")
    # Membuat peta interaktif dengan Folium
    m = folium.Map(location=[latitude, longitude], zoom_start=11)
    
    # Menambahkan penanda (marker) di peta
    folium.Marker(
        [latitude, longitude], 
        popup=f"Titik Pantai di {provinsi_pilihan}", 
        tooltip="Klik untuk info",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Menampilkan peta di Streamlit
    st_folium(m, width=400, height=350)

st.markdown("---")

# Tombol Prediksi ditaruh di tengah bawah
if st.button("🔍 Prediksi Potensi Sekarang", type="primary", use_container_width=True):
    try:
        provinsi_encoded = le_provinsi.transform([provinsi_pilihan])[0]
        data_input = np.array([[rating_angka, latitude, longitude, provinsi_encoded]])
        
        prediksi_encoded = model.predict(data_input)
        hasil_prediksi = le_target.inverse_transform(prediksi_encoded)[0]
        
        st.markdown("### 🎯 Hasil Analisis Potensi Wisata:")
        st.success(f"Berdasarkan koordinat peta dan parameter, potensi pantai ini diprediksi masuk kategori: **{hasil_prediksi.upper()}**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.caption("Developed with Streamlit & XGBoost | Geospatial Data Mining Project")
