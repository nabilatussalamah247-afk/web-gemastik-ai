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

# Layout menjadi 2 kolom
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 Parameter Input")
    
    # 🌟 TAMBAHAN: Input Nama Pantai
    nama_pantai = st.text_input("Nama Pantai", value="Pantai Mutun")
    
    daftar_provinsi = list(le_provinsi.classes_)
    provinsi_pilihan = st.selectbox("Pilih Provinsi", daftar_provinsi, index=daftar_provinsi.index("Lampung") if "Lampung" in daftar_provinsi else 0)
    
    rating_angka = st.slider("Rating Angka Pantai", 1.0, 5.0, 4.5, 0.1)
    
    latitude = st.number_input("Latitude", value=-5.5312, format="%.4f")
    longitude = st.number_input("Longitude", value=105.2715, format="%.4f")

with col2:
    st.subheader("🗺️ Peta Lokasi Pantai")
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    
    # 🌟 TAMBAHAN: Desain Kotak Informasi (Popup) menggunakan HTML
    popup_info = f"""
    <div style="font-family: Arial; font-size: 12px; min-width: 180px;">
        <h4 style="margin-top: 0px; margin-bottom: 5px; color: #1f77b4;">{nama_pantai}</h4>
        <b>Provinsi:</b> {provinsi_pilihan}<br>
        <b>Rating:</b> {rating_angka} ⭐<br>
        <b>Koordinat:</b> {latitude}, {longitude}
    </div>
    """
    
    # Menambahkan penanda (marker) dengan popup HTML
    folium.Marker(
        [latitude, longitude], 
        popup=folium.Popup(popup_info, max_width=300), 
        tooltip="Klik untuk detail pantai",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    st_folium(m, width=400, height=350)

st.markdown("---")

if st.button("🔍 Prediksi Potensi Sekarang", type="primary", use_container_width=True):
    try:
        provinsi_encoded = le_provinsi.transform([provinsi_pilihan])[0]
        data_input = np.array([[rating_angka, latitude, longitude, provinsi_encoded]])
        
        prediksi_encoded = model.predict(data_input)
        hasil_prediksi = le_target.inverse_transform(prediksi_encoded)[0]
        
        st.markdown("### 🎯 Hasil Analisis Potensi Wisata:")
        # 🌟 TAMBAHAN: Nama pantai ikut dipanggil di hasil prediksi
        st.success(f"Berdasarkan parameter, potensi **{nama_pantai}** diprediksi masuk kategori: **{hasil_prediksi.upper()}**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.caption("Developed with Streamlit & XGBoost | Geospatial Data Mining Project")
