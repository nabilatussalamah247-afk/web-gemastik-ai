import streamlit as st
import numpy as np
import joblib

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Prediksi Potensi Wisata Pantai",
    page_icon="🌊",
    layout="centered"
)

# 2. Load Model dan Encoder yang sudah didownload dari Colab
# Pastikan file .pkl ini berada di folder yang SAMA dengan file app.py ini!


@st.cache_resource
def load_artifacts():
    model = joblib.load('model_xgboost_wisata.pkl')
    le_target = joblib.load('label_encoder_target.pkl')
    le_provinsi = joblib.load('le_provinsi.pkl')
    return model, le_target, le_provinsi


model, le_target, le_provinsi = load_artifacts()

# 3. Tampilan Antarmuka Web (UI)
st.title("🌊 Prediksi Potensi Wisata Pantai Indonesia")
st.write("Aplikasi berbasis *Machine Learning* (XGBoost) untuk memprediksi tingkat potensi wisata pantai berdasarkan data geospasial dan rating.")

st.markdown("---")

# 4. Form Input untuk Pengguna di Sidebar / Bagian Utama
st.subheader("📝 Masukkan Parameter Wisata Pantai")

# Dropdown Provinsi berdasarkan data training
daftar_provinsi = list(le_provinsi.classes_)
provinsi_pilihan = st.selectbox("Pilih Provinsi", daftar_provinsi)

rating_angka = st.slider("Rating Angka Pantai",
                         min_value=1.0, max_value=5.0, value=4.5, step=0.1)
latitude = st.number_input("Latitude (Garis Lintang)",
                           value=-6.2000, format="%.4f")
longitude = st.number_input(
    "Longitude (Garis Bujur)", value=106.8166, format="%.4f")

# 5. Tombol Prediksi
if st.button("🔍 Prediksi Potensi Sekarang", type="primary"):
    try:
        # Ubah provinsi teks ke bentuk angka sesuai encoder
        provinsi_encoded = le_provinsi.transform([provinsi_pilihan])[0]

        # Susun data input ke bentuk array sesuai urutan fitur saat training:
        # ['Rating Angka', 'Latitude', 'Longitude', 'Provinsi']
        data_input = np.array(
            [[rating_angka, latitude, longitude, provinsi_encoded]])

        # Lakukan prediksi dengan model XGBoost
        prediksi_encoded = model.predict(data_input)
        hasil_prediksi = le_target.inverse_transform(prediksi_encoded)[0]

        # Tampilkan Hasil ke Web dengan gaya menarik
        st.markdown("### Hasil Analisis Potensi Wisata:")
        st.success(
            f"Berdasarkan model AI, potensi pantai di **{provinsi_pilihan}** ini diprediksi masuk kategori: **{hasil_prediksi.upper()}**")

        # Detail ringkas
        with st.expander("Lihat Detail Parameter Input"):
            st.write(f"- **Provinsi:** {provinsi_pilihan}")
            st.write(f"- **Rating:** {rating_angka}")
            st.write(f"- **Koordinat:** ({latitude}, {longitude})")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# Catatan kaki
st.markdown("---")
st.caption("Developed with Streamlit & XGBoost | Geospatial Data Mining Project")
