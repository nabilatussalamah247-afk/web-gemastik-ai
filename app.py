"""
BeachFinder Indonesia — Dashboard Peta Interaktif + Prediksi Kualitas Pantai
=============================================================================
Dibangun dengan Streamlit + Folium + XGBoost.

Struktur file yang dibutuhkan di folder yang sama (satu repo):
    app.py
    dataset_pantai_clean.csv
    model_xgboost_wisata.pkl
    label_encoder_target.pkl
    le_provinsi.pkl
    requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import os

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="BeachFinder Indonesia",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS KUSTOM — TAMPILAN MODERN
# =============================================================================
st.markdown(
    """
    <style>
    .main { background-color: #f7fafc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .metric-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%);
        border-radius: 14px;
        padding: 18px 20px;
        color: white;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25);
    }
    .metric-card h2 { margin: 0; font-size: 26px; font-weight: 700; }
    .metric-card p { margin: 2px 0 0 0; font-size: 13px; opacity: 0.9; }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: white;
    }
    .badge-green  { background-color: #16a34a; }
    .badge-blue   { background-color: #2563eb; }
    .badge-orange { background-color: #ea580c; }
    .badge-red    { background-color: #dc2626; }
    .badge-gray   { background-color: #6b7280; }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    h1, h2, h3 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# KONSTAN — SESUAI ACUAN PETA FOLIUM ASLI
# =============================================================================
COLOR_MAP = {
    "sangat bagus": {"marker": "green", "badge": "badge-green"},
    "bagus": {"marker": "blue", "badge": "badge-blue"},
    "cukup bagus": {"marker": "orange", "badge": "badge-orange"},
    "biasa": {"marker": "red", "badge": "badge-red"},
}
PREDIKAT_ORDER = ["Sangat Bagus", "Bagus", "Cukup Bagus", "Biasa"]

DATA_PATH = "dataset_pantai_clean.csv"
MODEL_PATH = "model_xgboost_wisata.pkl"
LE_TARGET_PATH = "label_encoder_target.pkl"
LE_PROVINSI_PATH = "le_provinsi.pkl"


# =============================================================================
# LOAD DATA & MODEL (DI-CACHE AGAR CEPAT)
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Nama Pantai", "Provinsi", "Rating Angka", "Predikat", "Latitude", "Longitude"])
    df["Predikat"] = df["Predikat"].str.strip().str.lower()
    return df


@st.cache_resource
def load_model_artifacts():
    """Mengembalikan (model, le_target, le_provinsi) atau None jika file belum ada."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LE_TARGET_PATH) and os.path.exists(LE_PROVINSI_PATH)):
        return None
    model = joblib.load(MODEL_PATH)
    le_target = joblib.load(LE_TARGET_PATH)
    le_provinsi = joblib.load(LE_PROVINSI_PATH)
    return model, le_target, le_provinsi


def stars_from_rating(rating: float) -> str:
    penuh = int(round(rating))
    penuh = max(0, min(5, penuh))
    return "⭐" * penuh


def badge_class(predikat_lower: str) -> str:
    return COLOR_MAP.get(predikat_lower, {}).get("badge", "badge-gray")


def marker_color(predikat_lower: str) -> str:
    return COLOR_MAP.get(predikat_lower, {}).get("marker", "gray")


df = load_data()
model_bundle = load_model_artifacts()

# =============================================================================
# SIDEBAR — FILTER
# =============================================================================
with st.sidebar:
    st.markdown("## 🏖️ BeachFinder")
    st.caption("Eksplorasi & prediksi kualitas pantai di Indonesia")
    st.markdown("---")

    st.markdown("### 🔎 Filter Data")

    all_provinsi = sorted(df["Provinsi"].unique().tolist())
    provinsi_pilihan = st.multiselect(
        "Provinsi",
        options=all_provinsi,
        default=all_provinsi,
    )

    kualitas_options = [p for p in PREDIKAT_ORDER if p.lower() in df["Predikat"].unique()]
    kualitas_pilihan = st.multiselect(
        "Kualitas / Predikat",
        options=kualitas_options,
        default=kualitas_options,
    )

    rating_min, rating_max = float(df["Rating Angka"].min()), float(df["Rating Angka"].max())
    rating_range = st.slider(
        "Rentang Rating",
        min_value=rating_min,
        max_value=rating_max,
        value=(rating_min, rating_max),
        step=0.1,
    )

    st.markdown("---")
    st.caption("Dibangun dengan Streamlit + Folium + XGBoost")

# Terapkan filter
kualitas_pilihan_lower = [k.lower() for k in kualitas_pilihan]
df_filtered = df[
    df["Provinsi"].isin(provinsi_pilihan)
    & df["Predikat"].isin(kualitas_pilihan_lower)
    & df["Rating Angka"].between(rating_range[0], rating_range[1])
].copy()

# =============================================================================
# HEADER
# =============================================================================
st.markdown("# 🏖️ BeachFinder Indonesia")
st.markdown("Peta interaktif destinasi pantai di seluruh Indonesia, lengkap dengan filter dan prediksi kualitas berbasis Machine Learning.")

# =============================================================================
# METRICS
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card"><h2>{len(df_filtered)}</h2><p>Pantai ditampilkan</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    avg_rating = df_filtered["Rating Angka"].mean() if len(df_filtered) else 0
    st.markdown(
        f'<div class="metric-card"><h2>{avg_rating:.2f} ⭐</h2><p>Rata-rata rating</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    n_sangat_bagus = (df_filtered["Predikat"] == "sangat bagus").sum()
    st.markdown(
        f'<div class="metric-card"><h2>{n_sangat_bagus}</h2><p>Predikat "Sangat Bagus"</p></div>',
        unsafe_allow_html=True,
    )
with col4:
    n_provinsi = df_filtered["Provinsi"].nunique()
    st.markdown(
        f'<div class="metric-card"><h2>{n_provinsi}</h2><p>Provinsi tercakup</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# =============================================================================
# TABS
# =============================================================================
tab_peta, tab_prediksi, tab_data = st.tabs(["🗺️ Peta Interaktif", "🔮 Prediksi Kualitas Pantai", "📋 Tabel Data"])

# -----------------------------------------------------------------------------
# TAB 1 — PETA
# -----------------------------------------------------------------------------
with tab_peta:
    if len(df_filtered) == 0:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
    else:
        center_lat = df_filtered["Latitude"].mean()
        center_lon = df_filtered["Longitude"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")
        cluster = MarkerCluster().add_to(m)

        for _, row in df_filtered.iterrows():
            predikat_lower = row["Predikat"]
            predikat_title = predikat_lower.title()
            color = marker_color(predikat_lower)
            stars = stars_from_rating(row["Rating Angka"])
            link = row.get("Link Google Maps", "")

            popup_html = f"""
            <div style="font-family: 'Segoe UI', sans-serif; width: 230px;">
                <h4 style="margin: 0 0 6px 0; color: #0f172a;">{row['Nama Pantai']}</h4>
                <p style="margin: 2px 0;">📍 <b>Provinsi:</b> {row['Provinsi']}</p>
                <p style="margin: 2px 0;">⭐ <b>Rating:</b> {row['Rating Angka']} {stars}</p>
                <p style="margin: 2px 0;">🏖️ <b>Kualitas:</b> {predikat_title}</p>
                {f'<a href="{link}" target="_blank">Buka di Google Maps ↗</a>' if isinstance(link, str) and link else ''}
            </div>
            """

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=row["Nama Pantai"],
                icon=folium.Icon(color=color, icon="umbrella-beach", prefix="fa"),
            ).add_to(cluster)

        st_folium(m, use_container_width=True, height=560, returned_objects=[])

        st.markdown(
            """
            <div style="display:flex; gap:16px; margin-top:8px; flex-wrap:wrap;">
                <span><span class="badge badge-green">●</span> Sangat Bagus</span>
                <span><span class="badge badge-blue">●</span> Bagus</span>
                <span><span class="badge badge-orange">●</span> Cukup Bagus</span>
                <span><span class="badge badge-red">●</span> Biasa</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# TAB 2 — PREDIKSI
# -----------------------------------------------------------------------------
with tab_prediksi:
    st.markdown("### 🔮 Prediksi Kualitas Pantai Baru")
    st.caption(
        "Masukkan data pantai baru (belum ada di dataset) untuk memprediksi predikat "
        "kualitasnya menggunakan model XGBoost yang telah dilatih."
    )

    if model_bundle is None:
        st.error(
            "File model belum ditemukan di folder aplikasi. Pastikan `model_xgboost_wisata.pkl`, "
            "`label_encoder_target.pkl`, dan `le_provinsi.pkl` berada di folder yang sama dengan `app.py`."
        )
    else:
        model, le_target, le_provinsi = model_bundle

        with st.form("form_prediksi"):
            c1, c2 = st.columns(2)
            with c1:
                provinsi_input = st.selectbox("Provinsi", options=sorted(le_provinsi.classes_.tolist()))
                rating_input = st.number_input("Rating (1.0 – 5.0)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            with c2:
                lat_input = st.number_input("Latitude", value=float(df["Latitude"].mean()), format="%.6f")
                lon_input = st.number_input("Longitude", value=float(df["Longitude"].mean()), format="%.6f")

            submitted = st.form_submit_button("Prediksi Kualitas", use_container_width=True)

        if submitted:
            try:
                provinsi_encoded = le_provinsi.transform([provinsi_input])[0]
                X_new = pd.DataFrame(
                    [[rating_input, lat_input, lon_input, provinsi_encoded]],
                    columns=["Rating Angka", "Latitude", "Longitude", "Provinsi"],
                )
                pred_encoded = model.predict(X_new)[0]
                pred_label = le_target.inverse_transform([pred_encoded])[0]
                pred_lower = str(pred_label).strip().lower()

                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_new)[0]
                    confidence = proba[pred_encoded] * 100

                st.markdown("#### Hasil Prediksi")
                st.markdown(
                    f'<span class="badge {badge_class(pred_lower)}" style="font-size:16px; padding:8px 18px;">'
                    f'{str(pred_label).title()}</span>',
                    unsafe_allow_html=True,
                )
                if proba is not None:
                    st.caption(f"Tingkat keyakinan model: {confidence:.1f}%")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# -----------------------------------------------------------------------------
# TAB 3 — TABEL DATA
# -----------------------------------------------------------------------------
with tab_data:
    st.markdown("### 📋 Data Pantai (sesuai filter)")
    tampil = df_filtered.copy()
    tampil["Predikat"] = tampil["Predikat"].str.title()
    kolom_tampil = ["Nama Pantai", "Provinsi", "Rating Angka", "Predikat", "Latitude", "Longitude"]
    st.dataframe(tampil[kolom_tampil].reset_index(drop=True), use_container_width=True, height=500)
