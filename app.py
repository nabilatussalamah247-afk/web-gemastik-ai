"""
BeachFinder Indonesia — Dashboard Peta Interaktif + Prediksi Kualitas Pantai
=============================================================================
Dibangun dengan Streamlit + Folium + XGBoost.

Struktur file yang dibutuhkan di folder yang sama (satu repo):
    app.py
    dataset_clean.csv
    model_xgboost_wisata.pkl     -> model lokasi-only (fitur: Latitude, Longitude, Provinsi)
    label_encoder_target.pkl
    le_provinsi.pkl
    requirements.txt
"""

import os
import altair as alt
import folium
from folium.plugins import MarkerCluster
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .main { background-color: #f4f7fb; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(120deg, #0891b2 0%, #0e7490 45%, #164e63 100%);
        border-radius: 20px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(14, 116, 144, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; right: -40px; top: -40px;
        width: 180px; height: 180px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p { margin: 8px 0 0 0; font-size: 15px; opacity: 0.92; max-width: 640px; }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .icon { font-size: 20px; }
    .metric-card h2 { margin: 6px 0 0 0; font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-card p { margin: 2px 0 0 0; font-size: 12.5px; color: #64748b; font-weight: 500; }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        color: white;
        letter-spacing: 0.2px;
    }
    .badge-green  { background-color: #16a34a; }
    .badge-blue   { background-color: #2563eb; }
    .badge-orange { background-color: #ea580c; }
    .badge-red    { background-color: #dc2626; }
    .badge-gray   { background-color: #6b7280; }

    .legend-dot {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 13px; color: #334155; font-weight: 500; margin-right: 4px;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #0891b2 !important;
    }

    /* ---------- Cards / notes ---------- */
    .info-card {
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .caution-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 4px solid #ea580c;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13.5px;
        color: #7c2d12;
    }
    .result-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    h1, h2, h3 { color: #0f172a; }
    div[data-testid="stTabs"] button p { font-weight: 600; font-size: 14.5px; }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# KONSTAN
# =============================================================================
COLOR_MAP = {
    "sangat bagus": {"marker": "green", "badge": "badge-green", "hex": "#16a34a"},
    "bagus": {"marker": "blue", "badge": "badge-blue", "hex": "#2563eb"},
    "biasa": {"marker": "red", "badge": "badge-red", "hex": "#dc2626"},
}
PREDIKAT_ORDER = ["Sangat Bagus", "Bagus", "Biasa"]

DATA_PATH = "dataset_clean.csv"
MODEL_PATH = "model_xgboost_wisata.pkl"
LE_TARGET_PATH = "label_encoder_target.pkl"
LE_PROVINSI_PATH = "le_provinsi.pkl"

KELAS_MINORITAS = {"bagus", "biasa"}


# =============================================================================
# LOAD DATA & MODEL (DI-CACHE)
# =============================================================================
@st.cache_data
def load_data():
  df = pd.read_csv(DATA_PATH)
  df = df.dropna(
      subset=[
          "Nama Pantai",
          "Provinsi",
          "Rating Angka",
          "Predikat",
          "Latitude",
          "Longitude",
      ]
  )
  df["Predikat"] = df["Predikat"].str.strip().str.lower()
  df["Predikat"] = df["Predikat"].replace({"cukup bagus": "biasa"})
  return df


@st.cache_resource
def load_model_artifacts():
  if not (
      os.path.exists(MODEL_PATH)
      and os.path.exists(LE_TARGET_PATH)
      and os.path.exists(LE_PROVINSI_PATH)
  ):
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


def hex_color(predikat_lower: str) -> str:
  return COLOR_MAP.get(predikat_lower, {}).get("hex", "#6b7280")


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

  kualitas_options = [
      p for p in PREDIKAT_ORDER if p.lower() in df["Predikat"].unique()
  ]
  kualitas_pilihan = st.multiselect(
      "Kualitas / Predikat",
      options=kualitas_options,
      default=kualitas_options,
  )

  rating_min, rating_max = float(df["Rating Angka"].min()), float(
      df["Rating Angka"].max()
  )
  rating_range = st.slider(
      "Rentang Rating",
      min_value=rating_min,
      max_value=rating_max,
      value=(rating_min, rating_max),
      step=0.1,
  )

  st.markdown("---")
  st.caption("Dibangun dengan Streamlit · Folium · XGBoost")

kualitas_pilihan_lower = [k.lower() for k in kualitas_pilihan]
df_filtered = df[
    df["Provinsi"].isin(provinsi_pilihan)
    & df["Predikat"].isin(kualitas_pilihan_lower)
    & df["Rating Angka"].between(rating_range[0], rating_range[1])
].copy()

# =============================================================================
# HERO / HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🏖️ BeachFinder Indonesia</h1>
        <p>Peta interaktif destinasi pantai di seluruh Indonesia, lengkap dengan filter,
        statistik ringkas, dan prediksi kualitas berbasis lokasi menggunakan XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# METRICS
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
metric_items = [
    ("🏝️", len(df_filtered), "Pantai ditampilkan"),
    (
        "⭐",
        f'{df_filtered["Rating Angka"].mean():.2f}'
        if len(df_filtered)
        else "0.00",
        "Rata-rata rating",
    ),
    (
        "🌟",
        int((df_filtered["Predikat"] == "sangat bagus").sum()),
        'Predikat "Sangat Bagus"',
    ),
    ("📍", df_filtered["Provinsi"].nunique(), "Provinsi tercakup"),
]
for col, (icon, value, label) in zip([col1, col2, col3, col4], metric_items):
  with col:
    st.markdown(
        f'<div class="metric-card"><span class="icon">{icon}</span>'
        f"<h2>{value}</h2><p>{label}</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab_peta, tab_prediksi, tab_data = st.tabs(
    ["🗺️ Peta Interaktif", "🔮 Prediksi Kualitas Pantai", "📋 Tabel Data"]
)

# -----------------------------------------------------------------------------
# TAB 1 — PETA
# -----------------------------------------------------------------------------
with tab_peta:
  if len(df_filtered) == 0:
    st.warning("Tidak ada data yang cocok dengan filter saat ini.")
  else:
    map_col, chart_col = st.columns([2.4, 1])

    with map_col:
      center_lat = df_filtered["Latitude"].mean()
      center_lon = df_filtered["Longitude"].mean()

      m = folium.Map(
          location=[center_lat, center_lon],
          zoom_start=5,
          tiles="CartoDB positron",
      )
      cluster = MarkerCluster().add_to(m)

      for _, row in df_filtered.iterrows():
        predikat_lower = row["Predikat"]
        predikat_title = predikat_lower.title()
        color = marker_color(predikat_lower)
        stars = stars_from_rating(row["Rating Angka"])
        link = row.get("Link Google Maps", "")

        # Ambil ulasan 1 - 3
        u1 = (
            row["Ulasan 1"]
            if pd.notna(row.get("Ulasan 1"))
            else "Belum ada ulasan"
        )
        u2 = (
            row["Ulasan 2"]
            if pd.notna(row.get("Ulasan 2"))
            else "Belum ada ulasan"
        )
        u3 = (
            row["Ulasan 3"]
            if pd.notna(row.get("Ulasan 3"))
            else "Belum ada ulasan"
        )

        popup_html = f"""
                <div style="font-family: 'Segoe UI', sans-serif; width: 250px; font-size: 11.5px;">
                    <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 13.5px;">{row['Nama Pantai']}</h4>
                    <p style="margin: 2px 0;">📍 <b>Provinsi:</b> {row['Provinsi']}</p>
                    <p style="margin: 2px 0;">⭐ <b>Rating:</b> {row['Rating Angka']} {stars}</p>
                    <p style="margin: 2px 0 6px 0;">🏖️ <b>Kualitas:</b> {predikat_title}</p>
                    <hr style="margin: 4px 0; border: 0; border-top: 1px solid #cbd5e1;">
                    <p style="margin: 2px 0;"><b>💬 Ulasan 1:</b> {u1}</p>
                    <p style="margin: 2px 0;"><b>💬 Ulasan 2:</b> {u2}</p>
                    <p style="margin: 2px 0 6px 0;"><b>💬 Ulasan 3:</b> {u3}</p>
                    {f'<a href="{link}" target="_blank">Buka di Google Maps ↗</a>' if isinstance(link, str) and link else ''}
                </div>
                """

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=row["Nama Pantai"],
            icon=folium.Icon(color=color, icon="umbrella-beach", prefix="fa"),
        ).add_to(cluster)

      st_folium(m, use_container_width=True, height=560, returned_objects=[])

      st.markdown(
          """
                <div style="display:flex; gap:16px; margin-top:10px; flex-wrap:wrap;">
                    <span class="legend-dot"><span class="dot" style="background:#16a34a;"></span>Sangat Bagus</span>
                    <span class="legend-dot"><span class="dot" style="background:#2563eb;"></span>Bagus</span>
                    <span class="legend-dot"><span class="dot" style="background:#dc2626;"></span>Biasa</span>
                </div>
                """,
          unsafe_allow_html=True,
      )

    with chart_col:
      st.markdown("##### Distribusi Kualitas")
      dist = df_filtered["Predikat"].value_counts().reset_index()
      dist.columns = ["Predikat", "Jumlah"]
      dist["Predikat"] = dist["Predikat"].str.title()
      dist["warna"] = dist["Predikat"].str.lower().map(hex_color)

      chart = (
          alt.Chart(dist)
          .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
          .encode(
              x=alt.X("Predikat:N", sort="-y", title=None),
              y=alt.Y("Jumlah:Q", title=None),
              color=alt.Color("warna:N", scale=None, legend=None),
              tooltip=["Predikat", "Jumlah"],
          )
          .properties(height=220)
      )
      st.altair_chart(chart, use_container_width=True)

      st.markdown("##### 🏆 Top 5 Pantai")
      top5 = df_filtered.sort_values("Rating Angka", ascending=False).head(5)
      for _, r in top5.iterrows():
        st.markdown(
            f"""<div class="info-card" style="margin-bottom:8px; padding:10px 14px;">
                    <b>{r['Nama Pantai']}</b><br>
                    <span style="font-size:12.5px; color:#64748b;">{r['Provinsi']} · {r['Rating Angka']} ⭐</span>
                    </div>""",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# TAB 2 — PREDIKSI
# -----------------------------------------------------------------------------
with tab_prediksi:
  st.markdown("### 🔮 Prediksi Kualitas Pantai Baru")
  st.caption(
      "Menaksir predikat kualitas sebuah titik pantai yang **belum punya rating**"
      " — hanya berdasarkan lokasinya (provinsi, latitude, longitude)."
  )

  st.markdown(
      """
        <div class="caution-box">
        ⚠️ <b>Perlu diketahui:</b> dataset training sangat timpang — sekitar 90%"
        " pantai berpredikat "Sangat Bagus". Akibatnya model jauh lebih akurat"
        " menebak kelas "Sangat Bagus" dibanding kelas lain ("Bagus", "Biasa")."
        " Anggap hasil prediksi ini sebagai <i>indikasi awal</i>, bukan"
        " penilaian pasti.
        </div>
        <br>
        """,
      unsafe_allow_html=True,
  )

  if model_bundle is None:
    st.error(
        "File model belum ditemukan di folder aplikasi. Pastikan"
        " `model_xgboost_wisata.pkl`, `label_encoder_target.pkl`, dan"
        " `le_provinsi.pkl` berada di folder yang sama dengan `app.py`."
    )
  else:
    model, le_target, le_provinsi = model_bundle

    with st.form("form_prediksi"):
      c1, c2, c3 = st.columns(3)
      with c1:
        provinsi_input = st.selectbox(
            "Provinsi", options=sorted(le_provinsi.classes_.tolist())
        )
      with c2:
        lat_input = st.number_input(
            "Latitude", value=float(df["Latitude"].mean()), format="%.6f"
        )
      with c3:
        lon_input = st.number_input(
            "Longitude", value=float(df["Longitude"].mean()), format="%.6f"
        )

      submitted = st.form_submit_button(
          "Prediksi Kualitas", use_container_width=True
      )

    if submitted:
      try:
        provinsi_encoded = le_provinsi.transform([provinsi_input])[0]
        X_new = pd.DataFrame(
            [[lat_input, lon_input, provinsi_encoded]],
            columns=["Latitude", "Longitude", "Provinsi"],
        )
        pred_encoded = model.predict(X_new)[0]
        pred_label = le_target.inverse_transform([pred_encoded])[0]
        pred_lower = str(pred_label).strip().lower()

        proba = None
        confidence = None
        if hasattr(model, "predict_proba"):
          proba = model.predict_proba(X_new)[0]
          confidence = proba[pred_encoded] * 100

        st.markdown(
            f"""
                    <div class="result-box">
                        <p style="margin:0 0 8px 0; font-size:13px; color:#166534; font-weight:600;">HASIL PREDIKSI</p>
                        <span class="badge {badge_class(pred_lower)}" style="font-size:18px; padding:10px 22px;">
                            {str(pred_label).title()}
                        </span>
                    </div>
                    """,
            unsafe_allow_html=True,
        )
        if confidence is not None:
          st.caption(f"Tingkat keyakinan model: {confidence:.1f}%")

        if pred_lower in KELAS_MINORITAS:
          st.info(
              "ℹ️ Kelas ini termasuk minoritas di data training — anggap hasil"
              " ini sebagai sinyal kasar, dan tetap cek langsung kondisi"
              " lapangan sebelum mengambil keputusan."
          )

      except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# -----------------------------------------------------------------------------
# TAB 3 — TABEL DATA
# -----------------------------------------------------------------------------
with tab_data:
  st.markdown("### 📋 Data Pantai (sesuai filter)")
  tampil = df_filtered.copy()
  tampil["Predikat"] = tampil["Predikat"].str.title()

  kolom_tampil = [
      "Nama Pantai",
      "Provinsi",
      "Rating Angka",
      "Predikat",
      "Jumlah Ulasan",
      "Ulasan 1",
      "Ulasan 2",
      "Ulasan 3",
      "Latitude",
      "Longitude",
  ]
  st.dataframe(
      tampil[kolom_tampil].reset_index(drop=True),
      use_container_width=True,
      height=500,
  )

  csv_bytes = tampil[kolom_tampil].to_csv(index=False).encode("utf-8")
  st.download_button(
      "⬇️ Unduh data terfilter (CSV)",
      data=csv_bytes,
      file_name="pantai_terfilter.csv",
      mime="text/csv",
  )
