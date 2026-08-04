"""
BeachFinder Indonesia — Dashboard Peta Interaktif + Prediksi Kualitas + Analisis NLP
=============================================================================
Dibangun dengan Streamlit + Folium + XGBoost + Geolocation + Text Sentiment Analysis.
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
from streamlit_geolocation import streamlit_geolocation

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
# INISIALISASI SESSION STATE UNTUK SPLASH SCREEN
# =============================================================================
if "show_splash" not in st.session_state:
  st.session_state.show_splash = True

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

    /* ---------- Splash Screen Styling ---------- */
    .splash-container {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-radius: 24px;
        color: white;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.3);
        margin-top: 20px;
    }
    .splash-container h1 { font-size: 38px; font-weight: 800; margin-bottom: 10px; color: #ffffff; }
    .splash-container p { font-size: 16px; color: #94a3b8; max-width: 600px; margin: 0 auto 30px auto; }

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
        content: ""; position: absolute; right: -40px; top: -40px;
        width: 180px; height: 180px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p { margin: 8px 0 0 0; font-size: 15px; opacity: 0.92; max-width: 640px; }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: white; border-radius: 16px; padding: 18px 20px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .metric-card h2 { margin: 6px 0 0 0; font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-card p { margin: 2px 0 0 0; font-size: 12.5px; color: #64748b; font-weight: 500; }

    /* ---------- Badges & Boxes ---------- */
    .badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white; }
    .badge-blue { background-color: #2563eb; }
    .badge-red { background-color: #dc2626; }
    .result-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 14px; padding: 20px; text-align: center; }
    .search-result-box { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 14px; padding: 20px; box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05); margin-bottom: 20px; }

    h1, h2, h3 { color: #0f172a; }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# KONTROL TAMPILAN: SPLASH SCREEN / DASHBOARD UTAMA
# =============================================================================
if st.session_state.show_splash:
  st.markdown(
      """
        <div class="splash-container">
            <h1>🏖️ Selamat Datang di BeachFinder Indonesia</h1>
            <p>Jelajahi keindahan destinasi pantai nusantara, temukan informasi ulasan terpercaya, analisis teks ulasan, dan prediksi kualitas pantai berbasis Machine Learning.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
  with col_img2:
    if os.path.exists("1.png"):
      st.image(
          "1.png",
          use_container_width=True,
          caption="Eksplorasi Pantai Nusantara",
      )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Mulai Eksplorasi Sekarang", use_container_width=True):
      st.session_state.show_splash = False
      st.rerun()

else:
  # =============================================================================
  # DASHBOARD UTAMA APLIKASI
  # =============================================================================
  COLOR_MAP = {
      "bagus": {"marker": "blue", "badge": "badge-blue", "hex": "#2563eb"},
      "biasa": {"marker": "red", "badge": "badge-red", "hex": "#dc2626"},
  }
  PREDIKAT_ORDER = ["Bagus", "Biasa"]

  DATA_PATH = "dataset_clean.csv"
  MODEL_PATH = "model_xgboost_wisata.pkl"
  LE_TARGET_PATH = "label_encoder_target.pkl"
  LE_PROVINSI_PATH = "le_provinsi.pkl"


  @st.cache_data
  def load_data():
    df = pd.read_csv(DATA_PATH)

    # Hapus kolom kategori kotor/kosong dari dataset
    kolom_tidak_pakai = [
        "Kategori Pantai",
        "Kategori 1",
        "Kategori 2",
        "Kategori 3",
        "Kategori 4",
        "Kategori 5",
    ]
    df = df.drop(
        columns=[col for col in kolom_tidak_pakai if col in df.columns],
        errors="ignore",
    )

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

    def map_provinsi(val):
      val_str = str(val).strip()
      if val_str in ["Ambon", "Pulau Buru", "Pulau Seram", "Pulau Wetar"]:
        return "Maluku"
      elif val_str in ["Ternate"]:
        return "Maluku Utara"
      return val_str

    df["Provinsi"] = df["Provinsi"].apply(map_provinsi)
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


  def hitung_jarak_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


  def analisis_ulasan_otomatis(u1, u2, u3):
    teks_gabungan = f"{str(u1)} {str(u2)} {str(u3)}".lower()
    if (
        teks_gabungan == "nan nan nan"
        or teks_gabungan.strip() == "belum ada ulasan"
    ):
      return "Belum cukup data ulasan teks untuk dianalisis oleh sistem."

    kata_positif = [
        "indah",
        "bagus",
        "bersih",
        "sejuk",
        "keren",
        "nyaman",
        "tenang",
        "luas",
        "ramai",
        "cantik",
        "mantap",
    ]
    kata_negatif = [
        "kotor",
        "rusak",
        "mahal",
        "macet",
        "sampah",
        "sempit",
        "jauh",
        "kurang",
    ]

    skor_positif = sum(1 for kata in kata_positif if kata in teks_gabungan)
    skor_negatif = sum(1 for kata in kata_negatif if kata in teks_gabungan)

    if skor_positif > skor_negatif:
      return (
          "💡 **Analisis Sistem (NLP):** Mayoritas pengunjung memberikan ulasan"
          " positif, memuji keindahan, kenyamanan, atau kebersihan lokasi"
          " pantai ini."
      )
    elif skor_negatif > skor_positif:
      return (
          "💡 **Analisis Sistem (NLP):** Terdapat beberapa catatan atau keluhan"
          " dari pengunjung terkait fasilitas, kebersihan, atau akses di"
          " sekitar pantai ini."
      )
    else:
      return (
          "💡 **Analisis Sistem (NLP):** Ulasan pengunjung bervariasi dengan"
          " kesan netral terhadap kondisi pantai."
      )


  df = load_data()
  model_bundle = load_model_artifacts()

  # Sidebar — Filter
  with st.sidebar:
    st.markdown("## 🏖️ BeachFinder")
    st.caption("Eksplorasi & prediksi kualitas pantai di Indonesia")
    st.markdown("---")
    st.markdown("### 🔎 Filter Data")

    all_provinsi = sorted(df["Provinsi"].unique().tolist())
    provinsi_pilihan = st.multiselect(
        "Provinsi", options=all_provinsi, default=all_provinsi
    )

    kualitas_options = [
        p for p in PREDIKAT_ORDER if p.lower() in df["Predikat"].unique()
    ]
    kualitas_pilihan = st.multiselect(
        "Kualitas / Predikat", options=kualitas_options, default=kualitas_options
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

  # Hero Section
  st.markdown(
      """
        <div class="hero">
            <h1>🏖️ BeachFinder Indonesia</h1>
            <p>Peta interaktif destinasi pantai di seluruh Indonesia, lengkap dengan pencarian, filter,
            analisis teks ulasan, dan prediksi kualitas berbasis lokasi menggunakan XGBoost.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  # Metric Cards
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
          int((df_filtered["Predikat"] == "bagus").sum()),
          'Predikat "Bagus"',
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

  # Tabs Utama
  tab_peta, tab_prediksi, tab_data = st.tabs(
      ["🗺️ Peta Interaktif", "🔮 Prediksi Kualitas Pantai", "📋 Tabel Data"]
  )

  with tab_peta:
    st.markdown("### 🔍 Cari Destinasi Pantai")
    st.caption(
        "Ketik atau pilih nama pantai untuk langsung melihat analisis lengkap,"
        " rating, predikat, dan ringkasan ulasan pengunjung."
    )

    list_nama_pantai = sorted(df["Nama Pantai"].unique().tolist())
    pilihan_pencarian = st.selectbox(
        "Pilih atau ketik nama pantai:",
        options=["-- Pilih / Cari Pantai --"] + list_nama_pantai,
    )

    if pilihan_pencarian != "-- Pilih / Cari Pantai --":
      data_pilih = df[df["Nama Pantai"] == pilihan_pencarian].iloc[0]
      p_pred = data_pilih["Predikat"].title()
      p_stars = stars_from_rating(data_pilih["Rating Angka"])
      p_link = (
          data_pilih["Link Google Maps"]
          if pd.notna(data_pilih.get("Link Google Maps"))
          else "#"
      )
      u1 = (
          data_pilih["Ulasan 1"]
          if pd.notna(data_pilih.get("Ulasan 1"))
          else "Belum ada ulasan"
      )
      u2 = (
          data_pilih["Ulasan 2"]
          if pd.notna(data_pilih.get("Ulasan 2"))
          else "Belum ada ulasan"
      )
      u3 = (
          data_pilih["Ulasan 3"]
          if pd.notna(data_pilih.get("Ulasan 3"))
          else "Belum ada ulasan"
      )

      # Analisis NLP otomatis ulasan
      kesimpulan_nlp = analisis_ulasan_otomatis(u1, u2, u3)

      st.markdown(
          f"""
            <div class="search-result-box">
                <h3 style="margin-top:0; color:#0f172a;">🌊 {data_pilih['Nama Pantai']}</h3>
                <hr style="margin: 8px 0 14px 0; border:0; border-top:1px solid #e2e8f0;">
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px;">
                        <p style="margin: 4px 0;">📍 <b>Provinsi:</b> {data_pilih['Provinsi']}</p>
                        <p style="margin: 4px 0;">⭐ <b>Rating Angka:</b> {data_pilih['Rating Angka']} {p_stars}</p>
                        <p style="margin: 4px 0;">🏖️ <b>Predikat Kualitas:</b> <span class="badge {badge_class(data_pilih['Predikat'])}">{p_pred}</span></p>
                        <p style="margin: 4px 0;">💬 <b>Jumlah Ulasan:</b> {data_pilih.get('Jumlah Ulasan', 'N/A')}</p>
                        <p style="margin: 8px 0 0 0;"><a href="{p_link}" target="_blank" style="text-decoration: none; color: #0284c7; font-weight: 600;">Buka Lokasi di Google Maps ↗</a></p>
                    </div>
                    <div style="flex: 1.5; min-width: 300px; background: #f8fafc; padding: 12px 16px; border-radius: 10px; border: 1px solid #e2e8f0;">
                        <p style="margin: 0 0 6px 0; font-weight: 700; color: #334155; font-size: 13px;">💬 Informasi Ulasan Pengunjung:</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 1:</b> {u1}</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 2:</b> {u2}</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 3:</b> {u3}</p>
                        <hr style="margin: 8px 0; border:0; border-top:1px solid #cbd5e1;">
                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #0369a1; font-style: italic;">{kesimpulan_nlp}</p>
                    </div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.markdown("---")
    st.markdown("### 📍 Cari Pantai Terdekat dari Lokasi Anda")
    st.caption(
        "Aktifkan izin lokasi di browser Anda untuk melacak destinasi pantai"
        " dalam radius 10 km."
    )

    user_loc = streamlit_geolocation()

    if user_loc and user_loc.get("latitude") and user_loc.get("longitude"):
      u_lat = user_loc["latitude"]
      u_lon = user_loc["longitude"]
      st.success(
          f"Lokasi terdeteksi! (Latitude: {u_lat:.4f}, Longitude:"
          f" {u_lon:.4f})"
      )

      df_lokasi = df.copy()
      df_lokasi["Jarak_Km"] = df_lokasi.apply(
          lambda row: hitung_jarak_km(
              u_lat, u_lon, row["Latitude"], row["Longitude"]
          ),
          axis=1,
      )

      radius_maks = 10.0
      df_terdekat = df_lokasi[df_lokasi["Jarak_Km"] <= radius_maks].sort_values(
          "Jarak_Km"
      )

      if len(df_terdekat) > 0:
        st.info(
            f"Ditemukan **{len(df_terdekat)} pantai** dalam radius {radius_maks}"
            " km dari posisi Anda:"
        )
        for _, r in df_terdekat.iterrows():
          st.markdown(
              f"""
                    <div class="search-result-box" style="padding:15px; margin-bottom:10px;">
                        <h4 style="margin:0 0 4px 0; color:#0f172a;">🌊 {r['Nama Pantai']} <span style="font-size:13px; color:#0284c7; font-weight:normal;">({r['Jarak_Km']:.2f} km dari Anda)</span></h4>
                        <p style="margin:2px 0; font-size:13px;">📍 Provinsi: {r['Provinsi']} | ⭐ Rating: {r['Rating Angka']} | 🏖️ Kualitas: <b>{r['Predikat'].title()}</b></p>
                        <p style="margin:2px 0; font-size:12.5px; color:#475569;">💬 Ulasan: {r.get('Ulasan 1', 'Belum ada ulasan')}</p>
                        <p style="margin:6px 0 0 0;"><a href="{r.get('Link Google Maps', '#')}" target="_blank" style="text-decoration:none; color:#0284c7; font-weight:600; font-size:12.5px;">Buka di Google Maps ↗</a></p>
                    </div>
                    """,
              unsafe_allow_html=True,
          )
      else:
        st.warning(
            f"Tidak ada pantai yang ditemukan dalam radius {radius_maks} km"
            " dari lokasi Anda saat ini."
        )
    else:
      st.info(
          "👆 Klik tombol di atas dan pilih **Allow / Izin** pada pop-up"
          " browser untuk mendeteksi posisi Anda."
      )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Peta Sebaran Destinasi Pantai")

    if len(df_filtered) == 0:
      st.warning("Tidak ada data yang cocok dengan filter saat ini.")
    else:
      center_lat = df_filtered["Latitude"].mean()
      center_lon = df_filtered["Longitude"].mean()

      m = folium.Map(
          location=[center_lat, center_lon],
          zoom_start=5,
          tiles="OpenStreetMap",
      )
      cluster = MarkerCluster().add_to(m)

      for _, row in df_filtered.iterrows():
        predikat_lower = row["Predikat"]
        predikat_title = predikat_lower.title()
        color = marker_color(predikat_lower)
        stars = stars_from_rating(row["Rating Angka"])
        link = row.get("Link Google Maps", "")

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
                <span class="legend-dot"><span class="dot" style="background:#2563eb;"></span>Bagus</span>
                <span class="legend-dot"><span class="dot" style="background:#dc2626;"></span>Biasa</span>
            </div>
            """,
          unsafe_allow_html=True,
      )

  with tab_prediksi:
    st.markdown("### 🔮 Prediksi Kualitas Pantai Baru")
    st.caption(
        "Menaksir predikat kualitas sebuah titik pantai baru berdasarkan wilayah"
        " provinsi yang dipilih (koordinat disesuaikan otomatis)."
    )

    if model_bundle is None:
      st.error("File model belum ditemukan di folder aplikasi.")
    else:
      model, le_target, le_provinsi = model_bundle

      with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
          provinsi_input = st.selectbox(
              "Provinsi", options=sorted(le_provinsi.classes_.tolist())
          )

        df_prov = df[df["Provinsi"] == provinsi_input]
        auto_lat = (
            float(df_prov["Latitude"].mean())
            if len(df_prov) > 0
            else float(df["Latitude"].mean())
        )
        auto_lon = (
            float(df_prov["Longitude"].mean())
            if len(df_prov) > 0
            else float(df["Longitude"].mean())
        )

        with c2:
          lat_input = st.number_input(
              "Latitude (Otomatis)",
              value=auto_lat,
              format="%.6f",
              disabled=True,
          )
        with c3:
          lon_input = st.number_input(
              "Longitude (Otomatis)",
              value=auto_lon,
              format="%.6f",
              disabled=True,
          )

        submitted = st.form_submit_button(
            "Prediksi Kualitas", use_container_width=True
        )

      if submitted:
        try:
          provinsi_encoded = le_provinsi.transform([provinsi_input])[0]
          X_new = pd.DataFrame(
              [[auto_lat, auto_lon, provinsi_encoded]],
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
        except Exception as e:
          st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

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
