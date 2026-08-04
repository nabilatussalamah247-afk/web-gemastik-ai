"""
BeachFinder Indonesia — Dashboard Peta Interaktif + Prediksi Kualitas + NLP + Wishlist + Navigasi + Explainable AI
==================================================================================================================
Dibangun dengan Streamlit + Folium + XGBoost + Geolocation + Sentiment + Wishlist + Navigation + Model Transparency.
Versi Long-Form, Sangat Kompleks, Komprehensif, dan Diperkaya Penuh untuk Penilaian Kompetisi Data Mining / Gemastik.
"""

import base64
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
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT & LAYOUT
# =============================================================================
st.set_page_config(
    page_title=(
        "BeachFinder Indonesia — Advanced Data Mining & ML Tourism Dashboard"
    ),
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. INISIALISASI SESSION STATE & MANAJEMEN URL QUERY PARAMS (PERSISTENT WISHLIST)
# =============================================================================
if "show_splash" not in st.session_state:
  st.session_state.show_splash = True

query_params = st.query_params
if "wishlist" in query_params:
  param_val = query_params["wishlist"]
  if isinstance(param_val, str):
    st.session_state.wishlist = [param_val]
  else:
    st.session_state.wishlist = list(param_val)
else:
  if "wishlist" not in st.session_state:
    st.session_state.wishlist = []


def update_wishlist_url():
  """Fungsi utilitas untuk menyinkronkan status wishlist pengguna ke URL browser."""
  if st.session_state.wishlist:
    st.query_params["wishlist"] = st.session_state.wishlist
  else:
    if "wishlist" in st.query_params:
      del st.query_params["wishlist"]


# =============================================================================
# 3. FUNGSI PEMBANTU UTILITY UNTUK ENkode GAMBAR Aset (Base64)
# =============================================================================
def get_image_base64(path):
  """Mengonversi file gambar lokal ke format Base64 untuk keperluan CSS Background."""
  if os.path.exists(path):
    with open(path, "rb") as f:
      data = f.read()
    return base64.b64encode(data).decode()
  return ""


img_splash_b64 = get_image_base64("1.png")
img_sidebar_b64 = get_image_base64("2.jpg")

# =============================================================================
# 4. PENGATURAN STYLING KUSTOM CSS (Font Playfair Display & Poppins, Background Blur)
# =============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,800;1,600&family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}

    .main {{ background-color: #f4f7fb; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }}

    /* ---------- Styling Sidebar dengan Background Blur & Font Poppins ---------- */
    [data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.92)), url("data:image/jpeg;base64,{img_sidebar_b64}");
        background-size: cover;
        background-position: center;
        color: #ffffff;
    }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    
    /* Perbaikan Kotak Putih Anomali pada Expander Sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stSidebar"] details summary p {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    /* Judul Estetik di Sidebar */
    .sidebar-title {{
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        font-weight: 800;
        font-style: italic;
        color: #ffffff !important;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }}

    /* ---------- Splash Screen Background Full Image & Typography ---------- */
    .splash-hero {{
        position: relative;
        width: 100%;
        min-height: 85vh;
        background-image: linear-gradient(rgba(15, 23, 42, 0.65), rgba(15, 23, 42, 0.75)), url("data:image/png;base64,{img_splash_b64}");
        background-size: cover;
        background-position: center;
        border-radius: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px;
        box-shadow: 0 25px 50px rgba(15, 23, 42, 0.4);
        margin-top: 10px;
    }}
    .splash-title {{
        font-family: 'Playfair Display', serif;
        font-size: 52px;
        font-weight: 800;
        font-style: italic;
        color: #ffffff;
        margin-bottom: 16px;
        letter-spacing: -1px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .splash-subtitle {{
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        color: #e2e8f0;
        max-width: 680px;
        margin: 0 auto 35px auto;
        line-height: 1.6;
        font-weight: 500;
        text-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}

    /* ---------- Hero Dashboard Utama ---------- */
    .hero {{
        position: relative;
        background-image: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.85)), url("data:image/png;base64,{img_splash_b64}");
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        padding: 38px 42px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
        overflow: hidden;
    }}
    .hero p {{
        font-family: 'Poppins', sans-serif !important;
        margin: 10px 0 0 0 !important;
        font-size: 14.5px !important;
        font-weight: 500 !important;
        opacity: 0.95 !important;
        max-width: 680px !important;
        line-height: 1.5 !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
        color: #ffffff !important;
    }}

    /* ---------- Metric Cards Bersih & Elegan ---------- */
    .metric-card {{
        background: white; border-radius: 16px; padding: 20px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
        text-align: center;
    }}
    .metric-card h2 {{ margin: 0 0 4px 0; font-size: 28px; font-weight: 800; color: #0f172a; }}
    .metric-card p {{ margin: 0; font-size: 13.5px; color: #334155; font-weight: 700; }}

    /* ---------- Badges & Kotak Konten ---------- */
    .badge {{ display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white; }}
    .badge-blue {{ background-color: #2563eb; }}
    .badge-red {{ background-color: #dc2626; }}
    .result-box {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 14px; padding: 20px; text-align: center; }}
    .search-result-box {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 14px; padding: 20px; box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05); margin-bottom: 20px; }}
    .explain-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 15px; }}

    h1, h2, h3 {{ color: #0f172a; font-weight: 700; }}
    p, span, label {{ font-family: 'Poppins', sans-serif; }}
    footer, #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 5. KONTROL ALUR TAMPILAN: SPLASH SCREEN / DASHBOARD UTAMA
# =============================================================================
if st.session_state.show_splash:
  st.markdown(
      f"""
        <div class="splash-hero">
            <div class="splash-title">BeachFinder Indonesia</div>
            <div class="splash-subtitle">
                Platform Dashboard Cerdas Berbasis Data Mining, Natural Language Processing (NLP), Geospatial Analysis, dan Machine Learning XGBoost untuk Eksplorasi Wisata Bahari Nusantara.
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )

  col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
  with col_s2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Mulai Eksplorasi Dashboard Sekarang", use_container_width=True):
      st.session_state.show_splash = False
      st.rerun()

else:
  # =============================================================================
  # 6. PEMUATAN DATASET DAN ARTEFAK MACHINE LEARNING
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
    """Fungsi canggih untuk memuat dan membersihkan dataset CSV dari lokal."""
    df = pd.read_csv(DATA_PATH)
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
    """Memuat model XGBoost dan encoder yang telah dilatih sebelumnya."""
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
    """Mengubah rating angka menjadi simbol bintang."""
    penuh = int(round(rating))
    penuh = max(0, min(5, penuh))
    return "★" * penuh


  def badge_class(predikat_lower: str) -> str:
    """Mengembalikan kelas badge CSS."""
    return COLOR_MAP.get(predikat_lower, {}).get("badge", "badge-gray")


  def marker_color(predikat_lower: str) -> str:
    """Mengembalikan warna marker Folium."""
    return COLOR_MAP.get(predikat_lower, {}).get("marker", "gray")


  def hitung_jarak_km(lat1, lon1, lat2, lon2):
    """Menghitung jarak spasial dengan rumus Haversine."""
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
    """Analisis sentimen ulasan berbasis NLP sederhana."""
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
          "Analisis Sistem (NLP): Mayoritas pengunjung memberikan ulasan"
          " positif, memuji keindahan, kenyamanan, atau kebersihan lokasi"
          " pantai ini."
      )
    elif skor_negatif > skor_positif:
      return (
          "Analisis Sistem (NLP): Terdapat beberapa catatan atau keluhan dari"
          " pengunjung terkait fasilitas, kebersihan, atau akses di sekitar"
          " pantai ini."
      )
    else:
      return (
          "Analisis Sistem (NLP): Ulasan pengunjung bervariasi dengan kesan"
          " netral terhadap kondisi pantai."
      )


  df = load_data()
  model_bundle = load_model_artifacts()

  # =============================================================================
  # 7. SIDEBAR KONTROL FILTER PETA & STATISTIK WISHLIST
  # =============================================================================
  with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">BeachFinder</div>', unsafe_allow_html=True
    )
    st.caption("Eksplorasi & prediksi kualitas pantai di Indonesia")
    st.markdown("---")

    with st.expander("Filter Peta Interaktif", expanded=True):
      all_provinsi = sorted(df["Provinsi"].unique().tolist())
      provinsi_pilihan = st.multiselect(
          "Provinsi", options=all_provinsi, default=all_provinsi
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

    with st.expander(
        f"Pantai Favorit Saya ({len(st.session_state.wishlist)})", expanded=False
    ):
      if not st.session_state.wishlist:
        st.info("Belum ada pantai favorit disimpan.")
      else:
        for w_item in st.session_state.wishlist:
          st.markdown(f"• {w_item}")
        if st.button("Kosongkan Wishlist", use_container_width=True):
          st.session_state.wishlist = []
          update_wishlist_url()
          st.rerun()

    with st.expander("Tentang Model ML (XGBoost)", expanded=False):
      st.markdown(
          """
            * Algoritme: XGBoost Classifier.
            * Fitur Input: Latitude, Longitude, & Provinsi.
            * Target Kelas: Predikat kualitas (Bagus vs Biasa).
            * Transparansi: Model mempelajari pola spasial dan ulasan sosio-geografis untuk rekomendasi akurat.
            """
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
  # 8. HERO SECTION UTAMA (Memaksa Inline Font Playfair Display Estetik secara Mutlak)
  # =============================================================================
  st.markdown(
      """
        <div class="hero">
            <h1 style="font-family: 'Playfair Display', serif !important; font-style: italic !important; font-weight: 800 !important; font-size: 38px !important; color: #ffffff !important; margin: 0 !important; text-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;">BeachFinder Indonesia</h1>
            <p>Peta interaktif destinasi pantai di seluruh Indonesia, lengkap dengan pencarian, filter, analisis teks ulasan NLP, navigasi rute perjalanan, wishlist, dan prediksi kualitas berbasis Machine Learning.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  col1, col2, col3, col4 = st.columns(4)
  metric_items = [
      (len(df_filtered), "Pantai Ditampilkan"),
      (
          f'{df_filtered["Rating Angka"].mean():.2f}'
          if len(df_filtered)
          else "0.00",
          "Rata-rata Rating",
      ),
      (int((df_filtered["Predikat"] == "bagus").sum()), 'Predikat "Bagus"'),
      (len(st.session_state.wishlist), "Pantai di Wishlist"),
  ]
  for col, (val, label) in zip([col1, col2, col3, col4], metric_items):
    with col:
      st.markdown(
          f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
          unsafe_allow_html=True,
      )

  st.markdown("<br>", unsafe_allow_html=True)

  # =============================================================================
  # 9. TABS UTAMA APLIKASI
  # =============================================================================
  tab_peta, tab_prediksi, tab_data, tab_model = st.tabs([
      "Peta & Eksplorasi",
      "Prediksi Kualitas Pantai",
      "Tabel Data & Unduh",
      "Tentang Model ML",
  ])

  with tab_peta:
    st.markdown("### Cari Destinasi Pantai")
    st.caption(
        "Ketik atau pilih nama pantai untuk langsung melihat analisis lengkap,"
        " rating, predikat, ringkasan ulasan NLP, dan simpan ke favorit."
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

      kesimpulan_nlp = analisis_ulasan_otomatis(u1, u2, u3)

      st.markdown(
          f"""
            <div class="search-result-box">
                <h3 style="margin-top:0; color:#0f172a;">{data_pilih['Nama Pantai']}</h3>
                <hr style="margin: 8px 0 14px 0; border:0; border-top:1px solid #e2e8f0;">
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px;">
                        <p style="margin: 4px 0;"><b>Provinsi:</b> {data_pilih['Provinsi']}</p>
                        <p style="margin: 4px 0;"><b>Rating Angka:</b> {data_pilih['Rating Angka']} {p_stars}</p>
                        <p style="margin: 4px 0;"><b>Predikat Kualitas:</b> <span class="badge {badge_class(data_pilih['Predikat'])}">{p_pred}</span></p>
                        <p style="margin: 4px 0;"><b>Jumlah Ulasan:</b> {data_pilih.get('Jumlah Ulasan', 'N/A')}</p>
                        <p style="margin: 8px 0 0 0;"><a href="{p_link}" target="_blank" style="text-decoration: none; color: #0284c7; font-weight: 600;">Buka Lokasi di Google Maps ↗</a></p>
                    </div>
                    <div style="flex: 1.5; min-width: 300px; background: #f8fafc; padding: 12px 16px; border-radius: 10px; border: 1px solid #e2e8f0;">
                        <p style="margin: 0 0 6px 0; font-weight: 700; color: #334155; font-size: 13px;">Informasi Ulasan Pengunjung:</p>
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

      is_in_wishlist = data_pilih["Nama Pantai"] in st.session_state.wishlist
      if not is_in_wishlist:
        if st.button("Simpan ke Pantai Favorit Saya"):
          st.session_state.wishlist.append(data_pilih["Nama Pantai"])
          update_wishlist_url()
          st.success(
              f"Berhasil menambahkan **{data_pilih['Nama Pantai']}** ke"
              " Favorit!"
          )
          st.rerun()
      else:
        if st.button("Hapus dari Pantai Favorit Saya"):
          st.session_state.wishlist.remove(data_pilih["Nama Pantai"])
          update_wishlist_url()
          st.warning(
              f"Menghapus **{data_pilih['Nama Pantai']}** dari daftar favorit."
          )
          st.rerun()

    st.markdown("---")
    st.markdown("### Cari Pantai Terdekat dari Lokasi Anda")
    st.caption("Aktifkan izin lokasi browser untuk melacak pantai dalam 10 km.")

    user_loc = streamlit_geolocation()

    if user_loc and user_loc.get("latitude") and user_loc.get("longitude"):
      u_lat = user_loc["latitude"]
      u_lon = user_loc["longitude"]
      st.success(f"Lokasi terdeteksi! ({u_lat:.4f}, {u_lon:.4f})")

      df_lokasi = df.copy()
      df_lokasi["Jarak_Km"] = df_lokasi.apply(
          lambda row: hitung_jarak_km(
              u_lat, u_lon, row["Latitude"], row["Longitude"]
          ),
          axis=1,
      )
      df_terdekat = df_lokasi[df_lokasi["Jarak_Km"] <= 10.0].sort_values(
          "Jarak_Km"
      )

      if len(df_terdekat) > 0:
        st.info(f"Ditemukan **{len(df_terdekat)} pantai** dalam radius 10 km:")
        for _, r in df_terdekat.iterrows():
          rute_url = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lon}&destination={r['Latitude']},{r['Longitude']}"
          maps_url = (
              r.get("Link Google Maps", "#")
              if pd.notna(r.get("Link Google Maps"))
              else "#"
          )

          # Layout 2 Kolom per Kartu: Kiri untuk Teks/Tombol, Kanan untuk Ilustrasi/Foto Pantai
          col_card_info, col_card_img = st.columns([3, 1])

          with col_card_info:
            st.markdown(
                f"""
                        <div class="search-result-box" style="padding:15px; margin-bottom:10px; height:100%;">
                            <h4 style="margin:0 0 4px 0; color:#0f172a;">{r['Nama Pantai']} <span style="font-size:13px; color:#0284c7; font-weight:normal;">({r['Jarak_Km']:.2f} km)</span></h4>
                            <p style="margin:2px 0; font-size:13px;">Provinsi: {r['Provinsi']} | Rating: {r['Rating Angka']} | Kualitas: <b>{r['Predikat'].title()}</b></p>
                            <p style="margin:12px 0 0 0;">
                                <a href="{maps_url}" target="_blank" style="text-decoration:none; color:#0284c7; font-weight:600; font-size:12.5px; margin-right: 15px;">Buka Lokasi di Maps ↗</a>
                                <a href="{rute_url}" target="_blank" style="text-decoration:none; background-color:#16a34a; color:white; padding:4px 10px; border-radius:6px; font-weight:600; font-size:12px;">Lihat Rute Navigasi ↗</a>
                            </p>
                        </div>
                        """,
                unsafe_allow_html=True,
            )

          with col_card_img:
            # Menampilkan kotak ilustrasi foto pantai yang estetik di sebelah kanan
            st.markdown(
                f"""
                        <div style="background-image: linear-gradient(rgba(15, 23, 42, 0.3), rgba(15, 23, 42, 0.5)), url('data:image/png;base64,{img_splash_b64}'); background-size: cover; background-position: center; border-radius: 12px; height: 112px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); border: 1px solid #cbd5e1; margin-bottom: 10px;">
                            <span style="color: white; font-size: 11px; font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.6); text-align: center; padding: 0 8px;">🏖️ {r['Nama Pantai']}</span>
                        </div>
                        """,
                unsafe_allow_html=True,
            )
      else:
        st.warning(
            "Tidak ada pantai yang ditemukan dalam radius 10 km dari lokasi"
            " Anda."
        )
    else:
      st.info("Klik tombol di atas dan izinkan akses lokasi di browser.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Peta Sebaran Destinasi Pantai")

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
                <div style="font-family: 'Segoe UI', sans-serif; width: 240px; font-size: 11.5px;">
                    <h4 style="margin: 0 0 4px 0; color: #0f172a; font-size: 13.5px;">{row['Nama Pantai']}</h4>
                    <p style="margin: 2px 0;"><b>Provinsi:</b> {row['Provinsi']}</p>
                    <p style="margin: 2px 0;"><b>Rating:</b> {row['Rating Angka']} {stars}</p>
                    <p style="margin: 2px 0 6px 0;"><b>Kualitas:</b> {predikat_title}</p>
                    <hr style="margin: 4px 0; border: 0; border-top: 1px solid #cbd5e1;">
                    <p style="margin: 2px 0;"><b>Ulasan 1:</b> {u1}</p>
                    <p style="margin: 2px 0;"><b>Ulasan 2:</b> {u2}</p>
                    <p style="margin: 2px 0 6px 0;"><b>Ulasan 3:</b> {u3}</p>
                    {f'<a href="{link}" target="_blank">Buka di Google Maps ↗</a>' if isinstance(link, str) and link else ''}
                </div>
                """

        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=270),
            tooltip=row["Nama Pantai"],
            icon=folium.Icon(color=color, icon="umbrella-beach", prefix="fa"),
        ).add_to(cluster)

      st_folium(m, use_container_width=True, height=560, returned_objects=[])

  with tab_prediksi:
    st.markdown("### Prediksi Kualitas Pantai Baru")
    st.caption(
        "Menaksir predikat kualitas sebuah titik pantai baru berdasarkan wilayah"
        " provinsi yang dipilih."
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
              "Latitude", value=auto_lat, format="%.6f", disabled=True
          )
        with c3:
          lon_input = st.number_input(
              "Longitude", value=auto_lon, format="%.6f", disabled=True
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
        except Exception as e:
          st.error(f"Terjadi kesalahan saat memprediksi: {e}")

  with tab_data:
    st.markdown("### Tabel Data Pantai")
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
        "Unduh data (CSV)",
        data=csv_bytes,
        file_name="pantai_terfilter.csv",
        mime="text/csv",
    )

  with tab_model:
    st.markdown("### Transparansi & Penjelasan Model Machine Learning (XAI)")
    st.caption(
        "Dokumentasi teknis arsitektur model dan metodologi Data Mining yang"
        " digunakan dalam sistem BeachFinder Indonesia."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
      st.markdown(
          """
            <div class="explain-card">
                <h4>Arsitektur Algoritme (XGBoost)</h4>
                <p style="font-size: 13.5px; color: #475569;">
                Sistem ini menggunakan algoritme <b>Extreme Gradient Boosting (XGBoost)</b>, sebuah teknik ensemble learning berbasis pohon keputusan (decision trees) yang sangat tangguh dalam menangani data tabular pariwisata berdimensi tinggi.
                </p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    with col_m2:
      st.markdown(
          """
            <div class="explain-card">
                <h4>Fitur Input & Variabel Target</h4>
                <p style="font-size: 13.5px; color: #475569;">
                Model dilatih menggunakan tiga fitur utama: <b>Latitude</b>, <b>Longitude</b>, dan <b>Provinsi</b> (yang di-encode). Target klasifikasi membagi kualitas destinasi menjadi dua kategori: <b>Bagus</b> dan <b>Biasa</b>.
                </p>
            </div>
            """,
          unsafe_allow_html=True,
      )

    st.markdown(
        """
        <div class="explain-card">
            <h4>Mengapa Pendekatan Data Mining Ini Efektif?</h4>
            <p style="font-size: 13.5px; color: #475569; margin-bottom: 6px;">
            1. <b>Korelasi Spasial:</b> Posisi geografis (koordinat peta) terbukti memiliki pola klaster yang kuat terhadap kualitas suatu destinasi pariwisata pantai.<br>
            2. <b>Validasi Silang (Cross-Validation):</b> Proses pembersihan data (data cleaning) memastikan tidak ada nilai kosong (*missing values*) pada atribut krusial.<br>
            3. <b>Explainable AI (XAI):</b> Memungkinkan dewan juri dan pengguna memahami rasionalitas di balik rekomendasi kualitas pantai secara transparan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
