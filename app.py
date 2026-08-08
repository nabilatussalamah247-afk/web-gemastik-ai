"""
BeachFinder Indonesia — Dashboard Peta Interaktif + Prediksi Kualitas + Top 5 Grafik Ulasan + NLP + Wishlist + Navigasi
=========================================================================================================================
Dibangun dengan Streamlit + Folium + XGBoost + Altair Charts + Geolocation + Sentiment + Wishlist + Navigation.
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
# 2. INISIALISASI SESSION STATE & MANAJEMEN URL QUERY PARAMS (STABIL BERBASIS COMMA-SEPARATED)
# =============================================================================
if "show_splash" not in st.session_state:
  st.session_state.show_splash = True

query_params = st.query_params
# Pemisah wishlist di URL memakai "|" (bukan ",") karena 37 nama pantai di
# dataset mengandung tanda koma (mis. "Pantai Tebing Karang, Labuang"),
# sehingga pemisahan berbasis koma akan memecah nama pantai itu jadi dua
# entri yang salah. "|" tidak muncul di nama pantai manapun.
WISHLIST_SEP = "|"

if "wishlist" in query_params:
  val_param = query_params["wishlist"]
  if isinstance(val_param, list):
    combined = WISHLIST_SEP.join(val_param)
    st.session_state.wishlist = [
        item.strip() for item in combined.split(WISHLIST_SEP) if item.strip()
    ]
  else:
    st.session_state.wishlist = [
        item.strip()
        for item in str(val_param).split(WISHLIST_SEP)
        if item.strip()
    ]
else:
  if "wishlist" not in st.session_state:
    st.session_state.wishlist = []


def update_wishlist_url():
  """Fungsi utilitas untuk menyinkronkan status wishlist ke URL browser secara aman."""
  if st.session_state.wishlist:
    st.query_params["wishlist"] = WISHLIST_SEP.join(st.session_state.wishlist)
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
img_wave_b64 = get_image_base64("3.jpg")

# =============================================================================
# 4. PENGATURAN STYLING KUSTOM CSS (Background Gradasi Elegan & Komponen Lainnya)
# =============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,600;0,9..144,700;0,9..144,900;1,9..144,600&family=Poppins:wght@400;500;600;700;800&display=swap');

    :root {{
        --teal-950: #062e2c;
        --teal-900: #0a3d3d;
        --teal-700: #0e5c58;
        --teal-500: #0f9b8e;
        --teal-300: #6bcfc0;
        --coral-500: #ff6b4a;
        --coral-600: #e8532f;
        --lime-400: #c9e265;
        --sand-50: #fdf8f0;
        --sand-100: #f7ecd9;
        --ink-900: #10241f;
        --ink-600: #3a5450;
    }}

    html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
    h1, h2, h3, .hero h1, .splash-title, .sidebar-title {{
        font-family: 'Fraunces', serif !important;
    }}

    /* Background utama: gradasi teal gelap di atas foto pantai */
    .main {{
        background-image: linear-gradient(160deg, rgba(6, 46, 44, 0.90), rgba(10, 61, 61, 0.80)), url("data:image/png;base64,{img_splash_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: var(--sand-50);
        border-radius: 28px;
        box-shadow: 0 25px 60px rgba(6, 46, 44, 0.45);
        margin-top: 20px;
        margin-bottom: 20px;
        padding-left: 2rem;
        padding-right: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-image: linear-gradient(180deg, rgba(6, 46, 44, 0.92), rgba(10, 61, 61, 0.96)), url("data:image/jpeg;base64,{img_sidebar_b64}");
        background-size: cover;
        background-position: center;
        color: #ffffff;
    }}

    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
        color: #f2fbf9 !important;
        font-weight: 600 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        background-color: transparent !important;
        border: none !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary {{
        background-color: rgba(255, 255, 255, 0.06) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover,
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary:focus {{
        background-color: rgba(255, 255, 255, 0.14) !important;
        border-color: var(--lime-400) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] details summary svg {{
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 700 !important;
    }}

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div:nth-child(1) {{
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div:nth-child(1):hover {{
        border-color: var(--lime-400) !important;
        background-color: rgba(255, 255, 255, 0.14) !important;
    }}

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] svg {{
        fill: #ffffff !important;
        color: #ffffff !important;
    }}

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input {{
        color: #ffffff !important;
    }}

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"],
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] div,
    [data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] > span {{
        background-color: var(--teal-500) !important;
        transition: background-color 0.2s ease;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]:hover {{
        background-color: var(--teal-700) !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] svg {{
        fill: #ffffff !important;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(135deg, var(--coral-500), var(--coral-600)) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(255, 107, 74, 0.35) !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 107, 74, 0.45) !important;
    }}
    [data-testid="stSidebar"] .stButton > button * {{
        color: #ffffff !important;
    }}

    .sidebar-title {{
        font-size: 32px;
        font-weight: 900;
        font-style: italic;
        color: #ffffff !important;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }}

    /* Splash */
    .splash-hero {{
        position: relative;
        width: 100%;
        min-height: 85vh;
        background-image: linear-gradient(160deg, rgba(6, 46, 44, 0.55), rgba(6, 46, 44, 0.82)), url("data:image/png;base64,{img_splash_b64}");
        background-size: cover;
        background-position: center;
        border-radius: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px 20px;
        box-shadow: 0 30px 60px rgba(6, 46, 44, 0.5);
        margin-top: 10px;
    }}
    .splash-title {{
        font-size: 56px;
        font-weight: 900;
        font-style: italic;
        color: #ffffff;
        margin-bottom: 16px;
        letter-spacing: -1px;
        text-shadow: 0 4px 16px rgba(0,0,0,0.35);
    }}
    .splash-subtitle {{
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        color: #d7f2ee;
        max-width: 680px;
        margin: 0 auto 35px auto;
        line-height: 1.6;
        font-weight: 500;
        text-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}

    /* Hero dashboard */
    .hero {{
        position: relative;
        background-image: linear-gradient(135deg, rgba(6, 46, 44, 0.88), rgba(15, 155, 142, 0.72)), url("data:image/png;base64,{img_splash_b64}");
        background-size: cover;
        background-position: center;
        border-radius: 22px;
        padding: 38px 42px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px rgba(6, 46, 44, 0.35);
        overflow: hidden;
    }}
    .hero::after {{
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(201, 226, 101, 0.35), transparent 70%);
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
        position: relative;
        z-index: 2;
    }}

    /* Ticker fakta berjalan */
    .fact-ticker {{
        background: var(--teal-950);
        border-radius: 999px;
        padding: 10px 0;
        margin-bottom: 24px;
        overflow: hidden;
        white-space: nowrap;
        border: 1px solid rgba(201, 226, 101, 0.25);
    }}
    .fact-ticker-track {{
        display: inline-block;
        padding-left: 100%;
        animation: ticker-scroll 38s linear infinite;
        color: var(--lime-400);
        font-weight: 600;
        font-size: 13.5px;
        letter-spacing: 0.2px;
    }}
    .fact-ticker-track span {{ margin: 0 28px; color: #eafff5; }}
    .fact-ticker-track span::before {{ content: "✦"; color: var(--lime-400); margin-right: 8px; }}
    @keyframes ticker-scroll {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-100%); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .fact-ticker-track {{ animation: none; }}
    }}

    /* Metric cards */
    .metric-card {{
        position: relative;
        background-color: #ffffff !important;
        background-image: linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.92)), url("data:image/jpeg;base64,{img_wave_b64}");
        background-repeat: no-repeat;
        background-position: bottom right;
        background-size: 120px auto;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid var(--sand-100);
        border-top: 3px solid var(--teal-500);
        box-shadow: 0 6px 18px rgba(6, 46, 44, 0.06);
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        overflow: hidden;
    }}
    .metric-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 16px 30px rgba(15, 155, 142, 0.2);
        border-top-color: var(--coral-500);
    }}
    .metric-card h2 {{ margin: 0 0 4px 0; font-size: 30px; font-weight: 800; color: var(--teal-950); position: relative; z-index: 2; font-family: 'Fraunces', serif !important; }}
    .metric-card p {{ margin: 0; font-size: 13px; color: var(--ink-600); font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; position: relative; z-index: 2; }}

    .badge {{ display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; color: white; }}
    .badge-teal {{ background: linear-gradient(135deg, var(--teal-500), var(--teal-700)); }}
    .badge-coral {{ background: linear-gradient(135deg, var(--coral-500), var(--coral-600)); }}
    .result-box {{ background: #f1fbf5; border: 1px solid #bfead9; border-radius: 16px; padding: 20px; text-align: center; }}
    .search-result-box {{ background: #ffffff; border: 1px solid var(--sand-100); border-radius: 16px; padding: 20px; box-shadow: 0 6px 18px rgba(6, 46, 44, 0.06); margin-bottom: 20px; transition: box-shadow 0.25s ease; }}
    .search-result-box:hover {{ box-shadow: 0 12px 28px rgba(15, 155, 142, 0.14); }}
    .explain-card {{ background: var(--sand-100); border: 1px solid #ecdfc4; border-left: 4px solid var(--teal-500); border-radius: 12px; padding: 18px; margin-bottom: 15px; transition: border-left-color 0.25s ease; }}
    .explain-card:hover {{ border-left-color: var(--coral-500); }}

    h1, h2, h3 {{ color: var(--teal-950); font-weight: 700; }}
    p, span, label {{ font-family: 'Poppins', sans-serif; }}
    footer, #MainMenu {{ visibility: hidden; }}

    /* Tombol utama non-sidebar */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        color: var(--ink-600);
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--teal-700) !important;
        font-weight: 800;
    }}

    /* Panel card native Streamlit (st.container(border=True)) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 16px !important;
        border: 1px solid var(--sand-100) !important;
        box-shadow: 0 6px 16px rgba(6, 46, 44, 0.05);
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 10px 26px rgba(15, 155, 142, 0.12);
    }}

    [data-testid="stForm"] {{
        background-color: #ffffff;
        border-radius: 18px !important;
        border: 1px solid var(--sand-100) !important;
        border-top: 3px solid var(--teal-500) !important;
        box-shadow: 0 8px 22px rgba(6, 46, 44, 0.07);
        padding: 1.4rem 1.6rem !important;
    }}
    [data-testid="stForm"] .stFormSubmitButton > button {{
        background: linear-gradient(135deg, var(--coral-500), var(--coral-600)) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 14px rgba(255, 107, 74, 0.3) !important;
    }}
    [data-testid="stForm"] .stFormSubmitButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 107, 74, 0.4) !important;
    }}

    /* Header section dengan ikon bulat + eyebrow */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 6px 0 4px 0;
    }}
    .section-header .icon-badge {{
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--teal-500), var(--teal-700));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        box-shadow: 0 4px 12px rgba(15, 155, 142, 0.3);
    }}
    .section-header .section-text h3 {{
        margin: 0 !important;
        font-family: 'Fraunces', serif !important;
        line-height: 1.1;
    }}
    .section-eyebrow {{
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--teal-500);
        margin: 0 0 2px 0;
    }}
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
      "bagus": {"marker": "cadetblue", "badge": "badge-teal", "hex": "#0f9b8e"},
      "biasa": {"marker": "orange", "badge": "badge-coral", "hex": "#ff6b4a"},
  }
  PREDIKAT_ORDER = ["Bagus", "Biasa"]

  # Mapping balik: provinsi hasil konsolidasi (dipakai di seluruh tampilan
  # aplikasi) -> label provinsi mentah yang dikenali oleh le_provinsi.pkl.
  # le_provinsi.pkl dilatih sebelum konsolidasi provinsi diterapkan, sehingga
  # "Maluku" dan "Maluku Utara" tidak ada di antara kelasnya. "Maluku" berasal
  # dari gabungan 4 label mentah (Ambon, Pulau Buru, Pulau Seram, Pulau Wetar);
  # "Ambon" dipakai sebagai representasi karena cakupannya paling relevan
  # untuk wilayah Maluku secara umum.
  PROVINSI_ENCODING_PROXY = {
      "Maluku": "Ambon",
      "Maluku Utara": "Ternate",
  }

  DATA_PATH = "dataset_clean.csv"
  MODEL_PATH = "model_xgboost_wisata.pkl"
  LE_TARGET_PATH = "label_encoder_target.pkl"
  LE_PROVINSI_PATH = "le_provinsi.pkl"


  @st.cache_data
  def load_data():
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

    if "Jumlah Ulasan" in df.columns:
      df["Jumlah Ulasan"] = (
          pd.to_numeric(df["Jumlah Ulasan"], errors="coerce").fillna(0).astype(int)
      )
    else:
      df["Jumlah Ulasan"] = 100

    pantai_yang_dihapus = [
        "Melasti Beach",
        "Pantai Melasti",
        "Pantai Baru",
    ]
    df = df[~df["Nama Pantai"].isin(pantai_yang_dihapus)]

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
    return "★" * penuh


  def badge_class(predikat_lower: str) -> str:
    return COLOR_MAP.get(predikat_lower, {}).get("badge", "badge-gray")


  def marker_color(predikat_lower: str) -> str:
    return COLOR_MAP.get(predikat_lower, {}).get("marker", "gray")


  def section_header(icon: str, eyebrow: str, title: str):
    """Render header section dengan ikon bulat + eyebrow, dipakai di semua tab."""
    st.markdown(
        f"""
          <div class="section-header">
              <div class="icon-badge">{icon}</div>
              <div class="section-text">
                  <p class="section-eyebrow">{eyebrow}</p>
                  <h3>{title}</h3>
              </div>
          </div>
          """,
        unsafe_allow_html=True,
    )


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

  # Label pencarian dibuat di sini (bukan di dalam tab_peta) supaya sudah
  # tersedia untuk sidebar (yang menampilkan wishlist) maupun semua tab —
  # sidebar dijalankan Streamlit lebih dulu daripada isi tab mana pun.
  nama_duplikat = set(
      df.loc[df.duplicated("Nama Pantai", keep=False), "Nama Pantai"]
  )

  def label_pencarian(row):
    if row["Nama Pantai"] in nama_duplikat:
      return f"{row['Nama Pantai']} ({row['Provinsi']})"
    return row["Nama Pantai"]

  df["_label_pencarian"] = df.apply(label_pencarian, axis=1)

  # =============================================================================
  # 7. SIDEBAR KONTROL FILTER PETA & STATISTIK WISHLIST
  # =============================================================================
  with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">BeachFinder</div>', unsafe_allow_html=True
    )
    st.caption("Eksplorasi & prediksi kualitas pantai di Indonesia")
    st.markdown("---")

    if st.button("🧹 Clear Cache & Refresh Data", use_container_width=True):
      st.cache_data.clear()
      st.success("Cache dibersihkan! Memuat ulang...")
      st.rerun()

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
          # Coba cocokkan dengan label pencarian dulu (format baru, sudah
          # disambiguasi provinsi); kalau tidak ketemu, fallback ke nama
          # polos untuk kompatibilitas dengan link wishlist lama.
          match_row = df[df["_label_pencarian"] == w_item]
          if match_row.empty:
            match_row = df[df["Nama Pantai"] == w_item]
          display_name = (
              match_row.iloc[0]["Nama Pantai"] if not match_row.empty else w_item
          )
          if (
              not match_row.empty
              and pd.notna(match_row.iloc[0].get("Link Google Maps"))
              and match_row.iloc[0].get("Link Google Maps") != "#"
          ):
            maps_url = match_row.iloc[0]["Link Google Maps"]
            st.markdown(
                f'• <a href="{maps_url}" target="_blank"'
                ' style="text-decoration: none; color: #38bdf8;'
                f' font-weight: 600;">{display_name} ↗</a>',
                unsafe_allow_html=True,
            )
          else:
            st.markdown(f"• {display_name}")

        st.markdown("<br>", unsafe_allow_html=True)
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
  # 8. HERO SECTION UTAMA
  # =============================================================================
  st.markdown(
      """
        <div class="hero">
            <h1 style="font-family: 'Fraunces', serif !important; font-style: italic !important; font-weight: 900 !important; font-size: 40px !important; color: #ffffff !important; margin: 0 !important; text-shadow: 0 2px 10px rgba(0,0,0,0.4) !important; position: relative; z-index: 2;">BeachFinder Indonesia</h1>
            <p>Peta interaktif destinasi pantai di seluruh Indonesia, lengkap dengan pencarian, filter, analisis teks ulasan NLP, navigasi rute perjalanan, wishlist, dan prediksi kualitas berbasis Machine Learning.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  # Ticker fakta berjalan — dihitung langsung dari dataset asli (df, bukan
  # df_filtered), supaya isinya tetap konsisten walau pengguna sedang
  # menyaring peta ke provinsi tertentu.
  top_provinsi = df["Provinsi"].value_counts().idxmax()
  top_provinsi_n = int(df["Provinsi"].value_counts().max())
  n_provinsi = df["Provinsi"].nunique()
  pantai_rating_5 = int((df["Rating Angka"] >= 4.8).sum())
  ulasan_terbanyak_row = df.loc[df["Jumlah Ulasan"].idxmax()]
  fakta_list = [
      f"📍 {top_provinsi} adalah provinsi dengan destinasi pantai terbanyak: {top_provinsi_n} lokasi",
      f"🌊 Data mencakup {n_provinsi} provinsi di seluruh Indonesia, dari Sabang sampai Merauke",
      f"⭐ {pantai_rating_5} pantai punya rating hampir sempurna (≥ 4.8)",
      f"💬 {ulasan_terbanyak_row['Nama Pantai']} punya ulasan terbanyak: {int(ulasan_terbanyak_row['Jumlah Ulasan'])} ulasan",
      f"🤖 Prediksi kualitas pantai baru ditenagai model XGBoost dari {len(df)} data pantai nyata",
  ]
  st.markdown(
      f"""
        <div class="fact-ticker">
            <div class="fact-ticker-track">
                {''.join(f'<span>{f}</span>' for f in fakta_list * 2)}
            </div>
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
  tab_peta, tab_prediksi, tab_top5, tab_data, tab_model = st.tabs([
      "Peta & Eksplorasi",
      "Prediksi Kualitas Pantai",
      "Top 5 Grafik Ulasan",
      "Tabel Data & Unduh",
      "Tentang Model ML",
  ])

  with tab_peta:
    section_header("🔍", "Pencarian", "Cari Destinasi Pantai")
    st.caption(
        "Ketik atau pilih nama pantai untuk langsung melihat analisis lengkap,"
        " rating, predikat, ringkasan ulasan NLP, dan simpan ke favorit."
    )

    with st.container(border=True):
      list_nama_pantai = sorted(df["_label_pencarian"].unique().tolist())
      pilihan_pencarian = st.selectbox(
          "Pilih atau ketik nama pantai:",
          options=["-- Pilih / Cari Pantai --"] + list_nama_pantai,
      )

    if pilihan_pencarian != "-- Pilih / Cari Pantai --":
      data_pilih = df[df["_label_pencarian"] == pilihan_pencarian].iloc[0]
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
                <h3 style="margin-top:0; color:var(--teal-950);">{data_pilih['Nama Pantai']}</h3>
                <hr style="margin: 8px 0 14px 0; border:0; border-top:1px solid var(--sand-100);">
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 250px;">
                        <p style="margin: 4px 0;"><b>Provinsi:</b> {data_pilih['Provinsi']}</p>
                        <p style="margin: 4px 0;"><b>Rating Angka:</b> {data_pilih['Rating Angka']} {p_stars}</p>
                        <p style="margin: 4px 0;"><b>Predikat Kualitas:</b> <span class="badge {badge_class(data_pilih['Predikat'])}">{p_pred}</span></p>
                        <p style="margin: 4px 0;"><b>Jumlah Ulasan:</b> {data_pilih.get('Jumlah Ulasan', 'N/A')}</p>
                        <p style="margin: 8px 0 0 0;"><a href="{p_link}" target="_blank" style="text-decoration: none; color: var(--teal-700); font-weight: 600;">Buka Lokasi di Google Maps ↗</a></p>
                    </div>
                    <div style="flex: 1.5; min-width: 300px; background: var(--sand-100); padding: 12px 16px; border-radius: 10px; border: 1px solid var(--sand-100);">
                        <p style="margin: 0 0 6px 0; font-weight: 700; color: var(--ink-600); font-size: 13px;">Informasi Ulasan Pengunjung:</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 1:</b> {u1}</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 2:</b> {u2}</p>
                        <p style="margin: 3px 0; font-size: 12.5px;">• <b>Ulasan 3:</b> {u3}</p>
                        <hr style="margin: 8px 0; border:0; border-top:1px solid var(--sand-100);">
                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #0369a1; font-style: italic;">{kesimpulan_nlp}</p>
                    </div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      # Disimpan sebagai label pencarian (bukan Nama Pantai polos) supaya
      # pantai dengan nama kembar di provinsi lain tidak salah tertaut saat
      # dibuka kembali dari wishlist/URL.
      wishlist_key = data_pilih["_label_pencarian"]
      is_in_wishlist = wishlist_key in st.session_state.wishlist
      if not is_in_wishlist:
        if st.button("Simpan ke Pantai Favorit Saya"):
          st.session_state.wishlist.append(wishlist_key)
          update_wishlist_url()
          st.success(
              f"Berhasil menambahkan **{data_pilih['Nama Pantai']}** ke"
              " Favorit!"
          )
          st.rerun()
      else:
        if st.button("Hapus dari Pantai Favorit Saya"):
          st.session_state.wishlist.remove(wishlist_key)
          update_wishlist_url()
          st.warning(
              f"Menghapus **{data_pilih['Nama Pantai']}** dari daftar favorit."
          )
          st.rerun()

    st.markdown("---")
    section_header("📍", "Lokasi Real-Time", "Cari Pantai Terdekat dari Lokasi Anda")
    st.caption(
        "Gunakan tombol di bawah untuk mendeteksi lokasi atau pilih koordinat"
        " otomatis."
    )

    with st.container(border=True):
      col_geo1, col_geo2 = st.columns([1, 2])
      with col_geo1:
        deteksi_lokasi = st.button(
            "📍 Deteksi Lokasi Saya", use_container_width=True
        )

    # Lokasi disimpan di session_state supaya hasil "pantai terdekat" tidak
    # hilang saat ada rerun lain (mis. menambah wishlist, ganti filter).
    # Sebelumnya nilai ini cuma dibaca sekali langsung dari klik tombol, jadi
    # hilang begitu Streamlit rerun karena interaksi apa pun.
    if deteksi_lokasi:
      user_loc = streamlit_geolocation()
      if user_loc and user_loc.get("latitude") and user_loc.get("longitude"):
        st.session_state.u_lat = user_loc["latitude"]
        st.session_state.u_lon = user_loc["longitude"]
      else:
        st.session_state.u_lat, st.session_state.u_lon = -5.4297, 105.2615
        st.info(
            "Menggunakan lokasi estimasi default (Bandar Lampung) karena"
            " pembatasan browser cloud."
        )

    u_lat = st.session_state.get("u_lat")
    u_lon = st.session_state.get("u_lon")

    if u_lat and u_lon:
      st.success(f"Lokasi aktif: ({u_lat:.4f}, {u_lon:.4f})")

      df_lokasi = df.copy()
      df_lokasi["Jarak_Km"] = df_lokasi.apply(
          lambda row: hitung_jarak_km(
              u_lat, u_lon, row["Latitude"], row["Longitude"]
          ),
          axis=1,
      )
      df_terdekat = df_lokasi[df_lokasi["Jarak_Km"] <= 15.0].sort_values(
          "Jarak_Km"
      )

      if len(df_terdekat) > 0:
        st.info(f"Ditemukan **{len(df_terdekat)} pantai** terdekat:")
        for _, r in df_terdekat.iterrows():
          rute_url = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lon}&destination={r['Latitude']},{r['Longitude']}"
          maps_url = (
              r.get("Link Google Maps", "#")
              if pd.notna(r.get("Link Google Maps"))
              else "#"
          )

          col_card_info, col_card_img = st.columns([3, 1])

          with col_card_info:
            st.markdown(
                f"""
                        <div class="search-result-box" style="padding:15px; margin-bottom:10px; height:100%;">
                            <h4 style="margin:0 0 4px 0; color:var(--teal-950);">{r['Nama Pantai']} <span style="font-size:13px; color:var(--teal-700); font-weight:normal;">({r['Jarak_Km']:.2f} km)</span></h4>
                            <p style="margin:2px 0; font-size:13px;">Provinsi: {r['Provinsi']} | Rating: {r['Rating Angka']} | Kualitas: <b>{r['Predikat'].title()}</b></p>
                            <p style="margin:12px 0 0 0;">
                                <a href="{maps_url}" target="_blank" style="text-decoration:none; color:var(--teal-700); font-weight:600; font-size:12.5px; margin-right: 15px;">Buka Lokasi di Maps ↗</a>
                                <a href="{rute_url}" target="_blank" style="text-decoration:none; background-color:#16a34a; color:white; padding:4px 10px; border-radius:6px; font-weight:600; font-size:12px;">Lihat Rute Navigasi ↗</a>
                            </p>
                        </div>
                        """,
                unsafe_allow_html=True,
            )

          with col_card_img:
            st.markdown(
                f"""
                        <div style="background-image: linear-gradient(rgba(6, 46, 44, 0.35), rgba(15, 155, 142, 0.4)), url('data:image/png;base64,{img_splash_b64}'); background-size: cover; background-position: center; border-radius: 12px; height: 112px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(6, 46, 44, 0.1); border: 1px solid var(--sand-100); margin-bottom: 10px;">
                            <span style="color: white; font-size: 11px; font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.6); text-align: center; padding: 0 8px;">🏖️ {r['Nama Pantai']}</span>
                        </div>
                        """,
                unsafe_allow_html=True,
            )
      else:
        st.warning("Tidak ada pantai yang ditemukan dalam radius terdekat.")
    else:
      st.info(
          "Silakan klik tombol **📍 Deteksi Lokasi Saya** di atas untuk"
          " menampilkan daftar pantai terdekat."
      )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🗺️", "Visualisasi Spasial", "Peta Sebaran Destinasi Pantai")

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
                    <h4 style="margin: 0 0 4px 0; color: var(--teal-950); font-size: 13.5px;">{row['Nama Pantai']}</h4>
                    <p style="margin: 2px 0;"><b>Provinsi:</b> {row['Provinsi']}</p>
                    <p style="margin: 2px 0;"><b>Rating:</b> {row['Rating Angka']} {stars}</p>
                    <p style="margin: 2px 0 6px 0;"><b>Kualitas:</b> {predikat_title}</p>
                    <hr style="margin: 4px 0; border: 0; border-top: 1px solid var(--sand-100);">
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
    section_header("🔮", "Machine Learning", "Prediksi Kualitas Pantai Baru")
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
          # Opsi provinsi diambil dari data yang sudah dikonsolidasi (df),
          # bukan langsung dari le_provinsi.classes_, supaya daftar ini
          # konsisten dengan provinsi yang ditampilkan di peta, tabel, dan
          # filter lain di aplikasi (mis. "Maluku" muncul, bukan "Ambon"
          # atau "Pulau Buru" sebagai provinsi terpisah).
          provinsi_input = st.selectbox(
              "Provinsi", options=sorted(df["Provinsi"].unique().tolist())
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
          provinsi_for_encoding = PROVINSI_ENCODING_PROXY.get(
              provinsi_input, provinsi_input
          )
          provinsi_encoded = le_provinsi.transform([provinsi_for_encoding])[0]
          expected_features = (
              model.get_booster().feature_names
              if hasattr(model, "get_booster")
              and model.get_booster().feature_names
              else ["Latitude", "Longitude", "Provinsi_Encoded"]
          )
          col_prov_name = (
              expected_features[2]
              if len(expected_features) > 2
              else "Provinsi_Encoded"
          )
          X_new = pd.DataFrame(
              [[auto_lat, auto_lon, provinsi_encoded]],
              columns=[
                  expected_features[0]
                  if len(expected_features) > 0
                  else "Latitude",
                  expected_features[1]
                  if len(expected_features) > 1
                  else "Longitude",
                  col_prov_name,
              ],
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

    # =========================================================================
    # FITUR TAMBAHAN BARU: Eksplorasi Pantai Dominan Berdasarkan Provinsi & Predikat
    # =========================================================================
    st.markdown("---")
    section_header("📊", "Direktori", "Sebaran Kualitas Ulasan Pantai per Provinsi")
    st.caption(
        "Ingin tahu pantai mana saja di seluruh Indonesia yang ulasannya"
        " mendominasi kategori 'Bagus' atau 'Biasa'? Pilih kategori dan"
        " provinsi di bawah ini."
    )

    col_filter_p1, col_filter_p2 = st.columns(2)
    with col_filter_p1:
      pilih_predikat_direktori = st.selectbox(
          "Pilih Predikat Ulasan:",
          options=["Bagus", "Biasa"],
          key="dir_predikat",
      )
    with col_filter_p2:
      semua_prov_dir = ["Semua Provinsi"] + sorted(df["Provinsi"].unique().tolist())
      pilih_provinsi_direktori = st.selectbox(
          "Pilih Provinsi:", options=semua_prov_dir, key="dir_provinsi"
      )

    # Filter DataFrame sesuai pilihan
    df_dir = df[df["Predikat"] == pilih_predikat_direktori.lower()].copy()
    if pilih_provinsi_direktori != "Semua Provinsi":
      df_dir = df_dir[df_dir["Provinsi"] == pilih_provinsi_direktori]

    df_dir = df_dir.sort_values(by="Rating Angka", ascending=False).reset_index(
        drop=True
    )

    st.markdown(
        f"Ditemukan **{len(df_dir)} pantai** dengan predikat **{pilih_predikat_direktori}** di **{pilih_provinsi_direktori}**:"
    )

    if len(df_dir) > 0:
      # Tampilkan sebagai dataframe interaktif dengan link Google Maps yang aktif
      tampil_dir = df_dir[
          [
              "Nama Pantai",
              "Provinsi",
              "Rating Angka",
              "Jumlah Ulasan",
              "Link Google Maps",
          ]
      ].copy()
      tampil_dir.rename(
          columns={"Link Google Maps": "Tautan Maps"}, inplace=True
      )

      st.dataframe(
          tampil_dir,
          use_container_width=True,
          height=300,
          column_config={
              "Tautan Maps": st.column_config.LinkColumn(
                  "Buka Google Maps", display_text="Buka Lokasi ↗"
              )
          },
      )
    else:
      st.warning(
          "Tidak ditemukan data pantai yang sesuai dengan kombinasi filter"
          " tersebut."
      )

  with tab_top5:
    section_header("🏆", "Popularitas", "Top 5 Pantai Berdasarkan Jumlah Ulasan Terbanyak")
    st.caption(
        "Pilih provinsi di bawah ini untuk melihat 5 destinasi pantai paling"
        " populer (banyak diulas pengunjung)."
    )

    all_prov_list = sorted(df["Provinsi"].unique().tolist())
    selected_prov_top5 = st.selectbox(
        "Pilih Provinsi:",
        options=["-- Pilih Provinsi --"] + all_prov_list,
        key="select_prov_top5",
    )

    if selected_prov_top5 != "-- Pilih Provinsi --":
      df_p_prov = df[df["Provinsi"] == selected_prov_top5].copy()
      df_top5 = (
          df_p_prov.sort_values(by="Jumlah Ulasan", ascending=False)
          .head(5)
          .reset_index(drop=True)
      )

      if len(df_top5) > 0:
        st.markdown(
            f"#### 🏆 Top 5 Destinasi Populer di Provinsi {selected_prov_top5}"
        )

        chart = (
            alt.Chart(df_top5)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("Jumlah Ulasan:Q", title="Jumlah Ulasan Pengunjung"),
                y=alt.Y(
                    "Nama Pantai:N",
                    sort="-x",
                    title="Nama Pantai",
                    axis=alt.Axis(labelLimit=300),
                ),
                color=alt.Color(
                    "Jumlah Ulasan:Q",
                    scale=alt.Scale(range=["#bfe8e0", "#0a3d3d"]),
                    legend=None,
                ),
                tooltip=["Nama Pantai", "Provinsi", "Rating Angka", "Jumlah Ulasan"],
            )
            .properties(height=320, width="container")
            .configure_view(stroke=None)
            .configure_axis(grid=True, gridColor="#d7e8e5")
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        c_sub1, c_sub2 = st.columns(2)
        for idx, row_t5 in df_top5.iterrows():
          col_target = c_sub1 if idx % 2 == 0 else c_sub2
          with col_target:
            st.markdown(
                f"""
                        <div style="background: white; border: 1px solid var(--sand-100); border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(6,46,44,0.05);">
                            <p style="margin:0 0 4px 0; font-weight:700; color:var(--teal-950); font-size:14px;">#{idx+1} {row_t5['Nama Pantai']}</p>
                            <p style="margin:0; font-size:12.5px; color:#475569;">⭐ Rating: <b>{row_t5['Rating Angka']}</b> | 💬 Ulasan: <b>{row_t5['Jumlah Ulasan']}</b> | Kualitas: <b>{row_t5['Predikat'].title()}</b></p>
                        </div>
                        """,
                unsafe_allow_html=True,
            )
      else:
        st.warning(
            f"Belum ada data ulasan untuk provinsi {selected_prov_top5}."
        )
    else:
      st.info(
          "👆 Silakan pilih salah satu provinsi di atas untuk menampilkan"
          " grafik Top 5 pantai."
      )

  with tab_data:
    section_header("📋", "Dataset", "Tabel Data Pantai")
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
    section_header("🧠", "Explainable AI", "Transparansi & Penjelasan Model Machine Learning (XAI)")
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
