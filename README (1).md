# 🏖️ BeachFinder Indonesia — Dashboard Peta & Prediksi Kualitas Pantai

Aplikasi web interaktif berbasis **Streamlit** untuk mengeksplorasi 1.558 destinasi pantai
di seluruh provinsi Indonesia, sekaligus memprediksi predikat kualitas pantai baru
menggunakan model **XGBoost** yang sudah dilatih di notebook.

## Struktur Project

Pastikan semua file berikut ada di **folder yang sama** (root repo):

```
├── app.py                      # Aplikasi utama Streamlit
├── dataset_clean.csv           # Dataset pantai
├── model_xgboost_wisata.pkl    # Model XGBoost hasil training (dari notebook kamu)
├── label_encoder_target.pkl    # Encoder untuk kolom Predikat
├── le_provinsi.pkl             # Encoder untuk kolom Provinsi
└── requirements.txt            # Daftar dependensi
```

> **Catatan:** File `.pkl` tidak disertakan di sini karena tidak diupload — cukup salin
> ketiga file `.pkl` yang sudah kamu hasilkan dari notebook (`joblib.dump(...)`) ke folder
> yang sama dengan `app.py`. Jika file model belum ada, tab "Peta Interaktif" dan
> "Tabel Data" tetap berfungsi normal; hanya tab "Prediksi Kualitas Pantai" yang akan
> menampilkan pesan bahwa model belum ditemukan.

## Langkah Instalasi

1. **Buat virtual environment (opsional tapi disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

4. Buka browser di `http://localhost:8501`.

## Fitur

- **🗺️ Peta Interaktif** — Peta Leaflet (via Folium) dengan marker ter-cluster untuk performa,
  warna marker sesuai predikat (biru = Bagus, merah = Biasa), dan popup berisi nama pantai,
  provinsi, rating berbintang, dan kualitas.
- **🔮 Prediksi Kualitas Pantai** — Form input Provinsi, Latitude, Longitude (tanpa Rating —
  lihat "Catatan Teknis" di bawah) yang memanggil model XGBoost untuk menaksir predikat
  kualitas sebuah lokasi pantai baru, lengkap dengan tingkat keyakinan model dan peringatan
  untuk kelas minoritas.
- **📋 Tabel Data** — Tabel data pantai sesuai hasil filter, bisa diurutkan/dicari langsung
  di UI Streamlit, plus tombol unduh CSV hasil filter.
- **Filter Sidebar** — Provinsi (multiselect), Kualitas/Predikat (multiselect), dan rentang
  Rating (slider). Semua elemen (peta, chart, metrik, tabel) otomatis diperbarui sesuai filter.
- **Panel statistik** — chart distribusi kualitas dan daftar Top 5 pantai berdasarkan rating,
  di samping peta.

## Deploy ke Streamlit Community Cloud

1. Push seluruh folder (termasuk file `.pkl`) ke repo GitHub kamu.
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan repo, pilih `app.py`
   sebagai entry point.
3. Streamlit Cloud otomatis membaca `requirements.txt` untuk instalasi dependensi.

## Catatan Teknis — Kenapa Model Tidak Pakai Rating Sebagai Fitur

Di dataset asli, `Predikat` ternyata **100% deterministik terhadap `Rating Angka`**.
Artinya kalau Rating dimasukkan sebagai fitur prediksi, model cuma "menghafal" pemetaan
rating → predikat, bukan belajar pola lokasi — dan itu tidak berguna untuk kasus nyata
(menaksir kualitas pantai yang **belum punya rating sama sekali**).

Karena itu, model di project ini dilatih ulang **hanya dari `Latitude`, `Longitude`,
dan `Provinsi`** (tanpa Rating). Target klasifikasi (`label_encoder_target.pkl`) memakai
**2 kelas: `Bagus` dan `Biasa`**.

<!-- TODO(Nabila): isi angka akurasi & recall per kelas yang aktual untuk model 2-kelas
     ini (dari notebook, sebelum dipakai untuk laporan/paper GEMASTIK), supaya klaim di
     dokumentasi ini sesuai dengan model_xgboost_wisata.pkl yang sebenarnya di-deploy. -->

Kolom `Provinsi` di form prediksi menampilkan provinsi yang sudah dikonsolidasi (sama
seperti yang ditampilkan di peta/tabel — mis. "Maluku", bukan "Ambon" atau "Pulau Buru"
sebagai provinsi terpisah). `le_provinsi.pkl` sendiri dilatih pada label provinsi mentah
sebelum konsolidasi ini diterapkan, jadi `app.py` melakukan mapping balik
(`PROVINSI_ENCODING_PROXY`) sebelum encoding untuk provinsi hasil gabungan ("Maluku" →
diwakili "Ambon", "Maluku Utara" → diwakili "Ternate") agar tidak terjadi error
"unseen label" saat prediksi.
