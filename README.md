# 🏖️ BeachFinder Indonesia — Dashboard Peta & Prediksi Kualitas Pantai

Aplikasi web interaktif berbasis **Streamlit** untuk mengeksplorasi 1.558 destinasi pantai
di seluruh provinsi Indonesia, sekaligus memprediksi predikat kualitas pantai baru
menggunakan model **XGBoost** yang sudah dilatih di notebook.

## Struktur Project

Pastikan semua file berikut ada di **folder yang sama** (root repo):

```
├── app.py                      # Aplikasi utama Streamlit
├── dataset_pantai_clean.csv    # Dataset pantai
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
  warna marker sesuai predikat (hijau = Sangat Bagus, biru = Bagus, oranye = Cukup Bagus,
  merah = Biasa), dan popup berisi nama pantai, provinsi, rating berbintang, dan kualitas.
- **🔮 Prediksi Kualitas Pantai** — Form input Provinsi, Rating, Latitude, Longitude yang
  memanggil model XGBoost untuk memprediksi predikat kualitas pantai baru, lengkap dengan
  tingkat keyakinan model.
- **📋 Tabel Data** — Tabel data pantai sesuai hasil filter, bisa diurutkan/dicari langsung
  di UI Streamlit.
- **Filter Sidebar** — Provinsi (multiselect), Kualitas/Predikat (multiselect), dan rentang
  Rating (slider). Semua elemen (peta, metrik, tabel) otomatis diperbarui sesuai filter.

## Deploy ke Streamlit Community Cloud

1. Push seluruh folder (termasuk file `.pkl`) ke repo GitHub kamu.
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan repo, pilih `app.py`
   sebagai entry point.
3. Streamlit Cloud otomatis membaca `requirements.txt` untuk instalasi dependensi.

## Catatan Teknis

- Fitur model (`Rating Angka`, `Latitude`, `Longitude`, `Provinsi`) dan urutan kolomnya
  disesuaikan persis dengan proses training di notebook (`Klasifikasi_Pantai_Indonesia_XGBOOST`)
  agar hasil prediksi konsisten.
- Kolom `Provinsi` di form prediksi hanya menampilkan provinsi yang dikenali oleh
  `le_provinsi.pkl` (hasil `LabelEncoder.classes_`), sehingga tidak akan terjadi error
  "unseen label" saat prediksi.
