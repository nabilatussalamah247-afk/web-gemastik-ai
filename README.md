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
  warna marker sesuai predikat (hijau = Sangat Bagus, biru = Bagus, merah = Biasa), dan popup
  berisi nama pantai, provinsi, rating berbintang, dan kualitas.
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

Di dataset asli, `Predikat` ternyata **100% deterministik terhadap `Rating Angka`**
(tidak ada rentang rating yang tumpang tindih antar kelas: Biasa = 1.0, Cukup Bagus
2.5–3.0, Bagus 3.3–4.0, Sangat Bagus 4.1–5.0). Artinya kalau Rating dimasukkan sebagai
fitur prediksi, model cuma "menghafal" pemetaan rating → predikat, bukan belajar pola
lokasi — dan itu tidak berguna untuk kasus nyata (menaksir kualitas pantai yang **belum
punya rating sama sekali**).

Karena itu, model di project ini dilatih ulang **hanya dari `Latitude`, `Longitude`,
dan `Provinsi`** (tanpa Rating). Kategori `Predikat` juga dikonsolidasi dari 4 menjadi
**3 kelas** ("Cukup Bagus" digabung ke "Biasa") baik di data yang ditampilkan maupun di
target model. Konsekuensinya:

- Akurasi keseluruhan pada data uji: **88,14%** — tapi ini didominasi kelas mayoritas.
- Recall per kelas: Sangat Bagus 97%, Bagus hanya 7%, Biasa tidak cukup data untuk
  dievaluasi dengan andal (hanya 12 baris di seluruh dataset setelah digabung).
- **Penyebabnya keterbatasan data**, bukan kesalahan model: ~90% dataset berlabel
  "Sangat Bagus", jadi model condong menebak kelas itu. Ini bukan sesuatu yang bisa
  diperbaiki lewat tuning — perlu data tambahan untuk kelas minoritas kalau ingin
  akurasi yang lebih seimbang.
- UI aplikasi sudah menampilkan peringatan ini secara eksplisit di tab prediksi, dan
  memberi catatan tambahan kalau hasil prediksi jatuh ke kelas minoritas.

Kolom `Provinsi` di form prediksi hanya menampilkan provinsi yang dikenali oleh
`le_provinsi.pkl` (hasil `LabelEncoder.classes_`), sehingga tidak akan terjadi error
"unseen label" saat prediksi.
