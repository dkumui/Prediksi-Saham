# Aplikasi Prediksi Harga Saham

Aplikasi machine learning untuk memprediksi harga penutupan (Close Price) saham pada hari berikutnya (H+1) berdasarkan data historis harga saham pada hari ini (H). Model yang digunakan adalah **XGBoost Regressor** dengan antarmuka web interaktif yang dibangun menggunakan **Gradio**.

## Fitur Utama

- **Pelatihan Model**: Melatih model XGBoost menggunakan data historis untuk menemukan pola
- **Evaluasi Kinerja**: Mengevaluasi model menggunakan metrik Mean Squared Error (MSE) pada data uji
- **Visualisasi Hasil**: Menghasilkan plot perbandingan antara harga aktual dan harga prediksi untuk analisis visual
- **Penyimpanan Model**: Menyimpan model yang telah dilatih beserta metadata (fitur yang digunakan) untuk digunakan kembali
- **Antarmuka Web**: Menyediakan UI sederhana berbasis Gradio untuk melakukan prediksi secara real-time

## Struktur Proyek

```
.
├── app.py                      # Skrip utama untuk menjalankan aplikasi Gradio
├── train.py                    # Skrip untuk menjalankan alur kerja pelatihan model
├── requirements.bat            # Daftar pustaka Python yang dibutuhkan
├── data/
│   └── Dataset Saham.csv       # Dataset saham untuk pelatihan model
├── outputs/
│   ├── models/                 # Direktori untuk menyimpan file model (.joblib)
│   └── plots/                  # Direktori untuk menyimpan plot hasil evaluasi
└── src/
    └── stock_logic/
        ├── __init__.py
        ├── config.py           # Konfigurasi path, fitur, dan parameter model
        ├── data_processing.py  # Kelas untuk memuat, memproses, dan membagi data
        ├── ml_models.py        # Fungsi untuk membangun model XGBoost
        ├── model_operations.py # Kelas untuk operasi model (latih, prediksi, simpan, muat)
        ├── performance_eval.py # Kelas untuk evaluasi performa model (MSE, plot)
        ├── training_workflow.py# Mengorkestrasi seluruh proses pelatihan
        └── app_interface.py    # Kelas untuk membangun dan menjalankan antarmuka Gradio
```

## Instalasi

### 1. Clone Repositori

```bash
git clone https://github.com/dkumui/Prediksi-Saham.git
cd Prediksi-Saham
```

### 2. Buat dan Aktifkan Virtual Environment (Direkomendasikan)

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.bat
```

**Requirements:**
```bat
pandas
scikit-learn
xgboost
matplotlib
gradio
joblib
```

## Penggunaan

### 1. Persiapan Data

- Pastikan Anda memiliki file dataset bernama `Dataset Saham.csv`
- Letakkan file tersebut di dalam direktori `data/`
- Kolom yang dibutuhkan: `Date`, `Open Price`, `High Price`, `Low Price`, `Close Price`, dan `Volume`

### 2. Melatih Model

Jalankan skrip pelatihan model:

```bash
python train.py
```

**Proses yang dilakukan:**
- Memuat data dari `data/Dataset Saham.csv`
- Memproses data dan membaginya menjadi set data latih dan uji
- Melatih model XGBoost
- Mengevaluasi model dan menyimpan plot perbandingan di `outputs/plots/`
- Menyimpan model yang telah dilatih ke `outputs/models/model_prediksi_harga_saham.joblib`

### 3. Menjalankan Aplikasi Prediksi

Setelah model berhasil dilatih, jalankan aplikasi Gradio:

```bash
python app.py
```

**Cara menggunakan aplikasi:**
1. Buka URL yang ditampilkan di terminal
2. Masukkan nilai-nilai fitur yang diminta:
   - Open Price (Harga Pembukaan)
   - High Price (Harga Tertinggi)
   - Low Price (Harga Terendah)
   - Close Price (Harga Penutupan)
   - Volume (Volume Perdagangan)
3. Klik **"Submit"** untuk mendapatkan prediksi harga penutupan hari esok

## Konfigurasi

Semua pengaturan utama dapat diubah dalam file `src/stock_logic/config.py`:

### Path File
- `CSV_FILE_PATH`: Lokasi file dataset
- `MODEL_SAVE_PATH`: Lokasi penyimpanan model
- `PLOT_SAVE_PATH`: Lokasi penyimpanan plot

### Kolom Data
- `TARGET_COLUMN_NAME`: Kolom target untuk prediksi
- `DATE_COLUMN`: Kolom tanggal
- `FEATURE_COLUMN_NAMES`: Daftar kolom fitur

### Parameter Model
- `XGBOOST_PARAMS`: Hyperparameter untuk XGBoost

## Model Machine Learning

**XGBoost Regressor** dipilih karena:
- Performa tinggi untuk data tabular
- Mampu menangani non-linearitas dalam data
- Robust terhadap outliers
- Feature importance yang dapat diinterpretasi
