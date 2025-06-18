# Import library 'os' untuk berinteraksi dengan sistem operasi, 
# terutama untuk menangani path file dan direktori.
import os

# Mendefinisikan direktori root proyek. 
# __file__ adalah path ke file config.py ini.
# os.path.abspath mendapatkan path absolutnya.
# os.path.dirname sekali untuk mendapatkan direktori 'stock_logic'.
# os.path.dirname dua kali untuk naik ke direktori root proyek.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === PENGATURAN PATH DIREKTORI ===
# Mendefinisikan path absolut untuk direktori 'data' dan 'outputs'.
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PARENT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Mendefinisikan path untuk sub-direktori di dalam 'outputs'.
MODEL_OUTPUT_SUBDIR = os.path.join(OUTPUT_PARENT_DIR, 'models') # Untuk menyimpan file model
PLOT_OUTPUT_SUBDIR = os.path.join(OUTPUT_PARENT_DIR, 'plots')   # Untuk menyimpan file gambar/plot

# === PEMBUATAN DIREKTORI OTOMATIS ===
# Blok kode ini memeriksa apakah direktori yang dibutuhkan sudah ada.
# Jika belum, maka direktori tersebut akan dibuat secara otomatis.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"[Config] Direktori data dibuat: {DATA_DIR}. Harap letakkan file CSV Anda di sini.")

if not os.path.exists(OUTPUT_PARENT_DIR):
    os.makedirs(OUTPUT_PARENT_DIR)

if not os.path.exists(MODEL_OUTPUT_SUBDIR):
    os.makedirs(MODEL_OUTPUT_SUBDIR)

if not os.path.exists(PLOT_OUTPUT_SUBDIR):
    os.makedirs(PLOT_OUTPUT_SUBDIR)

# === PENGATURAN PATH FILE ===
# Path lengkap ke file dataset CSV.
CSV_FILE_PATH = os.path.join(DATA_DIR, 'Dataset Saham.csv') 

# Path lengkap untuk menyimpan model yang telah dilatih dan plot hasil evaluasi.
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_SUBDIR, "model_prediksi_harga_saham.joblib")
PLOT_SAVE_PATH = os.path.join(PLOT_OUTPUT_SUBDIR, "plot_hasil_prediksi.png")

# === KONFIGURASI DATASET & FITUR ===
# Mendefinisikan nama-nama kolom penting dalam dataset.
TARGET_COLUMN_NAME = 'Close Price' # Kolom yang akan diprediksi (target).
DATE_COLUMN = 'Date'               # Kolom tanggal untuk di-set sebagai index.
# Daftar nama kolom yang akan digunakan sebagai fitur untuk training model.
FEATURE_COLUMN_NAMES = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume']


# === KONFIGURASI MODEL MACHINE LEARNING ===
# Jumlah pembagian (split) untuk validasi silang pada data time series.
N_SPLITS = 5

# Hyperparameter untuk model XGBoost Regressor.
# Parameter ini mengontrol bagaimana model akan dilatih.
XGBOOST_PARAMS = {
    'n_estimators': 1000,      # Jumlah pohon (model) yang akan dibangun.
    'learning_rate': 0.05,     # Seberapa cepat model belajar dari kesalahan.
    'max_depth': 6,            # Kedalaman maksimum setiap pohon.
    'subsample': 0.8,          # Fraksi sampel data yang digunakan untuk setiap pohon.
    'colsample_bytree': 0.8,   # Fraksi fitur (kolom) yang digunakan untuk setiap pohon.
    'verbosity': 0,            # Level output log (0 = silent).
    'random_state': 42         # Seed untuk reproduktifitas hasil.
}