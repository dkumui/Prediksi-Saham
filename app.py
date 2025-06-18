import os
import traceback
# Mengimpor modul-modul yang diperlukan dari paket stock_logic
from src.stock_logic import config, AppInterface

def main():
    """
    Fungsi utama untuk menjalankan aplikasi Gradio.
    Fungsi ini akan memeriksa keberadaan file model, membuat instance antarmuka,
    dan meluncurkannya.
    """
    # Mencetak header untuk tampilan di terminal
    print("="*50)
    print(" Aplikasi Prediksi Saham Gradio (Final) ".center(50, "="))
    print("="*50)

    # === Pemeriksaan Keberadaan Model ===
    # Sebelum meluncurkan UI, periksa dulu apakah file model yang sudah dilatih ada.
    # Ini untuk mencegah error jika aplikasi dijalankan sebelum model dilatih.
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[Error Kritis] File model '{config.MODEL_SAVE_PATH}' tidak ditemukan.") #
        print("  Harap latih model terlebih dahulu dengan menjalankan: python train.py") #
        print("="*50)
        return # Menghentikan eksekusi fungsi jika model tidak ada
    
    # === Blok Eksekusi Utama ===
    # Menggunakan try...except...finally untuk penanganan error yang baik.
    try:
        # Membuat instance dari kelas AppInterface
        ui_instance = AppInterface(
            model_file_path=config.MODEL_SAVE_PATH, # Path ke model yang akan dimuat
            ui_input_cols_ordered=config.FEATURE_COLUMN_NAMES # Daftar fitur sebagai input di UI
        )
        # Meluncurkan antarmuka pengguna (UI) Gradio
        ui_instance.launch() #
    except Exception as e:
        # Menangkap dan menampilkan error tak terduga yang mungkin terjadi saat inisialisasi atau peluncuran UI
        print(f"\n[Error Tak Terduga] Gagal menjalankan aplikasi Gradio:") #
        print(f"  Detail: {e}") #
        traceback.print_exc() # Mencetak traceback untuk mempermudah debugging
    finally:
        # Blok ini akan selalu dieksekusi, baik terjadi error maupun tidak.
        # Berguna untuk memberi pesan penutup.
        print("="*50)
        print(" Aplikasi Gradio Ditutup ".center(50, "="))  #
        print("="*50)

# === Titik Masuk Skrip ===
# Konstruksi `if __name__ == "__main__"` memastikan bahwa fungsi `main()` hanya akan
# dijalankan ketika file ini dieksekusi secara langsung (misal: `python app.py`),
# dan tidak akan berjalan jika file ini diimpor sebagai modul oleh skrip lain.
if __name__ == "__main__":
    main() #