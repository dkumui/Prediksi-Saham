import traceback
# Mengimpor modul-modul yang diperlukan dari paket stock_logic
from src.stock_logic import config, TrainingWorkflow

def main():
    """
    Fungsi utama untuk menjalankan alur kerja (workflow) pelatihan model.
    Fungsi ini akan membuat instance dari TrainingWorkflow dan mengeksekusinya.
    """
    # Mencetak header untuk tampilan di terminal
    print("="*50)
    print(" Skrip Pelatihan Model Saham (Final) ".center(50, "="))
    print("="*50)
    
    # === Blok Eksekusi Pelatihan ===
    # Menggunakan try...except...finally untuk menangani berbagai jenis error
    # yang mungkin terjadi selama proses pelatihan.
    try:
        # 1. Membuat instance dari kelas TrainingWorkflow
        #    'config' dilewatkan sebagai pengaturan aplikasi.
        workflow = TrainingWorkflow(app_settings=config) #
        # 2. Menjalankan seluruh proses workflow (load data, process, train, evaluate, save)
        workflow.execute() #
        print("\n[Sukses] Proses pelatihan model telah selesai.") #
    except FileNotFoundError as e:
        # Menangani secara spesifik jika file (misal: dataset .csv) tidak ditemukan.
        print(f"\n[Error Kritis] File tidak ditemukan: {e}") #
        print(f"  Pastikan file CSV ada di direktori '{config.DATA_DIR}' dan path di 'config.py' sudah benar.") #
    except ValueError as e:
        # Menangani error yang berkaitan dengan data atau konfigurasi yang tidak valid.
        # Contoh: kolom tidak ditemukan, data tidak cukup untuk di-split.
        print(f"\n[Error Data/Konfigurasi] {e}") #
    except Exception as e:
        # Menangkap semua jenis error lain yang tidak terduga.
        print(f"\n[Error Tak Terduga] Terjadi kesalahan saat pelatihan:") #
        print(f"  Detail: {e}") #
        traceback.print_exc() # Mencetak traceback untuk mempermudah debugging
    finally:
        # Blok ini akan selalu dieksekusi setelah blok try...except selesai.
        print("="*50)
        print(" Proses Pelatihan Selesai ".center(50, "=")) #
        print("="*50)

# === Titik Masuk Skrip ===
# Memastikan fungsi `main()` hanya dijalankan saat file dieksekusi secara langsung
# (`python train.py`).
if __name__ == "__main__":
    main() #