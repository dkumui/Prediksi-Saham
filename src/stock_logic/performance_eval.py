import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error 

class PerformanceEvaluator:
    def get_mse_score(self, y_actual, y_predicted):
        """
        Menghitung dan mencetak Mean Squared Error (MSE) antara nilai aktual dan prediksi.
        MSE adalah metrik umum untuk mengukur rata-rata kuadrat dari selisih (error).

        Args:
            y_actual (np.array): Nilai target yang sebenarnya.
            y_predicted (np.array): Nilai yang diprediksi oleh model.

        Returns:
            float: Skor MSE yang dihitung.
        """
        if y_actual is None or y_predicted is None:
            raise ValueError("y_actual dan y_predicted untuk evaluasi tidak boleh None.")
        print("[log] Menghitung Mean Squared Error (MSE)...")
        score = mean_squared_error(y_actual, y_predicted)
        print(f"[log] MSE: {score:.4f}")
        return score

    def create_results_plot(self, y_actual, y_predicted, plot_file_path):
        """
        Membuat dan menyimpan plot perbandingan antara nilai aktual dan prediksi.
        Plot ini sangat berguna untuk memvisualisasikan seberapa baik model mengikuti data aslinya.

        Args:
            y_actual (np.array): Nilai target yang sebenarnya.
            y_predicted (np.array): Nilai yang diprediksi oleh model.
            plot_file_path (str): Path lengkap untuk menyimpan file gambar plot.
        """
        if y_actual is None or y_predicted is None:
            raise ValueError("y_actual dan y_predicted untuk plot tidak boleh None.")
        
        print(f"[log] Membuat plot hasil prediksi dan menyimpan ke {plot_file_path}...")
        plt.figure(figsize=(14, 7))
        # Menggunakan indeks numerik untuk sumbu X
        time_indices = np.arange(len(y_actual))
        # Membuat plot garis untuk data aktual dan prediksi
        plt.plot(time_indices, y_actual, label='Harga Aktual (H+1)', color='navy', marker='o', markersize=5, linestyle='-')
        plt.plot(time_indices, y_predicted, label='Harga Prediksi (H+1)', color='crimson', marker='x', markersize=5, linestyle='--')
        # Menambahkan judul, label, legenda, dan grid untuk keterbacaan
        plt.title("Perbandingan Harga Saham Aktual vs. Prediksi untuk Hari Berikutnya (H+1)", fontsize=16)
        plt.xlabel("Indeks Waktu pada Set Pengujian", fontsize=12)
        plt.ylabel("Harga Saham", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        # Menyimpan plot ke file dan menutup figure untuk membebaskan memori
        plt.savefig(plot_file_path)
        plt.close()
        print("[log] Plot hasil prediksi berhasil disimpan.")