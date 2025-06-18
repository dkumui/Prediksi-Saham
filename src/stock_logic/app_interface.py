import gradio as gr
import numpy as np
from .model_operations import ModelOperations

class AppInterface:
    def __init__(self, model_file_path, ui_input_cols_ordered):
        """
        Inisialisasi (constructor) untuk antarmuka aplikasi Gradio.
        Fungsi ini memuat model prediksi yang sudah dilatih dari file dan menyiapkan
        urutan fitur yang diperlukan.

        Args:
            model_file_path (str): Path ke file model .joblib yang telah disimpan.
            ui_input_cols_ordered (list): Daftar nama kolom fitur sesuai urutan yang akan ditampilkan di UI.
        """
        self.model_file_path = model_file_path
        self.ui_input_cols_ordered = ui_input_cols_ordered
        
        try:
            # Memuat artifak model dan urutan fitur yang digunakan saat training
            self.pred_model, self.trained_model_feature_order = ModelOperations.load_prediction_model(self.model_file_path)
            
            if not self.trained_model_feature_order:
                raise ValueError("Daftar fitur (feature_columns_used) tidak ditemukan dalam model yang dimuat.")
            
            # Peringatan jika urutan kolom di UI berbeda dengan yang digunakan model
            if set(self.ui_input_cols_ordered) != set(self.trained_model_feature_order):
                print(f"[Peringatan UI] Set kolom input UI berbeda dari fitur yang digunakan model!")
                print(f"  UI mengharapkan (untuk label): {self.ui_input_cols_ordered}")
                print(f"  Model dilatih dengan (untuk prediksi): {self.trained_model_feature_order}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File model tidak ada di {self.model_file_path}. Latih model dulu.")
        except Exception as e:
            raise RuntimeError(f"Error saat memuat model untuk UI: {e}")

    def _predict_price(self, *input_values):
        """
        Fungsi inti yang melakukan prediksi berdasarkan input dari pengguna di UI Gradio.
        Fungsi ini diawali dengan underscore (_) untuk menandakan bahwa ini adalah metode internal.

        Args:
            *input_values: Nilai-nilai input dari komponen UI Gradio, diterima sebagai tuple.

        Returns:
            str: String yang diformat berisi hasil prediksi atau pesan error.
        """
        if len(input_values) != len(self.ui_input_cols_ordered):
            return f"Error: Jumlah input ({len(input_values)}) tidak cocok ({len(self.ui_input_cols_ordered)} fitur diharapkan)."
        
        try:
            # Mengubah input menjadi kamus (dictionary)
            input_data_dict = {label: float(val) for label, val in zip(self.ui_input_cols_ordered, input_values)}
            
            # Menyusun array input untuk model sesuai urutan fitur saat training
            model_input_array = np.array([input_data_dict[feature_name] for feature_name in self.trained_model_feature_order]).reshape(1, -1)
            
            # Melakukan prediksi
            predicted_value = self.pred_model.predict(model_input_array)
            # Mengembalikan hasil prediksi dalam format string yang rapi
            return f"Prediksi Harga Penutupan Besok: {predicted_value[0]:.2f}"
        except ValueError:
            return "Error: Pastikan semua input adalah angka."
        except KeyError as e:
             return f"Error: Fitur input '{str(e)}' tidak ditemukan. Periksa konsistensi nama fitur."
        except Exception as e:
            return f"Error saat prediksi: {str(e)}"

    def launch(self):
        """
        Membangun komponen-komponen UI Gradio dan meluncurkan server webnya.
        """
        # Membuat komponen input numerik untuk setiap fitur yang dibutuhkan
        gradio_input_components = [
            gr.Number(label=f"{col_label} Hari Ini") for col_label in self.ui_input_cols_ordered
        ]
        
        # Membuat objek antarmuka Gradio
        ui = gr.Interface(
            fn=self._predict_price,           # Fungsi yang akan dijalankan saat tombol 'Submit' ditekan
            inputs=gradio_input_components,   # Komponen input
            outputs=gr.Textbox(label="Hasil Prediksi Harga Saham Besok"), # Komponen output
            title="Prediksi Harga Saham",
            description=(f"Masukkan data {', '.join(self.ui_input_cols_ordered)} Hari Ini "
                         "Untuk Prediksi Harga Saham Besok."),
            allow_flagging='never',           # Menonaktifkan fitur 'flag' dari Gradio
            theme=gr.themes.Soft()            # Menggunakan tema visual 'Soft'
        )
        print("[log] Meluncurkan antarmuka Gradio... Akses melalui browser Anda.")
        # Meluncurkan aplikasi web, share=True untuk membuat link publik
        ui.launch(share=True)