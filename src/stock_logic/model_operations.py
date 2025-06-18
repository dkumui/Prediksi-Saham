import joblib 
from .ml_models import build_model 

class ModelOperations:
    def __init__(self, model_architecture="xgboost", model_hyperparams=None):
        """
        Inisialisasi (constructor) untuk kelas ModelOperations.
        Menyimpan arsitektur model dan hyperparameter yang akan digunakan.

        Args:
            model_architecture (str, optional): Nama arsitektur model. Defaults to "xgboost".
            model_hyperparams (dict, optional): Hyperparameter untuk model. Defaults to None.
        """
        self.model_architecture = model_architecture
        self.model_hyperparams = model_hyperparams
        self.trained_model = None # Variabel untuk menyimpan model setelah dilatih
        print(f"[log] ModelOperations diinisialisasi untuk arsitektur '{model_architecture}'.")

    def perform_training(self, X_train_data, y_train_data):
        """
        Melakukan proses pelatihan (training) model.
        Fungsi ini memanggil `build_model` untuk membuat instance model, kemudian melatihnya
        dengan data training yang disediakan.

        Args:
            X_train_data (np.array): Data fitur untuk training.
            y_train_data (np.array): Data target untuk training.
        """
        if X_train_data is None or y_train_data is None:
            raise ValueError("Data training (X_train_data, y_train_data) tidak boleh None.")
        print(f"[log] Melatih model {self.model_architecture}...")
        # Membangun model menggunakan factory function
        self.trained_model = build_model(model_type=self.model_architecture, params=self.model_hyperparams)
        # Melatih model dengan data (X_train, y_train)
        self.trained_model.fit(X_train_data, y_train_data)
        print("[log] Pelatihan model selesai.")

    def generate_predictions(self, X_input_data):
        """
        Menghasilkan prediksi menggunakan model yang sudah dilatih.

        Args:
            X_input_data (np.array): Data fitur yang akan diprediksi.

        Returns:
            np.array: Hasil prediksi dari model.
        """
        if self.trained_model is None:
            raise ValueError("Model belum dilatih.")
        if X_input_data is None:
            raise ValueError("Data input (X_input_data) untuk prediksi tidak boleh None.")
        
        print(f"[log] Melakukan prediksi pada {X_input_data.shape[0]} sampel...")
        # Menggunakan model yang sudah dilatih untuk memprediksi data input
        return self.trained_model.predict(X_input_data)

    def save_trained_model(self, output_path, training_feature_cols, training_target_col):
        """
        Menyimpan model yang telah dilatih beserta metadatanya ke sebuah file.
        Metadata penting seperti daftar fitur yang digunakan juga disimpan agar konsisten saat prediksi.

        Args:
            output_path (str): Path file untuk menyimpan model (misal, 'model.joblib').
            training_feature_cols (list): Daftar nama kolom fitur yang digunakan saat training.
            training_target_col (str): Nama kolom target yang digunakan saat training.
        """
        if self.trained_model is None:
            raise ValueError("Tidak ada model untuk disimpan (model belum dilatih).")
        
        # Membuat payload (muatan) yang berisi objek model dan metadata penting
        persistence_payload = {
            'model_artifact': self.trained_model,
            'architecture': self.model_architecture,
            'hyperparameters': self.model_hyperparams,
            'feature_columns_used': training_feature_cols, 
            'target_column_used': training_target_col
        }
        # Menggunakan joblib untuk menyimpan payload ke file
        joblib.dump(persistence_payload, output_path)
        print(f"[log] Model dan metadata disimpan ke {output_path}")

    @staticmethod
    def load_prediction_model(model_file_path):
        """
        Memuat model dan metadata dari file yang telah disimpan.
        Ini adalah 'staticmethod' karena bisa dipanggil tanpa harus membuat instance dari kelas ModelOperations.

        Args:
            model_file_path (str): Path ke file model yang akan dimuat.

        Returns:
            tuple: Berisi (objek model, daftar fitur yang digunakan saat training).
        """
        print(f"[log] Memuat model dan metadata dari {model_file_path}...")
        # Memuat payload dari file joblib
        loaded_payload = joblib.load(model_file_path)
        
        # Mengekstrak objek model dan daftar fitur dari payload
        model_artifact = loaded_payload.get('model_artifact')
        feature_cols_used = loaded_payload.get('feature_columns_used')
        
        if model_artifact is None or feature_cols_used is None:
            raise ValueError("File model korup atau kehilangan data esensial (model_artifact, feature_columns_used).")
            
        print("[log] Model dan metadata berhasil dimuat.")
        return model_artifact, feature_cols_used