from .data_processing import DataProcessor
from .model_operations import ModelOperations
from .performance_eval import PerformanceEvaluator

class TrainingWorkflow:
    def __init__(self, app_settings):
        """
        Inisialisasi (constructor) untuk kelas TrainingWorkflow.
        Menyiapkan semua komponen yang diperlukan untuk alur kerja, yaitu:
        DataProcessor, ModelOperations, dan PerformanceEvaluator, berdasarkan
        pengaturan (settings) yang diberikan.

        Args:
            app_settings (module): Modul 'config' yang berisi semua pengaturan aplikasi.
        """
        self.settings = app_settings
        # Inisialisasi objek untuk setiap langkah dalam workflow
        self.data_proc = DataProcessor(
            csv_path=self.settings.CSV_FILE_PATH,
            date_col_name=self.settings.DATE_COLUMN,
            target_col_label=self.settings.TARGET_COLUMN_NAME,
            feature_col_labels=self.settings.FEATURE_COLUMN_NAMES,
            num_splits=self.settings.N_SPLITS
        )
        self.model_ops = ModelOperations(
            model_architecture="xgboost",
            model_hyperparams=self.settings.XGBOOST_PARAMS
        )
        self.perf_eval = PerformanceEvaluator()
        print("[log] TrainingWorkflow diinisialisasi.")

    def execute(self):
        """
        Menjalankan keseluruhan alur kerja (workflow) training secara berurutan.
        Mulai dari memuat data, memproses, melatih model, mengevaluasi,
        hingga menyimpan model yang sudah jadi.
        """
        print("\n[Workflow] Memulai alur kerja training...")
        
        # Langkah 1: Memuat dan memproses data
        self.data_proc.load_dataset()
        self.data_proc.prepare_for_training()
        self.data_proc.split_time_series_data()
        
        # Pemeriksaan untuk memastikan data training tidak kosong setelah diproses
        if self.data_proc.X_train is None or self.data_proc.X_train.size == 0:
            print("[Error Workflow] Data training (X_train) kosong setelah split. Tidak bisa melanjutkan.")
            return

        # Langkah 2: Melatih model
        self.model_ops.perform_training(self.data_proc.X_train, self.data_proc.y_train)
        
        # Langkah 3: Evaluasi model pada data tes (jika ada)
        if self.data_proc.X_test is not None and len(self.data_proc.X_test) > 0:
            predictions_on_test = self.model_ops.generate_predictions(self.data_proc.X_test)
            # Hitung skor MSE
            self.perf_eval.get_mse_score(self.data_proc.y_test, predictions_on_test)
            # Buat dan simpan plot hasil
            self.perf_eval.create_results_plot(
                self.data_proc.y_test, 
                predictions_on_test, 
                self.settings.PLOT_SAVE_PATH
            )
        else:
            print("[Peringatan Workflow] Tidak ada data tes untuk evaluasi atau pembuatan plot.")
            
        # Langkah 4: Menyimpan model yang telah dilatih
        self.model_ops.save_trained_model(
            output_path=self.settings.MODEL_SAVE_PATH,
            training_feature_cols=self.settings.FEATURE_COLUMN_NAMES, 
            training_target_col=self.settings.TARGET_COLUMN_NAME
        )
        print("[Workflow] Alur kerja training selesai.\n")