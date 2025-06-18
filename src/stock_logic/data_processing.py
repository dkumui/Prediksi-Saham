import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit 

class DataProcessor:
    def __init__(self, csv_path, date_col_name, target_col_label, feature_col_labels, num_splits):
        """
        Inisialisasi (constructor) untuk kelas DataProcessor.
        Menyimpan semua konfigurasi yang dibutuhkan untuk pemrosesan data.

        Args:
            csv_path (str): Path ke file dataset CSV.
            date_col_name (str): Nama kolom tanggal dalam dataset.
            target_col_label (str): Nama kolom yang menjadi target prediksi.
            feature_col_labels (list): Daftar nama kolom yang menjadi fitur.
            num_splits (int): Jumlah pembagian untuk TimeSeriesSplit.
        """
        self.csv_path = csv_path
        self.date_col_name = date_col_name
        self.target_col_label = target_col_label
        self.feature_col_labels = feature_col_labels
        self.num_splits = num_splits
        
        # Inisialisasi variabel untuk menyimpan hasil pemrosesan
        self.df_raw = None
        self.X_prepared, self.y_prepared = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        print(f"[log] DataProcessor diinisialisasi untuk target '{target_col_label}' & fitur {feature_col_labels}")

    def load_dataset(self):
        """
        Memuat dataset dari file CSV yang path-nya telah ditentukan.
        Fungsi ini juga mengonversi kolom tanggal ke format datetime dan menjadikannya sebagai index DataFrame.
        Menangani error jika file tidak ditemukan.
        """
        print(f"[log] Memuat dataset dari {self.csv_path}...")
        try:
            self.df_raw = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"[Error Kritis] File CSV tidak ditemukan di: {self.csv_path}")
            print("  Pastikan path dan nama file di config.py sudah benar, dan file CSV ada di folder 'data/'.")
            raise
        
        # Mencoba mengonversi kolom tanggal ke format datetime
        try:
            self.df_raw[self.date_col_name] = pd.to_datetime(self.df_raw[self.date_col_name])
        except Exception:
            try:
                # Mencoba inferensi format tanggal otomatis jika format standar gagal
                self.df_raw[self.date_col_name] = pd.to_datetime(self.df_raw[self.date_col_name], infer_datetime_format=True)
            except Exception as e:
                print(f"[Error] Gagal mem-parsing kolom tanggal '{self.date_col_name}': {e}")
                raise
        # Menetapkan kolom tanggal sebagai index dari DataFrame
        self.df_raw.set_index(self.date_col_name, inplace=True)
        print("[log] Dataset berhasil dimuat.")

    def prepare_for_training(self):
        """
        Mempersiapkan data untuk training.
        Fungsi ini membuat fitur (X) dan target (y). Target (y) adalah harga penutupan hari berikutnya,
        yang didapatkan dengan menggeser (shift) kolom target ke atas satu baris.
        Data terakhir akan dibuang karena tidak memiliki target masa depan.
        """
        if self.df_raw is None:
            raise ValueError("Dataset mentah belum dimuat. Panggil load_dataset() dulu.")
        
        # Validasi bahwa semua kolom yang dibutuhkan ada di DataFrame
        if self.target_col_label not in self.df_raw.columns:
            raise ValueError(f"Kolom target '{self.target_col_label}' tidak ditemukan dalam DataFrame.")
        for col in self.feature_col_labels:
            if col not in self.df_raw.columns:
                raise ValueError(f"Kolom fitur '{col}' tidak ditemukan. Kolom tersedia: {self.df_raw.columns.tolist()}")

        print("[log] Mempersiapkan fitur dan target untuk training...")
        # X adalah data fitur pada hari H
        X_source = self.df_raw[self.feature_col_labels].copy()
        # y adalah harga penutupan pada hari H+1 (didapat dengan shift -1)
        y_source = self.df_raw[self.target_col_label].shift(-1).copy()

        # Menghapus baris terakhir dari X dan y karena y baris terakhir akan bernilai NaN (tidak punya masa depan)
        X_source = X_source.iloc[:-1]
        y_source = y_source.iloc[:-1]
        
        # Memastikan tidak ada nilai null di target, lalu menyamakan X dan y
        valid_indices = ~y_source.isnull()
        self.X_prepared = X_source[valid_indices].values
        self.y_prepared = y_source[valid_indices].values

        print(f"[log] Persiapan data training selesai. Bentuk X: {self.X_prepared.shape}, Bentuk y: {self.y_prepared.shape}")
        if self.X_prepared.size == 0 or self.y_prepared.size == 0:
            raise ValueError("X_prepared atau y_prepared kosong setelah persiapan. Periksa data Anda.")

    def split_time_series_data(self):
        """
        Membagi data menjadi set training dan testing menggunakan TimeSeriesSplit.
        Metode ini memastikan bahwa data testing selalu berada di masa depan relatif terhadap data training,
        yang sangat penting untuk data time series. Hanya hasil split terakhir yang digunakan.
        """
        if self.X_prepared is None or self.y_prepared is None:
            raise ValueError("Data belum dipersiapkan. Panggil prepare_for_training() dulu.")
        
        print(f"[log] Membagi data time series (n_splits={self.num_splits})...")
        if len(self.X_prepared) < self.num_splits + 1:
            raise ValueError(f"Data tidak cukup ({len(self.X_prepared)} sampel) untuk {self.num_splits} pembagian. Coba kurangi N_SPLITS di config.py.")

        ts_cv_splitter = TimeSeriesSplit(n_splits=self.num_splits)
        
        # Looping melalui setiap split, tapi hanya menyimpan yang terakhir
        # Split terakhir memberikan set data training terbesar dan set test terbaru
        for train_idx, test_idx in ts_cv_splitter.split(self.X_prepared):
            self.X_train, self.X_test = self.X_prepared[train_idx], self.X_prepared[test_idx]
            self.y_train, self.y_test = self.y_prepared[train_idx], self.y_prepared[test_idx]
        
        if self.X_train is None or self.X_test is None : 
             raise RuntimeError("Pembagian data train/test gagal. Periksa jumlah data dan n_splits.")
        print(f"[log] Pembagian data selesai. Bentuk Train: {self.X_train.shape}, Bentuk Test: {self.X_test.shape}")