from xgboost import XGBRegressor 

def build_model(model_type="xgboost", params=None):
    """
    Membangun dan mengembalikan instance model machine learning.
    Fungsi ini bertindak sebagai 'factory' untuk model. Saat ini hanya mendukung 'xgboost'.

    Args:
        model_type (str, optional): Tipe model yang ingin dibuat. Defaults to "xgboost".
        params (dict, optional): Kamus (dictionary) berisi hyperparameter untuk model. 
                                 Jika None, parameter default akan digunakan.

    Raises:
        ValueError: Jika tipe model yang diminta tidak didukung.

    Returns:
        object: Instance dari model machine learning yang siap untuk dilatih.
    """
    if model_type.lower() == "xgboost":
        # Jika tidak ada parameter yang diberikan, gunakan parameter default sederhana
        if params is None: 
            params = {
                'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
                'random_state': 42, 'verbosity': 0
            }
        print(f"[log] Membuat model XGBoost dengan params: {params}")
        # Membuat dan mengembalikan instance XGBRegressor dengan parameter yang diberikan
        return XGBRegressor(**params)
    else:
        # Jika tipe model lain diminta, lemparkan error
        raise ValueError(f"Tipe model '{model_type}' tidak didukung.")