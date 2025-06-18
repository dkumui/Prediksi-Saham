# File __init__.py ini mengubah direktori 'stock_logic' menjadi sebuah paket Python.
# Hal ini memungkinkan kita untuk mengimpor modul-modul di dalamnya dari skrip lain,
# contohnya: from stock_logic import DataProcessor

# Mengimpor kelas dan fungsi utama dari modul-modul lain dalam paket.
# Ini memudahkan akses sehingga kita bisa melakukan `from stock_logic import build_model`
# daripada `from stock_logic.ml_models import build_model`.
from . import config

from .ml_models import build_model
from .data_processing import DataProcessor
from .model_operations import ModelOperations
from .performance_eval import PerformanceEvaluator
from .training_workflow import TrainingWorkflow
from .app_interface import AppInterface

# Variabel __all__ mendefinisikan 'public API' dari paket ini.
# Ini adalah daftar nama (string) yang akan diimpor ketika seseorang menjalankan
# `from stock_logic import *`. Ini adalah praktik yang baik untuk mengontrol namespace.
__all__ = [
    'config',
    'build_model',
    'DataProcessor',
    'ModelOperations',
    'PerformanceEvaluator',
    'TrainingWorkflow',
    'AppInterface'
]

# Pesan ini akan dicetak sekali saat paket 'stock_logic' pertama kali diimpor
# dalam sebuah sesi Python.
print("[log] Paket 'stock_logic' diinisialisasi.")