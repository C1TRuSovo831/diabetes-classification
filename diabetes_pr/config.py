from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_PATH / "data" / "raw"
OUTPUTS_DIR = PROJECT_PATH / "outputs"
DATA_FILE_PATH = RAW_DATA_DIR / "diabetes_prediction_dataset.csv"

TARGET = "diabetes"
CONTINUOUS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]
FEATURES = CONTINUOUS + BINARY + CATEGORICAL
SEED = 42
K_VALUES = (3, 5, 9, 15, 31)
