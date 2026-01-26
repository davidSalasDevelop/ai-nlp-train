# config.py
"""
Centralized configuration for high-level project constants.
"""

# Model and tokenizer name from Hugging Face Hub
MODEL_NAME = "prajjwal1/bert-tiny"

# Path to the dataset
DATASET_PATH = 'dataset/dataset_v2.json'

# Directory to save the final, best-performing model
FINAL_MODEL_OUTPUT_DIR = "../output-models"

# MLflow configuration (used as environment variables)
MLFLOW_TRACKING_URI = "http://138.197.233.39:4200"
MLFLOW_USERNAME = "editoriapl"
MLFLOW_PASSWORD = "P1cod33ditor2026Goog13"