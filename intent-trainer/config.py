# config.py
"""
Centralized configuration for high-level project constants.
"""

# Model and tokenizer name from Hugging Face Hub
MODEL_NAME = "prajjwal1/bert-tiny"

# Path to the dataset
DATASET_PATH = 'small-intent-detector-cpu/dataset_v2.json'

# Directory to save the final, best-performing model
FINAL_MODEL_OUTPUT_DIR = "small-intent-detector-cpu/output"

# MLflow configuration (used as environment variables)
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"