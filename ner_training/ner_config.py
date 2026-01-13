# ner_config.py
MODEL_NAME = "prajjwal1/bert-tiny"
DATASET_PATH = "ner_training/ner_dataset.json"
NER_MODEL_OUTPUT_DIR = "ner_training/models/get_news_extractor"

ENTITY_LABELS = ["O", "B-SUBJECT", "I-SUBJECT", "B-DATE_RANGE", "I-DATE_RANGE"]

# --- HIPERPARÁMETROS AJUSTADOS ---
# Más épocas para darle más oportunidades de aprender de pocos datos.
TRAIN_EPOCHS = 60 
BATCH_SIZE = 4
# Tasa de aprendizaje ligeramente más baja para un aprendizaje más estable.
LEARNING_RATE = 3e-5 

# --- MLflow (sin cambios) ---
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"