# ner_config.py
MODEL_NAME = "prajjwal1/bert-tiny" #16mb
# MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased" #440mb
#MODEL_NAME = "Geotrend/distilbert-base-es-cased" #250mb
#MODEL_NAME = "dccuchile/albert-base-spanish" #48mb
DATASET_PATH = "ner_training/ner_dataset.json"
NER_MODEL_OUTPUT_DIR = "ner_training/models/get_news_extractor"

ENTITY_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]

# --- HIPERPARÁMETROS AJUSTADOS ---
# Más épocas para darle más oportunidades de aprender de pocos datos.
TRAIN_EPOCHS = 3 
BATCH_SIZE = 16
# Tasa de aprendizaje ligeramente más baja para un aprendizaje más estable.
LEARNING_RATE = 1e-5 

# --- MLflow (sin cambios) ---
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"