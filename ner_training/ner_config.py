# ner_config.py
"""
Configuración centralizada para el pipeline de entrenamiento del modelo NER.
"""
# Modelo base de Hugging Face
MODEL_NAME = "prajjwal1/bert-tiny"

# Archivos y directorios
DATASET_PATH = "ner_training/ner_dataset.json"
NER_MODEL_OUTPUT_DIR = "models/ner_model" 

# Etiquetas de entidad (formato IOB2)
ENTITY_LABELS = [
    "O",
    "B-SUBJECT",
    "I-SUBJECT",
    "B-DATE_RANGE",
    "I-DATE_RANGE"
]

# Hiperparámetros de entrenamiento
TRAIN_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# --- AÑADIDO: Configuración de MLflow ---
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"