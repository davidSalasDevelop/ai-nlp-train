# ner_config.py

# =============================================================================
# 1. DEFINICIÓN DEL MODELO Y RUTAS
# =============================================================================

#MODEL_NAME = "prajjwal1/bert-tiny" #16mb
#MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased" #440mb
#MODEL_NAME = "Geotrend/distilbert-base-es-cased" #250mb
#MODEL_NAME = "prajjwal1/bert-small" #125mb
#MODEL_NAME = "dccuchile/albert-base-spanish" #48mb
MODEL_NAME = "dccuchile/albert-base-spanish"


DATASET_PATH = "ner_training/ner_dataset.json"
NER_MODEL_OUTPUT_DIR = "ner_training/models/get_news_extractor"
CACHE_DIR = "ner_training/cache"

# =============================================================================
# 2. ESTRATEGIA: ¡ENTRENAR!
# =============================================================================
# Como ALBERT no sabe NER, es OBLIGATORIO entrenarlo.
DO_TRAINING = True 
INCLUDE_CUSTOM_DATASET = True
TEST_SIZE = 0.2
SEED = 42

# =============================================================================
# 3. ETIQUETAS COMPLETAS
# =============================================================================
ENTITY_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
    "B-DATE", "I-DATE"
]

# =============================================================================
# 4. HIPERPARÁMETROS PARA ALBERT
# =============================================================================
TRAIN_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 5e-5 # Tasa un poco más alta para que aprenda la tarea NER desde cero
MAX_LENGTH = 128

# =============================================================================
# 5. OPTIMIZACIÓN AVANZADA
# =============================================================================
FP16 = True                   
GRADIENT_ACCUMULATION_STEPS = 1 
WEIGHT_DECAY = 0.01           
WARMUP_STEPS = 100            # Pasos de calentamiento
WARMUP_RATIO = 0.0            # <--- ¡ESTA ERA LA VARIABLE QUE FALTABA!

# =============================================================================
# 6. ESTRATEGIA DE EVALUACIÓN Y GUARDADO
# =============================================================================
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"

# =============================================================================
# 7. EARLY STOPPING
# =============================================================================
USE_EARLY_STOPPING = False     # Puedes ponerlo en True si quieres que se detenga solo
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0

# =============================================================================
# 8. LOGGING Y MLFLOW
# =============================================================================
LOGGING_STEPS = 50
REPORT_TO = "mlflow"
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_EXPERIMENT_NAME = "NER-Albert-Lightweight"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"

# =============================================================================
# 9. DATALOADER
# =============================================================================
import os
NUM_WORKERS = 2