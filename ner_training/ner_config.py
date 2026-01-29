import os

# =============================================================================
# 1. MODELO Y RUTAS (MODELOS LIVIANOS <100MB)
# =============================================================================
# Modelos livianos recomendados (<100MB):
# - "dccuchile/albert-base-spanish": ~48MB
# - "mrm8488/electricidad-base-discriminator": ~50MB
# - "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-ner": ~250MB (un poco más)
# - "Recognai/bert-base-spanish-wwm-cased-xnli": ~440MB (más pesado)
MODEL_NAME = "dccuchile/albert-base-spanish"  # 48MB, buen rendimiento


OUTPUT_MODEL_NAME = "get_news_extractor.pt"

# Alternativas livianas:
# MODEL_NAME = "mrm8488/electricidad-base-discriminator"  # 50MB
# MODEL_NAME = "hackathon-somos-nlp-2023/bert-spanish-cased-finetuned-ner"  # 110MB
# MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"  # 500MB (más pesado)

NER_MODEL_OUTPUT_DIR = "../output-models"
CACHE_DIR = "cache"
HF_TOKEN = None  # Token opcional para datasets privados

# =============================================================================
# 2. CONFIGURACIÓN DE DATASETS
# =============================================================================

# Lista de archivos JSON propios para entrenar
# Ejemplo: ["datos1.json", "datos2.json", "datos3.json"]
# Se nescesitan almenos 10 datos para que se divida en entrenamiento y test
CUSTOM_DATASET_FILES = [
    "ner_dataset-test.json"  # Tu archivo actual
]
# Directorio donde están tus datasets
CUSTOM_DATASET_DIR = "./"
# Porcentaje para test (ej: 0.2 = 20% test, 80% train)
TEST_SIZE = 0.2

DO_TRAINING = True
INCLUDE_CUSTOM_DATASET = True
TEST_SIZE = 0.2
SEED = 42

# Límites para datasets
MAX_DATASETS_TO_LOAD = 4  # Máximo número de datasets a combinar
MAX_SAMPLES_PER_DATASET = 3000  # Máximo de ejemplos por dataset
MAX_TEST_SAMPLES_PER_DATASET = 500  # Máximo de ejemplos de test por dataset
MAX_TOTAL_TRAIN_SAMPLES = 10000  # Máximo total de ejemplos de entrenamiento
MAX_TOTAL_TEST_SAMPLES = 2000  # Máximo total de ejemplos de prueba
FORCE_REDOWNLOAD = False  # Forzar redescarga de datasets

# =============================================================================
# 3. ETIQUETAS DEL MODELO
# =============================================================================
ENTITY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-DATE", "I-DATE"]

# =============================================================================
# 4. HIPERPARÁMETROS DE ENTRENAMIENTO
# =============================================================================
TRAIN_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128

# =============================================================================
# 5. OPTIMIZACIÓN DE VELOCIDAD
# =============================================================================
FP16 = True
NUM_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
GRADIENT_ACCUMULATION_STEPS = 1

# =============================================================================
# 6. OPTIMIZADOR Y SCHEDULER
# =============================================================================
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 100
WARMUP_RATIO = 0.0

# =============================================================================
# 7. REGULARIZACIÓN AVANZADA
# =============================================================================
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING_FACTOR = 0.0

# =============================================================================
# 8. FUNCIÓN DE PÉRDIDA
# =============================================================================
USE_FOCAL_LOSS = False
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# =============================================================================
# 9. EVALUACIÓN Y GUARDADO
# =============================================================================
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.0

# =============================================================================
# 10. ARQUITECTURA DE RED NEURONAL
# =============================================================================
USE_CUSTOM_HEAD = True
CUSTOM_HEAD_ACTIVATION = 'ReLU'
CUSTOM_HEAD_LAYERS = [(256, 0.25)]

# =============================================================================
# 11. LOGGING Y MLFLOW
# =============================================================================
LOGGING_STRATEGY = "steps"
LOGGING_STEPS = 50
REPORT_TO = "mlflow"
MLFLOW_TRACKING_URI = "http://138.197.233.39:4200"
MLFLOW_EXPERIMENT_NAME = "Parameters-Extractor-Training"
MLFLOW_USERNAME = "editoriapl"
MLFLOW_PASSWORD = "P1cod33ditor2026Goog13"


# 0 = O (ninguna entidad)
# 1 = B-PER
# 2 = I-PER
# 3 = B-LOC
# 4 = B-ORG
# 5 = I-ORG
# 6 = I-LOC
# 7 = B-MISC
# 8 = I-MISC
# 9 = B-DATE
# 10 = I-DATE