import os

# =============================================================================
# 1. MODELO Y RUTAS
# =============================================================================
# El 'cerebro' base de nuestro modelo.
# - "dccuchile/albert-base-spanish": Ligero (~48MB), ideal para CPU/velocidad.
# - "dccuchile/bert-base-spanish-wwm-cased": Robusto (~440MB), más preciso.
# - "Geotrend/distilbert-base-es-cased": Equilibrio (~250MB).
MODEL_NAME = "dccuchile/albert-base-spanish"

DATASET_PATH = "ner_training/ner_dataset.json"
NER_MODEL_OUTPUT_DIR = "output-models/get_news_extractor"
CACHE_DIR = "ner_training/cache"

# =============================================================================
# 2. ESTRATEGIA DE EJECUCIÓN
# =============================================================================
DO_TRAINING = True
INCLUDE_CUSTOM_DATASET = True
TEST_SIZE = 0.2
SEED = 42

# =============================================================================
# 3. ETIQUETAS DEL MODELO
# =============================================================================
# Define todas las categorías que el modelo aprenderá a detectar.
# El número total de etiquetas aquí determina automáticamente el tamaño de la capa de salida.
ENTITY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-DATE", "I-DATE"]
# PREGUNTA 2: ¿La capa de salida es automática?
# SÍ. El script `ner_main.py` cuenta cuántas etiquetas hay en esta lista (ej: 11) y
# construye la capa final (logits) para que tenga exactamente ese tamaño de salida.
# Es crucial que esta lista contenga todas las etiquetas presentes en tus datasets.

# =============================================================================
# 10. ARQUITECTURA DE RED NEURONAL ("CABEZA" DE CLASIFICACIÓN)
# =============================================================================
# True = Reemplaza la capa final estándar por la que definas abajo.
USE_CUSTOM_HEAD = True

# Función de activación entre las capas ocultas ('ReLU', 'GELU', 'Tanh').
CUSTOM_HEAD_ACTIVATION = 'ReLU'

# Define las capas ocultas personalizadas entre el "cuerpo" de ALBERT y la capa de salida.
# Formato: [(tamaño_capa_1, dropout_1), (tamaño_capa_2, dropout_2), ...]
#
# PREGUNTA 1: ¿La capa oculta debe ser del mismo tamaño que la salida de ALBERT?
# NO, no tiene por qué. Es totalmente flexible.
# - La PRIMERA capa oculta que definas aquí recibirá la salida del cuerpo de ALBERT
#   (que es de tamaño 768 para ALBERT-Base).
# - Puedes hacerla más pequeña (ej: 256) para "comprimir" la información,
#   o más grande (ej: 1024) para darle más capacidad de aprender patrones complejos.
#   Hacerla más pequeña es lo más común (se conoce como "cuello de botella").
CUSTOM_HEAD_LAYERS = [(256, 0.25)] 

# Ejemplo de una cabeza más profunda:
# La salida de ALBERT (768) -> entra a una capa de 512 -> la salida (512) entra a una de 128 -> la salida (128) va a la capa final de logits.
# CUSTOM_HEAD_LAYERS = [(512, 0.3), (128, 0.2)]

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
OPTIMIZER_TYPE = "AdamW" 
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
# 11. LOGGING Y MLFLOW
# =============================================================================
LOGGING_STRATEGY = "steps"
LOGGING_STEPS = 50
REPORT_TO = "mlflow"
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_EXPERIMENT_NAME = "Parameters-Extractor-Training"
MLFLOW_USERNAME = "dsalasmlflow"
MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"