# web-app-only-ner.py

# ============================
# IMPORTACIONES
# ============================
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, pipeline
from transformers import AlbertTokenizer
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ============================
# CONFIGURACIONES
# ============================

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configuración de rutas
NER_MODEL_PT_PATH = Path("../output-models/web-app-ner-0POLMD.pt")
TOKENIZER_CACHE_DIR = Path("./tokenizer_cache")

# Configuración del modelo
DEFAULT_TOKENIZER_NAME = "dccuchile/albert-base-spanish"
VOCAB_FILE_NAME = "spiece.model"

# Configuración de arquitectura del clasificador
CLASSIFIER_DROPOUT_RATE = 0.2
CLASSIFIER_HIDDEN_SIZE = 768
CLASSIFIER_INTERMEDIATE_SIZE = 256

# Configuración del pipeline NER
NER_PIPELINE_DEVICE = -1  # CPU
NER_AGGREGATION_STRATEGY = "simple"
NER_PIPELINE_TASK = "token-classification"

# Configuración del servidor
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_RELOAD = False

# Variables globales
ner_extractor = None
device = None

#SAMPLE USAGE
# curl -X POST "http://localhost:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar una habitación"}'
# curl -X POST "http://vscode:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar una habitación"}'
# cd ai-nlp-train/ai-server , python web-app-only-ner.py

# ============================
# LIFECYCLE MANAGEMENT
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_extractor, device
    device = torch.device("cpu")
    logging.info(f"Usando dispositivo para la carga y ejecución de modelos: {device}")

    # --- Cargar Modelo NER ---
    logging.info(f"Cargando Modelo NER desde {NER_MODEL_PT_PATH}...")
    try:
        if not NER_MODEL_PT_PATH.exists():
            raise FileNotFoundError(f"¡Error! No se encontró el archivo del modelo NER en '{NER_MODEL_PT_PATH}'.")
        
        # Cargar checkpoint
        ner_checkpoint = torch.load(NER_MODEL_PT_PATH, map_location=device, weights_only=False)
        ner_tokenizer_name = ner_checkpoint['tokenizer_name']
        ner_id2label = ner_checkpoint['id2label']
        ner_num_labels = len(ner_id2label)
        ner_state_dict = ner_checkpoint['model_state_dict']

        # IMPORTANTE: EL TOKENIZADO ES MUY IMPORTANTE
        # Primero cargar el tokenizador de la forma correcta
        # Usar el mismo método que en el entrenamiento (AlbertTokenizer específico)
        logging.info(f"Cargando tokenizador desde: {ner_tokenizer_name}")
        
        # Opción 1: Intentar cargar con AutoTokenizer
        try:
            ner_tokenizer = AutoTokenizer.from_pretrained(ner_tokenizer_name)
            logging.info("✅ Tokenizador cargado exitosamente con AutoTokenizer")
        except Exception as tokenizer_error:
            logging.warning(f"AutoTokenizer falló: {tokenizer_error}")
            # Opción 2: Cargar específicamente AlbertTokenizer
            try:
                # Descargar el archivo de vocabulario
                vocab_file = hf_hub_download(
                    repo_id=ner_tokenizer_name,
                    filename=VOCAB_FILE_NAME,
                    cache_dir=TOKENIZER_CACHE_DIR
                )
                ner_tokenizer = AlbertTokenizer.from_pretrained(vocab_file)
                logging.info("✅ Tokenizador AlbertTokenizer cargado exitosamente desde huggingface_hub")
            except Exception as e:
                logging.error(f"Error cargando AlbertTokenizer desde huggingface_hub: {e}")
                # Opción 3: Usar el tokenizador desde el directorio cache si existe
                try:
                    ner_tokenizer = AlbertTokenizer.from_pretrained(ner_tokenizer_name)
                    logging.info("✅ Tokenizador AlbertTokenizer cargado desde caché local")
                except Exception as cache_error:
                    logging.error(f"Error cargando desde caché: {cache_error}")
                    raise
        
        # Cargar configuración del modelo
        ner_config = AutoConfig.from_pretrained(
            ner_tokenizer_name, 
            num_labels=ner_num_labels, 
            id2label=ner_id2label, 
            label2id={v: k for k, v in ner_id2label.items()}
        )
        
        # Crear modelo
        ner_model = AutoModelForTokenClassification.from_config(ner_config)
        
        # IMPORTANTE: Reconstruir la misma arquitectura que en el entrenamiento
        # Verificar si tiene clasificador personalizado en el estado
        has_custom_classifier = any('classifier' in key for key in ner_state_dict.keys())
        
        if has_custom_classifier:
            # Reconstruir exactamente la misma arquitectura que en ner_main.py
            # Por defecto, usar la misma arquitectura que entrenaste
            ner_model.classifier = nn.Sequential(
                nn.Dropout(CLASSIFIER_DROPOUT_RATE),
                nn.Linear(CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_INTERMEDIATE_SIZE),
                nn.ReLU(),
                nn.Linear(CLASSIFIER_INTERMEDIATE_SIZE, ner_num_labels)
            )
            logging.info("✅ Arquitectura personalizada reconstruida")
        
        # Cargar los pesos
        ner_model.load_state_dict(ner_state_dict)
        ner_model.to(device)
        ner_model.eval()
        
        logging.info(f"Estado del modelo cargado en: {device}")
        
        # Crear pipeline NER
        ner_extractor = pipeline(
            NER_PIPELINE_TASK, 
            model=ner_model, 
            tokenizer=ner_tokenizer, 
            device=NER_PIPELINE_DEVICE,
            aggregation_strategy=NER_AGGREGATION_STRATEGY
        )
        
        logging.info("✅ Modelo NER cargado exitosamente con arquitectura reconstruida.")
        
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR EL MODELO NER: {e}")
        # Si falla, intentar una carga simplificada como respaldo
        try:
            logging.info("Intentando carga simplificada como respaldo...")
            ner_checkpoint = torch.load(NER_MODEL_PT_PATH, map_location=device, weights_only=False)
            ner_tokenizer_name = ner_checkpoint['tokenizer_name']
            
            # Cargar tokenizador simple
            ner_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
            
            # Cargar modelo base y aplicar pesos
            ner_model = AutoModelForTokenClassification.from_pretrained(
                ner_tokenizer_name,
                num_labels=len(ner_checkpoint['id2label']),
                id2label=ner_checkpoint['id2label'],
                label2id=ner_checkpoint['label2id']
            )
            ner_model.load_state_dict(ner_checkpoint['model_state_dict'])
            ner_model.to(device).eval()
            
            ner_extractor = pipeline(
                NER_PIPELINE_TASK, 
                model=ner_model, 
                tokenizer=ner_tokenizer, 
                device=NER_PIPELINE_DEVICE,
                aggregation_strategy=NER_AGGREGATION_STRATEGY
            )
            logging.info("✅ Modelo NER cargado con método simplificado.")
        except Exception as backup_error:
            logging.error(f"❌ FALLO también en carga simplificada: {backup_error}")
    
    yield
    
    logging.info("La aplicación se está apagando. Liberando recursos.")

# ============================
# INSTANCIA DE FASTAPI
# ============================
app = FastAPI(
    title="Servicio de Extracción de Entidades Nombradas (NER)",
    description="API para extraer entidades nombradas de una oración.",
    version="1.0.0",
    lifespan=lifespan
)

# ============================
# DEFINICIONES DE PYDANTIC
# ============================

class PredictRequest(BaseModel):
    text: str

class Entity(BaseModel):
    label: str
    value: str
    score: float

class PredictResponse(BaseModel):
    ner_entities: List[Entity]

# ============================
# ENDPOINTS DE LA API
# ============================

@app.post("/predict", response_model=PredictResponse, summary="Extrae entidades nombradas de una oración.")
async def predict_single_sentence(request: PredictRequest):
    if ner_extractor is None:
        raise HTTPException(status_code=503, detail="El modelo NER no está cargado.")

    phrase = request.text
    ner_entities_list = []

    try:
        entities = ner_extractor(phrase)
        ner_entities_list = [
            {
                'label': e.get('entity_group', e.get('entity', 'UNKNOWN')), 
                'value': e['word'], 
                'score': float(e['score'])
            } 
            for e in entities
        ]
    except Exception as e:
        logging.error(f"Error prediciendo entidades NER para '{phrase}': {e}")

    return PredictResponse(ner_entities=ner_entities_list)

# ============================
# BLOQUE PARA EJECUTAR EL SERVIDOR
# ============================

if __name__ == "__main__":
    uvicorn.run(
        "web-app-ner-0POLMD:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD
    )