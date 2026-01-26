# web-app-only-ner.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, pipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variables globales
ner_extractor = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_extractor, device
    device = torch.device("cpu")
    logging.info(f"Usando dispositivo para la carga y ejecución de modelos: {device}")

    NER_MODEL_PT_PATH = Path("../output-models/get_news_extractor.pt")

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

        # IMPORTANTE: Primero cargar el tokenizador de la forma correcta
        # Usar el mismo método que en el entrenamiento (AlbertTokenizer específico)
        logging.info(f"Cargando tokenizador desde: {ner_tokenizer_name}")
        
        # Opción 1: Intentar cargar con AutoTokenizer
        try:
            ner_tokenizer = AutoTokenizer.from_pretrained(ner_tokenizer_name)
        except Exception as tokenizer_error:
            logging.warning(f"AutoTokenizer falló: {tokenizer_error}")
            # Opción 2: Cargar específicamente AlbertTokenizer
            from huggingface_hub import hf_hub_download
            from transformers import AlbertTokenizer
            try:
                # Descargar el archivo de vocabulario
                vocab_file = hf_hub_download(
                    repo_id=ner_tokenizer_name,
                    filename="spiece.model",
                    cache_dir="./tokenizer_cache"
                )
                ner_tokenizer = AlbertTokenizer.from_pretrained(vocab_file)
                logging.info("✅ Tokenizador AlbertTokenizer cargado exitosamente")
            except Exception as e:
                logging.error(f"Error cargando AlbertTokenizer: {e}")
                # Opción 3: Usar el tokenizador desde el directorio cache si existe
                ner_tokenizer = AlbertTokenizer.from_pretrained(ner_tokenizer_name)
        
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
            # basado en ner_config.CUSTOM_HEAD_LAYERS
            # Por defecto, usar la misma arquitectura que entrenaste
            ner_model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, ner_num_labels)
            )
            logging.info("✅ Arquitectura personalizada reconstruida")
        
        # Cargar los pesos
        ner_model.load_state_dict(ner_state_dict)
        ner_model.to(device)
        ner_model.eval()
        
        logging.info(f"Estado del modelo cargado en: {device}")
        
        # Crear pipeline NER
        ner_extractor = pipeline(
            "token-classification", 
            model=ner_model, 
            tokenizer=ner_tokenizer, 
            device=-1,  # Usar CPU
            aggregation_strategy="simple"
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
            ner_tokenizer = AutoTokenizer.from_pretrained("dccuchile/albert-base-spanish")
            
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
                "token-classification", 
                model=ner_model, 
                tokenizer=ner_tokenizer, 
                device=-1,
                aggregation_strategy="simple"
            )
            logging.info("✅ Modelo NER cargado con método simplificado.")
        except Exception as backup_error:
            logging.error(f"❌ FALLO también en carga simplificada: {backup_error}")
    
    yield
    
    logging.info("La aplicación se está apagando. Liberando recursos.")

# --- Instancia de FastAPI ---
app = FastAPI(
    title="Servicio de Extracción de Entidades Nombradas (NER)",
    description="API para extraer entidades nombradas de una oración.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Definiciones de Pydantic ---
class PredictRequest(BaseModel):
    text: str

class Entity(BaseModel):
    label: str
    value: str
    score: float

class PredictResponse(BaseModel):
    ner_entities: List[Entity]

# --- Endpoint de la API ---
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

# --- BLOQUE PARA EJECUTAR EL SERVIDOR DIRECTAMENTE ---
if __name__ == "__main__":
    uvicorn.run("web-app-only-ner:app", host="0.0.0.0", port=8000, reload=False)