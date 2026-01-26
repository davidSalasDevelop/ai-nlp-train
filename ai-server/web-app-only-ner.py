# web-app-only-ner.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, pipeline # Mantener estas importaciones para NER

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Eliminamos la clase IntentClassifierModel ya que no es necesaria ---

# --- Variables globales y lifespan ---
# Eliminamos intent_model, intent_tokenizer, id_to_intent
ner_extractor = None # Mantener ner_extractor
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_extractor, device # Solo necesitamos ner_extractor y device
    device = torch.device("cpu")
    logging.info(f"Usando dispositivo para la carga y ejecución de modelos: {device}")

    # Eliminamos la ruta del modelo de intenciones
    NER_MODEL_PT_PATH = Path("../ner_training/models/get_news_extractor/get_news_extractor.pt") # Mantener la ruta del modelo NER

    # --- Eliminamos todo el bloque de carga del Modelo de Intenciones ---

    # --- Cargar Modelo NER ---
    logging.info(f"Cargando Modelo NER desde {NER_MODEL_PT_PATH}...")
    try:
        if not NER_MODEL_PT_PATH.exists():
            raise FileNotFoundError(f"¡Error! No se encontró el archivo del modelo NER en '{NER_MODEL_PT_PATH}'.")
        
        ner_checkpoint = torch.load(NER_MODEL_PT_PATH, map_location=device, weights_only=False)
        ner_tokenizer_name = ner_checkpoint['tokenizer_name']
        ner_id2label = ner_checkpoint['id2label']
        ner_num_labels = len(ner_id2label)
        ner_state_dict = ner_checkpoint['model_state_dict']

        # Reconstrucción de la configuración y el modelo, tal como estaba en el original
        ner_config = AutoConfig.from_pretrained(ner_tokenizer_name, num_labels=ner_num_labels, id2label=ner_id2label, label2id={v: k for k, v in ner_id2label.items()})
        ner_model = AutoModelForTokenClassification.from_config(ner_config)
        # Asegúrate de que esta línea coincida con la arquitectura exacta guardada si es personalizada
        ner_model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, ner_num_labels))
        ner_model.load_state_dict(ner_state_dict)
        ner_model.to(device).eval()
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_tokenizer_name)
        ner_extractor = pipeline("token-classification", model=ner_model, tokenizer=ner_tokenizer, device=device if torch.cuda.is_available() else -1, aggregation_strategy="simple")
        logging.info("✅ Modelo NER cargado exitosamente con arquitectura reconstruida.")
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR EL MODELO NER: {e}")
    
    yield
    logging.info("La aplicación se está apagando. Liberando recursos.")

# --- Instancia de FastAPI ---
app = FastAPI(
    title="Servicio de Extracción de Entidades Nombradas (NER)", # Título actualizado
    description="API para extraer entidades nombradas de una oración.", # Descripción actualizada
    version="1.0.0",
    lifespan=lifespan
)

# --- Definiciones de Pydantic ---
class PredictRequest(BaseModel):
    text: str

class Entity(BaseModel): # Mantener la clase Entity
    label: str
    value: str
    score: float

# Eliminamos la clase Intent ya que no se usa

class PredictResponse(BaseModel):
    # Eliminamos intent_predictions: List[Intent]
    ner_entities: List[Entity] # Solo retornamos entidades NER

# --- Endpoint de la API ---
@app.post("/predict", response_model=PredictResponse, summary="Extrae entidades nombradas de una oración.") # Sumario actualizado
async def predict_single_sentence(request: PredictRequest):
    if ner_extractor is None: # Solo comprobamos ner_extractor
        raise HTTPException(status_code=503, detail="El modelo NER no está cargado.")

    phrase = request.text
    # Eliminamos intent_predictions_list
    ner_entities_list = [] # Mantener ner_entities_list

    # Eliminamos todo el bloque de predicción de intenciones

    try:
        entities = ner_extractor(phrase)
        ner_entities_list = [{'label': e['entity_group'], 'value': e['word'], 'score': float(e['score'])} for e in entities]
    except Exception as e:
        logging.error(f"Error prediciendo entidades NER para '{phrase}': {e}")

    return PredictResponse(ner_entities=ner_entities_list) # Retornamos únicamente las entidades NER

# --- BLOQUE PARA EJECUTAR EL SERVIDOR DIRECTAMENTE ---
if __name__ == "__main__":
    # Asegúrate de tener uvicorn instalado: pip install uvicorn
    # Cambia "web-app-only-ner:app" para que coincida con el nombre de tu archivo si lo guardas con otro nombre
    uvicorn.run("web-app-only-ner:app", host="0.0.0.0", port=8000, reload=False)