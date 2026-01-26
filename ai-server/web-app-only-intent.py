# app.py
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel # Eliminamos AutoModelForTokenClassification y pipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ¡NUEVA IMPORTACIÓN! Necesitas uvicorn para ejecutarlo desde el script.
import uvicorn

#SAMPLE USAGE
# curl -X POST "http://localhost:8000/predict" \
#     -H "Content-Type: application/json" \
#     -d '{"text": "quiero reservar una habitación"}'

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DEFINICIÓN DE IntentClassifierModel ---
class IntentClassifierModel(nn.Module):
    def __init__(self, checkpoint: dict):
        super().__init__()
        config_data = checkpoint['config']
        if isinstance(config_data, dict):
            config = AutoConfig.from_pretrained(checkpoint['tokenizer_name'], **config_data)
        else:
            config = config_data
        num_labels = len(checkpoint.get('id_to_intent', checkpoint.get('intent_to_id', {})))
        self.bert = AutoModel.from_pretrained(checkpoint['tokenizer_name'], config=config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# --- Variables globales y lifespan ---
intent_model = None
intent_tokenizer = None
id_to_intent = None
# Eliminamos ner_extractor de las variables globales
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global intent_model, intent_tokenizer, id_to_intent, device # Eliminamos ner_extractor
    device = torch.device("cpu")
    logging.info(f"Usando dispositivo para la carga y ejecución de modelos: {device}")

    INTENT_MODEL_PT_PATH = Path("../output-models/intent_classifier_final.pt")
    # Eliminamos la ruta del modelo NER

    # --- Cargar Modelo de Intenciones ---
    logging.info(f"Cargando Modelo de Intenciones desde {INTENT_MODEL_PT_PATH}...")
    try:
        if not INTENT_MODEL_PT_PATH.exists():
            raise FileNotFoundError(f"¡Error! No se encontró el archivo del modelo de intenciones en '{INTENT_MODEL_PT_PATH}'.")
        intent_checkpoint = torch.load(INTENT_MODEL_PT_PATH, map_location=device, weights_only=False)
        raw_mapping = intent_checkpoint['id_to_intent']
        first_key = next(iter(raw_mapping.keys()), None)
        # Adaptar la conversión de claves si son cadenas numéricas
        id_to_intent = {int(k): v for k, v in raw_mapping.items()} if isinstance(first_key, str) and first_key.isdigit() else {v: k for k, v in raw_mapping.items()}

        intent_model = IntentClassifierModel(intent_checkpoint)
        intent_model.load_state_dict(intent_checkpoint['model_state_dict'])
        intent_model.to(device).eval()
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_checkpoint['tokenizer_name'])
        logging.info("✅ Modelo de intenciones cargado exitosamente.")
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR EL MODELO DE INTENCIONES: {e}")

    # Eliminamos todo el bloque de carga del Modelo 2 (NER)
    
    yield
    logging.info("La aplicación se está apagando. Liberando recursos.")

# --- Instancia de FastAPI ---
app = FastAPI(
    title="Servicio de Predicción de Intenciones", # Título actualizado
    description="API para clasificar únicamente la intención de una oración.", # Descripción actualizada
    version="1.0.0",
    lifespan=lifespan
)

# --- Definiciones de Pydantic ---
class PredictRequest(BaseModel):
    text: str

# Eliminamos la clase Entity ya que no se usa

class Intent(BaseModel):
    intent: str
    confidence: float

class PredictResponse(BaseModel):
    intent_predictions: List[Intent]
    # Eliminamos ner_entities: List[Entity]

# --- Endpoint de la API ---
@app.post("/predict", response_model=PredictResponse, summary="Clasifica la intención de una oración.") # Sumario actualizado
async def predict_single_sentence(request: PredictRequest):
    if intent_model is None: # Eliminamos la comprobación de ner_extractor
        raise HTTPException(status_code=503, detail="El modelo de intenciones no está cargado.")

    phrase = request.text
    intent_predictions_list = [] # Eliminamos ner_entities_list

    try:
        inputs = intent_tokenizer(phrase, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            logits = intent_model(**inputs)
        probabilities = torch.softmax(logits, dim=1)[0]
        intent_predictions_list = sorted([{"intent": id_to_intent.get(i, f"unknown_{i}"), "confidence": prob.item()} for i, prob in enumerate(probabilities)], key=lambda x: x['confidence'], reverse=True)
    except Exception as e:
        logging.error(f"Error prediciendo intención para '{phrase}': {e}")
    
    # Eliminamos todo el bloque de predicción NER

    return PredictResponse(intent_predictions=intent_predictions_list) # Retornamos únicamente las intenciones

# --- BLOQUE PARA EJECUTAR EL SERVIDOR DIRECTAMENTE ---
# Este código solo se ejecuta si corres el script con: python app.py
if __name__ == "__main__":
    # Asegúrate de tener uvicorn instalado: pip install uvicorn
    # El string "web-app-only-intent:app" le dice a uvicorn que busque el objeto 'app' en el archivo 'web-app-only-intent.py'
    # 'reload=False' es adecuado para entornos de producción para evitar la recarga automática del servidor.
    uvicorn.run("web-app-only-intent:app", host="0.0.0.0", port=8000, reload=False)