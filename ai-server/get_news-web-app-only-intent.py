# app.py

# ============================
# IMPORTACIONES
# ============================
import logging
from pathlib import Path
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
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
INTENT_MODEL_PT_PATH = Path("../output-models/model_get_news.pt")

# Configuración del modelo
MODEL_MAX_LENGTH = 64

# Configuración del servidor
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8001
SERVER_RELOAD = False

#SAMPLE USAGE
# curl -X POST "http://localhost:8001/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar una habitación"}'
# curl -X POST "http://vscode:8001/predict"      -H "Content-Type: application/json"      -d '{"text": "que paso ayer en guatemala noticias"}'
# curl -X POST "http://vscode:8001/predict"      -H "Content-Type: application/json"      -d '{"text": "quiero reservar un vuel"}'

# ============================
# DEFINICIÓN DE MODELOS
# ============================

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

# ============================
# VARIABLES GLOBALES
# ============================
intent_model = None
intent_tokenizer = None
id_to_intent = None
device = None

# ============================
# LIFECYCLE MANAGEMENT
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global intent_model, intent_tokenizer, id_to_intent, device
    device = torch.device("cpu")
    logging.info(f"Usando dispositivo para la carga y ejecución de modelos: {device}")

    # --- Cargar Modelo de Intenciones ---
    logging.info(f"Cargando Modelo de Intenciones desde {INTENT_MODEL_PT_PATH}...")
    try:
        if not INTENT_MODEL_PT_PATH.exists():
            raise FileNotFoundError(f"¡Error! No se encontró el archivo del modelo de intenciones en '{INTENT_MODEL_PT_PATH}'.")
        
        intent_checkpoint = torch.load(INTENT_MODEL_PT_PATH, map_location=device, weights_only=False)
        
        # Procesar mapeo de identificadores a intenciones
        raw_mapping = intent_checkpoint['id_to_intent']
        first_key = next(iter(raw_mapping.keys()), None)
        
        # Adaptar la conversión de claves si son cadenas numéricas
        if isinstance(first_key, str) and first_key.isdigit():
            id_to_intent = {int(k): v for k, v in raw_mapping.items()}
        else:
            id_to_intent = {v: k for k, v in raw_mapping.items()}

        # Inicializar y cargar el modelo
        intent_model = IntentClassifierModel(intent_checkpoint)
        intent_model.load_state_dict(intent_checkpoint['model_state_dict'])
        intent_model.to(device).eval()
        
        # Cargar tokenizador
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_checkpoint['tokenizer_name'])
        
        logging.info("✅ Modelo de intenciones cargado exitosamente.")
        
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR EL MODELO DE INTENCIONES: {e}")
    
    yield
    
    logging.info("La aplicación se está apagando. Liberando recursos.")

# ============================
# INSTANCIA DE FASTAPI
# ============================
app = FastAPI(
    title="Servicio de Predicción de Intenciones",
    description="API para clasificar únicamente la intención de una oración.",
    version="1.0.0",
    lifespan=lifespan
)

# ============================
# DEFINICIONES DE PYDANTIC
# ============================

class PredictRequest(BaseModel):
    text: str

class Intent(BaseModel):
    intent: str
    confidence: float

class PredictResponse(BaseModel):
    intent_predictions: List[Intent]

# ============================
# ENDPOINTS DE LA API
# ============================

@app.post("/predict", response_model=PredictResponse, summary="Clasifica la intención de una oración.")
async def predict_single_sentence(request: PredictRequest):
    if intent_model is None:
        raise HTTPException(status_code=503, detail="El modelo de intenciones no está cargado.")

    phrase = request.text
    intent_predictions_list = []

    try:
        # Tokenizar entrada
        inputs = intent_tokenizer(
            phrase, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MODEL_MAX_LENGTH
        ).to(device)
        
        # Realizar predicción
        with torch.no_grad():
            logits = intent_model(**inputs)
        
        # Calcular probabilidades
        probabilities = torch.softmax(logits, dim=1)[0]
        
        # Formatear resultados
        intent_predictions_list = sorted([
            {
                "intent": id_to_intent.get(i, f"unknown_{i}"), 
                "confidence": prob.item()
            } 
            for i, prob in enumerate(probabilities)
        ], key=lambda x: x['confidence'], reverse=True)
        
    except Exception as e:
        logging.error(f"Error prediciendo intención para '{phrase}': {e}")
    
    return PredictResponse(intent_predictions=intent_predictions_list)

# ============================
# BLOQUE PARA EJECUTAR EL SERVIDOR
# ============================

if __name__ == "__main__":
    uvicorn.run(
        "get_news-web-app-only-intent:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD
    )