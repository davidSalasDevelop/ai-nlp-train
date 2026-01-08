# predict_model.py (Versión Final, Corregida y Profesional)
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import logging
from pathlib import Path
import os
from typing import TypedDict, List, Dict, Any
import pydantic

class Prediction(pydantic.BaseModel):
    intent: str
    confidence: float

class ModelConfig:
    SCRIPT_DIR = Path(__file__).parent
    MODEL_FILE = Path(os.getenv("MODEL_PATH", SCRIPT_DIR.parent / "output/intent_classifier_final.pt"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))

class ModelInfo(TypedDict):
    model: nn.Module
    tokenizer: Any
    id_to_intent: Dict[int, str] # Clave es un integer
    max_length: int

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionError(Exception):
    pass

class InferenceModel(nn.Module):
    def __init__(self, chkpt):
        super().__init__()
        self.bert = AutoModel.from_pretrained(chkpt['tokenizer_name'])
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, len(chkpt['id_to_intent']))
        self.load_state_dict(chkpt['model_state_dict'])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

def load_model() -> ModelInfo | None:
    logging.info(f"Iniciando la carga del modelo desde: {ModelConfig.MODEL_FILE.resolve()}")
    try:
        if not ModelConfig.MODEL_FILE.exists():
            logging.error(f"¡ERROR FATAL! El archivo del modelo no se encuentra: {ModelConfig.MODEL_FILE.resolve()}")
            return None
        checkpoint = torch.load(ModelConfig.MODEL_FILE, map_location=torch.device('cpu'))
        
        # --- CORRECCIÓN SUTIL PERO VITAL ---
        # Aseguramos que las claves del diccionario sean integers, como en tu script original
        id_to_intent = {int(k): v for k, v in checkpoint['id_to_intent'].items()}

        tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        model = InferenceModel(checkpoint)
        model.eval()
        logging.info("✅ Modelo y Tokenizer cargados exitosamente.")
        return {'model': model, 'tokenizer': tokenizer, 'id_to_intent': id_to_intent, 'max_length': checkpoint.get('max_length', ModelConfig.MAX_LENGTH)}
    except Exception as e:
        logging.exception(f"❌ Error crítico al cargar el modelo: {e}")
        return None

def predict(text: str, model_info: ModelInfo) -> List[Prediction]:
    try:
        encoding = model_info['tokenizer'](text, max_length=model_info['max_length'], padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model_info['model'](encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, 3)
            results = []
            for prob, idx in zip(top_probs, top_indices):
                # --- ¡AQUÍ ESTÁ LA CORRECCIÓN FINAL! ---
                # Usamos idx.item() (un integer) directamente, como en tu script original,
                # pero mantenemos la seguridad del .get()
                intent = model_info['id_to_intent'].get(idx.item(), "desconocido")
                confidence = round(prob.item() * 100, 2)
                results.append(Prediction(intent=intent, confidence=confidence))
        return results
    except Exception as e:
        raise PredictionError(f"Error durante la predicción para el texto: '{text}'") from e