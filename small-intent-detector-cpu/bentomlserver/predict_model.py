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
    id_to_intent: Dict[int, str]
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
        
        checkpoint = torch.load(ModelConfig.MODEL_FILE, map_location=torch.device('cpu'), weights_only=False)
        
        # --- MODIFICADO: Lógica robusta para manejar el mapeo de intenciones ---
        raw_mapping = checkpoint['id_to_intent']
        # Revisa la primera clave para ver qué tipo de diccionario es
        first_key = next(iter(raw_mapping.keys()))

        if isinstance(first_key, str) and not first_key.isdigit():
            # Las claves son nombres de intenciones (ej: 'get_news'). Es un dict 'intent_to_id'.
            # Lo invertimos para crear el 'id_to_intent' que necesitamos.
            logging.warning("El mapeo en el checkpoint es 'intent_to_id'. Invirtiéndolo a 'id_to_intent'.")
            id_to_intent = {v: k for k, v in raw_mapping.items()}
        else:
            # Las claves son números o strings de números (ej: 0 o '0'). Es 'id_to_intent'.
            # Solo nos aseguramos de que las claves sean integers.
            id_to_intent = {int(k): v for k, v in raw_mapping.items()}
        
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
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        id_to_intent = model_info['id_to_intent']
        max_length = model_info['max_length']
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probabilities = torch.softmax(logits, dim=1)
        predictions = []
        for i in range(probabilities.shape[1]):
            intent = id_to_intent[i]
            confidence = probabilities[0, i].item()
            predictions.append(Prediction(intent=intent, confidence=confidence))
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions
    except Exception as e:
        logging.exception(f"Error durante la predicción para el texto: '{text}'")
        raise PredictionError(f"No se pudo completar la predicción: {e}")

if __name__ == "__main__":
    model_assets = load_model()
    if model_assets:
        test_text = "Quiero ver las noticias de hoy"
        logging.info(f"--- Realizando una predicción de ejemplo ---")
        logging.info(f"Texto de entrada: '{test_text}'")
        try:
            all_predictions = predict(test_text, model_assets)
            best_prediction = all_predictions[0]
            print("\n--- Resultado de la Predicción ---")
            print(f"Texto:          '{test_text}'")
            print(f"Intención:      {best_prediction.intent}")
            print(f"Confianza:      {best_prediction.confidence:.2%}")
        except PredictionError as e:
            logging.error(f"No se pudo realizar la predicción de ejemplo: {e}")
    else:
        logging.error("El script no puede continuar porque el modelo no se pudo cargar.")