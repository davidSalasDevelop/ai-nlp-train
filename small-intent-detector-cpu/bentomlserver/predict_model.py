# predict_model.py (Versión Profesional y Robusta)
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import logging

# --- MEJORA: Configuración de Logging ---
# Se configura un logger para todo el módulo. Es la práctica estándar.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILE = "intent_classifier_final.pt"
MAX_LENGTH = 64

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

def load_model():
    logging.info(f"Iniciando la carga del modelo desde: {MODEL_FILE}")
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        
        model = InferenceModel(checkpoint)
        model.eval()
        
        logging.info("✅ Modelo y Tokenizer cargados exitosamente.")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'id_to_intent': checkpoint['id_to_intent'],
            'max_length': checkpoint.get('max_length', MAX_LENGTH)
        }
    except Exception as e:
        # --- MEJORA: Logging de Excepciones ---
        # logging.exception captura el error y el traceback completo.
        logging.exception(f"❌ Error crítico al cargar el modelo: {e}")
        return None

def predict(text: str, model_info: dict) -> list:
    try:
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        id_to_intent = model_info['id_to_intent']
        max_length = model_info['max_length']

        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                intent = id_to_intent.get(str(idx.item()), "desconocido")
                confidence = prob.item() * 100
                results.append({'intent': intent, 'confidence': round(confidence, 2)})
        
        return results
    except Exception as e:
        # --- MEJORA: Failsafe para la predicción ---
        logging.exception(f"Error durante la predicción para el texto: '{text}'")
        # Devolvemos una lista vacía para no romper el servicio. El error ya quedó registrado.
        return []