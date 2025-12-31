# predict.py - TÚ PONES EL PATH
# Predice con texto por defecto
#python predict.py

# Predice con tu texto
#python predict.py "Muéstrame noticias de tecnología"

# O así
#python predict.py ¿Cómo está el clima?

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import boto3
from botocore.client import Config
import io
import sys

# CONFIG - TÚ PONES ESTO
MINIO_ENDPOINT = "http://143.198.244.48:4201"
ACCESS_KEY = "mlflow_storage_admin"
SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"

# TÚ PONES ESTOS VALORES - según tu imagen
BUCKET = "mlflow"  # El bucket
MODEL_PATH = "6/25b329d0cd904c539ad2ae2d35d090ed/artifacts/model/model.pth"  # El path exacto

# Clase del modelo
class IntentEntityModel(nn.Module):
    def __init__(self, vocab_size, num_intents, num_entity_tags, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.intent_head = nn.Linear(d_model, num_intents)
        self.entity_head = nn.Linear(d_model, num_entity_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        encoded_text = self.transformer_encoder(embedded)
        intent_logits = self.intent_head(encoded_text[:, 0, :])
        entity_logits = self.entity_head(encoded_text)
        return intent_logits, entity_logits

# Cargar desde MinIO
def load_from_minio():
    print(f"📥 Cargando: s3://{BUCKET}/{MODEL_PATH}")
    
    s3 = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version='s3v4')
    )
    
    # Descarga directa
    response = s3.get_object(Bucket=BUCKET, Key=MODEL_PATH)
    model_bytes = response['Body'].read()
    
    print(f"✅ Descargado: {len(model_bytes)} bytes")
    
    # Cargar
    buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(buffer, map_location='cpu')
    
    # Crear modelo
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    model = IntentEntityModel(
        vocab_size=tokenizer.vocab_size,
        num_intents=len(checkpoint['intent_to_id']),
        num_entity_tags=len(checkpoint['entity_to_id']),
        d_model=checkpoint.get('args', {}).get('d_model', 128),
        nhead=checkpoint.get('args', {}).get('nhead', 4),
        num_layers=checkpoint.get('args', {}).get('num_layers', 2)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'intents': list(checkpoint['intent_to_id'].keys()),
        'id_to_intent': {v: k for k, v in checkpoint['intent_to_id'].items()}
    }

# Predecir
def predict(text, model_info):
    encoding = model_info['tokenizer'](text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        intent_logits, _ = model_info['model'](encoding['input_ids'])
    
    probs = torch.softmax(intent_logits, dim=1)[0]
    intent_idx = torch.argmax(probs).item()
    intent = model_info['id_to_intent'].get(intent_idx, "UNKNOWN")
    confidence = probs[intent_idx].item() * 100
    
    return intent, f"{confidence:.1f}%"

# MAIN
if __name__ == "__main__":
    # Cargar
    model_info = load_from_minio()
    print(f"🎯 Intents disponibles: {model_info['intents']}")
    
    # Texto
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "¿Qué tiempo hace en Madrid hoy?"
    
    # Predecir
    intent, confidence = predict(text, model_info)
    
    print(f"\n📝 Texto: {text}")
    print(f"🎯 Intención: {intent}")
    print(f"📈 Confianza: {confidence}")