# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import json
import mlflow
import os
import boto3
from botocore.client import Config
import argparse

# --- CLASES Y FUNCIONES (Las herramientas específicas de esta receta) ---
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

class IntentEntityModel(nn.Module):
    def __init__(self, vocab_size, num_intents, num_entity_tags, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.intent_head = nn.Linear(d_model, num_intents)
        self.entity_head = nn.Linear(d_model, num_entity_tags)
    def forward(self, input_ids):
        embedded = self.embedding(input_ids); encoded_text = self.transformer_encoder(embedded)
        return self.intent_head(encoded_text[:, 0, :]), self.entity_head(encoded_text)

def load_and_preprocess_data(filepath, tokenizer, intent_to_id, entity_to_id):
    with open(filepath, 'r') as f: data = json.load(f)
    texts, intent_labels, entity_labels_list = [], [], []
    for item in data:
        texts.append(item['text']); intent_labels.append(intent_to_id[item['intent']])
        encoding = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, padding=False)
        entity_tags = [entity_to_id.get('O')] * len(encoding['input_ids'])
        for entity in item['entities']:
            label, start_char, end_char = entity['label'], entity['start'], entity['end']
            is_first_token = True
            for i, (start, end) in enumerate(encoding['offset_mapping']):
                if start >= start_char and end <= end_char and start < end:
                    entity_tags[i] = entity_to_id.get(f'B-{label}' if is_first_token else f'I-{label}')
                    is_first_token = False
        entity_labels_list.append(entity_tags)
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    max_len = tokenized_inputs['input_ids'].shape[1]
    padded_entity_labels = [labels + [entity_to_id.get('O')] * (max_len - len(labels)) for labels in entity_labels_list]
    return (tokenized_inputs['input_ids'], torch.tensor(intent_labels, dtype=torch.long), torch.tensor(padded_entity_labels, dtype=torch.long))

def get_s3_client(endpoint, key, secret):
    return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret, config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))

# --- LA RECETA PRINCIPAL ---
if __name__ == "__main__":
    # 1. Recibir los ingredientes extra del cliente (Chef)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    # 2. Coger las credenciales del almacén (variables de entorno que el Chef le pasa al Ayudante)
    MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
    
    # 3. Ir al almacén (MinIO) a por los ingredientes principales (dataset)
    print("--- INICIANDO ENTRENAMIENTO EN CPU DEL SERVIDOR ---")
    local_dataset_path = "dataset_temp.json"
    s3 = get_s3_client(MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    s3.download_file("datasets", "nlu/dataset_v1.json", local_dataset_path)
    print(f"✅ Dataset descargado de MinIO.")

    # 4. Empezar a cocinar y avisar al Chef (MLflow)
    with mlflow.start_run() as run:
        print(f"🚀 Run de MLflow iniciado: {run.info.run_id}")
        mlflow.log_artifact(local_dataset_path, "dataset_usado")
        mlflow.log_param("num_epochs", args.num_epochs)
        mlflow.log_param("learning_rate", args.learning_rate)

        # 5. Preparar los ingredientes (procesar datos)
        intents = ["get_news", "check_weather", "get_user_info"]
        entities = ["TOPIC", "LOCATION", "DATE"]
        intent_to_id = {intent: i for i, intent in enumerate(intents)}
        entity_to_id = {'O': 0}
        for entity in entities: entity_to_id[f'B-{entity}'] = len(entity_to_id); entity_to_id[f'I-{entity}'] = len(entity_to_id)
        input_ids, intent_labels, entity_labels = load_and_preprocess_data(local_dataset_path, TOKENIZER, intent_to_id, entity_to_id)
        dataset = TensorDataset(input_ids, intent_labels, entity_labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 6. Cocinar el plato (entrenamiento en CPU)
        print(f"💪 Entrenando en CPU por {args.num_epochs} épocas... Esto tardará.")
        device = torch.device("cpu")
        model = IntentEntityModel(TOKENIZER.vocab_size, len(intent_to_id), len(entity_to_id)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        loss_fn_intent = nn.CrossEntropyLoss(); loss_fn_entity = nn.CrossEntropyLoss()
        for epoch in range(args.num_epochs):
            total_loss = 0
            for b_input_ids, b_intent_labels, b_entity_labels in dataloader:
                b_input_ids, b_intent_labels, b_entity_labels = b_input_ids.to(device), b_intent_labels.to(device), b_entity_labels.to(device)
                optimizer.zero_grad(); intent_logits, entity_logits = model(b_input_ids)
                loss = loss_fn_intent(intent_logits, b_intent_labels) + loss_fn_entity(entity_logits.view(-1, len(entity_to_id)), b_entity_labels.view(-1))
                loss.backward(); optimizer.step(); total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0: print(f"      Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)
        print("✅ Entrenamiento completado.")

        # 7. Entregar el plato cocinado al Chef para que lo guarde
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", registered_model_name="mi-modelo-nlu")
        print("🎉 ¡ÉXITO! El script ha finalizado.")
    
    os.remove(local_dataset_path)
