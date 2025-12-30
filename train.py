# ==============================================================================
# Archivo: train.py
# ==============================================================================

# --- 1. Importaciones Adicionales ---
import logging
import sys
import time
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import mlflow
import boto3
from botocore.client import Config
import psutil
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Configuración del Logging (para ver el progreso en la consola) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Clases y Funciones de Datos (sin cambios en su lógica interna) ---
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
        embedded = self.embedding(input_ids)
        encoded_text = self.transformer_encoder(embedded)
        intent_logits = self.intent_head(encoded_text[:, 0, :])
        entity_logits = self.entity_head(encoded_text)
        return intent_logits, entity_logits

def load_and_preprocess_data(filepath, tokenizer, intent_to_id, entity_to_id):
    with open(filepath, 'r') as f:
        data = json.load(f)
    texts, intent_labels, entity_labels_list = [], [], []
    for item in data:
        texts.append(item['text'])
        intent_labels.append(intent_to_id[item['intent']])
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

def evaluate_model(model, dataloader, device, id_to_intent):
    """Función para evaluar el modelo al final y generar artefactos de reporte."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for b_input_ids, b_intent_labels, _ in dataloader:
            b_input_ids, b_intent_labels = b_input_ids.to(device), b_intent_labels.to(device)
            intent_logits, _ = model(b_input_ids)
            preds = torch.argmax(intent_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_intent_labels.cpu().numpy())
    
    target_names = [id_to_intent[i] for i in sorted(id_to_intent.keys())]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(id_to_intent.keys()))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pd.DataFrame(cm, index=target_names, columns=target_names), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión de Intenciones')
    
    return report, fig

# --- LA RECETA PRINCIPAL ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
    
    logging.info("--- Iniciando el proceso principal del script ---")
    local_dataset_path = "dataset_temp.json"
    
    try:
        logging.info(f"Conectando a MinIO en {MINIO_ENDPOINT_URL} para descargar el dataset...")
        s3 = get_s3_client(MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
        s3.download_file("datasets", "nlu/dataset_v1.json", local_dataset_path)
        logging.info(f"✅ Dataset descargado de MinIO.")
    except Exception as e:
        logging.error(f"❌ ERROR CRÍTICO al descargar de MinIO: {e}")
        exit(1)

    with mlflow.start_run() as run:
        logging.info(f"🚀 Run de MLflow iniciado: {run.info.run_id}")
        mlflow.log_artifact(local_dataset_path, "dataset_usado")
        
        logging.info("📝 Registrando hiperparámetros del modelo y del run...")
        mlflow.log_params(vars(args))

        intents = ["get_news", "check_weather", "get_user_info"]
        entities = ["TOPIC", "LOCATION", "DATE"]
        intent_to_id = {intent: i for i, intent in enumerate(intents)}
        id_to_intent = {i: intent for intent, i in intent_to_id.items()}
        entity_to_id = {'O': 0}
        for entity in entities: entity_to_id[f'B-{entity}'] = len(entity_to_id); entity_to_id[f'I-{entity}'] = len(entity_to_id)
        
        input_ids, intent_labels, entity_labels = load_and_preprocess_data(local_dataset_path, TOKENIZER, intent_to_id, entity_to_id)
        
        dataset = TensorDataset(input_ids, intent_labels, entity_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2)
        logging.info(f"Datos divididos: {train_size} para entrenamiento, {val_size} para validación.")

        logging.info(f"💪 Entrenando en CPU por {args.num_epochs} épocas...")
        device = torch.device("cpu")
        model = IntentEntityModel(TOKENIZER.vocab_size, len(intent_to_id), len(entity_to_id), args.d_model, args.nhead, args.num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        loss_fn_intent = nn.CrossEntropyLoss(); loss_fn_entity = nn.CrossEntropyLoss()
        
        training_start_time = time.time()
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            total_loss, total_intent_loss, total_entity_loss = 0, 0, 0
            
            model.train()
            for b_input_ids, b_intent_labels, b_entity_labels in train_dataloader:
                b_input_ids, b_intent_labels, b_entity_labels = b_input_ids.to(device), b_intent_labels.to(device), b_entity_labels.to(device)
                optimizer.zero_grad()
                intent_logits, entity_logits = model(b_input_ids)
                loss_intent = loss_fn_intent(intent_logits, b_intent_labels)
                loss_entity = loss_fn_entity(entity_logits.view(-1, len(entity_to_id)), b_entity_labels.view(-1))
                loss = loss_intent + loss_entity
                loss.backward(); optimizer.step()
                total_loss += loss.item()
                total_intent_loss += loss_intent.item()
                total_entity_loss += loss_entity.item()
            
            avg_loss = total_loss / len(train_dataloader)
            avg_intent_loss = total_intent_loss / len(train_dataloader)
            avg_entity_loss = total_entity_loss / len(train_dataloader)
            epoch_duration = time.time() - epoch_start_time
            
            metrics_to_log = {
                "avg_loss_total": avg_loss,
                "avg_loss_intent": avg_intent_loss,
                "avg_loss_entity": avg_entity_loss,
                "epoch_duration_sec": epoch_duration,
                "cpu_usage_percent": psutil.cpu_percent(),
                "ram_usage_percent": psutil.virtual_memory().percent
            }
            mlflow.log_metrics(metrics_to_log, step=epoch)
            
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = args.num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} | Dur: {epoch_duration:.2f}s | Loss Total: {avg_loss:.4f} | ETA: {eta_minutes:.2f} min")

        logging.info("✅ Entrenamiento completado.")

        logging.info("📊 Realizando evaluación final del modelo...")
        report, confusion_matrix_fig = evaluate_model(model, val_dataloader, device, id_to_intent)
        
        logging.info("   Registrando artefactos de evaluación...")
        mlflow.log_dict(report, "final_classification_report.json")
        mlflow.log_figure(confusion_matrix_fig, "final_confusion_matrix.png")
        
        logging.info("   Registrando métricas de resumen...")
        mlflow.log_metric("final_accuracy", report["accuracy"])
        mlflow.log_metric("final_f1_macro_avg", report["macro avg"]["f1-score"])

        logging.info(f"📤 Registrando el modelo final...")
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", registered_model_name="mi-modelo-nlu")
        logging.info("🎉 ¡ÉXITO! El script ha finalizado.")
    
    os.remove(local_dataset_path)
