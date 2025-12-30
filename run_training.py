# run_training_with_minio.py
# Celda 1: Instalación de dependencias
# - mlflow: Para conectarse al servidor de tracking.
# - boto3: El cliente oficial de Python para S3, que MinIO usa.
# - transformers/torch/datasets: Para el modelo y el entrenamiento.
!pip install mlflow boto3 transformers torch datasets -q
print("✅ Librerías instaladas.")

# --------------------------------------------------------------------------
# --- 1. CONFIGURACIÓN: ¡¡MODIFICA ESTAS LÍNEAS!! ---
# --------------------------------------------------------------------------
# Celda 2: Configuración de tus Servidores y Proyecto

# --- Configuración de MLflow ---
# ¡¡CAMBIA ESTO por la URL PÚBLICA y puerto de tu servidor MLflow!!
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"  # URL de tu servidor MLflow

# --- Configuración de MinIO ---
# ¡¡CAMBIA ESTO por la URL PÚBLICA y puerto de tu servidor MinIO!!
MINIO_ENDPOINT_URL = "http://143.198.244.48:4202" # Endpoint de MinIO
MINIO_ACCESS_KEY = "mlflow_storage_admin"
MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"

# --- Configuración del Dataset en MinIO ---
MINIO_DATASET_BUCKET = "datasets"
MINIO_DATASET_OBJECT_NAME = "nlu/dataset_v1.json"

# --- Configuración del Experimento y Modelo en MLflow ---
MLFLOW_EXPERIMENT_NAME = "Entrenamiento NLU Bilingüe"
MLFLOW_MODEL_NAME = "mi-modelo-nlu" # El nombre que tendrá en el registro de modelos

# --- Variables de entorno para que MLflow sepa cómo conectarse a MinIO ---
import os
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT_URL
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY

print("✅ Configuración cargada. ¡Asegúrate de haber rellenado todos los campos!")


# --------------------------------------------------------------------------
# --- 2. DEFINICIÓN DEL MODELO Y PRE-PROCESAMIENTO (Sin cambios) ---
# --------------------------------------------------------------------------
# Celda 3: Definición del Modelo, Tokenizer y Funciones de Datos

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json

# --- Definiciones que no cambian ---
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
        token_offsets = encoding['offset_mapping']
        entity_tags = [entity_to_id.get('O')] * len(encoding['input_ids'])
        for entity in item['entities']:
            label, start_char, end_char = entity['label'], entity['start'], entity['end']
            is_first_token = True
            for i, (start, end) in enumerate(token_offsets):
                if start >= start_char and end <= end_char and start < end:
                    if is_first_token:
                        entity_tags[i] = entity_to_id.get(f'B-{label}')
                        is_first_token = False
                    else:
                        entity_tags[i] = entity_to_id.get(f'I-{label}')
        entity_labels_list.append(entity_tags)
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    max_len = tokenized_inputs['input_ids'].shape[1]
    padded_entity_labels = []
    for labels in entity_labels_list:
        padded_labels = labels + [entity_to_id.get('O')] * (max_len - len(labels))
        padded_entity_labels.append(padded_labels[:max_len])
    return (tokenized_inputs['input_ids'], torch.tensor(intent_labels, dtype=torch.long), torch.tensor(padded_entity_labels, dtype=torch.long))

print("✅ Clases y funciones del modelo listas.")


# --------------------------------------------------------------------------
# --- 3. FUNCIONES AUXILIARES PARA MINIO ---
# --------------------------------------------------------------------------

# Celda 4: Script Principal de Entrenamiento y Registro

import boto3
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

def main():
    # --- 1. Descargar Dataset desde MinIO ---
    print(f"⬇️ Descargando '{MINIO_DATASET_OBJECT_NAME}' desde MinIO...")
    local_dataset_path = "dataset_from_minio.json"
    try:
        s3 = boto3.client('s3', endpoint_url=MINIO_ENDPOINT_URL, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
        s3.download_file(MINIO_DATASET_BUCKET, MINIO_DATASET_OBJECT_NAME, local_dataset_path)
        print("   ✅ Dataset descargado con éxito.")
    except Exception as e:
        print(f"   ❌ ERROR: No se pudo descargar el dataset. Verifica tu configuración de MinIO. Error: {e}")
        return

    # --- 2. Conectarse a MLflow y Empezar un Experimento ---
    print(f"📡 Conectando a MLflow en {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"   🚀 Run iniciado en MLflow con ID: {run.info.run_id}")

        # --- 3. Registrar Artefactos y Parámetros ---
        print("   📝 Registrando artefactos y parámetros...")
        mlflow.log_artifact(local_dataset_path, "dataset_usado")
        params = {"num_epochs": 50, "batch_size": 4, "learning_rate": 5e-5}
        mlflow.log_params(params)

        # --- 4. Preparar Datos para el Entrenamiento ---
        print("   🥣 Preparando datos para el modelo...")
        intents = ["get_news", "check_weather", "get_user_info"]
        entities = ["TOPIC", "LOCATION", "DATE"]
        intent_to_id = {intent: i for i, intent in enumerate(intents)}
        entity_to_id = {'O': 0}
        for entity in entities:
            entity_to_id[f'B-{entity}'] = len(entity_to_id)
            entity_to_id[f'I-{entity}'] = len(entity_to_id)

        input_ids, intent_labels, entity_labels = load_and_preprocess_data(local_dataset_path, TOKENIZER, intent_to_id, entity_to_id)
        dataset = TensorDataset(input_ids, intent_labels, entity_labels)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

        # --- 5. Bucle de Entrenamiento ---
        print("   💪 ¡Iniciando entrenamiento en la GPU de Colab!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IntentEntityModel(TOKENIZER.vocab_size, len(intent_to_id), len(entity_to_id)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
        loss_fn_intent = nn.CrossEntropyLoss()
        loss_fn_entity = nn.CrossEntropyLoss()

        for epoch in range(params["num_epochs"]):
            total_loss = 0
            for b_input_ids, b_intent_labels, b_entity_labels in dataloader:
                b_input_ids, b_intent_labels, b_entity_labels = b_input_ids.to(device), b_intent_labels.to(device), b_entity_labels.to(device)
                optimizer.zero_grad()
                intent_logits, entity_logits = model(b_input_ids)
                loss = loss_fn_intent(intent_logits, b_intent_labels) + loss_fn_entity(entity_logits.view(-1, len(entity_to_id)), b_entity_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
              print(f"      Epoch {epoch+1}/{params['num_epochs']}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("avg_loss", avg_loss, step=epoch)
        
        print("   ✅ Entrenamiento completado.")

        # --- 6. Registrar el Modelo Final en MLflow/MinIO ---
        print(f"   📤 Registrando el modelo como '{MLFLOW_MODEL_NAME}'...")
        signature = ModelSignature(inputs=Schema([ColSpec("string", "text")]), outputs=Schema([ColSpec("string")]))
        
        # En lugar de guardar el modelo localmente, lo pasamos directamente a MLflow.
        # MLflow se encargará de guardarlo y subirlo a MinIO.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model", # Carpeta donde se guardará en MinIO
            registered_model_name=MLFLOW_MODEL_NAME,
            signature=signature,
        )
        print("---")
        print(f"🎉 ¡ÉXITO! Modelo entrenado y registrado en tu servidor MLflow.")

# --- Ejecutar todo el proceso ---
main()
