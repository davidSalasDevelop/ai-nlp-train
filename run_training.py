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

def get_s3_client():
    """Crea un cliente de boto3 para conectarse a MinIO."""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

def download_dataset_from_minio(bucket, object_name, local_path="dataset.json"):
    """Descarga el dataset desde MinIO a un archivo local temporal."""
    print(f"Descargando '{object_name}' desde el bucket '{bucket}' de MinIO...")
    try:
        s3 = get_s3_client()
        s3.download_file(bucket, object_name, local_path)
        print(f"Dataset descargado en '{local_path}'")
        return local_path
    except Exception as e:
        print(f"ERROR: No se pudo descargar el dataset desde MinIO. {e}")
        return None

# --------------------------------------------------------------------------
# --- 4. EL SCRIPT PRINCIPAL MODIFICADO ---
# --------------------------------------------------------------------------

def main():
    print(f"--- Conectando a MLflow en {MLFLOW_TRACKING_URI} ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Descargamos el dataset desde MinIO antes de empezar
    local_dataset_path = download_dataset_from_minio(MINIO_DATASET_BUCKET, MINIO_DATASET_OBJECT_NAME)
    if not local_dataset_path:
        return # Si no se puede descargar, detenemos la ejecución

    with mlflow.start_run() as run:
        print(f"--- Iniciando nuevo Run en MLflow: {run.info.run_id} ---")
        
        # --- REGISTRAMOS EL DATASET USADO ---
        # Esto le dice a MLflow "Este entrenamiento se hizo con ESTE archivo exacto"
        print("Registrando el dataset como un artefacto en MLflow...")
        mlflow.log_artifact(local_dataset_path, "dataset_usado")

        # --- PARÁMETROS DEL MODELO (se registrarán en MLflow) ---
        params = { "num_epochs": 50, "batch_size": 2, "learning_rate": 5e-5 }
        mlflow.log_params(params)

        # --- PREPARACIÓN DE DATOS (ahora usa la ruta local descargada) ---
        print("--- Preparando datos ---")
        # (El resto del código de preparación es idéntico, solo cambia la ruta)
        intents = ["get_news", "check_weather", "get_user_info"] # etc.
        # ... (código de mapeos idéntico)
        input_ids, _, intent_labels, entity_labels = load_and_preprocess_data(
            local_dataset_path, TOKENIZER, intent_to_id, entity_to_id
        )
        # ... (código de dataloader, modelo, optimizador y bucle de entrenamiento idéntico)

        # --- BUCLE DE ENTRENAMIENTO ---
        # ... (El bucle de entrenamiento completo va aquí, sin cambios)
        
        print("--- Entrenamiento completado ---")

        # --- REGISTRO DEL MODELO EN EL SERVIDOR MLFLOW ---
        # (Esta parte no cambia, MLflow ya sabe que debe subirlo a MinIO)
        print(f"--- Registrando el modelo como '{MLFLOW_MODEL_NAME}' en el servidor MLflow/MinIO ---")
        
        # ... (código para la firma y el registro del modelo idéntico)
        mlflow.pytorch.log_model(...)
        
        print("--- ¡Modelo registrado con éxito! ---")

if __name__ == "__main__":
    main()
