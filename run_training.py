# run_training_with_minio.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import json
import mlflow
import os
import boto3  # <--- Librería para S3/MinIO
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# --- 1. CONFIGURACIÓN: ¡¡MODIFICA ESTAS LÍNEAS!! ---
# --------------------------------------------------------------------------
# MLflow
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"  # URL de tu servidor MLflow
MLFLOW_EXPERIMENT_NAME = "Entrenamiento NLU con MinIO"
MLFLOW_MODEL_NAME = "mi-modelo-nlu-minio"

# --- CONFIGURACIÓN DE MINIO ---
MINIO_ENDPOINT_URL = "http://143.198.244.48:4202" # Endpoint de MinIO
MINIO_ACCESS_KEY = "mlflow_storage_admin"
MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"
MINIO_DATASET_BUCKET = "datasets" # Un bucket específico para tus datasets
MINIO_DATASET_OBJECT_NAME = "nlu/dataset_v1.json.txt" # Ruta al dataset dentro del bucket

# --------------------------------------------------------------------------
# --- 2. DEFINICIÓN DEL MODELO Y PRE-PROCESAMIENTO (Sin cambios) ---
# --------------------------------------------------------------------------
# ... (Copia aquí las clases y funciones de antes: TOKENIZER, IntentEntityModel, load_and_preprocess_data)
# No necesitan ninguna modificación.


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
