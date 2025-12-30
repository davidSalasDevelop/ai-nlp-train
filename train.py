# train.py (con logging detallado)

# --- 1. CONFIGURACIÓN DEL LOGGING (¡NUEVO!) ---
import logging
import sys

# Configuramos el logger para que imprima todo, incluyendo la hora y el nivel del mensaje.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Imprime los logs en la consola
    ]
)

logging.info("--- Script de entrenamiento iniciado ---")

# --- Importaciones normales ---
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

# --- 2. VERIFICACIÓN DEL ENTORNO (¡NUEVO!) ---
# Vamos a verificar que el entorno tiene lo que esperamos.
try:
    import shutil
    logging.info(f"Python ejecutable: {sys.executable}")
    logging.info(f"Ubicación de 'virtualenv' (si se usa): {shutil.which('virtualenv')}")
    logging.info(f"Ubicación de 'pip': {shutil.which('pip')}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"MLflow version: {mlflow.__version__}")
except Exception as e:
    logging.error(f"Error al verificar el entorno: {e}")

# --- CLASES Y FUNCIONES (Sin cambios) ---
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
class IntentEntityModel(nn.Module):
    # ... (Pega aquí la clase IntentEntityModel completa, sin cambios)
    # ...
def load_and_preprocess_data(filepath, tokenizer, intent_to_id, entity_to_id):
    # ... (Pega aquí la función load_and_preprocess_data completa, sin cambios)
    # ...
def get_s3_client(endpoint, key, secret):
    return boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret, config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))

# --- LA RECETA PRINCIPAL ---
if __name__ == "__main__":
    logging.info("--- Parseando argumentos ---")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    logging.info(f"Argumentos recibidos: {args}")

    logging.info("--- Cargando credenciales desde variables de entorno ---")
    MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
    if not all([MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY]):
        logging.error("¡Faltan credenciales de MinIO! Asegúrate de que están en backend_config.")
        exit(1)
    
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

    try:
        logging.info(f"Iniciando Run de MLflow...")
        with mlflow.start_run() as run:
            logging.info(f"🚀 Run de MLflow iniciado: {run.info.run_id}")
            # ... (El resto del código del bloque 'with' es igual, pero puedes añadir más logs si quieres)
            
            # ... Entrenamiento ...

            logging.info("✅ Entrenamiento completado.")
            logging.info(f"📤 Registrando el modelo...")
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", registered_model_name="mi-modelo-nlu")
            logging.info("🎉 ¡ÉXITO! El script ha finalizado.")
        
        os.remove(local_dataset_path)

    except Exception as e:
        logging.error(f"❌ ERROR CRÍTICO durante el run de MLflow: {e}")
        exit(1)
