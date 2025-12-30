# ==============================================================================
# Archivo: train.py - RUTA EXACTA DEL DATASET CORREGIDA
# ==============================================================================

# --- 1. Importaciones ---
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

# ==============================================================================
# CONFIGURACIÓN DE CREDENCIALES - TODO INTEGRADO
# ==============================================================================

# --- CREDENCIALES MLFLOW TRACKING SERVER ---
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_TRACKING_USERNAME = "dsalasmlflow"
MLFLOW_TRACKING_PASSWORD = "SALASdavidTECHmlFlow45542344"

# --- CREDENCIALES MINIO ---
MINIO_ENDPOINT_URL = "http://143.198.244.48:4202"
MINIO_ACCESS_KEY = "mlflow_storage_admin"
MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"

# --- RUTA EXACTA DEL DATASET EN MINIO ---
MINIO_BUCKET = "datasets"
MINIO_DATASET_PATH = "datasets/nlu/dataset_v1.json"  # ¡EXACTA!

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("="*60)
print("🚀 CONFIGURANDO ENTRENAMIENTO NLP")
print("="*60)

# Configurar MLflow
print(f"\n🔧 Configurando MLflow...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")

# Configurar variables de entorno
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT_URL
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_USE_HTTPS'] = '0'
os.environ['S3_VERIFY_SSL'] = '0'

print(f"✅ MLflow configurado")

# Tokenizer global
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# ==============================================================================
# FUNCIONES PARA MINIO (NO AWS)
# ==============================================================================

def get_minio_client():
    """Crea cliente para MINIO (S3 compatible)"""
    print(f"\n🔗 Creando cliente MinIO...")
    print(f"   Endpoint: {MINIO_ENDPOINT_URL}")
    print(f"   Bucket: {MINIO_BUCKET}")
    print(f"   Path: {MINIO_DATASET_PATH}")
    
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        ),
        region_name='us-east-1'
    )

def download_dataset_from_minio():
    """Descarga el dataset desde MinIO"""
    print(f"\n📥 Descargando dataset desde MinIO...")
    local_path = "dataset_temp.json"
    
    try:
        # Crear cliente MinIO
        s3_client = get_minio_client()
        
        # Verificar que el bucket existe
        print(f"   Verificando bucket '{MINIO_BUCKET}'...")
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
        print(f"   ✅ Bucket encontrado")
        
        # Verificar que el archivo existe
        print(f"   Verificando archivo '{MINIO_DATASET_PATH}'...")
        s3_client.head_object(Bucket=MINIO_BUCKET, Key=MINIO_DATASET_PATH)
        print(f"   ✅ Archivo encontrado")
        
        # Descargar archivo
        print(f"   Descargando a '{local_path}'...")
        s3_client.download_file(
            Bucket=MINIO_BUCKET,
            Key=MINIO_DATASET_PATH,
            Filename=local_path
        )
        
        # Verificar que se descargó
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            print(f"   ✅ Dataset descargado: {file_size} bytes")
            
            # Leer un poco para verificar
            with open(local_path, 'r') as f:
                data = json.load(f)
                print(f"   📊 Dataset contiene {len(data)} ejemplos")
                if len(data) > 0:
                    print(f"   Ejemplo 1: {data[0]}")
        else:
            raise Exception("Archivo no se descargó correctamente")
            
        return local_path
        
    except Exception as e:
        print(f"❌ ERROR descargando de MinIO: {e}")
        print(f"   Detalles:")
        print(f"   - Bucket: {MINIO_BUCKET}")
        print(f"   - Path: {MINIO_DATASET_PATH}")
        print(f"   - Endpoint: {MINIO_ENDPOINT_URL}")
        raise

# ==============================================================================
# CLASES Y FUNCIONES DEL MODELO
# ==============================================================================

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
    """Carga y preprocesa los datos"""
    print(f"\n📊 Cargando dataset desde {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"   Total ejemplos: {len(data)}")
    
    texts, intent_labels, entity_labels_list = [], [], []
    for idx, item in enumerate(data):
        if idx < 3:  # Mostrar primeros 3 ejemplos
            print(f"   Ejemplo {idx}: {item['text'][:50]}... -> {item['intent']}")
        
        texts.append(item['text'])
        intent_labels.append(intent_to_id[item['intent']])
        
        # Tokenizar texto
        encoding = tokenizer(item['text'], return_offsets_mapping=True, truncation=True, padding=False)
        
        # Inicializar todas las etiquetas como 'O'
        entity_tags = [entity_to_id.get('O')] * len(encoding['input_ids'])
        
        # Mapear entidades
        for entity in item['entities']:
            label, start_char, end_char = entity['label'], entity['start'], entity['end']
            is_first_token = True
            for i, (start, end) in enumerate(encoding['offset_mapping']):
                if start >= start_char and end <= end_char and start < end:
                    entity_tags[i] = entity_to_id.get(f'B-{label}' if is_first_token else f'I-{label}')
                    is_first_token = False
        
        entity_labels_list.append(entity_tags)
    
    # Tokenizar todos los textos
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    print(f"   Longitud máxima de secuencia: {tokenized_inputs['input_ids'].shape[1]}")
    
    # Padding para etiquetas de entidades
    max_len = tokenized_inputs['input_ids'].shape[1]
    padded_entity_labels = [
        labels + [entity_to_id.get('O')] * (max_len - len(labels)) 
        for labels in entity_labels_list
    ]
    
    return (
        tokenized_inputs['input_ids'], 
        torch.tensor(intent_labels, dtype=torch.long), 
        torch.tensor(padded_entity_labels, dtype=torch.long)
    )

def evaluate_model(model, dataloader, device, id_to_intent):
    """Evalúa el modelo y genera reportes"""
    print(f"\n📈 Evaluando modelo...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (b_input_ids, b_intent_labels, _) in enumerate(dataloader):
            b_input_ids = b_input_ids.to(device)
            b_intent_labels = b_intent_labels.to(device)
            
            intent_logits, _ = model(b_input_ids)
            preds = torch.argmax(intent_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_intent_labels.cpu().numpy())
            
            if batch_idx == 0:
                print(f"   Batch 0 - Predicciones: {preds[:5].tolist()}")
                print(f"   Batch 0 - Reales: {b_intent_labels[:5].tolist()}")
    
    # Generar reporte de clasificación
    target_names = [id_to_intent[i] for i in sorted(id_to_intent.keys())]
    report = classification_report(
        all_labels, all_preds, 
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # Generar matriz de confusión
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(id_to_intent.keys()))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pd.DataFrame(cm, index=target_names, columns=target_names), 
        annot=True, fmt='d', cmap='Blues', ax=ax
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión de Intenciones')
    
    print(f"   Accuracy: {report['accuracy']:.4f}")
    print(f"   Macro F1: {report['macro avg']['f1-score']:.4f}")
    
    return report, fig

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def train_model(args):
    """Función principal de entrenamiento"""
    
    print("\n" + "="*60)
    print("🏋️  INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    # 1. Descargar dataset de MinIO
    dataset_path = download_dataset_from_minio()
    
    # Configurar experimento de MLflow
    mlflow.set_experiment("nlp-training")
    
    with mlflow.start_run(run_name=f"train-d_model{args.d_model}-lr{args.learning_rate}") as run:
        print(f"\n🎯 MLflow Run ID: {run.info.run_id}")
        print(f"🔗 URL: http://143.198.244.48:4200/#/experiments/1/runs/{run.info.run_id}")
        
        # Loggear parámetros
        mlflow.log_params(vars(args))
        mlflow.log_param("minio_dataset_path", MINIO_DATASET_PATH)
        mlflow.log_param("tokenizer", "bert-base-multilingual-cased")
        
        # Loggear checkpoint inicial
        mlflow.log_metric("init_checkpoint", 1)
        
        # 2. Preparar datos
        print(f"\n📦 Preparando datos...")
        
        # Definir intents y entities
        intents = ["get_news", "check_weather", "get_user_info"]
        entities = ["TOPIC", "LOCATION", "DATE"]
        
        intent_to_id = {intent: i for i, intent in enumerate(intents)}
        id_to_intent = {i: intent for intent, i in intent_to_id.items()}
        
        entity_to_id = {'O': 0}
        for entity in entities:
            entity_to_id[f'B-{entity}'] = len(entity_to_id)
            entity_to_id[f'I-{entity}'] = len(entity_to_id)
        
        print(f"   Intents: {intents}")
        print(f"   Entities: {entities}")
        print(f"   Intent mapping: {intent_to_id}")
        print(f"   Entity mapping: {entity_to_id}")
        
        # Cargar y preprocesar datos
        input_ids, intent_labels, entity_labels = load_and_preprocess_data(
            dataset_path, TOKENIZER, intent_to_id, entity_to_id
        )
        
        mlflow.log_metric("init_checkpoint", 2)
        
        # Crear datasets
        dataset = TensorDataset(input_ids, intent_labels, entity_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2)
        
        print(f"   Train size: {train_size}, Val size: {val_size}")
        mlflow.log_metric("init_checkpoint", 3)
        
        # 3. Inicializar modelo
        print(f"\n🤖 Inicializando modelo...")
        
        device = torch.device("cpu")  # Usar CPU para simplicidad
        
        model = IntentEntityModel(
            vocab_size=TOKENIZER.vocab_size,
            num_intents=len(intent_to_id),
            num_entity_tags=len(entity_to_id),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        loss_fn_intent = nn.CrossEntropyLoss()
        loss_fn_entity = nn.CrossEntropyLoss()
        
        print(f"   Modelo en: {device}")
        print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters())}")
        mlflow.log_metric("init_checkpoint", 4)
        
        # 4. Entrenamiento
        print(f"\n🔥 Comenzando entrenamiento por {args.num_epochs} épocas...")
        
        training_start_time = time.time()
        
        for epoch in range(args.num_epochs):
            epoch_start_time = time.time()
            
            # Modo entrenamiento
            model.train()
            total_loss, total_intent_loss, total_entity_loss = 0, 0, 0
            
            for batch_idx, (b_input_ids, b_intent_labels, b_entity_labels) in enumerate(train_dataloader):
                # Mover a dispositivo
                b_input_ids = b_input_ids.to(device)
                b_intent_labels = b_intent_labels.to(device)
                b_entity_labels = b_entity_labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                intent_logits, entity_logits = model(b_input_ids)
                
                # Calcular pérdidas
                loss_intent = loss_fn_intent(intent_logits, b_intent_labels)
                loss_entity = loss_fn_entity(
                    entity_logits.view(-1, len(entity_to_id)), 
                    b_entity_labels.view(-1)
                )
                loss = loss_intent + loss_entity
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Acumular pérdidas
                total_loss += loss.item()
                total_intent_loss += loss_intent.item()
                total_entity_loss += loss_entity.item()
                
                # Loggear primer batch
                if epoch == 0 and batch_idx == 0:
                    print(f"   Batch 0 - Loss: {loss.item():.4f}")
                    mlflow.log_metric("first_batch_loss", loss.item())
            
            # Calcular métricas de la época
            avg_loss = total_loss / len(train_dataloader)
            avg_intent_loss = total_intent_loss / len(train_dataloader)
            avg_entity_loss = total_entity_loss / len(train_dataloader)
            epoch_duration = time.time() - epoch_start_time
            
            # Loggear métricas en MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_intent_loss": avg_intent_loss,
                "train_entity_loss": avg_entity_loss,
                "epoch_duration_sec": epoch_duration,
                "cpu_usage_percent": psutil.cpu_percent(),
                "ram_usage_percent": psutil.virtual_memory().percent
            }, step=epoch)
            
            # Calcular ETA
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = args.num_epochs - (epoch + 1)
            eta_minutes = (avg_epoch_time * remaining_epochs) / 60
            
            # Log en consola cada 5 épocas
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.num_epochs - 1:
                print(f"   Epoch {epoch+1}/{args.num_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {epoch_duration:.2f}s | "
                      f"ETA: {eta_minutes:.1f}min")
        
        print(f"\n✅ Entrenamiento completado en {(time.time() - training_start_time)/60:.1f} minutos")
        mlflow.log_metric("init_checkpoint", 5)
        
        # 5. Evaluación
        print(f"\n📊 Evaluando modelo...")
        
        report, confusion_fig = evaluate_model(model, val_dataloader, device, id_to_intent)
        
        # Guardar artefactos en MLflow
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_figure(confusion_fig, "confusion_matrix.png")
        
        # Loggear métricas finales
        mlflow.log_metrics({
            "final_accuracy": report["accuracy"],
            "final_f1_macro": report["macro avg"]["f1-score"],
            "final_f1_weighted": report["weighted avg"]["f1-score"]
        })
        
        mlflow.log_metric("init_checkpoint", 6)
        
        # 6. Guardar modelo
        print(f"\n💾 Guardando modelo en MLflow...")
        
        # Registrar el modelo
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="nlp-intent-entity-model"
        )
        
        print(f"   ✅ Modelo registrado como: nlp-intent-entity-model")
        
        # 7. Subir dataset usado como artefacto
        mlflow.log_artifact(dataset_path, "dataset")
        
        print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Ver en: http://143.198.244.48:4200/#/experiments/1/runs/{run.info.run_id}")
    
    return run.info.run_id

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 INICIANDO SCRIPT DE ENTRENAMIENTO NLP")
    print("="*60)
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo NLP")
    parser.add_argument("--num_epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Tasa de aprendizaje")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensión del modelo")
    parser.add_argument("--nhead", type=int, default=4, help="Número de cabezas de atención")
    parser.add_argument("--num_layers", type=int, default=2, help="Número de capas")
    
    args = parser.parse_args()
    
    print(f"\n⚙️  Parámetros de entrenamiento:")
    print(f"   Épocas: {args.num_epochs}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   d_model: {args.d_model}")
    print(f"   nhead: {args.nhead}")
    print(f"   num_layers: {args.num_layers}")
    
    # Variable para el archivo temporal
    temp_dataset_path = None
    
    try:
        # Ejecutar entrenamiento
        run_id = train_model(args)
        
        print("\n" + "="*60)
        print(f"✅ PROCESO COMPLETADO")
        print(f"🎯 Run ID: {run_id}")
        print(f"🔗 URL: http://143.198.244.48:4200/#/experiments/1/runs/{run_id}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        
        # Intentar loggear error en MLflow si hay un run activo
        try:
            mlflow.set_tag("error_fatal", str(e))
            mlflow.end_run(status="FAILED")
        except:
            pass
        
        sys.exit(1)
        
    finally:
        # Limpiar archivo temporal
        if temp_dataset_path and os.path.exists(temp_dataset_path):
            os.remove(temp_dataset_path)
            print(f"\n🧹 Archivo temporal eliminado: {temp_dataset_path}")