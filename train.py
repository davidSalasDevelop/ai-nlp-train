# ==============================================================================
# Archivo: train.py - VERSIÓN SIN REGISTRO DE MODELO
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
# CONFIGURACIÓN
# ==============================================================================

# --- CREDENCIALES MLFLOW ---
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
MLFLOW_TRACKING_USERNAME = "dsalasmlflow"
MLFLOW_TRACKING_PASSWORD = "SALASdavidTECHmlFlow45542344"

# --- CREDENCIALES MINIO ---
MINIO_ENDPOINT_URL = "http://143.198.244.48:4201"
MINIO_ACCESS_KEY = "mlflow_storage_admin"
MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("="*70)
print("🚀 ENTRENAMIENTO NLP - SIN REGISTRO DE MODELO")
print("="*70)
print(f"📡 MLflow: {MLFLOW_TRACKING_URI}")
print(f"🗄️  MinIO: {MINIO_ENDPOINT_URL}")
print("="*70)

# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configurar variables de entorno
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT_URL
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET_KEY
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['S3_USE_HTTPS'] = '0'
os.environ['S3_VERIFY_SSL'] = '0'

# Tokenizer global
TOKENIZER = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# ==============================================================================
# FUNCIONES PARA MINIO
# ==============================================================================

def get_minio_client():
    """Crea cliente MinIO"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )

def download_dataset():
    """Descarga el dataset de MinIO"""
    print(f"\n📥 Descargando dataset de MinIO...")
    
    local_path = "dataset_temp.json"
    bucket = "datasets"
    key = "datasets/nlu/dataset_v1.json"
    
    try:
        client = get_minio_client()
        client.download_file(Bucket=bucket, Key=key, Filename=local_path)
        
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            with open(local_path, 'r') as f:
                data = json.load(f)
                print(f"✅ Dataset descargado: {size} bytes, {len(data)} ejemplos")
                return local_path
        
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return None

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
    print(f"\n📊 Cargando dataset...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"   Total ejemplos: {len(data)}")
    
    texts, intent_labels, entity_labels_list = [], [], []
    for idx, item in enumerate(data):
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
    """Evalúa el modelo"""
    print(f"\n📈 Evaluando...")
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for b_input_ids, b_intent_labels, _ in dataloader:
            b_input_ids = b_input_ids.to(device)
            b_intent_labels = b_intent_labels.to(device)
            
            intent_logits, _ = model(b_input_ids)
            preds = torch.argmax(intent_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_intent_labels.cpu().numpy())
    
    target_names = [id_to_intent[i] for i in sorted(id_to_intent.keys())]
    report = classification_report(
        all_labels, all_preds, 
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_preds, labels=sorted(id_to_intent.keys()))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(cm, index=target_names, columns=target_names), 
        annot=True, fmt='d', cmap='Blues', ax=ax
    )
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')
    
    return report, fig

# ==============================================================================
# FUNCIÓN PRINCIPAL - SIN REGISTRAR MODELO
# ==============================================================================

def main():
    """Función principal"""
    
    # Parsear argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    
    args = parser.parse_args()
    
    print(f"\n⚙️  Parámetros:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # Variables para limpieza
    temp_files = []
    
    try:
        # 1. Descargar dataset
        dataset_path = download_dataset()
        if not dataset_path:
            print("❌ No se pudo descargar el dataset")
            return
        
        temp_files.append(dataset_path)
        
        # 2. Asegurar que no hay runs activos
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except:
            pass
        
        # 3. Crear o usar experimento
        EXPERIMENT_NAME = "nlp-training-final"
        
        try:
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                print(f"\n✅ Nuevo experimento: {EXPERIMENT_NAME}")
            else:
                experiment_id = experiment.experiment_id
                print(f"\n✅ Usando experimento existente: {EXPERIMENT_NAME}")
        except:
            experiment_id = "0"
        
        # 4. Iniciar run
        print(f"\n🎯 Iniciando run de entrenamiento...")
        
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"train-{int(time.time())}"
        ) as run:
            
            print(f"   Run ID: {run.info.run_id}")
            print(f"   🔗 URL: http://143.198.244.48:4200/#/experiments/{experiment_id}/runs/{run.info.run_id}")
            
            # Loggear parámetros
            mlflow.log_params(vars(args))
            mlflow.set_tag("dataset_source", "minio")
            mlflow.set_tag("experiment", EXPERIMENT_NAME)
            
            # 5. Preparar datos
            print(f"\n📦 Preparando datos...")
            
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
            
            # Cargar datos
            input_ids, intent_labels, entity_labels = load_and_preprocess_data(
                dataset_path, TOKENIZER, intent_to_id, entity_to_id
            )
            
            # Crear datasets
            dataset = TensorDataset(input_ids, intent_labels, entity_labels)
            train_size = max(1, int(0.8 * len(dataset)))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=2)
            
            print(f"   Dataset: {len(dataset)} ejemplos")
            print(f"   Train: {train_size}, Validation: {val_size}")
            
            # 6. Inicializar modelo
            print(f"\n🤖 Inicializando modelo...")
            
            device = torch.device("cpu")
            
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
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   Parámetros: {param_count:,}")
            mlflow.log_metric("model_parameters", param_count)
            
            # 7. Entrenamiento
            print(f"\n🔥 Entrenando por {args.num_epochs} épocas...")
            
            start_time = time.time()
            
            for epoch in range(args.num_epochs):
                model.train()
                total_loss = 0
                
                for b_input_ids, b_intent_labels, b_entity_labels in train_dataloader:
                    b_input_ids = b_input_ids.to(device)
                    b_intent_labels = b_intent_labels.to(device)
                    b_entity_labels = b_entity_labels.to(device)
                    
                    optimizer.zero_grad()
                    intent_logits, entity_logits = model(b_input_ids)
                    
                    loss_intent = loss_fn_intent(intent_logits, b_intent_labels)
                    loss_entity = loss_fn_entity(
                        entity_logits.view(-1, len(entity_to_id)), 
                        b_entity_labels.view(-1)
                    )
                    loss = loss_intent + loss_entity
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_dataloader)
                
                # Loggear métricas
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "epoch": epoch + 1,
                    "cpu_usage": psutil.cpu_percent()
                }, step=epoch)
                
                # Mostrar progreso
                if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == args.num_epochs - 1:
                    elapsed = time.time() - start_time
                    print(f"   Epoch {epoch+1:3d}/{args.num_epochs} | Loss: {avg_loss:.4f}")
            
            training_time = time.time() - start_time
            print(f"✅ Entrenamiento completado en {training_time:.1f}s")
            mlflow.log_metric("training_time_seconds", training_time)
            
            # 8. Evaluación
            print(f"\n📊 Evaluando modelo...")
            
            report, confusion_fig = evaluate_model(model, val_dataloader, device, id_to_intent)
            
            # Guardar artefactos
            mlflow.log_dict(report, "classification_report.json")
            mlflow.log_figure(confusion_fig, "confusion_matrix.png")
            
            # Métricas finales
            mlflow.log_metrics({
                "final_accuracy": report["accuracy"],
                "final_f1_macro": report["macro avg"]["f1-score"]
            })
            
            # 9. Guardar modelo COMO ARTEFACTO (sin registrar)
            print(f"\n💾 Guardando modelo como artefacto...")
            
            # Guardar modelo localmente primero
            model_path = "model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'intent_to_id': intent_to_id,
                'entity_to_id': entity_to_id,
                'args': vars(args)
            }, model_path)
            
            # Subir como artefacto
            mlflow.log_artifact(model_path, "model")
            temp_files.append(model_path)
            
            # También guardar el script del modelo
            with open("model_info.txt", "w") as f:
                f.write(f"IntentEntityModel\n")
                f.write(f"Run ID: {run.info.run_id}\n")
                f.write(f"Training time: {training_time:.1f}s\n")
                f.write(f"Parameters: {param_count:,}\n")
                f.write(f"Final loss: {avg_loss:.4f}\n")
                f.write(f"Accuracy: {report['accuracy']:.4f}\n")
            
            mlflow.log_artifact("model_info.txt", "model")
            temp_files.append("model_info.txt")
            
            # Subir dataset
            mlflow.log_artifact(dataset_path, "dataset")
            
            print(f"\n" + "="*70)
            print(f"🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
            print(f"🎯 Run ID: {run.info.run_id}")
            print(f"📊 Métricas finales:")
            print(f"   - Loss final: {avg_loss:.4f}")
            print(f"   - Accuracy: {report['accuracy']:.4f}")
            print(f"   - F1 Macro: {report['macro avg']['f1-score']:.4f}")
            print(f"🔗 URL: http://143.198.244.48:4200/#/experiments/{experiment_id}/runs/{run.info.run_id}")
            print("="*70)
            
            # Mostrar URL directa
            print(f"\n📋 PARA VER LOS RESULTADOS:")
            print(f"1. Abre: http://143.198.244.48:4200")
            print(f"2. Ve al experimento: {EXPERIMENT_NAME}")
            print(f"3. Busca el run: {run.info.run_id}")
            print(f"4. En 'Artifacts' encontrarás:")
            print(f"   • model/model.pth (el modelo entrenado)")
            print(f"   • model/model_info.txt (información del modelo)")
            print(f"   • dataset/ (el dataset usado)")
            print(f"   • classification_report.json")
            print(f"   • confusion_matrix.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            mlflow.set_tag("error", str(e))
            mlflow.end_run(status="FAILED")
        except:
            pass
        
    finally:
        # Limpiar archivos temporales
        for file_path in temp_files:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Eliminado: {file_path}")

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    main()