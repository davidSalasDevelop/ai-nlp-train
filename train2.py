# ==============================================================================
# train_lightweight.py - USA TU DATASET DE MINIO
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import mlflow
import json
import boto3
from botocore.client import Config
from io import BytesIO
import tempfile
import os
from typing import List, Dict
import argparse

# ==============================================================================
# CONFIGURACIÓN CON TUS DATOS
# ==============================================================================

class Config:
    # TUS CREDENCIALES
    MINIO_ENDPOINT = "http://143.198.244.48:4201"
    MINIO_ACCESS_KEY = "mlflow_storage_admin"
    MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"
    
    # TU DATASET EN MINIO
    MINIO_BUCKET = "datasets"
    DATASET_PATH = "datasets/nlu/dataset_v1.json"  # TU dataset
    
    # MODELO LIGERO
    BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 80 MB
    
    # MLflow
    MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
    MLFLOW_USERNAME = "dsalasmlflow"
    MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"
    
    # Hiperparámetros
    MAX_LENGTH = 64
    BATCH_SIZE = 8  # Pequeño para tu dataset de 6 ejemplos
    LEARNING_RATE = 3e-5
    EPOCHS = 10

# ==============================================================================
# DESCARGAR TU DATASET
# ==============================================================================

def download_your_dataset():
    """Descarga TU dataset específico de MinIO"""
    print(f"📥 Descargando TU dataset: {Config.DATASET_PATH}")
    
    s3 = boto3.client(
        's3',
        endpoint_url=Config.MINIO_ENDPOINT,
        aws_access_key_id=Config.MINIO_ACCESS_KEY,
        aws_secret_access_key=Config.MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )
    
    # Descargar a archivo temporal
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
        s3.download_fileobj(Config.MINIO_BUCKET, Config.DATASET_PATH, tmp)
        tmp_path = tmp.name
    
    # Cargar y analizar TU dataset
    with open(tmp_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Dataset cargado: {len(data)} ejemplos")
    
    # Mostrar análisis de TU dataset
    intents = [item['intent'] for item in data]
    unique_intents = list(set(intents))
    
    print(f"\n📊 ANÁLISIS DE TU DATASET:")
    print(f"   Total ejemplos: {len(data)}")
    print(f"   Intenciones únicas: {len(unique_intents)}")
    
    for intent in unique_intents:
        count = intents.count(intent)
        print(f"   - {intent}: {count} ejemplos")
    
    # Limpiar
    os.unlink(tmp_path)
    
    return data

# ==============================================================================
# PROCESAR TU DATASET ESPECÍFICO
# ==============================================================================

def process_your_dataset(data):
    """Procesa TU dataset específico con sus intenciones"""
    texts = []
    intents = []
    
    for item in data:
        texts.append(item['text'])
        intents.append(item['intent'])
    
    # Mapeo de intenciones (TUS intenciones reales)
    unique_intents = sorted(set(intents))
    intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    
    print(f"\n🎯 TUS INTENCIONES: {unique_intents}")
    print(f"   Mapeo: {intent_to_id}")
    
    # Convertir a IDs
    labels = [intent_to_id[intent] for intent in intents]
    
    return texts, labels, intent_to_id, id_to_intent

# ==============================================================================
# MODELO LIGERO
# ==============================================================================

class MiniLMClassifier(nn.Module):
    """Clasificador ligero con MiniLM"""
    def __init__(self, num_intents):
        super().__init__()
        
        # MiniLM pre-entrenado (80 MB)
        self.minilm = AutoModel.from_pretrained(Config.BASE_MODEL)
        
        # Clasificador simple
        self.classifier = nn.Linear(384, num_intents)  # 384 = dimensión MiniLM
        
    def forward(self, input_ids, attention_mask):
        outputs = self.minilm(input_ids=input_ids, attention_mask=attention_mask)
        
        # Usar el token [CLS] para clasificación
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Clasificar
        logits = self.classifier(cls_output)
        
        return logits

# ==============================================================================
# DATASET
# ==============================================================================

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================

def train_simple():
    """Entrenamiento simple con TU dataset"""
    
    # 1. Descargar TU dataset
    your_data = download_your_dataset()
    
    # 2. Procesar datos
    texts, labels, intent_to_id, id_to_intent = process_your_dataset(your_data)
    
    # Si hay muy pocos datos, crear dataset ampliado
    if len(texts) < 10:
        print("\n⚠️  Dataset muy pequeño. Ampliando con variaciones...")
        texts, labels = augment_dataset(texts, labels)
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    
    # 4. Dataset y DataLoader
    dataset = IntentDataset(texts, labels, tokenizer)
    
    # Split manual (porque es pequeño)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=min(Config.BATCH_SIZE, len(train_dataset)),
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(Config.BATCH_SIZE * 2, len(val_dataset)),
        shuffle=False
    )
    
    print(f"\n📦 Datos: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # 5. Modelo
    model = MiniLMClassifier(num_intents=len(intent_to_id))
    
    # 6. Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # 7. Configurar MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_TRACKING_USERNAME'] = Config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = Config.MLFLOW_PASSWORD
    
    mlflow.set_experiment("tu-nlp-training")
    
    with mlflow.start_run(run_name="miniLM-tu-dataset"):
        # Log parameters
        mlflow.log_params({
            'model': Config.BASE_MODEL,
            'dataset_size': len(your_data),
            'num_intents': len(intent_to_id),
            'intents': list(intent_to_id.keys()),
            'epochs': Config.EPOCHS,
            'batch_size': Config.BATCH_SIZE
        })
        
        mlflow.log_dict(intent_to_id, "intent_to_id.json")
        
        print(f"\n🔥 Entrenando con TU dataset...")
        
        best_val_acc = 0
        
        for epoch in range(Config.EPOCHS):
            # Training
            model.train()
            train_loss, train_correct = 0, 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                logits = model(batch['input_ids'], batch['attention_mask'])
                loss = loss_fn(logits, batch['label'])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == batch['label']).sum().item()
            
            # Validation
            model.eval()
            val_loss, val_correct = 0, 0
            
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch['input_ids'], batch['attention_mask'])
                    loss = loss_fn(logits, batch['label'])
                    
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch['label']).sum().item()
            
            # Métricas
            train_acc = train_correct / len(train_dataset)
            val_acc = val_correct / len(val_dataset)
            
            # Log
            mlflow.log_metrics({
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }, step=epoch)
            
            print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'intent_to_id': intent_to_id,
                    'id_to_intent': id_to_intent,
                    'val_accuracy': val_acc,
                    'config': {
                        'base_model': Config.BASE_MODEL,
                        'max_length': Config.MAX_LENGTH
                    }
                }
                
                torch.save(checkpoint, 'best_model.pt')
                mlflow.log_artifact('best_model.pt')
        
        # 8. Guardar modelo final
        print(f"\n💾 Guardando modelo final...")
        
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'intent_to_id': intent_to_id,
            'id_to_intent': id_to_intent,
            'tokenizer_config': tokenizer.init_kwargs
        }
        
        torch.save(final_checkpoint, 'final_model.pt')
        
        # Log final model
        mlflow.log_artifact('final_model.pt')
        
        # También guardar el modelo en formato MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"📊 Mejor val accuracy: {best_val_acc:.4f}")
        print(f"📦 Modelo guardado: final_model.pt (~80 MB)")
        print(f"🎯 TUS INTENCIONES: {list(intent_to_id.keys())}")

def augment_dataset(texts, labels):
    """Amplía el dataset pequeño con variaciones simples"""
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    
    # Variaciones para cada texto
    for text, label in zip(texts, labels):
        # Cambiar mayúsculas/minúsculas
        augmented_texts.append(text.lower())
        augmented_labels.append(label)
        
        augmented_texts.append(text.upper())
        augmented_labels.append(label)
        
        # Añadir signos de puntuación
        augmented_texts.append(text + "?")
        augmented_labels.append(label)
        
        augmented_texts.append("Por favor, " + text)
        augmented_labels.append(label)
        
        augmented_texts.append("Necesito " + text.lower())
        augmented_labels.append(label)
    
    print(f"   Dataset ampliado: {len(texts)} → {len(augmented_texts)} ejemplos")
    
    return augmented_texts, augmented_labels

# ==============================================================================
# PREDICT CON TU MODELO
# ==============================================================================

def predict_with_your_model():
    """Usa el modelo entrenado con TU dataset"""
    
    # Cargar checkpoint
    if not os.path.exists('final_model.pt'):
        print("❌ No hay modelo entrenado. Ejecuta primero: python train_lightweight.py")
        return
    
    checkpoint = torch.load('final_model.pt', map_location='cpu')
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)
    
    # Crear modelo
    model = MiniLMClassifier(num_intents=len(checkpoint['intent_to_id']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Mapeos
    intent_to_id = checkpoint['intent_to_id']
    id_to_intent = checkpoint['id_to_intent']
    
    print(f"\n🎯 Modelo cargado: {len(intent_to_id)} intenciones")
    print(f"   Intenciones: {list(intent_to_id.keys())}")
    
    # Predecir
    while True:
        text = input("\n📝 Escribe un texto (o 'salir'): ").strip()
        
        if text.lower() == 'salir':
            break
        
        # Tokenizar
        encoding = tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Inferencia
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = F.softmax(logits, dim=1)[0]
        
        # Obtener predicciones
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100
        
        print(f"\n🔍 Resultado:")
        print(f"   Texto: {text}")
        print(f"   🎯 Intención: {id_to_intent[pred_idx]}")
        print(f"   📈 Confianza: {confidence:.1f}%")
        
        # Mostrar todas las probabilidades
        print(f"\n📊 Todas las intenciones:")
        for idx, prob in enumerate(probs):
            intent_name = id_to_intent[idx]
            print(f"   - {intent_name}: {prob*100:.1f}%")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_simple()
    else:
        predict_with_your_model()