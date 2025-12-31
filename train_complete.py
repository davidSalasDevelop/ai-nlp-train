# train_complete.py - VERSIÓN CORREGIDA PARA CPU

import json
import torch
import torch.nn as nn
import torch.optim as optim  # Cambiado aquí
from transformers import AutoModel, AutoTokenizer  # Removido AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow
import argparse
from typing import Dict, List, Tuple
import os
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

class Config:
    # Modelo
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MAX_LENGTH = 128
    BATCH_SIZE = 8  # Reducido para CPU
    LEARNING_RATE = 2e-5
    EPOCHS = 10  # Reducido para CPU
    
    # MLflow
    MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
    MLFLOW_USERNAME = "dsalasmlflow"
    MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"
    
    # Entidades
    ENTITY_TYPES = {
        "get_user_info": ["SUBSCRIPTION", "START_DATE", "END_DATE", "PROMOTION", "PAYMENT_METHOD"],
        "get_news": ["TOPIC", "DATE_RANGE", "TAG", "SOURCE", "KEYWORD"],
        "get_date": ["DATE_TYPE", "FORMAT", "TIMEZONE"],
        "get_business_information": ["INFO_TYPE", "DEPARTMENT", "DOCUMENT"]
    }

# ==============================================================================
# CARGAR Y ANALIZAR DATASET
# ==============================================================================

def load_dataset(dataset_path: str):
    """Carga el dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_dataset(data):
    """Analiza estadísticas del dataset"""
    intents = list(set([item['intent'] for item in data]))
    
    stats = {
        "total_examples": len(data),
        "intents": intents,
        "intent_distribution": {intent: 0 for intent in intents},
        "languages": set()
    }
    
    for item in data:
        stats["intent_distribution"][item['intent']] += 1
        stats["languages"].add(item.get('language', 'es'))
    
    return stats

def create_mappings(intents):
    """Crea mapeos de intenciones y entidades"""
    # Mapeo de intenciones
    intent_to_id = {intent: i for i, intent in enumerate(sorted(intents))}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    
    # Mapeo de entidades
    entity_to_id = {'O': 0}
    entity_id = 1
    
    for intent, entities in Config.ENTITY_TYPES.items():
        for entity in entities:
            entity_to_id[f'B-{entity}'] = entity_id
            entity_id += 1
            entity_to_id[f'I-{entity}'] = entity_id
            entity_id += 1
    
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    
    return intent_to_id, id_to_intent, entity_to_id, id_to_entity

# ==============================================================================
# DATASET
# ==============================================================================

class NLUDataset(Dataset):
    def __init__(self, data, tokenizer, intent_to_id, entity_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.entity_to_id = entity_to_id
        self.max_length = Config.MAX_LENGTH
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenizar
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Intención
        intent_label = torch.tensor(self.intent_to_id[item['intent']], dtype=torch.long)
        
        # Entidades (simplificado para CPU)
        entity_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        # Solo procesar entidades si existen
        if item.get('entities'):
            for entity in item['entities']:
                # Encontrar token donde comienza la entidad
                tokens = self.tokenizer.encode(item['text'], add_special_tokens=False)
                if len(tokens) > 0 and len(tokens) < self.max_length:
                    entity_type = entity.get('label', 'O')
                    if entity_type != 'O':
                        entity_labels[1] = self.entity_to_id.get(f'B-{entity_type}', 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'intent_labels': intent_label,
            'entity_labels': entity_labels
        }

# ==============================================================================
# MODELO SIMPLIFICADO PARA CPU
# ==============================================================================

class SimpleIntentModel(nn.Module):
    """Modelo simplificado para CPU"""
    
    def __init__(self, num_intents: int, num_entities: int):
        super().__init__()
        
        # Backbone ligero
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Congelar la mayoría de capas para CPU
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Descongelar última capa
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        # Clasificador simple
        self.intent_classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_intents)
        )
        
        # Clasificador de entidades (opcional)
        self.entity_classifier = nn.Linear(bert_hidden_size, num_entities) if num_entities > 0 else None
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Usar [CLS] para intención
        pooled_output = outputs.last_hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(pooled_output)
        
        # Entidades (opcional)
        entity_logits = None
        if self.entity_classifier:
            entity_logits = self.entity_classifier(outputs.last_hidden_state)
        
        return intent_logits, entity_logits

# ==============================================================================
# ENTRENAMIENTO PARA CPU
# ==============================================================================

def train_model_simple(model, train_loader, val_loader, intent_to_id, device='cpu', epochs=10):
    """Entrenamiento simplificado para CPU"""
    
    # Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)  # Usando torch.optim
    
    # Loss functions
    intent_loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    print(f"\n🔥 Entrenando en {device}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            
            intent_logits, _ = model(input_ids, attention_mask)
            loss = intent_loss_fn(intent_logits, intent_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(intent_logits, dim=1)
            train_correct += (preds == intent_labels).sum().item()
            train_total += intent_labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent_labels = batch['intent_labels'].to(device)
                
                intent_logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(intent_logits, dim=1)
                val_correct += (preds == intent_labels).sum().item()
                val_total += intent_labels.size(0)
        
        val_acc = val_correct / val_total
        
        # Guardar historial
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'intent_to_id': intent_to_id,
                'val_accuracy': val_acc,
                'config': {
                    'model_name': Config.MODEL_NAME,
                    'max_length': Config.MAX_LENGTH
                }
            }
            
            torch.save(checkpoint, 'best_model_cpu.pt')
    
    return history, best_val_acc

# ==============================================================================
# MAIN SIMPLIFICADO
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento NLU para CPU")
    parser.add_argument('--dataset', type=str, default='dataset_v2.json', help='Path al dataset')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    
    args = parser.parse_args()
    
    print("="*60)
    print("🏋️‍♂️ ENTRENAMIENTO NLU - CPU OPTIMIZADO")
    print("="*60)
    
    # 1. Cargar dataset
    print("\n📥 Cargando dataset...")
    data = load_dataset(args.dataset)
    
    # 2. Analizar
    stats = analyze_dataset(data)
    
    print(f"\n📊 Estadísticas:")
    print(f"   Total ejemplos: {stats['total_examples']}")
    print(f"   Intenciones: {stats['intents']}")
    print(f"   Idiomas: {stats['languages']}")
    
    print(f"\n📈 Distribución:")
    for intent, count in stats['intent_distribution'].items():
        print(f"   {intent}: {count} ejemplos")
    
    # 3. Crear mapeos
    intent_to_id, id_to_intent, entity_to_id, id_to_entity = create_mappings(stats['intents'])
    
    print(f"\n🎯 Mapeos creados:")
    print(f"   Intenciones: {list(intent_to_id.keys())}")
    
    # 4. Tokenizer
    print("\n🔧 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 5. Dataset
    print("\n📦 Preparando datos...")
    dataset = NLUDataset(data, tokenizer, intent_to_id, entity_to_id)
    
    # Split simple
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Train: {len(train_dataset)} ejemplos")
    print(f"   Validation: {len(val_dataset)} ejemplos")
    
    # 6. Modelo
    print("\n🤖 Creando modelo optimizado para CPU...")
    model = SimpleIntentModel(
        num_intents=len(intent_to_id),
        num_entities=len(entity_to_id)
    )
    
    # 7. Dispositivo
    device = torch.device('cpu')
    print(f"💻 Dispositivo: {device}")
    
    # 8. Configurar MLflow (opcional)
    try:
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        os.environ['MLFLOW_TRACKING_USERNAME'] = Config.MLFLOW_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = Config.MLFLOW_PASSWORD
        
        mlflow.set_experiment("nlu-cpu-training")
        
        with mlflow.start_run(run_name=f"cpu-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_params({
                'model': Config.MODEL_NAME,
                'dataset_size': stats['total_examples'],
                'num_intents': len(intent_to_id),
                'epochs': args.epochs,
                'batch_size': args.batch_size
            })
            
            mlflow.log_dict(intent_to_id, "intent_to_id.json")
            
            # 9. Entrenar
            history, best_val_acc = train_model_simple(
                model, train_loader, val_loader, intent_to_id, 
                device, args.epochs
            )
            
            # 10. Evaluación final
            print(f"\n📊 Evaluación final...")
            
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    intent_labels = batch['intent_labels'].to(device)
                    
                    intent_logits, _ = model(input_ids, attention_mask)
                    preds = torch.argmax(intent_logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(intent_labels.cpu().numpy())
            
            # Convertir IDs a nombres
            preds_names = [id_to_intent[p] for p in all_preds]
            labels_names = [id_to_intent[l] for l in all_labels]
            
            # Reporte
            report = classification_report(labels_names, preds_names, output_dict=True)
            
            print("\n📈 Resultados por intención:")
            for intent in stats['intents']:
                if intent in report:
                    precision = report[intent]['precision']
                    recall = report[intent]['recall']
                    f1 = report[intent]['f1-score']
                    print(f"   {intent}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            
            # Guardar reporte
            with open('classification_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact('classification_report.json')
            
            # 11. Guardar modelo final
            print(f"\n💾 Guardando modelo final...")
            
            final_checkpoint = {
                'model_state_dict': model.state_dict(),
                'intent_to_id': intent_to_id,
                'id_to_intent': id_to_intent,
                'entity_to_id': entity_to_id,
                'id_to_entity': id_to_entity,
                'tokenizer_name': Config.MODEL_NAME,
                'max_length': Config.MAX_LENGTH,
                'val_accuracy': best_val_acc
            }
            
            torch.save(final_checkpoint, 'nlu_model_cpu.pt')
            
            # Log final model
            mlflow.log_artifact('nlu_model_cpu.pt')
            mlflow.pytorch.log_model(model, "model")
            
            print(f"\n✅ Entrenamiento completado!")
            print(f"🎯 Mejor val accuracy: {best_val_acc:.4f}")
            print(f"📦 Modelo guardado: nlu_model_cpu.pt (~80 MB)")
            print(f"🎯 Intenciones: {list(intent_to_id.keys())}")
            
    except Exception as e:
        print(f"⚠️  Error con MLflow: {e}")
        print("Continuando sin MLflow...")
        
        # Entrenar sin MLflow
        history, best_val_acc = train_model_simple(
            model, train_loader, val_loader, intent_to_id, 
            device, args.epochs
        )
        
        # Guardar modelo
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'intent_to_id': intent_to_id,
            'id_to_intent': id_to_intent,
            'entity_to_id': entity_to_id,
            'id_to_entity': id_to_entity,
            'tokenizer_name': Config.MODEL_NAME
        }
        
        torch.save(final_checkpoint, 'nlu_model_cpu.pt')
        
        print(f"\n✅ Entrenamiento completado sin MLflow!")
        print(f"🎯 Mejor val accuracy: {best_val_acc:.4f}")
        print(f"📦 Modelo guardado: nlu_model_cpu.pt")

# ==============================================================================
# PREDICT SIMPLE
# ==============================================================================

class SimplePredictor:
    """Predictor simple para CPU"""
    
    def __init__(self, model_path='nlu_model_cpu.pt'):
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.intent_to_id = checkpoint['intent_to_id']
        self.id_to_intent = checkpoint['id_to_intent']
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.get('tokenizer_name', Config.MODEL_NAME)
        )
        self.max_length = checkpoint.get('max_length', Config.MAX_LENGTH)
        
        # Crear modelo
        self.model = SimpleIntentModel(
            num_intents=len(self.intent_to_id),
            num_entities=len(checkpoint.get('entity_to_id', {}))
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Modelo cargado: {len(self.intent_to_id)} intenciones")
        print(f"🎯 Intenciones: {list(self.intent_to_id.keys())}")
    
    def predict(self, text):
        """Predice la intención"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            intent_logits, _ = self.model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(intent_logits, dim=1)[0]
        
        # Todas las probabilidades
        results = []
        for idx, prob in enumerate(probs):
            results.append({
                'intent': self.id_to_intent[idx],
                'confidence': f"{prob.item()*100:.1f}%",
                'score': prob.item()
            })
        
        # Ordenar por confianza
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        # Modo predicción
        if len(sys.argv) > 2:
            text = sys.argv[2]
            predictor = SimplePredictor()
            results = predictor.predict(text)
            
            print(f"\n📝 Texto: {text}")
            print(f"🎯 Predicciones:")
            for result in results[:3]:  # Top 3
                print(f"   {result['intent']}: {result['confidence']}")
        else:
            predictor = SimplePredictor()
            
            # Ejemplos de prueba
            test_texts = [
                "Quiero ver mi información de usuario",
                "Noticias sobre tecnología",
                "¿Qué fecha es hoy?",
                "Información de la empresa"
            ]
            
            for text in test_texts:
                results = predictor.predict(text)
                print(f"\n📝 '{text}'")
                print(f"   🎯 {results[0]['intent']} ({results[0]['confidence']})")
    else:
        # Modo entrenamiento
        main()