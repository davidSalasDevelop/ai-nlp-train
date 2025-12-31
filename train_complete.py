# train_complete.py - SISTEMA COMPLETO DE ENTRENAMIENTO

import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW
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
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 15
    
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

class IntentAnalyzer:
    """Analiza el dataset y extrae configuración"""
    
    @staticmethod
    def analyze_dataset(dataset_path: str) -> Dict:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraer intenciones
        intents = list(set([item['intent'] for item in data]))
        
        # Extraer entidades por intención
        intent_entities = {}
        for intent in intents:
            intent_entities[intent] = set()
        
        for item in data:
            intent = item['intent']
            for entity in item['entities']:
                intent_entities[intent].add(entity['label'])
        
        # Estadísticas
        stats = {
            "total_examples": len(data),
            "intents": intents,
            "intent_distribution": {intent: 0 for intent in intents},
            "languages": set(),
            "intent_entities": intent_entities
        }
        
        for item in data:
            stats["intent_distribution"][item['intent']] += 1
            stats["languages"].add(item['language'])
        
        return stats
    
    @staticmethod
    def create_intent_mapping(intents: List[str]) -> Tuple[Dict, Dict]:
        intent_to_id = {intent: i for i, intent in enumerate(sorted(intents))}
        id_to_intent = {i: intent for intent, i in intent_to_id.items()}
        return intent_to_id, id_to_intent
    
    @staticmethod
    def create_entity_mapping() -> Tuple[Dict, Dict]:
        # Crear mapeo completo de entidades
        entity_to_id = {'O': 0}
        entity_id = 1
        
        # Agregar todas las entidades por tipo
        for intent, entities in Config.ENTITY_TYPES.items():
            for entity in entities:
                entity_to_id[f'B-{entity}'] = entity_id
                entity_id += 1
                entity_to_id[f'I-{entity}'] = entity_id
                entity_id += 1
        
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        return entity_to_id, id_to_entity

# ==============================================================================
# DATASET Y PREPROCESAMIENTO
# ==============================================================================

class NLUDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, intent_to_id: Dict, entity_to_id: Dict):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
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
        
        # Entidades
        entity_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        # Mapear entidades a tokens
        if item['entities']:
            token_positions = self.tokenizer(
                item['text'],
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True
            )['offset_mapping']
            
            for entity in item['entities']:
                entity_start = entity['start']
                entity_end = entity['end']
                entity_type = entity['label']
                
                is_first = True
                for token_idx, (token_start, token_end) in enumerate(token_positions):
                    if token_start >= entity_start and token_end <= entity_end:
                        if is_first:
                            entity_labels[token_idx] = self.entity_to_id.get(f'B-{entity_type}', 0)
                            is_first = False
                        else:
                            entity_labels[token_idx] = self.entity_to_id.get(f'I-{entity_type}', 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'intent_labels': intent_label,
            'entity_labels': entity_labels
        }

# ==============================================================================
# MODELO COMPUESTO
# ==============================================================================

class JointIntentEntityModel(nn.Module):
    """Modelo que predice intención y entidades simultáneamente"""
    
    def __init__(self, num_intents: int, num_entities: int):
        super().__init__()
        
        # Backbone pre-entrenado
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Congelar primeras capas para fine-tuning eficiente
        for param in list(self.bert.parameters())[:50]:
            param.requires_grad = False
        
        # Clasificador de intenciones
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(bert_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_intents)
        )
        
        # Clasificador de entidades (por token)
        self.entity_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(bert_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_entities)
        )
        
    def forward(self, input_ids, attention_mask):
        # Obtener embeddings de BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Última capa oculta
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]
        
        # Clasificación de intención (usando [CLS])
        intent_logits = self.intent_classifier(pooled_output)
        
        # Clasificación de entidades (por token)
        entity_logits = self.entity_classifier(last_hidden_state)
        
        return intent_logits, entity_logits

# ==============================================================================
# ENTRENADOR
# ==============================================================================

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignorar padding
        
    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Mover a dispositivo
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            entity_labels = batch['entity_labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            intent_logits, entity_logits = self.model(input_ids, attention_mask)
            
            # Calcular pérdidas
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            entity_loss = self.entity_loss_fn(
                entity_logits.view(-1, entity_logits.size(-1)),
                entity_labels.view(-1)
            )
            
            # Pérdida combinada (ponderar intención más)
            loss = intent_loss + 0.3 * entity_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            intent_preds = torch.argmax(intent_logits, dim=1)
            intent_correct += (intent_preds == intent_labels).sum().item()
            total_samples += intent_labels.size(0)
            
            # Log cada 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader), intent_correct / total_samples
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        intent_correct = 0
        total_samples = 0
        all_intent_preds = []
        all_intent_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                intent_labels = batch['intent_labels'].to(self.device)
                entity_labels = batch['entity_labels'].to(self.device)
                
                intent_logits, entity_logits = self.model(input_ids, attention_mask)
                
                intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
                entity_loss = self.entity_loss_fn(
                    entity_logits.view(-1, entity_logits.size(-1)),
                    entity_labels.view(-1)
                )
                loss = intent_loss + 0.3 * entity_loss
                
                total_loss += loss.item()
                intent_preds = torch.argmax(intent_logits, dim=1)
                intent_correct += (intent_preds == intent_labels).sum().item()
                total_samples += intent_labels.size(0)
                
                all_intent_preds.extend(intent_preds.cpu().numpy())
                all_intent_labels.extend(intent_labels.cpu().numpy())
        
        return (total_loss / len(dataloader), 
                intent_correct / total_samples,
                all_intent_preds, all_intent_labels)

# ==============================================================================
# FUNCIONES PARA AÑADIR NUEVAS INTENCIONES
# ==============================================================================

def add_intent_to_config(intent_name: str, entity_types: List[str]):
    """Añade una nueva intención a la configuración"""
    Config.ENTITY_TYPES[intent_name] = entity_types
    
    # Guardar configuración actualizada
    config_path = "model_config_extended.json"
    config = {
        "entity_types": Config.ENTITY_TYPES,
        "model_name": Config.MODEL_NAME,
        "max_length": Config.MAX_LENGTH
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Intención '{intent_name}' añadida con entidades: {entity_types}")
    print(f"💾 Configuración guardada en {config_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento completo NLU")
    parser.add_argument('--dataset', type=str, default='dataset_v2.json', help='Path al dataset')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--add-intent', action='store_true', help='Añadir nueva intención')
    
    args = parser.parse_args()
    
    if args.add_intent:
        # Modo: añadir nueva intención
        print("\n➕ AÑADIR NUEVA INTENCIÓN")
        intent_name = input("Nombre de la intención: ").strip()
        
        print("🔧 Entidades a extraer (separadas por coma):")
        entities_input = input("Ej: PRODUCTO, PRECIO, CATEGORÍA: ").strip()
        entity_types = [e.strip().upper() for e in entities_input.split(",") if e.strip()]
        
        add_intent_to_config(intent_name, entity_types)
        
        print(f"\n🎯 Ahora puedes regenerar el dataset con la nueva intención:")
        print(f"   python generate_dataset.py --size 600")
        
        return
    
    print("="*60)
    print("🏋️‍♂️ ENTRENAMIENTO COMPLETO NLU")
    print("="*60)
    
    # 1. Analizar dataset
    print("\n📊 Analizando dataset...")
    analyzer = IntentAnalyzer()
    stats = analyzer.analyze_dataset(args.dataset)
    
    print(f"\n📈 Estadísticas:")
    print(f"   Total ejemplos: {stats['total_examples']}")
    print(f"   Intenciones: {stats['intents']}")
    print(f"   Idiomas: {stats['languages']}")
    
    print(f"\n📊 Distribución por intención:")
    for intent, count in stats['intent_distribution'].items():
        print(f"   {intent}: {count} ejemplos ({count/stats['total_examples']*100:.1f}%)")
    
    # 2. Crear mapeos
    intent_to_id, id_to_intent = analyzer.create_intent_mapping(stats['intents'])
    entity_to_id, id_to_entity = analyzer.create_entity_mapping()
    
    print(f"\n🎯 Mapeos creados:")
    print(f"   Intenciones: {intent_to_id}")
    print(f"   Entidades únicas: {len(entity_to_id) - 1}")  # Excluir 'O'
    
    # 3. Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 4. Crear dataset y dataloaders
    print("\n📦 Preparando datos...")
    full_dataset = NLUDataset(args.dataset, tokenizer, intent_to_id, entity_to_id)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"   Train: {len(train_dataset)} ejemplos")
    print(f"   Validation: {len(val_dataset)} ejemplos")
    
    # 5. Crear modelo
    print("\n🤖 Creando modelo...")
    model = JointIntentEntityModel(
        num_intents=len(intent_to_id),
        num_entities=len(entity_to_id)
    )
    
    # 6. Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Dispositivo: {device}")
    
    # 7. Optimizador
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    # 8. Entrenador
    trainer = Trainer(model, device)
    
    # 9. Configurar MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_TRACKING_USERNAME'] = Config.MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = Config.MLFLOW_PASSWORD
    
    mlflow.set_experiment("nlu-complete-training")
    
    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            'model': Config.MODEL_NAME,
            'dataset_size': stats['total_examples'],
            'num_intents': len(intent_to_id),
            'num_entities': len(entity_to_id),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': Config.LEARNING_RATE
        })
        
        mlflow.log_dict(intent_to_id, "intent_to_id.json")
        mlflow.log_dict(entity_to_id, "entity_to_id.json")
        
        print(f"\n🔥 Comenzando entrenamiento...")
        print("-" * 60)
        
        best_val_acc = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 40)
            
            # Entrenamiento
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch)
            
            # Validación
            val_loss, val_acc, val_preds, val_labels = trainer.validate(val_loader)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Guardar checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'intent_to_id': intent_to_id,
                    'id_to_intent': id_to_intent,
                    'entity_to_id': entity_to_id,
                    'id_to_entity': id_to_entity,
                    'val_accuracy': val_acc,
                    'config': {
                        'model_name': Config.MODEL_NAME,
                        'max_length': Config.MAX_LENGTH
                    }
                }
                
                torch.save(checkpoint, 'best_model.pt')
                mlflow.log_artifact('best_model.pt')
        
        # 10. Evaluación final
        print(f"\n📊 Evaluación final...")
        
        # Classification report
        from sklearn.metrics import classification_report
        
        # Convertir IDs a nombres
        val_preds_names = [id_to_intent[p] for p in val_preds]
        val_labels_names = [id_to_intent[l] for l in val_labels]
        
        report = classification_report(val_labels_names, val_preds_names, output_dict=True)
        
        print("\n📈 Classification Report:")
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
            'tokenizer_config': tokenizer.init_kwargs
        }
        
        torch.save(final_checkpoint, 'nlu_complete_model.pt')
        
        # Log final model
        mlflow.log_artifact('nlu_complete_model.pt')
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"🎯 Mejor val accuracy: {best_val_acc:.4f}")
        print(f"📦 Modelo guardado: nlu_complete_model.pt")
        print(f"🎯 Intenciones soportadas: {list(intent_to_id.keys())}")
        print(f"🏷️  Entidades extraíbles: {list(entity_to_id.keys())[1:10]}...")  # Mostrar primeras 10

if __name__ == "__main__":
    main()