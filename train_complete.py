# train_small_model.py - UN SOLO MODELO PEQUEÑO QUE SÍ EXISTE

import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os
import time

# ==============================================================================
# UN SOLO MODELO PEQUEÑO - 100% GARANTIZADO
# ==============================================================================

MODEL_NAME = "prajjwal1/bert-tiny"  # 17 MB - EL MÁS PEQUEÑO QUE EXISTE

class Config:
    # Modelo
    MODEL_NAME = MODEL_NAME
    MAX_LENGTH = 64
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 10
    
    # MLflow
    MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"

# ==============================================================================
# MODELO
# ==============================================================================

class TinyModel(nn.Module):
    def __init__(self, num_intents):
        super().__init__()
        
        # Modelo de 17 MB
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        
        # Clasificador simple
        self.classifier = nn.Linear(hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

# ==============================================================================
# ENTRENAMIENTO
# ==============================================================================

def train():
    print("="*60)
    print("🏋️‍♂️ ENTRENAMIENTO CON BERT-TINY (17 MB)")
    print("="*60)
    
    # 1. Verificar modelo
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        print(f"✅ Modelo cargado: {Config.MODEL_NAME}")
    except:
        print(f"❌ Error: Modelo no disponible")
        return
    
    # 2. Cargar dataset
    with open('dataset_v2.json', 'r') as f:
        data = json.load(f)
    
    # 3. Preparar datos
    texts = [item['text'] for item in data]
    intents = [item['intent'] for item in data]
    
    unique_intents = sorted(set(intents))
    intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    
    labels = [intent_to_id[intent] for intent in intents]
    
    print(f"\n📊 Dataset: {len(data)} ejemplos")
    print(f"🎯 Intenciones: {unique_intents}")
    
    # 4. Tokenizar
    encodings = tokenizer(
        texts,
        max_length=Config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 5. Dataset
    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f"📦 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 6. Modelo
    model = TinyModel(num_intents=len(unique_intents))
    device = torch.device('cpu')
    model.to(device)
    
    # 7. Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 8. Entrenar
    print(f"\n🔥 Entrenando {Config.EPOCHS} épocas...")
    print("-" * 50)
    
    best_val_acc = 0
    
    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids, attention_mask, batch_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_labels).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        
        # Validation
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'intent_to_id': intent_to_id,
                'id_to_intent': id_to_intent,
                'val_accuracy': val_acc,
                'tokenizer_name': Config.MODEL_NAME,
                'max_length': Config.MAX_LENGTH
            }
            
            torch.save(checkpoint, 'tiny_model.pt')
            print(f"💾 Modelo guardado: tiny_model.pt (Acc: {val_acc:.4f})")
    
    # 9. Guardar final
    print(f"\n✅ Entrenamiento completado!")
    print(f"🎯 Mejor accuracy: {best_val_acc:.4f}")
    print(f"📦 Modelo: tiny_model.pt (~17 MB)")
    
    # 10. Probar
    print(f"\n🧪 Probando modelo...")
    
    model.eval()
    test_texts = [
        "Quiero ver mi información",
        "Noticias de hoy",
        "¿Qué fecha es?",
        "Datos de la empresa"
    ]
    
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
        
        print(f"📝 '{text}' → {id_to_intent[pred_idx]} ({probs[pred_idx]*100:.1f}%)")

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    train()