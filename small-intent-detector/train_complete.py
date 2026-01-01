# train_final_model.py - VERSIÓN CON MLFLOW COMPLETO

import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import os
import time
import mlflow
import urllib.parse
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN SIMPLE
# ==============================================================================

MODEL_NAME = "prajjwal1/bert-tiny"

class Config:
    # Modelo
    MODEL_NAME = MODEL_NAME
    MAX_LENGTH = 64
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 10
    
    # MLflow
    MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
    MLFLOW_USERNAME = "dsalasmlflow"
    MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"
    
    # Output
    FINAL_MODEL_NAME = "intent_classifier_final.pt"

# ==============================================================================
# MODELO SIMPLE
# ==============================================================================

class TinyModel(nn.Module):
    def __init__(self, num_intents):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_intents)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

# ==============================================================================
# MLFLOW COMPLETO
# ==============================================================================

def setup_mlflow():
    """Configuración completa de MLflow con autenticación"""
    try:
        print(f"🔧 Configurando MLflow...")
        print(f"   URI: {Config.MLFLOW_TRACKING_URI}")
        print(f"   Usuario: {Config.MLFLOW_USERNAME}")
        
        # Construir URI con credenciales
        parsed_url = urllib.parse.urlparse(Config.MLFLOW_TRACKING_URI)
        secure_uri = f"{parsed_url.scheme}://{Config.MLFLOW_USERNAME}:{Config.MLFLOW_PASSWORD}@{parsed_url.netloc}"
        
        mlflow.set_tracking_uri(secure_uri)
        print(f"   ✅ Tracking URI configurada")
        
        # Crear experimento único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Intent-Classifier-{timestamp}"
        
        print(f"   📁 Experimento: {experiment_name}")
        
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"   ✅ Experimento creado")
        except Exception as e:
            print(f"   ⚠️  Usando experimento existente: {e}")
            mlflow.set_experiment(experiment_name)
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando MLflow: {e}")
        print(f"⚠️  Continuando sin MLflow...")
        return False

def log_to_mlflow(func, *args, **kwargs):
    """Función segura para logging a MLflow"""
    try:
        if mlflow.active_run():
            return func(*args, **kwargs)
        return None
    except Exception as e:
        print(f"⚠️  Error en MLflow logging: {e}")
        return None

# ==============================================================================
# ENTRENAMIENTO - CON MLFLOW COMPLETO
# ==============================================================================

def train():
    print("="*60)
    print("🤖 ENTRENANDO MODELO FINAL DE INTENCIONES")
    print("="*60)
    
    # Limpiar modelo anterior si existe
    if os.path.exists(Config.FINAL_MODEL_NAME):
        os.remove(Config.FINAL_MODEL_NAME)
        print(f"🗑️  Modelo anterior eliminado")
    
    # Setup MLflow
    mlflow_enabled = setup_mlflow()
    run_id = None
    
    if mlflow_enabled:
        try:
            run_name = f"final-model-{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            run_info = mlflow.active_run()
            run_id = run_info.info.run_id if run_info else None
            
            print(f"\n📊 MLflow Run iniciado:")
            print(f"   Nombre: {run_name}")
            print(f"   ID: {run_id}")
        except Exception as e:
            print(f"⚠️  No se pudo iniciar run: {e}")
            mlflow_enabled = False
    
    try:
        # 1. Cargar tokenizer
        print(f"\n📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # 2. Cargar dataset
        print(f"📂 Cargando dataset...")
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
        print(f"🎯 Intenciones: {len(unique_intents)}")
        print(f"📋 Clases: {', '.join(unique_intents)}")
        
        # Log PARÁMETROS a MLflow
        if mlflow_enabled:
            params = {
                "model_name": Config.MODEL_NAME,
                "max_length": Config.MAX_LENGTH,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "epochs": Config.EPOCHS,
                "num_intents": len(unique_intents),
                "dataset_size": len(data),
                "train_split": 0.8,
                "start_time": datetime.now().isoformat()
            }
            
            log_to_mlflow(mlflow.log_params, params)
            print(f"📋 Parámetros registrados en MLflow: {len(params)}")
            
            # Log info de intenciones como tag
            log_to_mlflow(mlflow.set_tag, "intents", ", ".join(unique_intents))
            log_to_mlflow(mlflow.set_tag, "model_type", "bert-tiny")
        
        # 4. Tokenizar
        encodings = tokenizer(
            texts,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 5. Crear datasets
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print(f"📦 División: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # 6. Crear modelo
        print(f"\n🧠 Creando modelo...")
        model = TinyModel(num_intents=len(unique_intents))
        device = torch.device('cpu')
        model.to(device)
        
        # 7. Optimizador
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 8. ENTRENAR
        print(f"\n🔥 Entrenando {Config.EPOCHS} épocas...")
        print("-" * 60)
        
        best_val_acc = 0
        
        for epoch in range(Config.EPOCHS):
            start_time = time.time()
            
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
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_dataset)
            
            # Validation
            model.eval()
            val_correct = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, batch_labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch_labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / len(val_dataset)
            epoch_time = time.time() - start_time
            
            # Mostrar progreso
            print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Log MÉTRICAS a MLflow
            if mlflow_enabled:
                metrics = {
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "epoch_time": epoch_time
                }
                
                log_to_mlflow(mlflow.log_metrics, metrics, step=epoch)
            
            # Guardar MEJOR modelo (solo el mejor)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Crear checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'intent_to_id': intent_to_id,
                    'id_to_intent': id_to_intent,
                    'val_accuracy': val_acc,
                    'tokenizer_name': Config.MODEL_NAME,
                    'max_length': Config.MAX_LENGTH,
                    'config': {
                        'batch_size': Config.BATCH_SIZE,
                        'learning_rate': Config.LEARNING_RATE,
                        'epochs': Config.EPOCHS,
                        'num_intents': len(unique_intents)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Guardar modelo FINAL
                torch.save(checkpoint, Config.FINAL_MODEL_NAME)
                print(f"💾 NUEVO MEJOR MODELO: {Config.FINAL_MODEL_NAME} (Acc: {val_acc:.4f})")
                
                # Log mejor accuracy a MLflow
                if mlflow_enabled:
                    log_to_mlflow(mlflow.log_metric, "best_val_accuracy", best_val_acc)
        
        # 9. RESULTADOS FINALES
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*60}")
        
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"   • Mejor Accuracy: {best_val_acc:.4f}")
        print(f"   • Modelo guardado: {Config.FINAL_MODEL_NAME}")
        print(f"   • Tamaño: {os.path.getsize(Config.FINAL_MODEL_NAME) / (1024*1024):.1f} MB")
        
        # Log métricas finales a MLflow
        if mlflow_enabled:
            final_metrics = {
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "final_train_loss": avg_train_loss,
                "final_val_loss": avg_val_loss,
                "total_epochs": Config.EPOCHS
            }
            
            log_to_mlflow(mlflow.log_metrics, final_metrics)
            
            # Log tags adicionales
            log_to_mlflow(mlflow.set_tag, "final_accuracy", f"{best_val_acc:.4f}")
            log_to_mlflow(mlflow.set_tag, "status", "completed")
        
        # 10. PROBAR MODELO FINAL
        print(f"\n🧪 PROBANDO MODELO FINAL...")
        print("-" * 40)
        
        # Cargar modelo final
        checkpoint = torch.load(Config.FINAL_MODEL_NAME, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_cases = [
            "Quiero ver mi información",
            "Noticias de hoy",
            "¿Qué fecha es?",
            "Datos de la empresa"
        ]
        
        test_results = []
        for text in test_cases:
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
                confidence = probs[pred_idx].item() * 100
            
            intent = id_to_intent[pred_idx]
            print(f"📝 '{text}'")
            print(f"   → {intent} ({confidence:.1f}%)")
            
            test_results.append({
                "text": text,
                "predicted_intent": intent,
                "confidence": confidence
            })
        
        # 11. LIMPIAR TEMPORALES
        print(f"\n🧹 Limpiando archivos temporales...")
        temp_files = []
        for file in os.listdir('.'):
            if (file.endswith('.pt') and file != Config.FINAL_MODEL_NAME) or \
               file.endswith('_local.json') or \
               file.endswith('_temp.json'):
                os.remove(file)
                temp_files.append(file)
        
        if temp_files:
            print(f"🗑️  Eliminados: {', '.join(temp_files)}")
        
        print(f"\n{'='*60}")
        print(f"🎉 ¡MODELO FINAL LISTO PARA USAR!")
        print(f"{'='*60}")
        print(f"\n📦 ARCHIVO FINAL:")
        print(f"   • {Config.FINAL_MODEL_NAME}")
        print(f"   • {os.path.getsize(Config.FINAL_MODEL_NAME) / (1024*1024):.1f} MB")
        print(f"   • Accuracy: {best_val_acc:.4f}")
        
        if mlflow_enabled and run_id:
            print(f"\n📊 MLflow:")
            print(f"   • Run ID: {run_id}")
            print(f"   • Dashboard: {Config.MLFLOW_TRACKING_URI}")
            print(f"   • Parámetros registrados: ✓")
            print(f"   • Métricas registradas: ✓")
            print(f"   • Gráficas disponibles: ✓")
        
        print(f"\n✨ ¡Entrenamiento completado exitosamente!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido")
        
        # Limpiar si existe modelo parcial
        if os.path.exists(Config.FINAL_MODEL_NAME):
            os.remove(Config.FINAL_MODEL_NAME)
        
        # Marcar como interrumpido en MLflow
        if mlflow_enabled:
            log_to_mlflow(mlflow.set_tag, "status", "interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Limpiar si existe modelo parcial
        if os.path.exists(Config.FINAL_MODEL_NAME):
            os.remove(Config.FINAL_MODEL_NAME)
        
        # Marcar como fallido en MLflow
        if mlflow_enabled:
            log_to_mlflow(mlflow.set_tag, "status", "failed")
        
    finally:
        # Cerrar MLflow
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.end_run()
                print(f"📊 MLflow Run cerrado")
            except Exception as e:
                print(f"⚠️  Error cerrando MLflow: {e}")

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ENTRENANDO MODELO FINAL DE INTENCIONES")
    print("="*60)
    
    start_time = time.time()
    
    try:
        train()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")