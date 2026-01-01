# train_small_model.py - UN SOLO MODELO PEQUEÑO QUE SÍ EXISTE CON MLFLOW

import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os
import time
import mlflow
import mlflow.pytorch
import urllib.parse
from datetime import datetime

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
    MLFLOW_USERNAME = "dsalasmlflow"
    MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"
    MLFLOW_EXPERIMENT_NAME = "BERT-Tiny-Intent-Classification-v2"  # NUEVO NOMBRE

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
# MLFLOW SETUP - VERSIÓN ROBUSTA
# ==============================================================================

def setup_mlflow():
    """Configure MLflow tracking with authentication"""
    print(f"🔧 Configurando MLflow...")
    print(f"   URI: {Config.MLFLOW_TRACKING_URI}")
    print(f"   Usuario: {Config.MLFLOW_USERNAME}")
    
    try:
        # Construir URI con credenciales
        parsed_url = urllib.parse.urlparse(Config.MLFLOW_TRACKING_URI)
        secure_uri = f"{parsed_url.scheme}://{Config.MLFLOW_USERNAME}:{Config.MLFLOW_PASSWORD}@{parsed_url.netloc}{parsed_url.path}"
        
        print(f"   URI segura: {parsed_url.scheme}://{Config.MLFLOW_USERNAME}:******@{parsed_url.netloc}")
        
        # Set tracking URI
        mlflow.set_tracking_uri(secure_uri)
        
        # Verificar conexión
        print(f"   Probando conexión a MLflow...")
        try:
            # Listar experimentos para verificar conexión
            experiments = mlflow.search_experiments()
            print(f"   ✅ Conexión exitosa. Experimentos disponibles: {len(experiments)}")
        except Exception as e:
            print(f"   ⚠️  Advertencia: No se pudo listar experimentos: {e}")
        
        # Crear un nuevo experimento con nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{Config.MLFLOW_EXPERIMENT_NAME}_{timestamp}"
        
        print(f"   Creando experimento: {experiment_name}")
        
        try:
            # Intentar crear experimento nuevo
            experiment_id = mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"   ✅ Experiment creado: {experiment_name} (ID: {experiment_id})")
            Config.MLFLOW_EXPERIMENT_NAME = experiment_name  # Actualizar nombre
        except Exception as e:
            print(f"   ⚠️  Error creando experimento: {e}")
            print(f"   Intentando usar experimento existente...")
            try:
                mlflow.set_experiment(experiment_name)
            except:
                # Usar experimento por defecto
                mlflow.set_experiment("Default")
                Config.MLFLOW_EXPERIMENT_NAME = "Default"
                print(f"   Usando experimento: Default")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando MLflow: {e}")
        print(f"⚠️  Continuando sin MLflow...")
        return False

# ==============================================================================
# MANEJO DE MLFLOW RUN
# ==============================================================================

def start_mlflow_run():
    """Iniciar un run de MLflow de forma robusta"""
    try:
        # Generar nombre único para el run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"bert-tiny-{timestamp}"
        
        # Iniciar run
        mlflow.start_run(run_name=run_name)
        run_info = mlflow.active_run()
        
        if run_info:
            print(f"📊 MLflow Run iniciado:")
            print(f"   Nombre: {run_name}")
            print(f"   Run ID: {run_info.info.run_id}")
            print(f"   Experimento: {Config.MLFLOW_EXPERIMENT_NAME}")
            return run_info.info.run_id
        else:
            print(f"⚠️  No se pudo obtener información del run")
            return None
            
    except Exception as e:
        print(f"⚠️  No se pudo iniciar run de MLflow: {e}")
        return None

def end_mlflow_run(status="FINISHED"):
    """Finalizar run de MLflow de forma segura"""
    try:
        if mlflow.active_run():
            mlflow.end_run(status=status)
            print(f"📊 MLflow Run finalizado con estado: {status}")
    except Exception as e:
        print(f"⚠️  Error finalizando run de MLflow: {e}")

# ==============================================================================
# LOGGING SEGURO
# ==============================================================================

def safe_mlflow_log(func, *args, **kwargs):
    """Función segura para logging a MLflow"""
    try:
        if mlflow.active_run():
            return func(*args, **kwargs)
        return None
    except Exception as e:
        print(f"⚠️  Error en MLflow logging: {e}")
        return None

def log_metrics_safe(metrics, step=None):
    """Log metrics de forma segura"""
    safe_mlflow_log(mlflow.log_metrics, metrics, step=step)

def log_params_safe(params):
    """Log params de forma segura"""
    safe_mlflow_log(mlflow.log_params, params)

def log_artifact_safe(path, artifact_path=None):
    """Log artifact de forma segura"""
    if os.path.exists(path):
        safe_mlflow_log(mlflow.log_artifact, path, artifact_path)
    else:
        print(f"⚠️  Archivo no encontrado: {path}")

# ==============================================================================
# ENTRENAMIENTO PRINCIPAL
# ==============================================================================

def train():
    print("="*60)
    print("🏋️‍♂️ ENTRENAMIENTO CON BERT-TINY (17 MB) Y MLFLOW")
    print("="*60)
    
    # Variables de estado
    mlflow_run_id = None
    mlflow_enabled = False
    
    # Setup MLflow
    mlflow_enabled = setup_mlflow()
    
    # Iniciar run de MLflow si está habilitado
    if mlflow_enabled:
        mlflow_run_id = start_mlflow_run()
        if mlflow_run_id:
            print(f"✅ MLflow habilitado")
        else:
            mlflow_enabled = False
            print(f"⚠️  MLflow deshabilitado (no se pudo iniciar run)")
    
    try:
        # 1. Cargar modelo y tokenizer
        print(f"\n📥 Cargando modelo y tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            print(f"✅ Modelo cargado: {Config.MODEL_NAME}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return
        
        # 2. Cargar dataset
        print(f"📂 Cargando dataset...")
        try:
            with open('dataset_v2.json', 'r') as f:
                data = json.load(f)
            print(f"✅ Dataset cargado: {len(data)} ejemplos")
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return
        
        # 3. Preparar datos
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        unique_intents = sorted(set(intents))
        intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
        id_to_intent = {i: intent for intent, i in intent_to_id.items()}
        
        labels = [intent_to_id[intent] for intent in intents]
        
        print(f"\n📊 RESUMEN DEL DATASET:")
        print(f"   • Total ejemplos: {len(data)}")
        print(f"   • Intenciones únicas: {len(unique_intents)}")
        print(f"   • Lista de intenciones: {', '.join(unique_intents)}")
        
        # Log parameters to MLflow
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
        
        if mlflow_enabled:
            log_params_safe(params)
            
            # Guardar información de intents localmente
            intent_info = {
                "intent_mapping": id_to_intent,
                "intent_counts": {intent: intents.count(intent) for intent in unique_intents},
                "sample_texts": texts[:3]  # Solo primeros 3 para ejemplo
            }
            
            with open("intent_info.json", "w") as f:
                json.dump(intent_info, f, indent=2)
            log_artifact_safe("intent_info.json")
        
        # 4. Tokenizar datos
        print(f"\n🔤 Tokenizando datos...")
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
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        print(f"📦 DIVISIÓN DE DATOS:")
        print(f"   • Training: {len(train_dataset)} ejemplos")
        print(f"   • Validation: {len(val_dataset)} ejemplos")
        
        # 6. Crear modelo
        print(f"\n🧠 Creando modelo...")
        model = TinyModel(num_intents=len(unique_intents))
        device = torch.device('cpu')
        model.to(device)
        print(f"✅ Modelo creado en {device}")
        
        # 7. Configurar optimizador y pérdida
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 8. ENTRENAMIENTO
        print(f"\n🔥 INICIANDO ENTRENAMIENTO ({Config.EPOCHS} épocas)")
        print("="*80)
        
        best_val_acc = 0
        training_history = []
        
        for epoch in range(Config.EPOCHS):
            epoch_start_time = time.time()
            
            # ===== TRAINING =====
            model.train()
            train_loss = 0
            train_correct = 0
            train_batches = 0
            
            print(f"\n📈 Época {epoch+1}/{Config.EPOCHS} - Training...")
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calcular métricas
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == batch_labels).sum().item()
                train_batches += 1
                
                # Mostrar progreso cada 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # Calcular métricas de training
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_dataset)
            
            # ===== VALIDATION =====
            model.eval()
            val_correct = 0
            val_loss = 0
            
            print(f"🧪 Época {epoch+1}/{Config.EPOCHS} - Validation...")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    input_ids, attention_mask, batch_labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch_labels).sum().item()
            
            # Calcular métricas de validation
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / len(val_dataset)
            epoch_time = time.time() - epoch_start_time
            
            # Guardar métricas
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(avg_val_loss),
                "val_accuracy": float(val_acc),
                "epoch_time": float(epoch_time)
            }
            
            training_history.append(epoch_metrics)
            
            # Log a MLflow
            if mlflow_enabled:
                log_metrics_safe({
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc, 
                    "epoch_time": epoch_time
                }, step=epoch)
            
            # Mostrar resultados de la época
            print(f"\n📊 ÉPOCA {epoch+1} COMPLETADA")
            print(f"   • Train Loss: {avg_train_loss:.4f}")
            print(f"   • Train Accuracy: {train_acc:.4f} ({train_correct}/{len(train_dataset)})")
            print(f"   • Val Loss: {avg_val_loss:.4f}")
            print(f"   • Val Accuracy: {val_acc:.4f} ({val_correct}/{len(val_dataset)})")
            print(f"   • Tiempo: {epoch_time:.1f} segundos")
            print(f"   • Mejor Accuracy Actual: {best_val_acc:.4f}")
            
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
                    'max_length': Config.MAX_LENGTH,
                    'config': params,
                    'training_history': training_history
                }
                
                # Guardar localmente
                torch.save(checkpoint, 'tiny_model.pt')
                print(f"💾 ¡NUEVO MEJOR MODELO GUARDADO!")
                print(f"   Archivo: tiny_model.pt")
                print(f"   Accuracy: {val_acc:.4f}")
                
                # Log a MLflow
                if mlflow_enabled:
                    try:
                        log_artifact_safe('tiny_model.pt')
                        log_metrics_safe({"best_val_accuracy": best_val_acc})
                    except Exception as e:
                        print(f"⚠️  No se pudo guardar en MLflow: {e}")
            
            print("-" * 60)
        
        # 9. FINALIZAR ENTRENAMIENTO
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*60}")
        
        # Guardar historial de entrenamiento
        with open("training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
        print(f"📝 Historial guardado: training_history.json")
        
        # Log final a MLflow
        if mlflow_enabled:
            try:
                log_artifact_safe("training_history.json")
                
                # Crear resumen
                summary = {
                    "status": "completed",
                    "best_accuracy": float(best_val_acc),
                    "total_epochs": Config.EPOCHS,
                    "dataset_size": len(data),
                    "model": Config.MODEL_NAME,
                    "intents": unique_intents,
                    "run_id": mlflow_run_id,
                    "experiment": Config.MLFLOW_EXPERIMENT_NAME,
                    "completion_time": datetime.now().isoformat()
                }
                
                with open("training_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                log_artifact_safe("training_summary.json")
                
                # Log métricas finales
                log_metrics_safe({
                    "final_train_accuracy": train_acc,
                    "final_val_accuracy": val_acc,
                    "best_val_accuracy": best_val_acc
                })
                
                print(f"\n📊 MLflow - Run completado exitosamente!")
                print(f"🔗 URL: {Config.MLFLOW_TRACKING_URI}")
                print(f"🔍 Buscar: {Config.MLFLOW_EXPERIMENT_NAME}")
                
            except Exception as e:
                print(f"⚠️  Error finalizando MLflow: {e}")
        
        # 10. PROBAR MODELO
        print(f"\n🧪 PROBANDO MODELO ENTRENADO...")
        print("-" * 40)
        
        # Cargar el mejor modelo guardado
        try:
            checkpoint = torch.load('tiny_model.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modelo cargado desde tiny_model.pt")
        except Exception as e:
            print(f"⚠️  No se pudo cargar el modelo guardado, usando el actual: {e}")
        
        model.eval()
        
        # Textos de prueba
        test_texts = [
            "Quiero ver mi información personal",
            "¿Qué noticias hay hoy?",
            "¿Cuál es la fecha actual?",
            "Necesito datos de la empresa",
            "Muestra mis datos de usuario",
            "Últimas noticias del día"
        ]
        
        print(f"\n📋 PREDICCIONES DE PRUEBA:")
        for text in test_texts:
            # Tokenizar
            encoding = tokenizer(
                text,
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Predecir
            with torch.no_grad():
                logits = model(encoding['input_ids'], encoding['attention_mask'])
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100
            
            print(f"📝 '{text}'")
            print(f"   → {id_to_intent[pred_idx]} ({confidence:.1f}% confianza)")
        
        print(f"\n{'='*60}")
        print(f"🎉 ¡PROCESO FINALIZADO EXITOSAMENTE!")
        print(f"{'='*60}")
        
        # Resumen final
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   • Modelo: {Config.MODEL_NAME}")
        print(f"   • Mejor Accuracy: {best_val_acc:.4f}")
        print(f"   • Épocas completadas: {Config.EPOCHS}")
        print(f"   • Tamaño del dataset: {len(data)} ejemplos")
        print(f"   • Intenciones detectadas: {len(unique_intents)}")
        
        if mlflow_enabled and mlflow_run_id:
            print(f"   • MLflow Run ID: {mlflow_run_id}")
            print(f"   • Experimento MLflow: {Config.MLFLOW_EXPERIMENT_NAME}")
        
        print(f"\n💾 ARCHIVOS GENERADOS:")
        print(f"   • tiny_model.pt - Modelo entrenado")
        print(f"   • training_history.json - Métricas de entrenamiento")
        print(f"   • intent_info.json - Información de intenciones")
        
        print(f"\n✨ ¡Listo para usar! ✨")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido por el usuario")
        
        # Guardar estado actual si es posible
        if training_history:
            with open("training_interrupted.json", "w") as f:
                json.dump({
                    "interrupted_at_epoch": len(training_history),
                    "history": training_history,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"💾 Estado guardado: training_interrupted.json")
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        
        # Guardar información del error
        with open("training_error.json", "w") as f:
            json.dump({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "model": Config.MODEL_NAME,
                "epochs_completed": len(training_history) if 'training_history' in locals() else 0
            }, f, indent=2)
        
    finally:
        # Siempre cerrar el run de MLflow si está abierto
        if mlflow_enabled:
            end_mlflow_run(status="FINISHED")

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELO BERT-TINY")
    print("="*60)
    
    start_time = time.time()
    
    try:
        train()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total del proceso: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
        
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        exit(1)