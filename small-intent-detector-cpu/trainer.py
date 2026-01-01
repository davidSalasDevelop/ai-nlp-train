#!/usr/bin/env python3
"""
train_final_model.py - Entrenamiento con monitoreo de CPU y RAM en MLflow
"""

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
import threading
import sys

# ==============================================================================
# CONFIGURACIÓN
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
    
    # System Monitoring
    MONITOR_CPU_RAM = True
    MONITOR_INTERVAL = 15  # segundos entre mediciones
    
    # Output
    FINAL_MODEL_NAME = "intent_classifier_final.pt"

# ==============================================================================
# MONITOREO DE SISTEMA (CPU y RAM)
# ==============================================================================

def get_system_metrics():
    """Obtener métricas de CPU y RAM"""
    try:
        # Intentar importar psutil
        import psutil
        
        # CPU Metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)  # Convertir a GB
        memory_total_gb = memory.total / (1024**3)
        
        # Process-specific metrics
        process = psutil.Process()
        process_cpu = process.cpu_percent(interval=0.1)
        process_memory_mb = process.memory_info().rss / (1024**2)  # MB
        
        return {
            # CPU
            "system_cpu_percent": cpu_percent,
            "system_cpu_count": cpu_count,
            
            # Memory System
            "system_memory_percent": memory_percent,
            "system_memory_used_gb": round(memory_used_gb, 2),
            "system_memory_total_gb": round(memory_total_gb, 2),
            
            # Process
            "process_cpu_percent": process_cpu,
            "process_memory_mb": round(process_memory_mb, 2),
            
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        print("⚠️  psutil no instalado. Usando métricas básicas.")
        return None
    except Exception as e:
        print(f"⚠️  Error obteniendo métricas del sistema: {e}")
        return None

def monitor_system_continuously(stop_event):
    """Monitorear sistema en segundo plano"""
    def monitor():
        print("🔍 Iniciando monitoreo de CPU y RAM...")
        
        while not stop_event.is_set():
            try:
                metrics = get_system_metrics()
                if metrics and mlflow.active_run():
                    # Loggear métricas en MLflow
                    mlflow.log_metrics({
                        "cpu_usage_percent": metrics["system_cpu_percent"],
                        "memory_usage_percent": metrics["system_memory_percent"],
                        "process_memory_mb": metrics["process_memory_mb"]
                    })
                
                # Esperar antes de la siguiente medición
                time.sleep(Config.MONITOR_INTERVAL)
                
            except Exception as e:
                print(f"⚠️  Error en monitoreo del sistema: {e}")
                time.sleep(5)
    
    # Iniciar thread
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

# ==============================================================================
# MLFLOW SETUP
# ==============================================================================

def setup_mlflow():
    """Configurar MLflow con métricas del sistema"""
    try:
        print(f"🔧 Configurando MLflow...")
        print(f"   URI: {Config.MLFLOW_TRACKING_URI}")
        print(f"   Usuario: {Config.MLFLOW_USERNAME}")
        
        # Construir URI con credenciales
        parsed_url = urllib.parse.urlparse(Config.MLFLOW_TRACKING_URI)
        secure_uri = f"{parsed_url.scheme}://{Config.MLFLOW_USERNAME}:{Config.MLFLOW_PASSWORD}@{parsed_url.netloc}"
        
        mlflow.set_tracking_uri(secure_uri)
        print(f"   ✅ Tracking URI configurada")
        
        # Configurar monitoreo
        if Config.MONITOR_CPU_RAM:
            print(f"   📊 System Metrics: CPU y RAM")
            print(f"   ⏱️  Intervalo: cada {Config.MONITOR_INTERVAL} segundos")
        
        # Crear experimento único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Intent-Classifier-CPU-RAM-{timestamp}"
        
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

# ==============================================================================
# MODELO
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
# ENTRENAMIENTO PRINCIPAL
# ==============================================================================

def train():
    print("="*60)
    print("🤖 ENTRENANDO MODELO CON MONITOREO DE CPU/RAM")
    print("="*60)
    
    # Verificar si psutil está instalado
    try:
        import psutil
        print(f"✅ psutil instalado: v{psutil.__version__}")
    except ImportError:
        print("❌ psutil no está instalado")
        print("💡 Instálalo con: pip install psutil")
        Config.MONITOR_CPU_RAM = False
    
    # Limpiar modelo anterior
    if os.path.exists(Config.FINAL_MODEL_NAME):
        os.remove(Config.FINAL_MODEL_NAME)
        print(f"🗑️  Modelo anterior eliminado")
    
    # Setup MLflow
    mlflow_enabled = setup_mlflow()
    run_id = None
    monitor_thread = None
    stop_event = None
    
    if mlflow_enabled:
        try:
            run_name = f"final-model-cpu-ram-{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            run_info = mlflow.active_run()
            run_id = run_info.info.run_id if run_info else None
            
            print(f"\n📊 MLflow Run:")
            print(f"   Nombre: {run_name}")
            print(f"   ID: {run_id}")
            
            # Iniciar monitoreo del sistema
            if Config.MONITOR_CPU_RAM:
                stop_event = threading.Event()
                monitor_thread = monitor_system_continuously(stop_event)
                print(f"   📈 System Monitoring iniciado")
            
        except Exception as e:
            print(f"⚠️  No se pudo iniciar run: {e}")
            mlflow_enabled = False
    
    try:
        # 1. Cargar tokenizer
        print(f"\n📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # 2. Cargar dataset
        print(f"📂 Cargando dataset...")
        with open('small-intent-detector-cpu/dataset_v2.json', 'r') as f:
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
        
        # Log PARÁMETROS iniciales a MLflow
        if mlflow_enabled:
            # Información del sistema inicial
            initial_metrics = get_system_metrics()
            if initial_metrics:
                system_params = {
                    "system_cpu_count": initial_metrics.get("system_cpu_count", 0),
                    "system_memory_total_gb": initial_metrics.get("system_memory_total_gb", 0),
                    "system_platform": sys.platform,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}"
                }
                mlflow.log_params(system_params)
            
            # Parámetros del modelo
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
            
            mlflow.log_params(params)
            print(f"📋 Parámetros registrados: {len(params)}")
            
            mlflow.set_tag("intents", ", ".join(unique_intents))
            mlflow.set_tag("model_type", "bert-tiny")
            mlflow.set_tag("monitoring", "cpu_ram")
        
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
        epoch_times = []
        
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
            epoch_times.append(epoch_time)
            
            # Mostrar progreso
            print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Log MÉTRICAS de entrenamiento
            if mlflow_enabled:
                metrics = {
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "epoch_time": epoch_time
                }
                
                mlflow.log_metrics(metrics, step=epoch)
            
            # Guardar MEJOR modelo
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
                    'system_info': get_system_metrics() if Config.MONITOR_CPU_RAM else {},
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(checkpoint, Config.FINAL_MODEL_NAME)
                print(f"💾 NUEVO MEJOR MODELO: {Config.FINAL_MODEL_NAME} (Acc: {val_acc:.4f})")
                
                if mlflow_enabled:
                    mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # 9. RESULTADOS FINALES
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*60}")
        
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        
        print(f"\n📊 RESULTADOS:")
        print(f"   • Mejor Accuracy: {best_val_acc:.4f}")
        print(f"   • Tiempo promedio/época: {avg_epoch_time:.1f}s")
        print(f"   • Tiempo total: {sum(epoch_times):.1f}s")
        print(f"   • Modelo final: {Config.FINAL_MODEL_NAME}")
        
        # Log métricas finales
        if mlflow_enabled:
            final_metrics = {
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "avg_epoch_time": avg_epoch_time,
                "total_training_time": sum(epoch_times)
            }
            
            mlflow.log_metrics(final_metrics)
            
            # Log sistema final
            if Config.MONITOR_CPU_RAM:
                final_sys_metrics = get_system_metrics()
                if final_sys_metrics:
                    mlflow.log_metrics({
                        "final_cpu_percent": final_sys_metrics.get("system_cpu_percent", 0),
                        "final_memory_percent": final_sys_metrics.get("system_memory_percent", 0)
                    })
            
            mlflow.set_tag("final_accuracy", f"{best_val_acc:.4f}")
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("avg_epoch_time", f"{avg_epoch_time:.1f}s")
        
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
        
        print(f"\n{'='*60}")
        print(f"🎉 ¡PROCESO COMPLETADO CON MONITOREO!")
        print(f"{'='*60}")
        
        if mlflow_enabled and run_id:
            print(f"\n📊 MLflow Dashboard:")
            print(f"   • URL: {Config.MLFLOW_TRACKING_URI}")
            print(f"   • Run ID: {run_id}")
            print(f"   • System Metrics: CPU y RAM ✓")
            print(f"   • Mira la pestaña 'System metrics' para gráficas")
        
        print(f"\n✨ ¡Entrenamiento completado exitosamente!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido")
        
        if mlflow_enabled:
            mlflow.set_tag("status", "interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if mlflow_enabled:
            mlflow.set_tag("status", "failed")
        
    finally:
        # Detener monitoreo del sistema
        if Config.MONITOR_CPU_RAM and stop_event:
            stop_event.set()
            print(f"🔍 Monitoreo de sistema detenido")
        
        # Cerrar MLflow
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.end_run()
                print(f"📊 MLflow Run cerrado")
            except Exception as e:
                print(f"⚠️  Error cerrando MLflow: {e}")
        
        # Limpiar archivos temporales
        for file in os.listdir('.'):
            if file.endswith('.pt') and file != Config.FINAL_MODEL_NAME:
                try:
                    os.remove(file)
                except:
                    pass

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 ENTRENANDO CON MONITOREO DE CPU/RAM")
    print("="*60)
    
    # Verificar e instalar psutil si es necesario
    try:
        import psutil
        print(f"✅ psutil instalado: v{psutil.__version__}")
    except ImportError:
        print("📦 Instalando psutil para monitoreo...")
        os.system("pip install psutil")
        print("✅ psutil instalado")
    
    start_time = time.time()
    
    try:
        train()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")