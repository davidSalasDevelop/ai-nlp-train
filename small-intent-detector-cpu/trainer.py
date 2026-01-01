#!/usr/bin/env python3
"""
train_final_model.py - Entrenamiento con monitoreo DETALLADO SIN UPLOAD DE ARTIFACTS
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
from collections import deque

# ==============================================================================
# CONFIGURACIÓN DETALLADA
# ==============================================================================

MODEL_NAME = "prajjwal1/bert-tiny"

class Config:
    # Modelo
    MODEL_NAME = MODEL_NAME
    MAX_LENGTH = 64
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    EPOCHS = 10
    
    # MLflow (SIN ARTIFACTS - SOLO MÉTRICAS)
    MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
    MLFLOW_USERNAME = "dsalasmlflow"
    MLFLOW_PASSWORD = "SALASdavidTECHmlFlow45542344"
    
    # System Monitoring DETALLADO
    MONITOR_DETAILED = True
    MONITOR_INTERVAL = 10  # segundos entre mediciones DETALLADAS
    LOG_BATCH_METRICS = True  # Loggear métricas por batch también
    LOG_PER_BATCH_INTERVAL = 5  # Cada cuantos batches loggear sistema
    
    # Output
    FINAL_MODEL_NAME = "small-intent-detector-cpu/intent_classifier_final.pt"

# ==============================================================================
# MONITOREO DETALLADO DEL SISTEMA
# ==============================================================================

class SystemMonitor:
    """Monitor detallado del sistema"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.batch_count = 0
        self.epoch_count = 0
        self.start_time = time.time()
        
    def get_detailed_metrics(self):
        """Obtener métricas DETALLADAS del sistema"""
        try:
            import psutil
            
            process = psutil.Process()
            
            # CPU
            cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_times = psutil.cpu_times_percent(interval=0.5)
            
            # Memoria
            system_memory = psutil.virtual_memory()
            process_memory = process.memory_info()
            
            metrics = {
                'timestamp': time.time(),
                
                # CPU Sistema
                'system_cpu_percent': cpu_percent,
                'system_cpu_user': cpu_times.user,
                'system_cpu_system': cpu_times.system,
                'system_cpu_core_0': cpu_per_core[0] if len(cpu_per_core) > 0 else 0,
                'system_cpu_core_1': cpu_per_core[1] if len(cpu_per_core) > 1 else 0,
                
                # Memoria Sistema
                'system_memory_percent': system_memory.percent,
                'system_memory_used_gb': system_memory.used / (1024**3),
                'system_memory_available_gb': system_memory.available / (1024**3),
                
                # Memoria Proceso
                'process_memory_rss_mb': process_memory.rss / (1024**2),
                'process_memory_percent': process.memory_percent(),
                'process_cpu_percent': process.cpu_percent(interval=0.1),
                
                # Contexto
                'batch_count': self.batch_count,
                'epoch_count': self.epoch_count,
                'runtime_seconds': time.time() - self.start_time
            }
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error obteniendo métricas: {e}")
            return None
    
    def increment_batch(self):
        self.batch_count += 1
    
    def increment_epoch(self):
        self.epoch_count += 1
    
    def get_summary_stats(self):
        """Obtener estadísticas resumen"""
        if not self.metrics_history:
            return {}
        
        recent = list(self.metrics_history)[-10:] if len(self.metrics_history) >= 10 else list(self.metrics_history)
        
        if not recent:
            return {}
        
        summary = {
            'avg_cpu_percent': sum(m.get('system_cpu_percent', 0) for m in recent) / len(recent),
            'avg_process_cpu': sum(m.get('process_cpu_percent', 0) for m in recent) / len(recent),
            'avg_memory_percent': sum(m.get('system_memory_percent', 0) for m in recent) / len(recent),
            'avg_process_memory_mb': sum(m.get('process_memory_rss_mb', 0) for m in recent) / len(recent),
            'max_process_memory_mb': max(m.get('process_memory_rss_mb', 0) for m in recent),
            'total_batches': self.batch_count,
            'total_epochs': self.epoch_count,
            'total_runtime': time.time() - self.start_time
        }
        
        return summary

def monitor_detailed_continuously(stop_event, monitor):
    """Monitoreo continuo SIN ARTIFACTS"""
    def detailed_monitor():
        print("🔍 Monitoreo DETALLADO iniciado...")
        
        last_log_time = time.time()
        measurement_count = 0
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                
                if current_time - last_log_time >= Config.MONITOR_INTERVAL:
                    metrics = monitor.get_detailed_metrics()
                    
                    if metrics and mlflow.active_run():
                        measurement_count += 1
                        
                        # SOLO MÉTRICAS - SIN ARTIFACTS
                        try:
                            mlflow.log_metrics({
                                "monitor_cpu_system": metrics["system_cpu_percent"],
                                "monitor_cpu_process": metrics["process_cpu_percent"],
                                "monitor_memory_system": metrics["system_memory_percent"],
                                "monitor_memory_process_mb": round(metrics["process_memory_rss_mb"], 2),
                                "monitor_batch_count": metrics["batch_count"],
                                "monitor_measurement": measurement_count
                            }, step=measurement_count)
                            
                            if measurement_count % 6 == 0:  # Cada minuto
                                print(f"📊 Monitor [{measurement_count}]: "
                                      f"CPU={metrics['system_cpu_percent']:.1f}%/{metrics['process_cpu_percent']:.1f}%, "
                                      f"RAM={metrics['system_memory_percent']:.1f}%/{metrics['process_memory_rss_mb']:.1f}MB")
                            
                        except Exception as e:
                            print(f"⚠️  Error loggeando métricas (continuando): {e}")
                    
                    last_log_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Error en monitoreo: {e}")
                time.sleep(5)
    
    thread = threading.Thread(target=detailed_monitor, daemon=True)
    thread.start()
    return thread

# ==============================================================================
# MLFLOW SETUP SIN ARTIFACTS
# ==============================================================================

def setup_mlflow_no_artifacts():
    """Configurar MLflow SIN soporte para artifacts"""
    try:
        print(f"🔧 Configurando MLflow (SIN artifacts)...")
        print(f"   URI: {Config.MLFLOW_TRACKING_URI}")
        print(f"   Usuario: {Config.MLFLOW_USERNAME}")
        
        # Deshabilitar S3/Artifacts completamente
        os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9999'  # URL inválida
        os.environ['AWS_ACCESS_KEY_ID'] = 'dummy'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'dummy'
        
        # Construir URI con credenciales
        parsed_url = urllib.parse.urlparse(Config.MLFLOW_TRACKING_URI)
        secure_uri = f"{parsed_url.scheme}://{Config.MLFLOW_USERNAME}:{Config.MLFLOW_PASSWORD}@{parsed_url.netloc}"
        
        mlflow.set_tracking_uri(secure_uri)
        print(f"   ✅ Tracking URI configurada")
        
        # Crear experimento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Intent-Detailed-Monitor-{timestamp}"
        
        print(f"   📁 Experimento: {experiment_name}")
        
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            print(f"   ✅ Experimento creado")
        except Exception as e:
            print(f"   ⚠️  Usando experimento existente")
            mlflow.set_experiment(experiment_name)
        
        print(f"   📊 System Metrics: CADA {Config.MONITOR_INTERVAL} segundos")
        print(f"   ⚠️  Artifacts: DESHABILITADOS (solo métricas)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configurando MLflow: {e}")
        print(f"⚠️  Continuando sin MLflow...")
        return False

# ==============================================================================
# FUNCIONES SEGURAS PARA MLFLOW (SIN ARTIFACTS)
# ==============================================================================

def safe_mlflow_log(func, *args, **kwargs):
    """Log seguro a MLflow - omite errores de artifacts"""
    try:
        if mlflow.active_run():
            return func(*args, **kwargs)
        return None
    except Exception as e:
        error_msg = str(e)
        if "credentials" in error_msg.lower() or "s3" in error_msg.lower():
            # Ignorar errores de S3/credentials
            return None
        else:
            print(f"⚠️  Error MLflow: {error_msg[:100]}...")
            return None

def log_metrics_safe(metrics, step=None):
    """Log metrics de forma segura"""
    return safe_mlflow_log(mlflow.log_metrics, metrics, step=step)

def log_params_safe(params):
    """Log params de forma segura"""
    return safe_mlflow_log(mlflow.log_params, params)

def set_tag_safe(key, value):
    """Set tag de forma segura"""
    return safe_mlflow_log(mlflow.set_tag, key, value)

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
    print("="*70)
    print("🤖 ENTRENAMIENTO CON MONITOREO DETALLADO")
    print("="*70)
    
    # Verificar psutil
    try:
        import psutil
        print(f"✅ psutil v{psutil.__version__} instalado")
    except ImportError:
        print("📦 Instalando psutil...")
        os.system("pip install psutil -q")
        import psutil
    
    # Inicializar monitor
    monitor = SystemMonitor()
    
    # Limpiar modelo anterior
    if os.path.exists(Config.FINAL_MODEL_NAME):
        os.remove(Config.FINAL_MODEL_NAME)
        print(f"🗑️  Modelo anterior eliminado")
    
    # Setup MLflow SIN artifacts
    mlflow_enabled = setup_mlflow_no_artifacts()
    run_id = None
    monitor_thread = None
    stop_event = None
    
    if mlflow_enabled:
        try:
            run_name = f"detailed-monitor-{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            run_info = mlflow.active_run()
            run_id = run_info.info.run_id if run_info else None
            
            print(f"\n📊 MLflow Run:")
            print(f"   Nombre: {run_name}")
            print(f"   ID: {run_id}")
            
            if Config.MONITOR_DETAILED:
                stop_event = threading.Event()
                monitor_thread = monitor_detailed_continuously(stop_event, monitor)
                print(f"   🔍 Monitor detallado INICIADO")
            
        except Exception as e:
            print(f"⚠️  No se pudo iniciar run: {e}")
            mlflow_enabled = False
    
    try:
        # Loggear información inicial
        if mlflow_enabled:
            import psutil
            initial_params = {
                "monitor_interval": Config.MONITOR_INTERVAL,
                "system_cpu_cores": psutil.cpu_count(),
                "system_memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch_version": torch.__version__,
                "start_time": datetime.now().isoformat()
            }
            log_params_safe(initial_params)
        
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
        
        # Loggear parámetros
        if mlflow_enabled:
            params = {
                "model_name": Config.MODEL_NAME,
                "max_length": Config.MAX_LENGTH,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
                "epochs": Config.EPOCHS,
                "num_intents": len(unique_intents),
                "dataset_size": len(data)
            }
            log_params_safe(params)
            set_tag_safe("intents", ", ".join(unique_intents))
        
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
        
        print(f"📦 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
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
        epoch_stats = []
        
        for epoch in range(Config.EPOCHS):
            monitor.increment_epoch()
            epoch_start_time = time.time()
            
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            batch_times = []
            
            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()
                
                optimizer.zero_grad()
                
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == batch_labels).sum().item()
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                monitor.increment_batch()
                
                # Log batch metrics (cada X batches)
                if (batch_idx + 1) % Config.LOG_PER_BATCH_INTERVAL == 0 and mlflow_enabled:
                    batch_metrics = monitor.get_detailed_metrics()
                    if batch_metrics:
                        log_metrics_safe({
                            "batch_loss": loss.item(),
                            "batch_memory_mb": batch_metrics["process_memory_rss_mb"]
                        })
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_dataset)
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
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
            epoch_time = time.time() - epoch_start_time
            
            # Obtener estadísticas
            summary = monitor.get_summary_stats()
            
            # Mostrar progreso
            print(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"CPU: {summary.get('avg_cpu_percent', 0):.1f}% | "
                  f"RAM: {summary.get('avg_process_memory_mb', 0):.0f}MB")
            
            # Log epoch metrics
            if mlflow_enabled:
                log_metrics_safe({
                    "epoch_train_loss": avg_train_loss,
                    "epoch_train_accuracy": train_acc,
                    "epoch_val_loss": avg_val_loss,
                    "epoch_val_accuracy": val_acc,
                    "epoch_time": epoch_time,
                    "epoch_cpu_avg": summary.get('avg_cpu_percent', 0),
                    "epoch_memory_avg_mb": summary.get('avg_process_memory_mb', 0)
                }, step=epoch)
            
            # Guardar MEJOR modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
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
                    'system_stats': summary,
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(checkpoint, Config.FINAL_MODEL_NAME)
                print(f"💾 Modelo guardado (Acc: {val_acc:.4f})")
                
                if mlflow_enabled:
                    log_metrics_safe({"best_val_accuracy": best_val_acc})
        
        # 9. RESULTADOS FINALES
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*60}")
        
        final_summary = monitor.get_summary_stats()
        
        print(f"\n📊 RESULTADOS:")
        print(f"   • Mejor Accuracy: {best_val_acc:.4f}")
        print(f"   • Épocas: {Config.EPOCHS}")
        print(f"   • Batches: {monitor.batch_count}")
        print(f"   • CPU promedio: {final_summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"   • RAM promedio: {final_summary.get('avg_process_memory_mb', 0):.0f}MB")
        print(f"   • Mediciones: {len(monitor.metrics_history)}")
        
        # Log final metrics
        if mlflow_enabled:
            log_metrics_safe({
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "final_cpu_avg": final_summary.get('avg_cpu_percent', 0),
                "final_memory_avg_mb": final_summary.get('avg_process_memory_mb', 0),
                "total_measurements": len(monitor.metrics_history)
            })
            
            set_tag_safe("final_accuracy", f"{best_val_acc:.4f}")
            set_tag_safe("status", "completed")
        
        # 10. PROBAR MODELO
        print(f"\n🧪 Probando modelo...")
        
        checkpoint = torch.load(Config.FINAL_MODEL_NAME, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_cases = ["Quiero ver mi información", "Noticias de hoy"]
        for text in test_cases:
            encoding = tokenizer(text, max_length=Config.MAX_LENGTH, 
                               padding='max_length', truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                logits = model(encoding['input_ids'], encoding['attention_mask'])
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
            
            print(f"📝 '{text}' → {id_to_intent[pred_idx]} ({probs[pred_idx].item()*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print(f"🎉 ¡PROCESO COMPLETADO!")
        print(f"{'='*60}")
        
        if mlflow_enabled and run_id:
            print(f"\n📊 MLflow Dashboard:")
            print(f"   • URL: {Config.MLFLOW_TRACKING_URI}")
            print(f"   • Run ID: {run_id}")
            print(f"   • Métricas registradas: ✓")
            print(f"   • System metrics: CADA {Config.MONITOR_INTERVAL}s")
        
        print(f"\n✨ ¡Entrenamiento exitoso!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido")
        if mlflow_enabled:
            set_tag_safe("status", "interrupted")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if mlflow_enabled:
            set_tag_safe("status", "failed")
        
    finally:
        # Detener monitoreo
        if Config.MONITOR_DETAILED and stop_event:
            stop_event.set()
            print(f"\n🔍 Monitoreo detenido")
        
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
    print("🚀 ENTRENAMIENTO CON MONITOREO DETALLADO")
    print("="*60)
    
    start_time = time.time()
    
    try:
        train()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {total_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")