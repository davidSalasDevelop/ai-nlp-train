#!/usr/bin/env python3
"""
train_final_model.py - Entrenamiento con monitoreo DETALLADO de CPU/RAM cada 10 segundos
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
    
    # MLflow
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
        self.metrics_history = deque(maxlen=1000)  # Guardar historial
        self.batch_count = 0
        self.epoch_count = 0
        self.start_time = time.time()
        
    def get_detailed_metrics(self):
        """Obtener métricas DETALLADAS del sistema"""
        try:
            import psutil
            
            # Obtener proceso actual
            process = psutil.Process()
            
            # ===== CPU DETALLADO =====
            # CPU por núcleo
            cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Estadísticas de CPU
            cpu_times = psutil.cpu_times_percent(interval=0.5)
            
            # ===== MEMORIA DETALLADA =====
            # Memoria del sistema
            system_memory = psutil.virtual_memory()
            
            # Memoria del proceso DETALLADA
            process_memory = process.memory_info()
            process_memory_full = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
            
            # ===== DISCO DETALLADO =====
            disk_io = psutil.disk_io_counters()
            
            # ===== RED DETALLADA =====
            net_io = psutil.net_io_counters()
            
            # ===== PROCESO DETALLADO =====
            process_cpu = process.cpu_percent(interval=0.1)
            process_threads = process.num_threads()
            process_ctx_switches = process.num_ctx_switches()
            
            metrics = {
                # Timestamp
                'timestamp': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                
                # ===== CPU SISTEMA DETALLADO =====
                'system_cpu_percent': cpu_percent,
                'system_cpu_user': cpu_times.user,
                'system_cpu_system': cpu_times.system,
                'system_cpu_idle': cpu_times.idle,
                'system_cpu_iowait': getattr(cpu_times, 'iowait', 0),
                
                # CPU por núcleo (solo primeros 4 para no saturar)
                'system_cpu_core_0': cpu_per_core[0] if len(cpu_per_core) > 0 else 0,
                'system_cpu_core_1': cpu_per_core[1] if len(cpu_per_core) > 1 else 0,
                'system_cpu_core_2': cpu_per_core[2] if len(cpu_per_core) > 2 else 0,
                'system_cpu_core_3': cpu_per_core[3] if len(cpu_per_core) > 3 else 0,
                
                # ===== MEMORIA SISTEMA DETALLADO =====
                'system_memory_percent': system_memory.percent,
                'system_memory_used_gb': system_memory.used / (1024**3),
                'system_memory_available_gb': system_memory.available / (1024**3),
                'system_memory_free_gb': system_memory.free / (1024**3),
                'system_memory_total_gb': system_memory.total / (1024**3),
                'system_memory_cached_gb': system_memory.cached / (1024**3) if hasattr(system_memory, 'cached') else 0,
                
                # ===== MEMORIA PROCESO DETALLADO =====
                'process_memory_rss_mb': process_memory.rss / (1024**2),  # Resident Set Size
                'process_memory_vms_mb': process_memory.vms / (1024**2),  # Virtual Memory Size
                'process_memory_percent': process.memory_percent(),
                'process_memory_shared_mb': process_memory.shared / (1024**2) if hasattr(process_memory, 'shared') else 0,
                
                # Memoria detallada si está disponible
                'process_memory_uss_mb': process_memory_full.uss / (1024**2) if process_memory_full and hasattr(process_memory_full, 'uss') else 0,
                'process_memory_pss_mb': process_memory_full.pss / (1024**2) if process_memory_full and hasattr(process_memory_full, 'pss') else 0,
                
                # ===== PROCESO DETALLADO =====
                'process_cpu_percent': process_cpu,
                'process_threads': process_threads,
                'process_ctx_switches_voluntary': process_ctx_switches.voluntary if hasattr(process_ctx_switches, 'voluntary') else 0,
                'process_ctx_switches_involuntary': process_ctx_switches.involuntary if hasattr(process_ctx_switches, 'involuntary') else 0,
                
                # ===== DISCO =====
                'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                
                # ===== RED =====
                'network_sent_mb': net_io.bytes_sent / (1024**2),
                'network_recv_mb': net_io.bytes_recv / (1024**2),
                
                # ===== CONTEXTO =====
                'batch_count': self.batch_count,
                'epoch_count': self.epoch_count,
                'runtime_seconds': time.time() - self.start_time
            }
            
            # Guardar en historial
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  Error obteniendo métricas detalladas: {e}")
            return None
    
    def increment_batch(self):
        """Incrementar contador de batches"""
        self.batch_count += 1
    
    def increment_epoch(self):
        """Incrementar contador de épocas"""
        self.epoch_count += 1
    
    def get_summary_stats(self):
        """Obtener estadísticas resumen del historial"""
        if not self.metrics_history:
            return {}
        
        # Calcular promedios de las últimas 10 mediciones
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
    """Monitoreo continuo DETALLADO cada 10 segundos"""
    def detailed_monitor():
        print("🔍 Iniciando monitoreo DETALLADO cada 10 segundos...")
        
        last_log_time = time.time()
        measurement_count = 0
        
        while not stop_event.is_set():
            try:
                current_time = time.time()
                
                # Medir cada 10 segundos exactos
                if current_time - last_log_time >= Config.MONITOR_INTERVAL:
                    metrics = monitor.get_detailed_metrics()
                    
                    if metrics and mlflow.active_run():
                        measurement_count += 1
                        
                        # Loggear métricas DETALLADAS
                        mlflow.log_metrics({
                            # CPU Sistema
                            "monitor_cpu_system_percent": metrics["system_cpu_percent"],
                            "monitor_cpu_system_user": metrics["system_cpu_user"],
                            "monitor_cpu_system_system": metrics["system_cpu_system"],
                            
                            # CPU Proceso
                            "monitor_cpu_process_percent": metrics["process_cpu_percent"],
                            
                            # Memoria Sistema
                            "monitor_memory_system_percent": metrics["system_memory_percent"],
                            "monitor_memory_system_used_gb": round(metrics["system_memory_used_gb"], 3),
                            "monitor_memory_system_available_gb": round(metrics["system_memory_available_gb"], 3),
                            
                            # Memoria Proceso DETALLADA
                            "monitor_memory_process_rss_mb": round(metrics["process_memory_rss_mb"], 2),
                            "monitor_memory_process_vms_mb": round(metrics["process_memory_vms_mb"], 2),
                            "monitor_memory_process_percent": round(metrics["process_memory_percent"], 2),
                            
                            # Contexto
                            "monitor_batch_count": metrics["batch_count"],
                            "monitor_epoch_count": metrics["epoch_count"],
                            "monitor_measurement_count": measurement_count,
                            "monitor_runtime_seconds": int(metrics["runtime_seconds"])
                        }, step=measurement_count)
                        
                        print(f"📊 Monitor [{measurement_count}]: "
                              f"CPU={metrics['system_cpu_percent']:.1f}%/{metrics['process_cpu_percent']:.1f}%, "
                              f"RAM={metrics['system_memory_percent']:.1f}%/{metrics['process_memory_rss_mb']:.1f}MB")
                    
                    last_log_time = current_time
                
                # Esperar corto tiempo antes de revisar nuevamente
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  Error en monitoreo detallado: {e}")
                time.sleep(5)
    
    # Iniciar thread
    thread = threading.Thread(target=detailed_monitor, daemon=True)
    thread.start()
    return thread

# ==============================================================================
# MLFLOW SETUP DETALLADO
# ==============================================================================

def setup_mlflow_detailed():
    """Configurar MLflow para métricas detalladas"""
    try:
        print(f"🔧 Configurando MLflow para monitoreo DETALLADO...")
        print(f"   URI: {Config.MLFLOW_TRACKING_URI}")
        print(f"   Usuario: {Config.MLFLOW_USERNAME}")
        print(f"   📊 Intervalo monitoreo: cada {Config.MONITOR_INTERVAL} segundos")
        
        # Construir URI con credenciales
        parsed_url = urllib.parse.urlparse(Config.MLFLOW_TRACKING_URI)
        secure_uri = f"{parsed_url.scheme}://{Config.MLFLOW_USERNAME}:{Config.MLFLOW_PASSWORD}@{parsed_url.netloc}"
        
        mlflow.set_tracking_uri(secure_uri)
        print(f"   ✅ Tracking URI configurada")
        
        # Crear experimento detallado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Intent-Detailed-Monitor-{timestamp}"
        
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
# ENTRENAMIENTO CON MONITOREO DETALLADO
# ==============================================================================

def train():
    print("="*70)
    print("🤖 ENTRENAMIENTO CON MONITOREO DETALLADO CADA 10 SEGUNDOS")
    print("="*70)
    
    # Verificar psutil
    try:
        import psutil
        print(f"✅ psutil instalado: v{psutil.__version__}")
    except ImportError:
        print("❌ psutil no instalado. Instalando...")
        os.system("pip install psutil")
        import psutil
    
    # Inicializar monitor
    monitor = SystemMonitor()
    
    # Limpiar modelo anterior
    if os.path.exists(Config.FINAL_MODEL_NAME):
        os.remove(Config.FINAL_MODEL_NAME)
        print(f"🗑️  Modelo anterior eliminado")
    
    # Setup MLflow detallado
    mlflow_enabled = setup_mlflow_detailed()
    run_id = None
    monitor_thread = None
    stop_event = None
    
    if mlflow_enabled:
        try:
            run_name = f"detailed-monitor-{int(time.time())}"
            mlflow.start_run(run_name=run_name)
            run_info = mlflow.active_run()
            run_id = run_info.info.run_id if run_info else None
            
            print(f"\n📊 MLflow Run DETALLADO:")
            print(f"   Nombre: {run_name}")
            print(f"   ID: {run_id}")
            print(f"   📈 Monitoreo: CADA {Config.MONITOR_INTERVAL} SEGUNDOS")
            
            # Iniciar monitoreo detallado
            if Config.MONITOR_DETAILED:
                stop_event = threading.Event()
                monitor_thread = monitor_detailed_continuously(stop_event, monitor)
                print(f"   🔍 Monitor detallado INICIADO")
            
        except Exception as e:
            print(f"⚠️  No se pudo iniciar run: {e}")
            mlflow_enabled = False
    
    try:
        # Loggear información inicial detallada
        if mlflow_enabled:
            initial_metrics = monitor.get_detailed_metrics()
            if initial_metrics:
                mlflow.log_params({
                    "monitor_interval_seconds": Config.MONITOR_INTERVAL,
                    "system_cpu_cores": psutil.cpu_count(),
                    "system_memory_total_gb": round(initial_metrics["system_memory_total_gb"], 2),
                    "system_platform": sys.platform,
                    "python_version": sys.version,
                    "torch_version": torch.__version__,
                    "transformers_version": "4.30.0"  # Versión actual
                })
                print(f"📋 Parámetros del sistema registrados")
        
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
        
        print(f"\n📊 DATASET DETALLADO:")
        print(f"   • Total ejemplos: {len(data):,}")
        print(f"   • Intenciones únicas: {len(unique_intents)}")
        print(f"   • Distribución:")
        for intent in unique_intents:
            count = intents.count(intent)
            percent = (count / len(data)) * 100
            print(f"     - {intent}: {count} ({percent:.1f}%)")
        
        # Loggear parámetros de entrenamiento
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
                "start_time": datetime.now().isoformat(),
                "monitoring_type": "detailed_10s",
                "log_batch_metrics": Config.LOG_BATCH_METRICS
            }
            
            mlflow.log_params(params)
            print(f"📋 {len(params)} parámetros registrados")
            
            mlflow.set_tag("intents", ", ".join(unique_intents))
            mlflow.set_tag("model", "bert-tiny-detailed")
            mlflow.set_tag("monitoring", "detailed_10s")
        
        # 4. Tokenizar
        print(f"\n🔤 Tokenizando {len(texts)} textos...")
        start_tokenize = time.time()
        encodings = tokenizer(
            texts,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tokenize_time = time.time() - start_tokenize
        print(f"   ✅ Tokenizado en {tokenize_time:.1f} segundos")
        
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
        
        print(f"\n📦 DIVISIÓN DE DATOS:")
        print(f"   • Train: {len(train_dataset):,} ejemplos")
        print(f"   • Validation: {len(val_dataset):,} ejemplos")
        print(f"   • Batches/train: {len(train_loader)}")
        print(f"   • Batches/val: {len(val_loader)}")
        
        # 6. Crear modelo
        print(f"\n🧠 Creando modelo...")
        model = TinyModel(num_intents=len(unique_intents))
        device = torch.device('cpu')
        model.to(device)
        
        # 7. Optimizador
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 8. ENTRENAMIENTO DETALLADO
        print(f"\n🔥 INICIANDO ENTRENAMIENTO DETALLADO ({Config.EPOCHS} épocas)")
        print("="*70)
        
        best_val_acc = 0
        epoch_stats = []
        global_batch_counter = 0
        
        for epoch in range(Config.EPOCHS):
            monitor.increment_epoch()
            epoch_start_time = time.time()
            
            print(f"\n📈 ÉPOCA {epoch+1}/{Config.EPOCHS}")
            print(f"   {'='*50}")
            
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
                
                # Calcular métricas
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == batch_labels).sum().item()
                
                # Tiempo del batch
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Incrementar contador de batches
                monitor.increment_batch()
                global_batch_counter += 1
                
                # Loggear métricas por batch (cada X batches)
                if Config.LOG_BATCH_METRICS and mlflow_enabled and (batch_idx + 1) % Config.LOG_PER_BATCH_INTERVAL == 0:
                    batch_metrics = monitor.get_detailed_metrics()
                    if batch_metrics:
                        mlflow.log_metrics({
                            "batch_loss": loss.item(),
                            "batch_time": batch_time,
                            "batch_memory_mb": batch_metrics["process_memory_rss_mb"],
                            "batch_cpu": batch_metrics["process_cpu_percent"],
                            "global_batch": global_batch_counter
                        }, step=global_batch_counter)
                
                # Mostrar progreso del batch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                    current_metrics = monitor.get_detailed_metrics()
                    
                    print(f"   Batch {batch_idx + 1:3d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Time: {batch_time:.2f}s | "
                          f"CPU: {current_metrics['process_cpu_percent'] if current_metrics else 0:.1f}% | "
                          f"RAM: {current_metrics['process_memory_rss_mb'] if current_metrics else 0:.0f}MB")
            
            # Calcular métricas de training
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_dataset)
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            # Validation
            model.eval()
            val_correct = 0
            val_loss = 0
            
            print(f"   {'─'*50}")
            print(f"   Validando...")
            
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
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / len(val_dataset)
            epoch_time = time.time() - epoch_start_time
            
            # Obtener estadísticas del sistema para esta época
            epoch_metrics = monitor.get_detailed_metrics()
            summary = monitor.get_summary_stats()
            
            # Guardar estadísticas de la época
            epoch_stat = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_acc,
                'val_loss': avg_val_loss,
                'val_accuracy': val_acc,
                'epoch_time': epoch_time,
                'avg_batch_time': avg_batch_time,
                'cpu_avg': summary.get('avg_cpu_percent', 0),
                'memory_avg_mb': summary.get('avg_process_memory_mb', 0),
                'memory_max_mb': summary.get('max_process_memory_mb', 0),
                'batches_processed': monitor.batch_count
            }
            epoch_stats.append(epoch_stat)
            
            # Mostrar resultados DETALLADOS de la época
            print(f"\n   📊 ÉPOCA {epoch+1} COMPLETADA")
            print(f"   {'─'*50}")
            print(f"   • Train Loss: {avg_train_loss:.4f}")
            print(f"   • Train Accuracy: {train_acc:.4f} ({train_correct}/{len(train_dataset)})")
            print(f"   • Val Loss: {avg_val_loss:.4f}")
            print(f"   • Val Accuracy: {val_acc:.4f} ({val_correct}/{len(val_dataset)})")
            print(f"   • Tiempo época: {epoch_time:.1f}s")
            print(f"   • Tiempo/batch: {avg_batch_time:.3f}s")
            print(f"   • Sistema - CPU: {summary.get('avg_cpu_percent', 0):.1f}% avg")
            print(f"   • Sistema - RAM: {summary.get('avg_process_memory_mb', 0):.0f}MB avg, "
                  f"{summary.get('max_process_memory_mb', 0):.0f}MB max")
            print(f"   • Batches totales: {monitor.batch_count}")
            
            # Loggear métricas de época a MLflow
            if mlflow_enabled:
                mlflow.log_metrics({
                    "epoch_train_loss": avg_train_loss,
                    "epoch_train_accuracy": train_acc,
                    "epoch_val_loss": avg_val_loss,
                    "epoch_val_accuracy": val_acc,
                    "epoch_time": epoch_time,
                    "epoch_avg_batch_time": avg_batch_time,
                    "epoch_cpu_avg": summary.get('avg_cpu_percent', 0),
                    "epoch_memory_avg_mb": summary.get('avg_process_memory_mb', 0),
                    "epoch_memory_max_mb": summary.get('max_process_memory_mb', 0),
                    "epoch_batches": monitor.batch_count
                }, step=epoch)
            
            # Guardar MEJOR modelo
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
                    'config': {
                        'batch_size': Config.BATCH_SIZE,
                        'learning_rate': Config.LEARNING_RATE,
                        'epochs': Config.EPOCHS,
                        'num_intents': len(unique_intents)
                    },
                    'system_stats': summary,
                    'epoch_stats': epoch_stats,
                    'monitoring_interval': Config.MONITOR_INTERVAL,
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(checkpoint, Config.FINAL_MODEL_NAME)
                print(f"\n   💾 NUEVO MEJOR MODELO GUARDADO!")
                print(f"   • Archivo: {Config.FINAL_MODEL_NAME}")
                print(f"   • Accuracy: {val_acc:.4f}")
                
                if mlflow_enabled:
                    mlflow.log_metric("best_val_accuracy", best_val_acc)
            
            print(f"   {'='*50}")
        
        # 9. RESULTADOS FINALES DETALLADOS
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO!")
        print(f"{'='*70}")
        
        # Estadísticas finales
        total_time = time.time() - monitor.start_time
        final_summary = monitor.get_summary_stats()
        
        print(f"\n📈 RESUMEN FINAL DETALLADO:")
        print(f"   {'─'*50}")
        print(f"   • Mejor Accuracy: {best_val_acc:.4f}")
        print(f"   • Total épocas: {Config.EPOCHS}")
        print(f"   • Total batches: {monitor.batch_count}")
        print(f"   • Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"   • Modelo final: {Config.FINAL_MODEL_NAME}")
        print(f"   • Tamaño modelo: {os.path.getsize(Config.FINAL_MODEL_NAME) / (1024**2):.1f} MB")
        
        print(f"\n   📊 ESTADÍSTICAS DEL SISTEMA:")
        print(f"   • CPU promedio: {final_summary.get('avg_cpu_percent', 0):.1f}%")
        print(f"   • CPU proceso: {final_summary.get('avg_process_cpu', 0):.1f}%")
        print(f"   • RAM sistema: {final_summary.get('avg_memory_percent', 0):.1f}%")
        print(f"   • RAM proceso: {final_summary.get('avg_process_memory_mb', 0):.0f}MB avg")
        print(f"   • RAM proceso max: {final_summary.get('max_process_memory_mb', 0):.0f}MB")
        print(f"   • Mediciones tomadas: {len(monitor.metrics_history)}")
        
        # Loggear métricas finales a MLflow
        if mlflow_enabled:
            final_metrics = {
                "final_train_accuracy": train_acc,
                "final_val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "total_training_time": total_time,
                "total_batches": monitor.batch_count,
                "total_epochs": Config.EPOCHS,
                "final_cpu_avg": final_summary.get('avg_cpu_percent', 0),
                "final_memory_avg_mb": final_summary.get('avg_process_memory_mb', 0),
                "final_memory_max_mb": final_summary.get('max_process_memory_mb', 0),
                "total_monitor_measurements": len(monitor.metrics_history)
            }
            
            mlflow.log_metrics(final_metrics)
            
            # Guardar estadísticas detalladas como JSON
            stats_file = "training_detailed_stats.json"
            with open(stats_file, 'w') as f:
                json.dump({
                    'epoch_stats': epoch_stats,
                    'system_summary': final_summary,
                    'monitor_history': list(monitor.metrics_history)[-100],  # Últimas 100 mediciones
                    'config': {
                        'monitor_interval': Config.MONITOR_INTERVAL,
                        'log_batch_metrics': Config.LOG_BATCH_METRICS
                    }
                }, f, indent=2)
            
            mlflow.log_artifact(stats_file)
            os.remove(stats_file)
            
            mlflow.set_tag("final_accuracy", f"{best_val_acc:.4f}")
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("total_time", f"{total_time:.1f}s")
            mlflow.set_tag("total_batches", str(monitor.batch_count))
            mlflow.set_tag("monitoring_measurements", str(len(monitor.metrics_history)))
        
        # 10. PROBAR MODELO FINAL
        print(f"\n🧪 PROBANDO MODELO FINAL...")
        print(f"{'─'*50}")
        
        # Cargar modelo final
        checkpoint = torch.load(Config.FINAL_MODEL_NAME, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        test_cases = [
            "Quiero ver mi información personal",
            "¿Qué noticias hay hoy?",
            "¿Cuál es la fecha actual?",
            "Necesito datos de la empresa",
            "Muestra mis datos de usuario",
            "Últimas noticias del día"
        ]
        
        for i, text in enumerate(test_cases, 1):
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
            print(f"   {i}. '{text[:40]}...'")
            print(f"      → {intent} ({confidence:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"🎉 ¡ENTRENAMIENTO CON MONITOREO DETALLADO COMPLETADO!")
        print(f"{'='*70}")
        
        if mlflow_enabled and run_id:
            print(f"\n📊 MLflow Dashboard DETALLADO:")
            print(f"   • URL: {Config.MLFLOW_TRACKING_URI}")
            print(f"   • Run ID: {run_id}")
            print(f"   • System Metrics: CADA {Config.MONITOR_INTERVAL} segundos ✓")
            print(f"   • Batch Metrics: {Config.LOG_BATCH_METRICS} ✓")
            print(f"   • Mediciones tomadas: {len(monitor.metrics_history)}")
            print(f"\n   📈 MIRA ESTAS GRÁFICAS:")
            print(f"   • CPU Usage Over Time")
            print(f"   • Memory Usage Over Time")
            print(f"   • Process Memory Over Time")
            print(f"   • Batch Loss Over Time")
            print(f"   • Epoch Accuracy Over Time")
        
        print(f"\n✨ ¡Monitoreo detallado completado exitosamente!")
        
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
        # Detener monitoreo
        if Config.MONITOR_DETAILED and stop_event:
            stop_event.set()
            print(f"\n🔍 Monitoreo detallado detenido")
            print(f"   • Total mediciones: {len(monitor.metrics_history)}")
        
        # Cerrar MLflow
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.end_run()
                print(f"📊 MLflow Run cerrado")
            except Exception as e:
                print(f"⚠️  Error cerrando MLflow: {e}")
        
        # Limpiar
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
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO CON MONITOREO DETALLADO CADA 10 SEGUNDOS")
    print("="*70)
    
    # Instalar psutil si es necesario
    try:
        import psutil
        print(f"✅ psutil v{psutil.__version__} instalado")
    except ImportError:
        print("📦 Instalando psutil para monitoreo detallado...")
        os.system("pip install psutil -q")
        import psutil
        print(f"✅ psutil instalado")
    
    start_time = time.time()
    
    try:
        train()
        
        total_time = time.time() - start_time
        print(f"\n⏱️  TIEMPO TOTAL DEL PROCESO: {total_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")