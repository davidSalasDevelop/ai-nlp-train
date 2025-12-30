# ==============================================================================
#             TEST SIMPLE DE CONEXIÓN MLFLOW - EXPERIMENTO DEFAULT
# ==============================================================================
import os
import mlflow
import sys

# --- Paso 1: Configuración básica ---
print("🚀 INICIANDO TEST DE CONEXIÓN MLFLOW")
print("="*60)

# Configurar servidor MLflow
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"✅ Tracking URI configurado: {MLFLOW_TRACKING_URI}")

# Credenciales (si el servidor las requiere)
os.environ['MLFLOW_TRACKING_USERNAME'] = "dsalasmlflow"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "SALASdavidTECHmlFlow45542344"

# --- Paso 2: Verificar conexión ---
print("\n🔍 Verificando conexión al servidor...")
try:
    # Usar el experimento por defecto (ID: 0)
    default_experiment = mlflow.get_experiment("0")
    print(f"✅ Conectado al servidor MLflow")
    print(f"   Experiment default: {default_experiment.name}")
    print(f"   Experiment ID: {default_experiment.experiment_id}")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    print("   Verifica que el servidor esté corriendo en:")
    print(f"   {MLFLOW_TRACKING_URI}")
    exit(1)

# --- Paso 3: Ejecutar test en EXPERIMENTO DEFAULT ---
print("\n" + "="*60)
print("📊 CREANDO RUN EN EXPERIMENTO DEFAULT")
print("="*60)

# Configurar explícitamente el experimento default (ID: 0)
mlflow.set_experiment("0")

# Iniciar un run simple en el experimento default
with mlflow.start_run(
    run_name="test-default-experiment",
    experiment_id="0"  # Forzar experimento default
) as run:
    print(f"\n🎯 Run ID: {run.info.run_id}")
    print(f"📈 Experiment: Default (ID: 0)")
    print(f"📍 Ubicación: {run.info.artifact_uri}")
    
    # Loggear algunas métricas básicas
    mlflow.log_metric("test_checkpoint", 1.0)
    mlflow.log_metric("accuracy_test", 0.87)
    mlflow.log_metric("loss_test", 0.13)
    
    # Loggear algunos parámetros
    mlflow.log_param("test_type", "default_experiment_test")
    mlflow.log_param("python_version", sys.version.split()[0])
    mlflow.log_param("mlflow_version", mlflow.__version__)
    
    # Loggear tags
    mlflow.set_tag("environment", "testing")
    mlflow.set_tag("author", "dsalas")
    mlflow.set_tag("purpose", "connection_test")
    mlflow.set_tag("experiment_type", "default")
    mlflow.set_tag("status", "success")
    
    print(f"\n✅ Métricas registradas en Default:")
    print(f"   - test_checkpoint: 1.0")
    print(f"   - accuracy_test: 0.87")
    print(f"   - loss_test: 0.13")
    
    print(f"\n✅ Parámetros registrados:")
    print(f"   - test_type: default_experiment_test")
    print(f"   - python_version: {sys.version.split()[0]}")
    print(f"   - mlflow_version: {mlflow.__version__}")
    
    # URL directa al run
    run_url = f"http://143.198.244.48:4200/#/experiments/0/runs/{run.info.run_id}"
    print(f"\n🔗 Run URL: {run_url}")

# --- Paso 4: Verificar que se guardó en Default ---
print("\n" + "="*60)
print("🔎 VERIFICANDO RUN EN EXPERIMENTO DEFAULT")
print("="*60)

try:
    # Buscar runs específicamente en el experimento default (ID: 0)
    runs = mlflow.search_runs(
        experiment_ids=["0"],  # Solo experimento default
        filter_string=f"tags.mlflow.runName = 'test-default-experiment'",
        max_results=1
    )
    
    if len(runs) > 0:
        run_data = runs.iloc[0]
        print(f"✅ Run guardado exitosamente en Default Experiment")
        print(f"   📍 Run ID: {run_data['run_id']}")
        print(f"   📅 Start Time: {run_data['start_time']}")
        print(f"   🔖 Status: {run_data['status']}")
        
        # Mostrar métricas guardadas
        print(f"\n   📊 Métricas almacenadas:")
        if 'metrics.test_checkpoint' in run_data:
            print(f"      - test_checkpoint: {run_data['metrics.test_checkpoint']}")
        if 'metrics.accuracy_test' in run_data:
            print(f"      - accuracy_test: {run_data['metrics.accuracy_test']}")
        
        # Mostrar parámetros
        print(f"\n   ⚙️  Parámetros almacenados:")
        if 'params.test_type' in run_data:
            print(f"      - test_type: {run_data['params.test_type']}")
            
    else:
        print("⚠️  Run no encontrado en Default Experiment")
        
except Exception as e:
    print(f"⚠️  Error al verificar run: {e}")

# --- Paso 5: Mostrar información del experimento default ---
print("\n" + "="*60)
print("📋 INFORMACIÓN DEL EXPERIMENTO DEFAULT")
print("="*60)

try:
    # Obtener todos los runs del experimento default
    all_default_runs = mlflow.search_runs(experiment_ids=["0"])
    print(f"📈 Total runs en Default Experiment: {len(all_default_runs)}")
    
    # Contar por estado
    if len(all_default_runs) > 0:
        status_counts = all_default_runs['status'].value_counts()
        print(f"\n📊 Distribución por estado:")
        for status, count in status_counts.items():
            print(f"   {status}: {count} runs")
    
    print(f"\n🔗 URL del experimento default:")
    print(f"   http://143.198.244.48:4200/#/experiments/0")
    
except Exception as e:
    print(f"⚠️  Error al obtener info del experimento: {e}")

print("\n" + "="*60)
print("🏁 TEST COMPLETADO - TODO EN EXPERIMENTO DEFAULT")
print("="*60)
print("\n✅ RESULTADO FINAL:")
print("   1. ✅ Conexión establecida con servidor MLflow")
print("   2. ✅ Run creado en EXPERIMENTO DEFAULT (ID: 0)")
print("   3. ✅ Métricas y parámetros guardados")
print(f"   4. ✅ Puedes verlo en: http://143.198.244.48:4200/#/experiments/0")
print(f"   5. ✅ Run específico: {run_url}")