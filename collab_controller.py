# ==============================================================================
# Celda 1: Instalar la librería de MLflow
# ==============================================================================
!pip install mlflow -q
print("✅ MLflow instalado en el entorno de Colab.")

# ==============================================================================
# Celda 2: Configuración de Conexión y Parámetros del Proyecto
# Rellena las variables de esta celda con tus datos.
# ==============================================================================
import os
import mlflow

# --- 1. La dirección de tu "Chef" (Servidor MLflow) ---
# ¡¡CAMBIA ESTO!! Pon la IP pública y puerto de tu servidor MLflow.
MLFLOW_TRACKING_URI = "http://143.198.244.48:4200"

# --- 2. Dónde encontrar el "Libro de Recetas" (Tu Repositorio de Git) ---
# ¡¡CAMBIA ESTO!! Pon la URL HTTPS de tu repositorio de GitHub.
PROJECT_URI = "https://github.com/tu_usuario/tu_repositorio.git"

# --- 3. Las credenciales de tu "Almacén" (Servidor MinIO) ---
# Estas se enviarán al servidor para que el "Ayudante de Cocina" pueda acceder a los ingredientes.
MINIO_ENDPOINT_URL = "http://143.198.244.48:4202"
MINIO_ACCESS_KEY = "mlflow_storage_admin"
MINIO_SECRET_KEY = "P@ssw0rd_St0r@g3_2025!"

# --- 4. "Ingredientes Extra" para la receta (Parámetros de Entrenamiento) ---
# Puedes cambiar estos valores para experimentar.
TRAINING_PARAMETERS = {
    "num_epochs": 10,       # Un número bajo para una prueba rápida
    "learning_rate": 0.0001
}

# --- 5. Apuntar el mando a distancia al Chef ---
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI

print(f"✅ Configuración lista. Se apuntará al servidor MLflow en: {MLFLOW_TRACKING_URI}")
print(f"✅ Se usará el código del repositorio: {PROJECT_URI}")



# ==============================================================================
# Celda 3: Enviar la Orden de Entrenamiento al Servidor
# ==============================================================================

print(f"🚀 Enviando orden para ejecutar el proyecto '{PROJECT_URI}' en el servidor...")
print(f"   Parámetros que se enviarán: {TRAINING_PARAMETERS}")

# Preparamos las credenciales de MinIO para enviarlas de forma segura al trabajo remoto.
backend_config = {
    "ENV_VARS": {
        "MINIO_ENDPOINT_URL": MINIO_ENDPOINT_URL,
        "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
        "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    }
}

try:
    # --- ¡LA ORDEN! ---
    # Le decimos a MLflow que ejecute el punto de entrada "main" del proyecto
    # que se encuentra en la URI de Git.
    submitted_run = mlflow.run(
        uri=PROJECT_URI,
        entry_point="main",
        parameters=TRAINING_PARAMETERS,
        backend="local", # Esto le dice a MLflow que ejecute el trabajo en la misma máquina del servidor.
        backend_config=backend_config
    )

    print("\n---")
    print("✅ ¡Orden enviada al servidor con éxito!")
    print("👀 Ahora puedes ir a la interfaz web de MLflow para monitorear el progreso del 'Run'.")

    # Esta parte espera a que el trabajo en el servidor termine.
    # Si el entrenamiento es muy largo, puedes detener la ejecución de esta celda
    # y el entrenamiento seguirá corriendo en tu servidor.
    print("\n⏳ Esperando a que el trabajo remoto finalice para mostrar el estado final...")
    run_status = submitted_run.wait()
    if run_status:
        final_status = submitted_run.get_status()
        print(f"🎉 ¡El trabajo remoto ha finalizado con estado: {final_status}!")
    else:
        print("🔴 El trabajo remoto falló o no se pudo obtener su estado final. Revisa los logs en la UI de MLflow.")

except Exception as e:
    print(f"\n❌ Ocurrió un error al intentar lanzar el trabajo remoto. Revisa la configuración y los logs.")
    print(f"   Error detallado: {e}")
    
