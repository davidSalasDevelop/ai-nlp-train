---

## **3. La Fábrica de Modelos: Un Vistazo Profundo**

### **`main.py`: El Director de Orquesta**

Este es el script principal que se ejecuta para iniciar el proceso de entrenamiento. No realiza el trabajo pesado directamente, sino que orquesta los diferentes módulos en la secuencia correcta.

*   **Proceso de Ejecución:**
    1.  **Preparación:** Configura la conexión con el servidor de MLflow para el reporte de métricas.
    2.  **Llama al Cocinero:** Invoca a `data_loader.py` para cargar, procesar y preparar los datos.
    3.  **Construye el Cerebro:** Utiliza el diseño definido en `model.py` para instanciar un modelo vacío.
    4.  **Configura la Maquinaria:** Define todos los hiperparámetros de entrenamiento (épocas, tasa de aprendizaje, tamaño de lote, etc.) usando el objeto `TrainingArguments` de Hugging Face.
    5.  **Contrata al Asistente:** Pasa el `SystemMetricsCallback` (definido en `callbacks.py`) al `Trainer` para que el monitoreo de recursos se active durante el entrenamiento.
    6.  **Inicia el Entrenamiento:** Llama al método `.train()` del `Trainer`, que es el motor principal que maneja los bucles de entrenamiento, la evaluación y la optimización.
    7.  **Guarda el Producto Final:** Una vez finalizado el entrenamiento, guarda el mejor modelo obtenido en el formato estándar de Hugging Face y, adicionalmente, crea el archivo autocontenido `intent_classifier.pt`.

### **`config.py`: El Panel de Control Central**

Este archivo centraliza todas las variables de configuración importantes. Es el primer lugar donde debes mirar si necesitas ajustar el comportamiento del pipeline.

*   **Perillas Configurables:**
    *   `MODEL_NAME`: El identificador del modelo base de Hugging Face (ej: `prajjwal1/bert-tiny`). Cambiar esto es como cambiar el motor del sistema.
    *   `DATASET_PATH`: La ruta al archivo JSON con los datos de entrenamiento.
    *   `FINAL_MODEL_OUTPUT_DIR`: La carpeta de destino para el modelo final.
    *   `MLFLOW_...`: Las credenciales y la URI para conectarse al servidor de MLflow.

### **`data_loader.py`: El Cocinero de Datos**

Los modelos de IA no entienden texto. Este módulo se encarga de convertir los datos textuales en un formato numérico que el modelo puede procesar.

*   **Receta de Preparación:**
    1.  **Cargar:** Utiliza la librería `datasets` para leer el archivo `dataset_v2.json` de manera eficiente.
    2.  **Etiquetar:** Asigna un ID numérico único a cada etiqueta de intención (ej: `get_news` se convierte en `0`).
    3.  **Tokenizar:** Proceso clave donde cada frase se descompone en piezas (tokens) y cada token se convierte en un número según el diccionario del `tokenizer`. El resultado es una representación puramente numérica de las frases.
    4.  **Dividir:** Separa los datos en un conjunto de entrenamiento (80%) para el aprendizaje y un conjunto de validación (20%) para evaluar el rendimiento en cada época.

### **`model.py`: El Diseño del Cerebro (La Arquitectura)**

Define la estructura de la red neuronal que estamos entrenando.

*   **Componentes:**
    1.  **`__init__` (Constructor):** Define las dos capas del modelo:
        *   `self.bert`: Un modelo `BERT` pre-entrenado que ya posee un conocimiento profundo del lenguaje. Actúa como la base de nuestro modelo.
        *   `self.classifier`: Una capa lineal simple que se añade encima de BERT. Esta es la única parte que se entrena desde cero para nuestra tarea específica de clasificación de intenciones.
    2.  **`forward` (Flujo de Pensamiento):** Describe cómo fluyen los datos a través del modelo. La entrada numérica pasa por `bert` para obtener una representación contextual y luego por `classifier` para obtener la puntuación final de cada intención.

### **`callbacks.py`: El Asistente de Monitoreo**

El `Trainer` de Hugging Face es eficiente entrenando, pero no monitorea los recursos del sistema por defecto. El `SystemMetricsCallback` es una extensión personalizada que hemos creado para añadir esta funcionalidad.

*   **Funcionamiento:**
    *   El `Trainer` tiene "puntos de control" en su ejecución (ej: al final de cada registro de logs).
    *   El método `on_log` de nuestro callback se activa en cada uno de estos puntos.
    *   Al activarse, utiliza las librerías `psutil` (para CPU/RAM) y `nvidia_smi` (para GPU) para consultar el estado actual del hardware.
    *   Finalmente, envía estas métricas a MLflow, permitiéndonos visualizar gráficos de uso de recursos alineados con el progreso del entrenamiento.

---

## **4. El Servidor de Predicciones: Usando Nuestro Modelo**

### **`predict_model.py`: El Consultor Experto**

Este script es un módulo autocontenido que encapsula toda la lógica para cargar el archivo `.pt` y realizar predicciones. Puede ser importado y utilizado por cualquier otra aplicación.

*   **Habilidades:**
    *   `load_model()`: Su función principal. "Despierta" el cerebro entrenado siguiendo estos pasos:
        1.  Carga el archivo `intent_classifier.pt` de forma segura (con `weights_only=False`).
        2.  Extrae los componentes: los pesos del modelo, la configuración de la arquitectura y el mapeo de `id_to_intent`.
        3.  Reconstruye la arquitectura del modelo (`InferenceModel`) y carga los pesos entrenados en ella.
        4.  Inicializa el `tokenizer` correspondiente.
    *   `predict()`: Con el modelo ya en memoria, esta función toma un texto, lo convierte a formato numérico (tokeniza), lo pasa al modelo y traduce la salida numérica a una intención legible con su respectiva confianza.
    *   `if __name__ == "__main__"`: Proporciona un "modo de prueba". Al ejecutar `python predict_model.py` directamente, carga el modelo y realiza una predicción de ejemplo para verificar que todo funciona correctamente.

### **`api.py`: La Recepcionista Digital**

Este script utiliza el framework **FastAPI** para crear un servidor web. Su rol es exponer la funcionalidad de `predict_model.py` a través de una API HTTP.

*   **Tareas:**
    *   **`lifespan`:** Una función especial de FastAPI. Al iniciar el servidor, invoca a `load_model()` para cargar el modelo en memoria una sola vez, evitando tener que recargarlo en cada petición.
    *   **Endpoints (Puertas de Servicio):**
        *   `@app.get("/health")`: Una ruta simple para verificar que el servidor está activo.
        *   `@app.post("/classify")`: La ruta principal. Acepta peticiones con un texto, se lo pasa a la función `predict()` y devuelve el resultado en formato JSON.
    *   **Uvicorn (El Motor):** `api.py` solo define el servidor. **Uvicorn** es el servidor ASGI que se encarga de ejecutarlo y gestionar las conexiones de red.

---

## **5. Resumen de Comandos: Cómo Usar Todo**

1.  **Instalar Herramientas:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Fabricar un Nuevo Cerebro (Entrenar):**
    ```bash
    python main.py
    ```

3.  **Hacer una Prueba Rápida del Cerebro:**
    ```bash
    python predict_model.py
    ```

4.  **Abrir la Oficina de Consultas (Iniciar la API):**
    ```bash
    uvicorn api:app --reload
    ```