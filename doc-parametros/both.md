# Ficha Técnica del Sistema: Pipeline de NLU de Dos Etapas (.pt)

**Versión del Sistema:** 1.1
**Fecha de Creación:** 2026-01-12
**Arquitectura:** Pipeline de dos etapas (Clasificación de Intención + Reconocimiento de Entidades) basado en checkpoints `.pt`.

---

## 1. Resumen del Sistema

Este documento describe un sistema de Procesamiento del Lenguaje Natural (NLU) diseñado para analizar una consulta de usuario y extraer tanto su **intención principal** como los **parámetros relevantes** asociados.

El sistema opera bajo una **arquitectura de pipeline de dos etapas**, utilizando dos modelos de Machine Learning distintos y especializados. Cada modelo es entrenado y guardado como un **archivo autocontenido `.pt`**. En la fase de inferencia, estos archivos `.pt` se cargan para reconstruir los modelos y procesar las consultas en secuencia.

**Funcionalidad Principal:**
*   **Input:** Una cadena de texto.
*   **Output:** Una estructura JSON que contiene la intención detectada, la confianza y un diccionario de parámetros (entidades) extraídos.

Este enfoque de usar archivos `.pt` desacopla completamente el entorno de entrenamiento del de inferencia, facilitando el despliegue y la portabilidad de los modelos.

---

## 2. Descripción de la Arquitectura del Pipeline

### **Etapa de Entrenamiento:**

*   Existen dos pipelines de entrenamiento independientes (`main.py` para intenciones y `ner_main.py` para entidades).
*   Cada pipeline utiliza el `Trainer` de Hugging Face para el fine-tuning.
*   Al finalizar el entrenamiento, cada pipeline empaqueta los pesos del mejor modelo (`state_dict`), su configuración (`config`) y los metadatos necesarios (mapeo de etiquetas, nombre del tokenizer) en un **único archivo `.pt`**.
    *   `intent_classifier.pt`
    *   `get_news_extractor.pt`

### **Etapa de Inferencia (predict.py):**

El sistema procesa el texto de entrada a través de los dos modelos cargados desde sus respectivos archivos `.pt`:

1.  **Carga de Modelos:** Al iniciar, el sistema carga `intent_classifier.pt` y `get_news_extractor.pt`. Reconstruye las arquitecturas de los modelos en memoria y carga los pesos entrenados desde los checkpoints.
2.  **Etapa 1: Modelo de Clasificación de Intenciones (El "Qué"):** La frase del usuario se procesa con el Modelo 1 para obtener un ranking de posibles intenciones. Se aplica un umbral de confianza para seleccionar las intenciones candidatas.
3.  **Etapa 2: Modelo de Reconocimiento de Entidades (El "Detalle"):** La frase del usuario se procesa con el Modelo 2 para etiquetar cada token y extraer una lista de todas las posibles entidades (`SUBJECT`, `DATE_RANGE`, etc.).
4.  **Etapa 3: Fusión y Lógica de Negocio:** Un orquestador de software filtra las entidades extraídas basándose en las intenciones detectadas y un mapa de validez, construyendo la respuesta final estructurada.

---

## 3. Detalles Técnicos de los Modelos

Ambos modelos comparten la misma arquitectura base pero tienen "cabezas" de clasificación diferentes.

### 3.1. Modelo Base Común

*   **Nombre del Modelo (`MODEL_NAME`):** `prajjwal1/bert-tiny`
*   **Arquitectura:** BERT
*   **Parámetros (Base):** ~4.4 Millones
*   **Dimensión del Embedding (`hidden_size`):** **128**

### 3.2. Modelo 1: Clasificador de Intenciones (desde `intent_classifier.pt`)

*   **Cabeza de Clasificación:** Una **única capa lineal (`nn.Linear`)** que opera sobre el embedding de la frase completa (vector del token `[CLS]`).
    *   **Entrada:** Vector de 128 dimensiones.
    *   **Salida:** Vector de N dimensiones (N = número de intenciones).
*   **Hiperparámetros Clave (Ejemplo):**
    *   **Épocas:** 10, **Tamaño de Lote:** 8, **Tasa de Aprendizaje:** 3e-5

### 3.3. Modelo 2: Extractor de Parámetros (desde `get_news_extractor.pt`)

*   **Cabeza de Clasificación:** Una **única capa lineal (`nn.Linear`)** que opera sobre **cada embedding de token** individualmente.
    *   **Entrada:** Matriz de `(longitud_secuencia) x 128`.
    *   **Salida:** Matriz de `(longitud_secuencia) x M` (M = número de etiquetas de entidad IOB2).
*   **Hiperparámetros Clave:**
    *   **Épocas:** 40, **Tamaño de Lote:** 4, **Tasa de Aprendizaje:** 5e-5

---

## 4. Uso Previsto e Inferencia

El sistema está diseñado para ser desplegado como una API RESTful.

*   **Proceso de Carga:** La API utiliza la lógica de `predict.py` para cargar los modelos desde los archivos `.pt` al arrancar.
*   **Endpoint Principal:** `/process` (o `/extract`).
*   **Request:** Un objeto JSON con una clave `text`.
*   **Proceso de Inferencia:** Se ejecuta el pipeline de dos etapas descrito anteriormente, realizando la predicción de forma manual (tokenización -> forward pass -> softmax/argmax -> post-procesamiento) sin usar la función `pipeline()` de alto nivel de Hugging Face.
*   **Response:** Un objeto JSON que contiene la intención (o `unknown`), la confianza y un diccionario de parámetros con los valores extraídos y formateados.
