# Ficha Técnica del Modelo: Extractor de Parámetros de Noticias

**Versión del Modelo:** 1.0
**Fecha de Creación:** 2026-01-12
**Tipo de Modelo:** Reconocimiento de Entidades Nombradas (NER) / Clasificación de Tokens

---

## 1. Resumen del Modelo

Este modelo es una red neuronal especializada diseñada para una única tarea: **analizar una frase y extraer parámetros específicos relacionados con la intención "get_news"**.

**Funcionalidad Principal:**
*   **Input:** Una cadena de texto (una frase de un usuario).
*   **Output:** Una lista de entidades extraídas del texto. Cada entidad tiene un tipo (`SUBJECT`, `DATE_RANGE`) y el texto correspondiente.
*   **Detección de Intención Implícita:** La intención `get_news` se considera detectada si el modelo logra extraer al menos una entidad. Si no extrae ninguna entidad, se asume que la intención del usuario es `unknown` (desconocida) para este modelo.

Este modelo está diseñado para ser ligero y rápido, priorizando la eficiencia en la inferencia para su uso en APIs en tiempo real.

---

## 2. Arquitectura de la Red Neuronal

El modelo se basa en una arquitectura de **Transfer Learning**, utilizando un modelo de lenguaje pre-entrenado como base y añadiendo una "cabeza" especializada para nuestra tarea.

### 2.1. Modelo Base: El Motor de Contextualización

*   **Nombre del Modelo (`MODEL_NAME`):** `prajjwal1/bert-tiny`
*   **Arquitectura:** BERT (Bidirectional Encoder Representations from Transformers).
*   **Parámetros Totales (Base):** ~4.4 Millones.
*   **Capas de Transformer:** 2 capas.
*   **Cabezas de Atención:** 2 cabezas por capa.
*   **Dimensión del Embedding (`hidden_size`):** **128**.

**Función:** Este bloque actúa como un potente **extractor de características contextuales**. Su trabajo es tomar la secuencia de tokens de la frase de entrada y generar una secuencia de vectores (embeddings), uno por cada token. Cada vector de 128 números representa el significado de ese token en el contexto específico de la frase.

### 2.2. Cabeza de Clasificación: El Especialista en Entidades

Añadida sobre el modelo base, esta es la parte que se entrena específicamente para nuestra tarea de NER.

*   **Tipo:** Clasificación de Tokens (`Token Classification Head`).
*   **Implementación:** Se utiliza la clase `AutoModelForTokenClassification` de Hugging Face, que añade una **única capa lineal** (`nn.Linear`) seguida de una función de activación Softmax (implícita en la pérdida) sobre la salida de cada token de BERT.
*   **Configuración de la Capa Lineal:**
    *   **Dimensiones de Entrada:** 128 (coincide con el `hidden_size` de `bert-tiny`). Cada vector de token de 128 números es una entrada.
    *   **Dimensiones de Salida:** 5 (coincide con el número de `ENTITY_LABELS` en `ner_config.py`).
*   **Funcionamiento:**
    1.  Recibe la matriz de embeddings contextuales de BERT (forma: `longitud_secuencia x 128`).
    2.  Pasa **cada uno** de los vectores de 128 a través de la capa lineal.
    3.  Para cada token, la capa produce un vector de 5 puntuaciones (logits), una por cada posible etiqueta (`O`, `B-SUBJECT`, `I-SUBJECT`, etc.).
    4.  La etiqueta con la puntuación más alta es la predicción del modelo para ese token.

---

## 3. Parámetros y Proceso de Entrenamiento

El modelo fue entrenado usando la técnica de **Fine-Tuning**. Esto significa que los pesos del modelo base `bert-tiny` se ajustaron ligeramente, mientras que los pesos de la "cabeza" de clasificación de tokens se aprendieron desde cero.

### 3.1. Dataset de Entrenamiento

*   **Fuente (`DATASET_PATH`):** `ner_training/ner_dataset.json`
*   **Descripción:** Un conjunto de frases de ejemplo. Cada frase está etiquetada con una lista de `entities`, que especifica el texto, la posición (`start`, `end`) y el `label` de cada parámetro a extraer.
*   **Esquema de Etiquetado (NER):** Se utiliza el formato **IOB2 (Inside, Outside, Beginning)** para convertir las entidades a nivel de caracter en etiquetas a nivel de token. Las etiquetas utilizadas son:
    *   `O`: (Outside) El token no es parte de ninguna entidad.
    *   `B-SUBJECT`: (Beginning) El token es el inicio de una entidad de tipo `SUBJECT`.
    *   `I-SUBJECT`: (Inside) El token está dentro de una entidad `SUBJECT`, pero no es el primero.
    *   `B-DATE_RANGE`: Inicio de una entidad `DATE_RANGE`.
    *   `I-DATE_RANGE`: Dentro de una entidad `DATE_RANGE`.

### 3.2. Hiperparámetros de Entrenamiento

Estos son los parámetros clave que definieron el comportamiento del proceso de entrenamiento, configurados en `ner_config.py` y `ner_main.py`.

| Parámetro                  | Valor (`ner_config.py`) | Descripción Conceptual                                                                                              |
| :------------------------- | :---------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Épocas (`TRAIN_EPOCHS`)**    | `40`                    | El número de veces que el modelo revisó el dataset de entrenamiento completo.                                       |
| **Tamaño de Lote (`BATCH_SIZE`)** | `4`                     | El número de frases que el modelo procesó a la vez antes de actualizar sus pesos.                                 |
| **Tasa de Aprendizaje (`LEARNING_RATE`)** | `5e-5` (0.00005)        | La "velocidad" o el "tamaño del paso" con el que el modelo ajustó sus pesos para corregir errores.            |
| **Optimizador**            | `AdamW` (por defecto)   | El algoritmo matemático utilizado para minimizar el error y actualizar los pesos del modelo de manera eficiente.        |
| **Función de Pérdida**       | `CrossEntropyLoss` (por defecto) | La función matemática que calcula el "error" o "pérdida" para cada token, comparando la predicción con la etiqueta correcta. |
| **Métrica Principal (`metric_for_best_model`)** | `eval_f1`               | Se utilizó el **F1-Score** en el set de validación para decidir cuál de los modelos guardados en cada época era el "mejor". |

### 3.3. Monitoreo

*   **Plataforma:** MLflow
*   **Nombre del Experimento:** `GetNews-Extractor-Training`
*   **Métricas Registradas:**
    *   **Métricas de Entrenamiento:** `train_loss`
    *   **Métricas de Evaluación:** `eval_loss`, `eval_precision`, `eval_recall`, `eval_f1`
    *   **Métricas del Sistema:** `system/cpu_usage_percent`, `system/ram_usage_percent`, `process/cpu_usage_percent`, `process/ram_usage_mb`, y métricas de GPU (si está disponible).

---

## 4. Uso Previsto e Inferencia

Este modelo está diseñado para ser consumido a través de una API.

*   **Input Esperado:** Una cadena de texto, como `"noticias sobre Arévalo de ayer"`.
*   **Proceso de Inferencia:**
    1.  El texto es tokenizado.
    2.  El modelo predice una etiqueta (`O`, `B-SUBJECT`, etc.) para cada token.
    3.  Un paso de post-procesamiento (`aggregation_strategy="simple"`) agrupa los tokens consecutivos (ej: `Aré` + `##valo`) en una única entidad (`Arévalo`).
    4.  Si se extraen entidades (`SUBJECT` o `DATE_RANGE`), se asume la intención `get_news`.
    5.  Las entidades `DATE_RANGE` se procesan adicionalmente para convertirlas a un formato de fecha estándar (`YYYY-MM-DD`).
*   **Output Final:** Un objeto JSON que contiene la intención (`get_news` o `unknown`) y un diccionario de parámetros con los valores extraídos.

## 5. Limitaciones y Consideraciones

*   **Dominio Específico:** El modelo solo ha sido entrenado para reconocer las entidades `SUBJECT` y `DATE_RANGE` en el contexto de la intención `get_news`. No generalizará a otros dominios.
*   **Dependencia del Dataset:** La calidad y variedad de las frases en `ner_dataset.json` determinan el rendimiento del modelo. Patrones de frases no vistos en el entrenamiento pueden resultar en extracciones incorrectas.
*   **Capacidad del Modelo:** Al ser un modelo `tiny`, su capacidad para manejar un gran número de tipos de entidades o contextos muy complejos es limitada.