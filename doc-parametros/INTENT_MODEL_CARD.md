# Ficha Técnica del Modelo: Clasificador de Intenciones

**Versión del Modelo:** 1.0
**Fecha de Creación:** 2026-01-12
**Tipo de Modelo:** Clasificación de Texto / Clasificación de Secuencia

---

## 1. Resumen del Modelo

Este modelo es una red neuronal especializada diseñada para una única tarea: **analizar una frase completa y asignarle una etiqueta de "intención" de un conjunto predefinido.**

**Funcionalidad Principal:**
*   **Input:** Una cadena de texto (una frase de un usuario).
*   **Output:** Un ranking de posibles intenciones, cada una con un puntaje de confianza (probabilidad). La intención con el puntaje más alto es la predicción principal del modelo.

Este modelo está optimizado para ser extremadamente ligero y rápido, ideal para ser el primer paso en un pipeline conversacional, donde la velocidad de respuesta es crítica.

---

## 2. Arquitectura de la Red Neuronal

Al igual que el modelo NER, este se basa en una arquitectura de **Transfer Learning**, utilizando un modelo de lenguaje pre-entrenado como base y añadiendo una "cabeza" especializada para la clasificación de frases.

### 2.1. Modelo Base: El Motor de Contextualización

*   **Nombre del Modelo (`MODEL_NAME`):** `prajjwal1/bert-tiny`
*   **Arquitectura:** BERT (Bidirectional Encoder Representations from Transformers).
*   **Parámetros Totales (Base):** ~4.4 Millones.
*   **Capas de Transformer:** 2 capas.
*   **Cabezas de Atención:** 2 cabezas por capa.
*   **Dimensión del Embedding (`hidden_size`):** **128**.

**Función:** Este bloque actúa como un **motor de contextualización y resumen**. Su trabajo es tomar la secuencia de tokens de la frase, procesarla, y generar un **único vector de 128 números** que representa el significado semántico de la frase completa. Esto se logra extrayendo el embedding contextualizado del token especial `[CLS]`.

### 2.2. Cabeza de Clasificación: El Especialista en Intenciones

Añadida sobre el modelo base, esta es la parte que se entrena específicamente para nuestra tarea de clasificación de intenciones.

*   **Tipo:** Clasificación de Secuencia (`Sequence Classification Head`).
*   **Implementación:** Se define en la clase `TinyModel` como una **única capa lineal** (`nn.Linear`).
*   **Configuración de la Capa Lineal:**
    *   **Dimensiones de Entrada:** 128 (coincide con el `hidden_size` de la salida de `bert-tiny`). El vector-resumen de 128 números de la frase es la entrada.
    *   **Dimensiones de Salida:** 4 (o el número total de intenciones únicas en el dataset de entrenamiento).
*   **Funcionamiento:**
    1.  Recibe el vector-resumen de 128 números de BERT.
    2.  Este vector se multiplica por la matriz de pesos de la capa lineal (forma: `128 x 4`).
    3.  El resultado es un vector de 4 puntuaciones crudas (logits), una por cada posible intención.
    4.  Una función Softmax convierte estos logits en una distribución de probabilidad, que representa la confianza del modelo en cada intención.

---

## 3. Parámetros y Proceso de Entrenamiento

El modelo fue entrenado usando **Fine-Tuning**. Los pesos del modelo `bert-tiny` se ajustaron sutilmente, mientras que los pesos de la capa lineal de clasificación se aprendieron desde cero.

### 3.1. Dataset de Entrenamiento

*   **Fuente (`DATASET_PATH`):** `small-intent-detector-cpu/dataset_v2.json`
*   **Descripción:** Un conjunto de frases de ejemplo. Cada frase está etiquetada con una única `intent` correspondiente. El dataset también incluye campos como `language` y `entities`, pero estos **fueron ignorados** durante el entrenamiento de este modelo específico.
*   **Preprocesamiento:**
    *   **Tokenización:** Las frases se convierten a secuencias de IDs numéricos.
    *   **Padding / Truncation:** Todas las secuencias se ajustan a una longitud máxima fija (`max_length=64`) para un procesamiento eficiente en lotes.

### 3.2. Hiperparámetros de Entrenamiento

Estos son los parámetros clave que definieron el comportamiento del proceso de entrenamiento, configurados en `config.py` y `main.py`.

| Parámetro                  | Valor                   | Descripción Conceptual                                                                                                                              |
| :------------------------- | :---------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Épocas (`EPOCHS`)**          | `10`                    | El número de veces que el modelo revisó el dataset de entrenamiento completo.                                                                       |
| **Tamaño de Lote (`BATCH_SIZE`)** | `8`                     | El número de frases que el modelo procesó a la vez antes de actualizar sus pesos.                                                                 |
| **Tasa de Aprendizaje (`LEARNING_RATE`)** | `3e-5` (0.00003)        | La "velocidad" o el "tamaño del paso" con el que el modelo ajustó sus pesos para corregir errores.                                              |
| **Optimizador**            | `AdamW` (por defecto)   | El algoritmo matemático utilizado para minimizar el error y actualizar los pesos del modelo de manera eficiente.                                        |
| **Función de Pérdida**       | `CrossEntropyLoss` (implícita) | La función matemática que calcula el "error" o "pérdida" para cada frase, comparando la predicción del modelo con la etiqueta de intención correcta. |
| **Métrica Principal (`metric_for_best_model`)** | `loss`                  | Se utilizó la **pérdida de validación** (`eval_loss`) para decidir cuál de los modelos guardados en cada época era el "mejor" y debía ser conservado.  |

### 3.3. Monitoreo

*   **Plataforma:** MLflow
*   **Nombre del Experimento:** `Intent-TrainerAPI`
*   **Métricas Registradas:**
    *   **Métricas de Entrenamiento:** `train_loss`, `epoch_train_accuracy`
    *   **Métricas de Evaluación:** `eval_loss`, `epoch_val_accuracy`
    *   **Métricas del Sistema:** Todas las métricas de CPU, RAM y GPU (si está disponible) fueron registradas a través de un callback personalizado.

---

## 4. Uso Previsto e Inferencia

Este modelo está diseñado para ser el **primer paso** en un pipeline de NLU.

*   **Input Esperado:** Una cadena de texto, como `"muéstrame las noticias de ayer"`.
*   **Proceso de Inferencia:**
    1.  El texto es tokenizado y procesado por el modelo completo.
    2.  El modelo produce un vector de probabilidades para todas las intenciones conocidas.
    3.  La aplicación que lo consume puede tomar la intención con la probabilidad más alta o aplicar una lógica más compleja (ej: considerar múltiples intenciones si superan un umbral de confianza).
*   **Output Final:** Un ranking de objetos JSON, cada uno conteniendo una `intent` y su `confidence`.

## 5. Limitaciones y Consideraciones

*   **Clasificación de Etiqueta Única:** El modelo, por diseño, asume que cada frase tiene una sola intención principal. Aunque se puede analizar el ranking de probabilidades, no está optimizado para la detección multi-intención.
*   **"Ceguera" a los Parámetros:** Este modelo no tiene la capacidad de extraer parámetros o entidades del texto (ej: no puede decir *qué* noticias o de *qué fecha*). Su única función es clasificar la frase en su totalidad.
*   **Dependencia del Dataset:** Su conocimiento se limita estrictamente a las intenciones presentes en su dataset de entrenamiento. No puede identificar intenciones no vistas.