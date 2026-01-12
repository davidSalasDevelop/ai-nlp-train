# Ficha Técnica del Sistema: Pipeline de NLU de Dos Etapas

**Versión del Sistema:** 1.0
**Fecha de Creación:** 2026-01-12
**Arquitectura:** Pipeline de dos etapas (Clasificación de Intención + Reconocimiento de Entidades)

---

## 1. Resumen del Sistema

Este documento describe un sistema de Procesamiento del Lenguaje Natural (NLU) diseñado para analizar una consulta de usuario y extraer tanto su **intención principal** como los **parámetros relevantes** asociados a dicha intención.

El sistema opera bajo una **arquitectura de pipeline de dos etapas**, utilizando dos modelos de Machine Learning distintos y especializados que trabajan en secuencia para producir un resultado estructurado y accionable.

**Funcionalidad Principal:**
*   **Input:** Una cadena de texto (una frase de un usuario).
*   **Output:** Una estructura JSON que contiene la intención detectada, un puntaje de confianza, y un diccionario de parámetros (entidades) extraídos del texto.

El sistema está diseñado para ser modular, permitiendo que cada componente (intenciones y entidades) sea entrenado y actualizado de forma independiente.

---

## 2. Descripción de la Arquitectura del Pipeline

El sistema procesa el texto de entrada a través de dos modelos secuenciales:

### **Etapa 1: Modelo de Clasificación de Intenciones (El "Qué")**

Este es el primer filtro. Su única responsabilidad es leer la frase completa y determinar la intención general del usuario.

*   **Nombre del Modelo:** Clasificador de Intenciones (Modelo 1)
*   **Tipo:** Clasificación de Texto / Clasificación de Secuencia
*   **Función:** Toma la frase entera, la procesa a través de una arquitectura BERT y produce un ranking de probabilidades para un conjunto predefinido de intenciones.
*   **Lógica de Decisión:** Se aplica un umbral de confianza (ej: > 50%) para determinar si una o más intenciones son lo suficientemente probables como para ser consideradas válidas. Si ninguna intención supera el umbral, el proceso puede detenerse y devolver una intención `unknown`.

### **Etapa 2: Modelo de Reconocimiento de Entidades (El "Detalle")**

Este modelo se activa después de la Etapa 1. Su trabajo es realizar un análisis más profundo del texto para "señalar" y extraer piezas específicas de información.

*   **Nombre del Modelo:** Extractor de Parámetros (`get_news_extractor`) (Modelo 2)
*   **Tipo:** Reconocimiento de Entidades Nombradas (NER) / Clasificación de Tokens
*   **Función:** Toma la frase entera y asigna una etiqueta (`O`, `B-SUBJECT`, `I-SUBJECT`, etc.) a cada token individual.
*   **Post-procesamiento:** Los tokens etiquetados se agrupan para reconstruir los parámetros completos (ej: los tokens `Aré` y `##valo` se combinan para formar la entidad `SUBJECT: "Arévalo"`).

### **Etapa 3: Fusión y Lógica de Negocio**

Un orquestador de software toma las salidas de ambos modelos y las combina.

1.  Se obtienen las intenciones válidas de la Etapa 1.
2.  Se obtiene una lista de todas las entidades encontradas en la Etapa 2.
3.  Para cada intención válida, el sistema consulta un **mapa de validez** (`INTENT_ENTITY_MAPPING`) para decidir qué entidades extraídas son relevantes para esa intención.
4.  Se construye la salida final, asociando los parámetros correctos con cada intención detectada.

---

## 3. Detalles Técnicos de los Modelos

Ambos modelos comparten la misma arquitectura base pero tienen "cabezas" de clasificación diferentes, especializadas para sus respectivas tareas.

### 3.1. Modelo Base Común

*   **Nombre del Modelo (`MODEL_NAME`):** `prajjwal1/bert-tiny`
*   **Arquitectura:** BERT (Bidirectional Encoder Representations from Transformers).
*   **Parámetros (Base):** ~4.4 Millones.
*   **Dimensión del Embedding (`hidden_size`):** **128**.
*   **Función:** Actúa como un motor de contextualización, convirtiendo el texto en representaciones numéricas ricas en significado.

### 3.2. Modelo 1: Clasificador de Intenciones

*   **Cabeza de Clasificación:** Una **única capa lineal (`nn.Linear`)** que opera sobre el embedding de la frase completa (vector del token `[CLS]`).
    *   **Entrada:** Vector de 128 dimensiones.
    *   **Salida:** Vector de N dimensiones (donde N es el número de intenciones).
*   **Dataset:** Entrenado con `dataset_v2.json`, utilizando solo los campos `text` e `intent`.
*   **Hiperparámetros Clave:**
    *   **Épocas:** 10
    *   **Tamaño de Lote:** 8
    *   **Tasa de Aprendizaje:** 3e-5
*   **Métrica Principal:** `eval_loss` (Pérdida de validación).

### 3.3. Modelo 2: Extractor de Parámetros (NER)

*   **Cabeza de Clasificación:** Una **única capa lineal (`nn.Linear`)** que opera sobre **cada embedding de token** individualmente.
    *   **Entrada:** Matriz de `(longitud_secuencia) x 128` dimensiones.
    *   **Salida:** Matriz de `(longitud_secuencia) x M` dimensiones (donde M es el número de etiquetas de entidad IOB2).
*   **Dataset:** Entrenado con `ner_dataset.json`, utilizando los campos `text` y `entities` para crear etiquetas a nivel de token.
*   **Hiperparámetros Clave:**
    *   **Épocas:** 40
    *   **Tamaño de Lote:** 4
    *   **Tasa de Aprendizaje:** 5e-5
*   **Métrica Principal:** `eval_f1` (F1-Score a nivel de entidad).

---

## 4. Monitoreo del Entrenamiento

Ambos modelos fueron entrenados utilizando el mismo sistema de monitoreo para garantizar la trazabilidad y la reproducibilidad.

*   **Plataforma:** MLflow
*   **Experimentos:** Se crearon experimentos separados para cada modelo (`Intent-TrainerAPI`, `GetNews-Extractor-Training`) para mantener los resultados organizados.
*   **Métricas Registradas:**
    *   **Métricas de Rendimiento:** Pérdida (`loss`), Precisión (`accuracy`), F1-Score, Precision, Recall.
    *   **Métricas del Sistema:** Uso de CPU, RAM y GPU (si aplica) a lo largo del tiempo, registradas a través de un `SystemMetricsCallback` personalizado.

---

## 5. Uso Previsto e Inferencia

El sistema está diseñado para ser desplegado como una API RESTful.

*   **Endpoint Principal:** `/process` (o `/extract`).
*   **Request:** Un objeto JSON con una clave `text`.
*   **Proceso de Inferencia:**
    1.  El texto es enviado al Modelo 1 para obtener un ranking de intenciones.
    2.  El texto es enviado al Modelo 2 para obtener una lista de entidades.
    3.  La lógica de fusión combina los resultados según las reglas de negocio.
*   **Response:** Un objeto JSON que contiene la intención (o `unknown`), la confianza y un diccionario de parámetros (ej: `subject`, `from_date`, `to_date`) con los valores extraídos y formateados.