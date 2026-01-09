
Básicamente, la pregunta que estamos tratando de responder es: **"¿Qué tan cerca vivirá esta nueva intención de sus vecinas en el mapa de 128 dimensiones que BERT crea?"**

Si vive en una "zona aislada", su costo es bajo. Si vive en un "barrio densamente poblado y ruidoso", su costo es muy alto.

Aquí te presento un método práctico y profesional para estimar este costo, usando el propio modelo BERT como herramienta de medición.

---

### **Método Práctico: Medición de la Similitud de Coseno**

La **similitud de coseno** es una métrica matemática que mide el ángulo entre dos vectores.

*   Un resultado de **1.0** significa que los vectores apuntan exactamente en la misma dirección (son semánticamente idénticos).
*   Un resultado de **0.0** significa que son ortogonales (no tienen ninguna relación semántica).
*   Un resultado de **-1.0** significa que apuntan en direcciones opuestas (son semánticamente opuestos).

Usaremos esta métrica para ver qué tan "confundible" es una nueva intención con las que ya existen.

#### **Paso 1: Define tus Intenciones (La Nueva y sus Posibles Rivales)**

Supongamos que tu modelo ya conoce estas intenciones:

*   `consultar_estado_pedido`
*   `cancelar_pedido`
*   `ver_catalogo_productos`

Y ahora quieres añadir una nueva intención, que llamaremos **`INTENCION_CANDIDATA`**:

*   `consultar_fecha_entrega_pedido`

Intuitivamente, sabemos que esta nueva intención es muy similar a `consultar_estado_pedido`. Vamos a cuantificarlo.

#### **Paso 2: Genera Frases Representativas (Prototipos)**

Para cada intención (la nueva y sus rivales), escribe 3-5 frases "prototípicas" que un usuario real diría. Estas son las frases que mejor capturan la esencia de cada intención.

*   **`consultar_estado_pedido` (Existente):**
    *   "Quisiera saber dónde está mi paquete"
    *   "Cuál es el estado de mi orden"
    *   "Rastrear mi compra"
*   **`ver_catalogo_productos` (Existente):**
    *   "Muéstrame los productos que vendes"
    *   "Qué hay en tu catálogo"
    *   "Quiero ver tus artículos"
*   **`consultar_fecha_entrega_pedido` (Candidata):**
    *   "Cuándo llegará mi pedido"
    *   "Cuál es la fecha estimada de entrega"
    *   "Dime el día que recibo mi paquete"

#### **Paso 3: Convierte cada Frase en su "Huella Dactilar" de 128 Números**

Ahora, usaremos el motor de BERT (sin el clasificador) para convertir cada una de estas frases en su vector-resumen de 128 dimensiones.

Aquí tienes un script de Python para hacerlo:

```python
# calculate_semantic_similarity.py
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

# --- CONFIGURACIÓN ---
MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")

# --- MODELO (Solo BERT, sin clasificador) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """Convierte una frase en su vector-resumen de 128 dimensiones."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # Usamos la estrategia del token [CLS]
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# --- FRASES PROTOTÍPICAS ---
prototypes = {
    "consultar_estado_pedido": [
        "Quisiera saber dónde está mi paquete",
        "Cuál es el estado de mi orden",
        "Rastrear mi compra"
    ],
    "ver_catalogo_productos": [
        "Muéstrame los productos que vendes",
        "Qué hay en tu catálogo",
        "Quiero ver tus artículos"
    ],
    "consultar_fecha_entrega_pedido": [ # La intención candidata
        "Cuándo llegará mi pedido",
        "Cuál es la fecha estimada de entrega",
        "Dime el día que recibo mi paquete"
    ]
}

# --- CÁLCULO DE EMBEDDINGS ---
# Calcula el embedding promedio para cada intención
intent_embeddings = {}
for intent, sentences in prototypes.items():
    embeddings = [get_sentence_embedding(s) for s in sentences]
    # Apilamos los tensores y calculamos la media a lo largo de la dimensión del lote
    intent_embeddings[intent] = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)

# --- CÁLCULO DEL COSTO SEMÁNTICO ---
print("--- Calculando Costo Semántico para 'consultar_fecha_entrega_pedido' ---\n")

candidate_intent = "consultar_fecha_entrega_pedido"
candidate_embedding = intent_embeddings[candidate_intent]

for existing_intent, existing_embedding in intent_embeddings.items():
    if existing_intent == candidate_intent:
        continue
    
    # Calcular la similitud de coseno entre la candidata y cada una de las existentes
    similarity = cosine_similarity(candidate_embedding, existing_embedding).item()
    
    print(f"Similitud con '{existing_intent}': {similarity:.4f}")

```

#### **Paso 4: Analiza los Resultados para Determinar el Costo**

Al ejecutar el script anterior, obtendrás una salida como esta:

```
--- Calculando Costo Semántico para 'consultar_fecha_entrega_pedido' ---

Similitud con 'consultar_estado_pedido': 0.9587
Similitud con 'ver_catalogo_productos': 0.6123
```

Ahora, puedes interpretar estos números para asignar un "costo":

| Similitud de Coseno Máxima | Costo Semántico para la Red Neuronal | Interpretación y Acción Recomendada                                                                                                                                                                                                                                                          |
| :------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **< 0.80**                 | **BAJO**                             | ¡Excelente! La nueva intención vive en una "zona aislada". La capa lineal de 128 neuronas tendrá una tarea muy fácil para dibujar una frontera clara. Necesitarás una cantidad estándar de datos para que aprenda bien.                                                                           |
| **0.80 - 0.90**            | **MEDIO**                            | Precaución. La nueva intención es una "vecina cercana" de otra. El modelo `bert-tiny` puede separarlas, pero es un desafío. **Acción:** Necesitarás un dataset más grande y variado para esta intención, con ejemplos que enfaticen las diferencias sutiles (ej., usar "cuándo", "fecha", "día"). |
| **> 0.90**                 | **ALTO**                             | ¡Peligro! La nueva intención es semánticamente casi idéntica a una existente. Desde la perspectiva de BERT, son casi el mismo punto en el mapa. Le estás pidiendo a tu simple capa lineal que dibuje una frontera en un espacio extremadamente reducido y concurrido. Es muy probable que el modelo se confunda. |

**Acciones recomendadas para un Costo Semántico ALTO (> 0.90):**

1.  **Fusionar Intenciones:** ¿Realmente necesitas dos intenciones separadas? Quizás podrías fusionarlas en una sola, como `consultar_informacion_pedido`, y luego usar el reconocimiento de entidades (NER) para extraer si el usuario preguntó por el "estado" o la "fecha".
2.  **Re-entrenar con Datos de Contraste:** Si debes mantenerlas separadas, tu dataset de entrenamiento DEBE incluir ejemplos diseñados específicamente para confundir al modelo y enseñarle la diferencia. Por ejemplo, frases que incluyan tanto "estado" como "fecha" y estén etiquetadas correctamente.
3.  **Considerar un Modelo más Grande:** Si tienes muchas intenciones de costo semántico alto, es una señal clara de que `bert-tiny` no tiene la "resolución" o capacidad suficiente para tu problema. Necesitas un "cartógrafo" más hábil como `bert-base`.

Este método te da una **métrica cuantitativa y objetiva** para predecir qué tan difícil será para tu red neuronal de 128 entradas aprender una nueva intención, permitiéndote tomar decisiones informadas sobre el diseño de tu sistema.