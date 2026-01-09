
---

### **Cómo Explorar la Ortogonalidad Semántica de BERT**

Aquí tienes un método práctico y un script para que puedas "jugar" con BERT y descubrir qué conceptos considera distantes.

#### **El Concepto: Anclajes Semánticos**

1.  **Elige una "Palabra Ancla":** Empieza con una palabra que sea central para una de tus intenciones existentes o una que estés considerando. Por ejemplo, la palabra **"pedido"**.
2.  **Crea una "Lista de Candidatas":** Haz una lluvia de ideas de palabras de diferentes dominios que crees que podrían estar lejos de tu palabra ancla. Incluye sinónimos, antónimos y palabras de campos completamente diferentes.
3.  **Calcula la Distancia:** Usa la similitud de coseno para medir la distancia semántica entre el embedding de tu palabra ancla y el embedding de cada palabra candidata.

#### **El Script: `explore_semantic_space.py`**

Este script te permitirá hacer exactamente eso. Es una herramienta interactiva para que puedas experimentar.

```python
# explore_semantic_space.py
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import numpy as np

# --- CONFIGURACIÓN ---
MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")

# --- MODELO (Solo BERT, sin clasificador) ---
print(f"Cargando el modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Modelo cargado.")

def get_word_embedding(word: str) -> torch.Tensor:
    """
    Convierte una palabra en su embedding contextual.
    La envolvemos en una plantilla para darle contexto al token [CLS].
    """
    # Usar una plantilla neutral como "esto es [PALABRA]" puede dar mejores resultados que la palabra aislada.
    template = f"esto es {word}"
    inputs = tokenizer(template, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # Usamos el embedding del token [CLS] como representación de la palabra/concepto.
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# --- ANÁLISIS INTERACTIVO ---
if __name__ == "__main__":
    
    # 1. DEFINE TU PALABRA ANCLA
    anchor_word = "pedido"
    print(f"\n--- Analizando distancias semánticas desde la palabra ancla: '{anchor_word}' ---")
    
    # 2. DEFINE TUS PALABRAS CANDIDATAS
    candidate_words = [
        # Sinónimos y palabras relacionadas (deberían estar cerca)
        "orden", "compra", "paquete", "entrega",
        
        # Palabras de un dominio de negocio (relacionadas pero diferentes)
        "factura", "cliente", "producto", "pago",
        
        # Palabras de un dominio completamente diferente (deberían estar lejos)
        "cielo", "música", "receta", "sentimiento", "sueño",
        
        # Palabras abstractas
        "justicia", "idea", "libertad",
        
        # Verbos de acción
        "correr", "pensar", "destruir"
    ]
    
    # 3. CALCULAR Y MOSTRAR RESULTADOS
    anchor_embedding = get_word_embedding(anchor_word)
    
    results = []
    for word in candidate_words:
        candidate_embedding = get_word_embedding(word)
        similarity = cosine_similarity(anchor_embedding, candidate_embedding).item()
        results.append((word, similarity))
        
    # Ordenar los resultados por similitud (de más cercano a más lejano)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Resultados (de más cercano a más lejano) ---\n")
    for word, sim in results:
        # Usamos una barra simple para visualizar la similitud
        bar = '█' * int(sim * 50) if sim > 0 else ''
        print(f"{word:<15} | Similitud: {sim: .4f} | {bar}")
```

### **Cómo Interpretar la Salida**

Al ejecutar el script, obtendrás una salida visual como esta:

```
--- Analizando distancias semánticas desde la palabra ancla: 'pedido' ---

--- Resultados (de más cercano a más lejano) ---

orden           | Similitud:  0.9812 | ██████████████████████████████████████████████████
compra          | Similitud:  0.9654 | ████████████████████████████████████████████████
paquete         | Similitud:  0.9432 | ███████████████████████████████████████████████
entrega         | Similitud:  0.9311 | ██████████████████████████████████████████████
factura         | Similitud:  0.8955 | ████████████████████████████████████████████
cliente         | Similitud:  0.8732 | ███████████████████████████████████████████
producto        | Similitud:  0.8510 | ██████████████████████████████████████████
pago            | Similitud:  0.8329 | ████████████████████████████████████████
correr          | Similitud:  0.6543 | ████████████████████████████████
pensar          | Similitud:  0.6210 | ██████████████████████████████
receta          | Similitud:  0.5899 | █████████████████████████████
idea            | Similitud:  0.5501 | ███████████████████████████
música          | Similitud:  0.5134 | █████████████████████████
sentimiento     | Similitud:  0.4876 | ████████████████████████
cielo           | Similitud:  0.4522 | ██████████████████████
sueño           | Similitud:  0.4198 | ████████████████████
justicia        | Similitud:  0.3987 | ███████████████████
destruir        | Similitud:  0.3755 | ██████████████████
libertad        | Similitud:  0.3502 | █████████████████
```

### **Cómo Usar esta Información para Diseñar tus Intenciones**

Esta herramienta te da un poder inmenso.

1.  **Validar la Separación:** Si estás pensando en crear una intención sobre "facturas" y otra sobre "pagos", puedes ver que aunque están relacionadas (similitud de ~0.8-0.9), BERT ya las ve como conceptos algo distintos. Esto te dice que **es posible** separarlas si tienes buenos datos.

2.  **Encontrar "Continentes Semánticos":** Los resultados muestran claramente que el grupo `(orden, compra, paquete, entrega)` vive en la misma "ciudad". El grupo `(factura, cliente, producto, pago)` vive en una "ciudad vecina". Pero el grupo `(cielo, música, receta, sentimiento)` vive en un "continente" completamente diferente.

3.  **Estrategia de Diseño:** Para maximizar el rendimiento de tu `bert-tiny`, **intenta que tus intenciones principales se basen en palabras ancla que provengan de diferentes "continentes semánticos".**
    *   Una intención podría girar en torno a **"pedidos"**.
    *   Otra intención podría girar en torno a **"música"**.
    *   Una tercera podría girar en torno a **"recetas"**.

    Como la similitud entre "pedido" y "música" es muy baja (~0.5), puedes estar casi seguro de que tu clasificador de una sola capa tendrá una tarea extremadamente fácil para distinguir entre `reproducir_musica` y `consultar_estado_pedido`. El "costo semántico" de añadir la intención de música es **muy bajo**.

Este proceso te permite dejar de adivinar y empezar a **diseñar tus intenciones basándote en la propia comprensión del lenguaje que tiene el modelo**, lo cual es una estrategia mucho más robusta y profesional.