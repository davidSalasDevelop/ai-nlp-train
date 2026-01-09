
Lo que tienes que hacer es un proceso de tres pasos:
1.  **TÚ generas los embeddings** para un conjunto de frases de ejemplo usando tu modelo `bert-tiny`.
2.  **TÚ guardas esos embeddings** en el formato de archivo que el TensorFlow Projector puede leer.
3.  **TÚ subes esos archivos** al sitio web del TensorFlow Projector para visualizarlos.

Es un proceso muy poderoso porque te permitirá "ver" exactamente cómo tu modelo agrupa las frases de tus diferentes intenciones.

Aquí tienes una guía completa paso a paso, incluyendo el script de Python que necesitas.

---

### **Paso 1: Generar los Archivos para el Proyector**

Primero, necesitamos crear un script que genere dos archivos a partir de tus datos:

1.  **`vectors.tsv`**: Un archivo de texto donde cada línea contiene los 128 números de un embedding, separados por tabulaciones.
2.  **`metadata.tsv`**: Un archivo de texto donde cada línea contiene la "etiqueta" de cada vector. En nuestro caso, será la frase original y su intención.

#### **El Script: `generate_embeddings_for_projector.py`**

Crea un nuevo archivo con este nombre y pega el siguiente código. He incluido frases de ejemplo para 4 intenciones diferentes. ¡Puedes y debes modificarlas para que se parezcan a tus datos reales!

```python
# generate_embeddings_for_projector.py
import torch
from transformers import AutoTokenizer, AutoModel
import os

print("--- Iniciando la generación de embeddings para TensorFlow Projector ---")

# --- CONFIGURACIÓN ---
MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "projector_data" # Carpeta para guardar los archivos

# --- FRASES DE EJEMPLO (¡Modifica esto con tus propios ejemplos!) ---
# Intenta tener un número similar de ejemplos por cada intención.
sample_sentences = {
    "get_news": [
        "cuáles son las noticias de hoy",
        "dame los titulares más recientes",
        "quiero saber qué está pasando en el mundo",
        "resumen de noticias por favor",
        "noticias de última hora"
    ],
    "get_profile": [
        "muéstrame la información de mi perfil",
        "quiero ver los datos de mi cuenta",
        "puedo ver mi información personal",
        "acceder a mi perfil",
        "dónde veo mi cuenta"
    ],
    "get_business_information": [
        "dame los datos de la empresa",
        "información sobre la visión del negocio",
        "cuál es la misión de la compañía",
        "necesito datos corporativos",
        "háblame sobre la empresa"
    ],
    "check_inbox": [
        "revisar mi bandeja de entrada",
        "tengo nuevos correos electrónicos",
        "muéstrame mis mensajes",
        "abrir el correo",
        "ver mis emails"
    ]
}

# --- Cargar Modelo y Tokenizer ---
print(f"Cargando el modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Modelo cargado.")

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """Convierte una frase en su vector-resumen de 128 dimensiones."""
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# --- Procesar Frases y Generar Embeddings ---
print("\nGenerando embeddings para las frases de ejemplo...")
all_vectors = []
all_metadata = []

for intent, sentences in sample_sentences.items():
    for sentence in sentences:
        # Generar el embedding
        vector = get_sentence_embedding(sentence).squeeze().tolist() # Convertir a lista de Python
        all_vectors.append(vector)
        
        # Guardar la intención y la frase como metadatos
        all_metadata.append(f"{intent}\t{sentence}")

print(f"✅ Se generaron {len(all_vectors)} embeddings.")

# --- Guardar Archivos para el Proyector ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
vectors_path = os.path.join(OUTPUT_DIR, "vectors.tsv")
metadata_path = os.path.join(OUTPUT_DIR, "metadata.tsv")

print(f"\nGuardando archivos en la carpeta '{OUTPUT_DIR}'...")

# Guardar los vectores
with open(vectors_path, "w", encoding="utf-8") as f:
    for vector in all_vectors:
        # Unir los 128 números con tabulaciones
        f.write("\t".join([str(x) for x in vector]) + "\n")

# Guardar los metadatos
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write("Intencion\tFrase\n") # Escribir una cabecera
    for meta in all_metadata:
        f.write(meta + "\n")

print(f"✅ ¡Listo! Sube los siguientes archivos al TensorFlow Projector:")
print(f"   - Vectores: {vectors_path}")
print(f"   - Metadatos: {metadata_path}")
```

#### **Ejecuta el Script**

Abre tu terminal, activa tu entorno virtual y corre el script:
`python generate_embeddings_for_projector.py`

Cuando termine, tendrás una nueva carpeta llamada `projector_data` con los dos archivos que necesitas (`vectors.tsv` y `metadata.tsv`).

---

### **Paso 2: Visualizar en TensorFlow Projector**

Ahora, vamos al sitio web.

1.  **Abre el TensorFlow Projector:** Ve a [**projector.tensorflow.org**](https://projector.tensorflow.org/) en tu navegador.
2.  **Carga tus Datos:** En el panel de la izquierda, verás una sección llamada "Load".
    *   Haz clic en el primer botón **"Choose file"**. Navega a tu carpeta `projector_data` y selecciona **`vectors.tsv`**.
    *   Espera un momento a que se procese.
    *   Haz clic en el segundo botón **"Choose file"** (debajo de "Load metadata"). Navega a tu carpeta `projector_data` y selecciona **`metadata.tsv`**.

    

---

### **Paso 3: Analiza el Resultado**

¡Listo! Ahora verás una nube de puntos en 3D en la pantalla principal. Cada punto es una de tus frases.

**¿Qué debes buscar?**

*   **¡Busca los Cúmulos (Clusters)!** La pregunta más importante es: **¿Las frases de la misma intención forman grupos de puntos cercanos?**
    *   Si ves una "nube" de puntos claramente separada que corresponde a `get_news`, otra para `get_profile`, etc., **¡es una noticia fantástica!** Significa que tu modelo `bert-tiny` ya ve estas intenciones como conceptos semánticamente distintos. Esto confirma que a tu clasificador de una sola capa le resultará muy fácil aprender a separarlos.
*   **Interactúa con los Puntos:** Haz clic en cualquier punto de la nube. En el panel de la derecha, verás la información de los metadatos que cargaste (la intención y la frase). Esto te ayuda a entender por qué ciertos puntos están cerca de otros.
*   **Busca "Puentes" o "Superposiciones":** Si ves que los puntos de dos intenciones diferentes están muy mezclados, es una **señal de alerta**. Es la misma advertencia que te daba el cálculo de similitud de coseno: estas dos intenciones son semánticamente muy cercanas para tu modelo, y probablemente se confundirá al clasificarlas.

Este proceso te da una prueba visual y poderosa de la "separabilidad" de tus intenciones en el espacio de embeddings de **tu modelo específico**. Es una de las herramientas de diagnóstico más intuitivas que existen.