

Piensa en el archivo del modelo `bert-tiny` como un archivo ZIP. Dentro de ese ZIP, hay muchos componentes, y uno de ellos es esa tabla de embeddings estáticos. Podemos usar Python para "descomprimirlo" y mirar esa pieza específica.

Hay dos maneras de "verlo":

1.  **Programáticamente:** Extraer la matriz de pesos directamente en Python para inspeccionar sus números.
2.  **Visualmente:** Extraer la matriz completa y su vocabulario asociado, y subirlos al TensorFlow Projector para explorar la "nube" de todos los tokens.

---

### **1. Verlo Programáticamente (La Inspección Directa)**

Este método te permite acceder a la matriz de embeddings y ver sus propiedades y algunos de sus vectores.

#### **El Script: `inspect_static_embeddings.py`**

```python
# inspect_static_embeddings.py
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")

# --- Cargar el Modelo Completo ---
print(f"Cargando el modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
print("✅ Modelo cargado.")

# --- Acceder a la Capa de Embeddings Estáticos ---
# La tabla de embeddings de palabras está en esta ruta dentro del modelo
static_embedding_layer = model.embeddings.word_embeddings

# Extraer la matriz de pesos completa. Esta es la tabla de búsqueda.
embedding_matrix = static_embedding_layer.weight

# --- Inspeccionar la Matriz ---
print("\n--- Inspección de la Matriz de Embeddings Estáticos ---")

# 1. Ver el tamaño de la matriz
# El resultado será (tamaño_vocabulario, dimension_embedding), ej: (30522, 128)
print(f"Forma de la matriz: {embedding_matrix.shape}")
print(f"Esto significa que hay {embedding_matrix.shape[0]} tokens en el vocabulario.")
print(f"Y cada token tiene un embedding estático de {embedding_matrix.shape[1]} números.")

# 2. Ver el embedding de un token específico
# Vamos a ver el embedding del token especial [CLS]
cls_token_id = tokenizer.cls_token_id
print(f"\nID del token [CLS]: {cls_token_id}")

# Extraer el vector correspondiente de la matriz (la fila 101)
cls_embedding_vector = embedding_matrix[cls_token_id]

print(f"Embedding estático para [CLS] (primeros 10 de 128 valores):\n{cls_embedding_vector[:10]}")

# 3. Ver el embedding de una palabra común como "casa"
try:
    casa_token_id = tokenizer.convert_tokens_to_ids("casa")
    casa_embedding_vector = embedding_matrix[casa_token_id]
    print(f"\nID del token 'casa': {casa_token_id}")
    print(f"Embedding estático para 'casa' (primeros 10 de 128 valores):\n{casa_embedding_vector[:10]}")
except KeyError:
    print("\nEl token 'casa' no está en el vocabulario como una sola palabra.")

# Los números en sí mismos no tienen un significado interpretable para los humanos.
# Su significado proviene de su posición relativa a otros vectores en el espacio de 128 dimensiones.
```

Al ejecutar este script, verás la prueba tangible de que esta tabla existe, su tamaño exacto y los vectores numéricos que contiene.

---

### **2. Verlo Visualmente (La Exploración Interactiva)**

Este método es mucho más intuitivo. Extraeremos **toda la matriz de embeddings** y todo el **vocabulario del tokenizer** y los subiremos al TensorFlow Projector.

#### **El Script: `export_static_embeddings_for_projector.py`**

Este script generará los archivos `vectors.tsv` y `metadata.tsv` para **todos los tokens** del vocabulario de `bert-tiny`.

```python
# export_static_embeddings_for_projector.py
import torch
from transformers import AutoTokenizer, AutoModel
import os

print("--- Iniciando la exportación de embeddings estáticos para TensorFlow Projector ---")

# --- CONFIGURACIÓN ---
MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "projector_static_embeddings"

# --- Cargar Modelo y Tokenizer ---
print(f"Cargando el modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
print("✅ Modelo cargado.")

# --- Extraer la Matriz de Embeddings y el Vocabulario ---
embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu() # Mover a CPU y quitar del grafo de gradientes
vocab = tokenizer.get_vocab() # Esto es un dict {'token': id}

# Invertir el vocabulario para tener {id: 'token'}
id_to_token = {v: k for k, v in vocab.items()}

print(f"Se encontraron {len(vocab)} tokens en el vocabulario.")

# --- Guardar Archivos ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
vectors_path = os.path.join(OUTPUT_DIR, "vectors.tsv")
metadata_path = os.path.join(OUTPUT_DIR, "metadata.tsv")

print(f"\nGuardando archivos en la carpeta '{OUTPUT_DIR}'...")
# ADVERTENCIA: ¡Esto puede generar archivos grandes!

# Guardar los vectores
with open(vectors_path, "w", encoding="utf-8") as f:
    for i in range(len(vocab)):
        vector = embedding_matrix[i].tolist()
        f.write("\t".join([str(x) for x in vector]) + "\n")

# Guardar los metadatos (los tokens) en el mismo orden que los vectores
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write("Token\n") # Cabecera
    for i in range(len(vocab)):
        f.write(id_to_token[i] + "\n")

print(f"✅ ¡Listo! Sube los siguientes archivos al TensorFlow Projector:")
print(f"   - Vectores: {vectors_path}")
print(f"   - Metadatos: {metadata_path}")
```

#### **Pasos para Visualizar**

1.  **Ejecuta el script:** `python export_static_embeddings_for_projector.py`. Esto creará una carpeta `projector_static_embeddings` con dos archivos.
2.  **Ve al TensorFlow Projector:** [projector.tensorflow.org](https://projector.tensorflow.org/)
3.  **Carga tus archivos:**
    *   Haz clic en "Load".
    *   Sube `vectors.tsv` para los vectores.
    *   Sube `metadata.tsv` para los metadatos.
4.  **¡Explora!** Ahora estás viendo el **cuaderno de bocetos** completo de `bert-tiny`.
    *   Busca la palabra "rey". Verás que sus vecinos más cercanos en este espacio estático son "reina", "príncipe", "monarca".
    *   Busca la palabra "correr". Verás cerca a "caminar", "saltar", "movimiento".

### **Conclusión Clave**

Lo que verás en el TensorFlow Projector con este método es la **base de conocimiento lingüístico estático** que `bert-tiny` utiliza como **punto de partida**. Es la "materia prima" antes de que el "motor de contextualización" se encienda y transforme estos vectores en las representaciones dinámicas y ricas que realmente impulsan tu clasificador de intenciones.