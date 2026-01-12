
---

### **1. TensorFlow Projector (El Estándar de la Industria)**

Esta es la herramienta más famosa y completa para visualizar embeddings. Fue creada por Google y es increíblemente poderosa.

*   **Enlace:** [**projector.tensorflow.org**](https://projector.tensorflow.org/)
*   **¿Qué es?:** Es una aplicación web que te permite cargar un conjunto de vectores (embeddings) y explorarlos en un espacio 3D interactivo. Para empezar, ya viene con varios datasets pre-cargados, incluyendo los embeddings clásicos de Word2Vec.
*   **¿Cómo te ayuda?:**
    1.  **Exploración Visual:** Puedes rotar, hacer zoom y moverte por una "nube" de miles de puntos, donde cada punto es una palabra.
    2.  **Búsqueda de Vecinos:** La función más útil. Escribes una palabra (ej: "rey") y la herramienta te resalta en la nube las palabras cuyos vectores están más cerca. Verás que "reina", "príncipe" y "castillo" se iluminan.
    3.  **Analogías Vectoriales:** Te permite realizar la famosa operación `rey - hombre + mujer`, y verás que el punto más cercano al resultado es "reina". Esto demuestra visualmente que el espacio ha aprendido relaciones semánticas complejas.
*   **Cómo usarlo para tu propósito:**
    1.  Ve al sitio.
    2.  En el panel de la derecha, en la sección de búsqueda, escribe una de tus palabras ancla, por ejemplo, **"pedido"**.
    3.  La herramienta te mostrará una lista de los vecinos más cercanos. Verás palabras como "orden", "envío", "entrega", "compra", etc.
    4.  Ahora, busca una palabra de otro dominio, como **"música"**. Verás que sus vecinos son "canción", "álbum", "artista", "sonido".
    5.  Al hacer esto, estás confirmando visualmente que los dos conceptos viven en "barrios" completamente diferentes del espacio semántico.

