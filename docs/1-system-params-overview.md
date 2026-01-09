Limitaciones del embeeding bert tiny 
 -> 512 tokens (límite total) ÷ 1.2 tokens/palabra (promedio) ≈ 426 palabras
 -> Por eficiencia las reducimos a 200 palabras, se puede consultar esto con redaccion.
 -> 4 intenciones bien definidas de bajo costo semantico.
 -> tiniy bert tiene salida de un array de 128 en la ultima capa resumen
 -> el programa actual tiene solo 1 capa de entrada de 128 como red neural a una salida de solo 4 intenciones
        capa de entrada 128
        capa de logit o salida 4

-> EMBEEDING
   Tipo: Embedding de Frase, Contextual y Dinámico.
       Generador: prajjwal1/bert-tiny.
       Proceso de Generación: Salida del token [CLS] de la última capa oculta del modelo Transformer.
       Dimensionalidad: 128.
       Características Notables:
       Ventaja: Extremadamente rápido y ligero, ideal para APIs en producción.
       Desventaja: Menor "resolución semántica" que embeddings de mayor dimensión (como los de bert-base), lo que limita el número de intenciones sutilmente diferentes que puede distinguir eficazmente.