Hay que comprender que un modelo de ia , es de solo lectura , porque solo tiene pesos.
Ya depende del hardware como aplicara esa base de datos de pesos y formulas.
No fluye nada a travez de el.
El modelo de IA no es un cerebro , es una archivo de lectura , que la GPU en computacion paralela lo
lee.
EL MODELO DE IA NO ES UN SOFTWARE 
El modelo NO es el programa.
Es solo una base de datos.
El motor de inferencia es el encargado de orquestar todo el trabajo y el verdadero software.

Torch como motor de inferencia


Limitaciones del embeeding bert tiny - DETECCION DE INTENCION
 -> 512 tokens (límite total) ÷ 1.2 tokens/palabra (promedio) ≈ 426 palabras
 -> Por eficiencia las reducimos a 200 palabras, se puede consultar esto con redaccion.
 -> 4 intenciones bien definidas de bajo costo semantico.
 -> tiniy bert tiene salida de un array de 128 en la ultima capa resumen
 -> el programa actual tiene solo 1 capa de entrada de 128 como red neural a una salida de solo 4 intenciones
        capa de entrada 128
        capa de logit o salida 4

-> BERT - MOTOR DE CONTEXTUALIZACIO CON EMBEDDING ESTATICO , PORQUE ES IMPORTANTE ?
Una vez agregado el la dimension del contexto , el vector resultante del token puede cambiar drasticamente y un token puede hacercarse a otra en el mapa. 

   Tipo: Embedding de Frase, Contextual y Dinámico.
       Generador: prajjwal1/bert-tiny.
       Proceso de Generación: Salida del token [CLS] de la última capa oculta del modelo Transformer.
       Dimensionalidad: 128.
       Características Notables:
       Ventaja: Extremadamente rápido y ligero, ideal para APIs en producción.
       Desventaja: Menor "resolución semántica" que embeddings de mayor dimensión (como los de bert-base), lo que limita el número de intenciones sutilmente diferentes que puede distinguir eficazmente.


-> TOKEN CLASSIFICATION 
   Para el token classification usa 
   Una Transformación Lineal en una red Feed-Forward, la cual goza de Independencia Computacional entre sus elementos.
   Por tanto unicamente usa una red neural distribuida en computacion paralela o GPU

   Como funciona , la red neural solo son un conjunto de pesos estaticos , solamente es de lectura , la gpu hace el resto.

   LOS DATOS (Tus 10 palabras vectorizadas , osea la base de datos completa de informacion):
        Son 10 ESTUDIANTES (Núcleos CUDA) sentados en pupitres.
        Cada estudiante tiene SU PROPIA HOJA con una palabra diferente.
        Es un poco mas de trabajo para CUDA per es casi despreciable , en caso de GPU en CPU sera un poco mas notirio
        Estudiante 1 tiene "get_news".
        Estudiante 2 tiene "tecnología".
        Estudiante 3 tiene "es".