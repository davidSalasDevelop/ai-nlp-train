Limitaciones del embeeding bert tiny 
 -> 512 tokens (límite total) ÷ 1.2 tokens/palabra (promedio) ≈ 426 palabras
 -> Por eficiencia las reducimos a 200 palabras, se puede consultar esto con redaccion.
 -> 4 intenciones bien definidas de bajo costo semantico.
 -> tiniy bert tiene salida de un array de 128 en la ultima capa resumen
 -> el programa actual tiene solo 1 capa de entrada de 128 como red neural a una salida de solo 4 intenciones
        capa de entrada 128
        capa de logit o salida 4