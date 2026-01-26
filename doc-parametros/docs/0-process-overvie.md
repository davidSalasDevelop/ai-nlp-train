El flujo del agente sera sencillo de la siguiente manera 


[Se ingresa la oracion] -> 
[Se detecta la intencion y luego se ejecuta el model de parametros acorde a la intencion] -> 
[Se extraen los parametros de cada intencion, usando la misma oracion del chat como entrada] -> 
Se ejecuta la accion o acciones.

Eje:

1) Quiero noticias de areval de este anio y dame la hora en guatemala
2) Inteciones :
        get_news
        get_time
    3) get_news_params
        person - arevalo
        date.  - este anio
    4) get_date_params
        loc - guatemala

Consideraciones en el dataset

El mismo texto con el que se entrena la intencion , es el mismo texto con el que se entrena el modelo para extraer los parametros.
De esta manera se podran ejecutar las intenciones y los parametros correctamente.

Los modelos de parametros recibiran la misma oracion para extraer los parametros

Recuerda que la cantidad de intenciones detectadas define todo y que el mismo model puede detectar multiples intenciones o se puede hacer en cascada, si hay un unknown por ejemplo:

get_news : confidence 50%
get_date : confidence 49%

Significa que las dos intenciones son fuertes en la oracion.

recuerda separar los modelos de la siguiente manera

Primer par de modelos contiene las siguientes intenciones ej:
get_news
get_date
get_definition

get_news_params
get_date_params
get_definition_params

Son 6 modelos , pero facilmente se puede crear dos modelos 3 y 3 o si la situacion lo amerita , separar los modelos 
para que sean independientes la cuestion esque 
TIENEN QUE SER EL MISMO DATASET

y todo esta exactamente en los datos.

CONSIDERACION DE ENTRENAMIENTO 
Se usa un model bert base como motor de embedding que se entrena al mismo tiempo, por lo tanto hay que usar palabras 
que se usara para el publico destino , por ejemplo 
publico destino 
Guatemala 
Usa ??

Las oraciones deben estar relacionadas a eso.

Usar palabras como areval , guatemala , mixco , etc...