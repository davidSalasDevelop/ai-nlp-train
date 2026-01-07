# service.py (Versión Definitiva con Sintaxis Moderna y Tipos Primitivos)
import bentoml
import logging

# NO importamos Text o JSON de bentoml.io, ya no es necesario
# from bentoml.io import JSON, Text 

from predict_model import load_model, predict

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('bentoml.service')

log.info("🚀 Inicializando el servicio de clasificación de intenciones...")
model_info = load_model()

if not model_info:
    log.error("❌ INICIO FALLIDO: No se pudo cargar el modelo.")
    raise RuntimeError("Error crítico: No se pudo cargar el modelo.")

log.info("✅ Modelo cargado y listo para recibir peticiones.")

@bentoml.service
class IntentClassifier:

    # --- SINTAXIS MODERNA Y FINAL ---
    # Usamos tipos primitivos de Python (str, dict). BentoML los convertirá
    # automáticamente a texto plano y JSON. Esto evita todos los errores
    # de inferencia y compatibilidad que hemos visto.
    @bentoml.api
    def classify(self, input_text: str, ctx: bentoml.Context) -> dict:
        log.info(f"Petición recibida para clasificar: '{input_text}'")
        
        # Validación de Entrada
        if not input_text or not input_text.strip():
            log.warning("Petición rechazada: el texto de entrada está vacío.")
            ctx.response.status_code = 400
            return {"error": "El texto de entrada no puede estar vacío."}

        try:
            predictions = predict(input_text, model_info)
            
            if not predictions:
                log.error("La predicción devolvió un resultado vacío, revisa los logs del modelo.")
                ctx.response.status_code = 500
                return {"error": "Ocurrió un error interno durante la predicción."}

            log.info(f"Predicciones generadas exitosamente.")
            # La salida es un diccionario, que BentoML convertirá a JSON.
            return {"predictions": predictions}

        except Exception as e:
            log.exception("Excepción no controlada en el endpoint /classify")
            ctx.response.status_code = 500
            return {"error": "Ocurrió un error interno inesperado."}

    @bentoml.api(route="/health")
    def health(self, ctx: bentoml.Context) -> dict:
        log.info("Health check recibido.")
        return {"status": "ok"}