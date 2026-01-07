# service.py (Versión Profesional y Robusta)
import bentoml
from bentoml.io import JSON, Text
import logging

from predict_model import load_model, predict

# --- MEJORA: Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('bentoml.service')

log.info("🚀 Inicializando el servicio de clasificación de intenciones...")
model_info = load_model()

if not model_info:
    log.error("❌ INICIO FALLIDO: No se pudo cargar el modelo. El servicio no podrá funcionar.")
    # En un sistema real, esto podría alertar a un sistema de monitoreo.
    raise RuntimeError("Error crítico: No se pudo cargar el modelo.")

log.info("✅ Modelo cargado y listo para recibir peticiones.")

@bentoml.service
class IntentClassifier:

    @bentoml.api(input=Text(), output=JSON())
    def classify(self, input_text: str, ctx: bentoml.Context) -> dict:
        log.info(f"Petición recibida para clasificar: '{input_text}'")
        
        # --- MEJORA: Validación de Entrada ---
        if not input_text or not input_text.strip():
            log.warning("Petición rechazada: el texto de entrada está vacío.")
            # Se establece el código de estado HTTP a 400 Bad Request
            ctx.response.status_code = 400
            return {"error": "El texto de entrada no puede estar vacío."}

        try:
            # --- MEJORA: Manejo de Errores a nivel de API ---
            predictions = predict(input_text, model_info)
            
            if not predictions:
                log.error("La predicción devolvió un resultado vacío, revisa los logs del modelo.")
                ctx.response.status_code = 500
                return {"error": "Ocurrió un error interno durante la predicción."}

            log.info(f"Predicciones generadas exitosamente.")
            return {"predictions": predictions}

        except Exception as e:
            log.exception("Excepción no controlada en el endpoint /classify")
            ctx.response.status_code = 500
            return {"error": "Ocurrió un error interno inesperado."}

    # --- MEJORA: Punto de Verificación de Salud (Health Check) ---
    @bentoml.api(route="/health", input=Text(), output=JSON())
    def health(self, _: str, ctx: bentoml.Context) -> dict:
        """
        Endpoint simple para verificar que el servicio está vivo y funcionando.
        """
        log.info("Health check recibido.")
        return {"status": "ok"}