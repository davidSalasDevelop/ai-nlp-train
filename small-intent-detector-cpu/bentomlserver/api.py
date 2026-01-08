# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from contextlib import asynccontextmanager

from predict_model import load_model, predict, PredictionError, ModelInfo, Prediction

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    predictions: list[Prediction]

model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Cargando el modelo al iniciar...")
    model_data["info"] = load_model()
    if not model_data["info"]:
        raise RuntimeError("❌ INICIO FALLIDO: No se pudo cargar el modelo.")
    logging.info("✅ Modelo cargado. La aplicación está lista.")
    yield
    logging.info("🔌 Apagando la aplicación.")
    model_data.clear()

app = FastAPI(lifespan=lifespan, title="API de Clasificación de Intenciones")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/classify")
def classify_intent(request: ClassifyRequest) -> ClassifyResponse:
    input_text = request.text
    logging.info(f"Petición recibida para clasificar: '{input_text}'")
    if not input_text or not input_text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío.")
    try:
        prediction_results = predict(input_text, model_data["info"])
        return ClassifyResponse(predictions=prediction_results)
    except PredictionError as e:
        logging.error(f"Error de predicción controlado: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error al procesar la predicción.")
    except Exception as e:
        logging.exception(f"Excepción no controlada en el endpoint")
        raise HTTPException(status_code=500, detail="Ocurrió un error interno inesperado.")

    
# Add this block at the end of api.py

if __name__ == "__main__":
    # This will run the server when you execute `python api.py`
    # Note: --reload is not as effective here as when run from the command line.
    uvicorn.run(app, host="0.0.0.0", port=3000)