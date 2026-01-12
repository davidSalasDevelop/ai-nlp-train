# api.py
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
from predict import NewsExtractor, GetNewsResult
import pydantic

# --- Configuración y Carga de Modelos ---
NER_MODEL_PATH = Path("ner_training/models/get_news_extractor")
pipeline_holder = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Cargando modelo extractor al iniciar la API...")
    pipeline_holder["pipeline"] = NewsExtractor(model_path=NER_MODEL_PATH)
    yield
    pipeline_holder.clear()
    logging.info("🔌 Modelo descargado. Apagando.")

app = FastAPI(lifespan=lifespan, title="API de Extracción de Parámetros de Noticias")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Endpoints de la API ---
class QueryRequest(pydantic.BaseModel):
    text: str

@app.post("/extract", response_model=GetNewsResult)
def process_text(request: QueryRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío.")
    
    pipeline = pipeline_holder.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="El modelo no está listo.")
    
    try:
        result = pipeline.process_query(request.text)
        return result
    except Exception as e:
        logging.exception("Error no controlado durante la predicción.")
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.")

@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in pipeline_holder else "loading"}

if __name__ == "__main__":
    import uvicorn
    # Para ejecutar directamente `python ner_training/api.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)