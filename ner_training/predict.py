# predict.py
import logging
from pathlib import Path
from typing import Dict
import torch
from transformers import pipeline
import pydantic
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class GetNewsResult(pydantic.BaseModel):
    intent: str
    parameters: Dict[str, str]

def parse_relative_date(date_str: str) -> dict:
    # (La función es idéntica a la de respuestas anteriores, la omito por brevedad)
    now = datetime.now()
    date_str = date_str.lower()
    start_date, end_date = "unknown", "unknown"
    # ... (lógica de fechas)
    if "presente año" in date_str:
        start_date = now.replace(month=1, day=1).strftime('%Y-%m-%d')
        end_date = now.replace(month=12, day=31).strftime('%Y-%m-%d')
    # ... (resto de la lógica de fechas)
    return {"from_date": start_date, "to_date": end_date}

class NewsExtractor:
    def __init__(self, model_path: Path):
        logging.info("Inicializando el extractor de noticias (Modelo NER)...")
        device = 0 if torch.cuda.is_available() else -1
        
        logging.info(f"Cargando modelo NER desde {model_path}...")
        self.ner_pipeline = pipeline(
            "token-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=device,
            aggregation_strategy="simple"
        )
        logging.info("✅ Extractor de noticias listo.")
        
    def process_query(self, text: str) -> GetNewsResult:
        ner_results = self.ner_pipeline(text)
        logging.info(f"Entidades crudas extraídas por NER: {ner_results}")

        if not ner_results: # Si el modelo no extrae nada, es una intención desconocida
            return GetNewsResult(intent="unknown", parameters={})

        parameters = {
            "subject": "unknown",
            "from_date": "unknown",
            "to_date": "unknown"
        }

        for entity in ner_results:
            entity_type = entity['entity_group']
            entity_value = entity['word']
            
            if entity_type == "SUBJECT":
                parameters["subject"] = entity_value
            elif entity_type == "DATE_RANGE":
                date_params = parse_relative_date(entity_value)
                parameters.update(date_params)
        
        return GetNewsResult(intent="get_news", parameters=parameters)