# predict.py
"""
Módulo autocontenido para el pipeline de inferencia de dos etapas.
Carga un modelo de intenciones y un modelo NER desde archivos .pt,
y orquesta la predicción completa.
Puede ser ejecutado directamente para realizar una prueba.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForTokenClassification
import pydantic
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Estructuras de Datos (Pydantic) ---
class FullResult(pydantic.BaseModel):
    intent: str
    confidence: float
    parameters: Dict[str, Any]

# --- Lógica de Procesamiento de Fechas ---
def parse_relative_date(date_str: str) -> dict:
    now = datetime.now()
    date_str = date_str.lower()
    start_date, end_date = "unknown", "unknown"

    if "presente año" in date_str:
        start_date = now.replace(month=1, day=1).strftime('%Y-%m-%d')
        end_date = now.replace(month=12, day=31).strftime('%Y-%m-%d')
    elif "ayer" in date_str:
        date = now - timedelta(days=1)
        start_date = end_date = date.strftime('%Y-%m-%d')
    elif "mañana" in date_str:
        date = now + timedelta(days=1)
        start_date = end_date = date.strftime('%Y-%m-%d')
    elif "hace un mes" in date_str:
        target_date = now - relativedelta(months=1)
        start_date = target_date.replace(day=1).strftime('%Y-%m-%d')
        last_day_of_month = (target_date.replace(day=1) + relativedelta(months=1) - timedelta(days=1))
        end_date = last_day_of_month.strftime('%Y-%m-%d')
    elif "hace" in date_str and "años" in date_str:
        match = re.search(r'(\d+)', date_str)
        if match:
            years_ago = int(match.group(1))
            target_year_date = now - relativedelta(years=years_ago)
            start_date = target_year_date.replace(month=1, day=1).strftime('%Y-%m-%d')
            end_date = target_year_date.replace(month=12, day=31).strftime('%Y-%m-%d')
    elif "hoy" in date_str:
        start_date = end_date = now.strftime('%Y-%m-%d')
        
    return {"from_date": start_date, "to_date": end_date}

# --- Arquitecturas de Inferencia (para reconstruir desde .pt) ---

class IntentClassifierModel(nn.Module):
    """Arquitectura para reconstruir el modelo de intenciones."""
    def __init__(self, config, tokenizer_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(tokenizer_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# --- Clase Orquestadora del Pipeline ---
class PredictionPipeline:
    def __init__(self, intent_model_path: Path, ner_model_path: Path):
        self.INTENT_ENTITY_MAPPING = {"get_news": ["SUBJECT", "DATE_RANGE"]}
        self.CONFIDENCE_THRESHOLD = 0.50
        
        logging.info("Inicializando pipeline de dos etapas desde archivos .pt...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Cargar Modelo de Intenciones desde .pt ---
        self._load_intent_model(intent_model_path)
        
        # --- 2. Cargar Modelo NER desde .pt ---
        self._load_ner_model(ner_model_path)

        logging.info("✅ Pipeline de predicción listo.")

    def _load_intent_model(self, model_path: Path):
        logging.info(f"Cargando Modelo 1 (Intenciones) desde {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = AutoConfig.from_dictionary(checkpoint['config'])
        
        self.intent_model = IntentClassifierModel(
            config=config,
            tokenizer_name=checkpoint['tokenizer_name'],
            num_labels=len(checkpoint['id_to_intent'])
        )
        self.intent_model.load_state_dict(checkpoint['model_state_dict'])
        self.intent_model.to(self.device).eval()
        
        self.intent_tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        # Aseguramos que las claves sean integers
        self.id_to_intent = {int(k): v for k, v in checkpoint['id_to_intent'].items()}
        logging.info("✅ Modelo de intenciones cargado.")

    def _load_ner_model(self, model_path: Path):
        logging.info(f"Cargando Modelo 2 (NER) desde {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = AutoConfig.from_dictionary(checkpoint['config'])
        
        self.ner_model = AutoModelForTokenClassification.from_config(config)
        self.ner_model.load_state_dict(checkpoint['model_state_dict'])
        self.ner_model.to(self.device).eval()
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        self.id_to_ner_label = checkpoint['id2label']
        logging.info("✅ Modelo NER cargado.")

    def _group_entities(self, ner_prediction: List[Dict]) -> List[Dict]:
        """Agrupa sub-tokens en entidades completas."""
        grouped_entities = []
        for entity in ner_prediction:
            if entity['entity'].startswith('B-'):
                grouped_entities.append(entity)
            elif entity['entity'].startswith('I-') and grouped_entities and \
                 entity['entity'].split('-')[1] == grouped_entities[-1]['entity'].split('-')[1]:
                
                last_entity = grouped_entities[-1]
                # Agregamos el texto del sub-token, con un espacio si no empieza con ##
                token_text = entity['word']
                if not token_text.startswith('##'):
                    last_entity['word'] += ' '
                last_entity['word'] += token_text.replace('##', '')
                last_entity['end'] = entity['end']
                last_entity['score'] = (last_entity['score'] + entity['score']) / 2
        return grouped_entities

    def process_query(self, text: str) -> List[FullResult]:
        final_results = []
        
        # --- ETAPA 1: PREDICCIÓN DE INTENCIÓN ---
        inputs = self.intent_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            logits = self.intent_model(**inputs)
        probabilities = torch.softmax(logits, dim=1)[0]
        
        intent_predictions = []
        for i, prob in enumerate(probabilities):
            intent_predictions.append({"label": self.id_to_intent.get(i, f"unknown_{i}"), "score": prob.item()})
        intent_predictions.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"Ranking de intenciones: {intent_predictions[:3]}")

        valid_intents = [p for p in intent_predictions if p['score'] >= self.CONFIDENCE_THRESHOLD]
        
        if not valid_intents:
            return [FullResult(intent="unknown", confidence=intent_predictions[0]['score'], parameters={})]
        
        # --- ETAPA 2: EXTRACCIÓN DE ENTIDADES ---
        ner_inputs = self.ner_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ner_logits = self.ner_model(**ner_inputs).logits
        
        predictions = torch.argmax(ner_logits, dim=2)[0]
        tokens = self.ner_tokenizer.convert_ids_to_tokens(ner_inputs["input_ids"][0])
        
        raw_entities = []
        for token, pred_id in zip(tokens, predictions):
            label = self.id_to_ner_label.get(pred_id.item())
            if label and label != 'O' and not token in [self.ner_tokenizer.cls_token, self.ner_tokenizer.sep_token]:
                raw_entities.append({'entity': label, 'word': token})

        ner_results = self._group_entities(raw_entities)
        logging.info(f"Entidades extraídas por NER: {ner_results}")

        # --- ETAPA 3: FUSIÓN Y LÓGICA DE NEGOCIO ---
        for intent_pred in valid_intents:
            detected_intent = intent_pred['label']
            default_params = {"subject": "unknown", "from_date": "unknown", "to_date": "unknown"}
            parameters = default_params.copy()
            
            if detected_intent in self.INTENT_ENTITY_MAPPING:
                valid_entity_types = self.INTENT_ENTITY_MAPPING[detected_intent]
                
                for entity in ner_results:
                    entity_type = entity['entity'].split('-')[1] # Ej: 'B-SUBJECT' -> 'SUBJECT'
                    if entity_type not in valid_entity_types:
                        continue
                    
                    entity_value = entity['word']
                    if entity_type == "SUBJECT":
                        parameters["subject"] = entity_value
                    elif entity_type == "DATE_RANGE":
                        date_params = parse_relative_date(entity_value)
                        parameters.update(date_params)
            
            final_results.append(FullResult(
                intent=detected_intent,
                confidence=intent_pred['score'],
                parameters=parameters
            ))
            
        return final_results

# --- Bloque de Ejecución para Pruebas ---
if __name__ == "__main__":
    
    print("\n--- INICIANDO PRUEBA LOCAL DEL PIPELINE DE PREDICCIÓN DESDE ARCHIVOS .pt ---")

    # Rutas a los archivos .pt que has entrenado
    INTENT_MODEL_PT_PATH = Path("ner_training/models/intent_model/intent_classifier.pt")
    NER_MODEL_PT_PATH = Path("ner_training/models/get_news_extractor/get_news_extractor.pt")

    if not INTENT_MODEL_PT_PATH.exists() or not NER_MODEL_PT_PATH.exists():
        logging.error("¡Error! No se encontraron los archivos .pt de los modelos.")
    else:
        full_pipeline = PredictionPipeline(
            intent_model_path=INTENT_MODEL_PT_PATH,
            ner_model_path=NER_MODEL_PT_PATH
        )
        
        test_phrases = [
            "Deseo noticias sobre Arévalo del presente año",
            "Qué pasó ayer con el congreso",
            "Quiero pedir una pizza",
            "Cuál es el clima de hoy"
        ]

        for phrase in test_phrases:
            print("\n" + "="*60)
            logging.info(f"PROCESANDO CONSULTA DE PRUEBA: '{phrase}'")
            
            results = full_pipeline.process_query(phrase)
            results_dict = [r.dict() for r in results]
            
            print(json.dumps(results_dict, indent=2, ensure_ascii=False))

    print("\n--- PRUEBA LOCAL FINALIZADA ---")