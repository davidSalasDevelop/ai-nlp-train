# predict.py (VERSIÓN DEFINITIVA, COPIANDO LA LÓGICA QUE SÍ FUNCIONA)
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForTokenClassification, pipeline

import pydantic
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntentClassifierModel(nn.Module):
    def __init__(self, checkpoint: dict):
        super().__init__()
        config_data = checkpoint['config']
        if isinstance(config_data, dict):
            config = AutoConfig.from_pretrained(checkpoint['tokenizer_name'], **config_data)
        else:
            config = config_data
        num_labels = len(checkpoint.get('id_to_intent', checkpoint.get('intent_to_id', {})))
        self.bert = AutoModel.from_pretrained(checkpoint['tokenizer_name'], config=config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

def debug_pipeline(intent_model_path: Path, ner_model_path: Path, test_phrases: List[str]):
    
    device = torch.device("cpu")
    
    # --- Cargar Modelo 1 (Intenciones) ---
    logging.info(f"Cargando Modelo 1 (Intenciones) desde {intent_model_path}...")
    try:
        intent_checkpoint = torch.load(intent_model_path, map_location=device, weights_only=False)
        raw_mapping = intent_checkpoint['id_to_intent']
        first_key = next(iter(raw_mapping.keys()), None)
        if isinstance(first_key, str) and not first_key.isdigit():
            id_to_intent = {v: k for k, v in raw_mapping.items()}
        else:
            id_to_intent = {int(k): v for k, v in raw_mapping.items()}
        intent_checkpoint['id_to_intent'] = id_to_intent
        intent_model = IntentClassifierModel(intent_checkpoint)
        intent_model.load_state_dict(intent_checkpoint['model_state_dict'])
        intent_model.to(device).eval()
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_checkpoint['tokenizer_name'])
        logging.info("✅ Modelo de intenciones cargado.")
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR MODELO 1: {e}")
        return

    # --- Cargar Modelo 2 (NER) ---
    logging.info(f"Cargando Modelo 2 (NER) desde {ner_model_path}...")
    try:
        # --- LA PUTA LÓGICA COPIADA DE predict_ner_only.py ---
        ner_checkpoint = torch.load(ner_model_path, map_location=device, weights_only=False)
        ner_config = AutoConfig.from_pretrained(ner_checkpoint['tokenizer_name'], **ner_checkpoint['config'])
        ner_model = AutoModelForTokenClassification.from_config(ner_config)
        ner_model.load_state_dict(ner_checkpoint['model_state_dict'])
        ner_model.to(device).eval()
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_checkpoint['tokenizer_name'])
        
        # Creamos el pipeline que SÍ funciona
        ner_extractor = pipeline(
            "token-classification",
            model=ner_model,
            tokenizer=ner_tokenizer,
            device=device,
            aggregation_strategy="simple"
        )
        logging.info("✅ Modelo NER cargado como pipeline.")
    except Exception as e:
        logging.exception(f"❌ FALLO AL CARGAR MODELO 2: {e}")
        return

    # --- Bucle de Pruebas ---
    for phrase in test_phrases:
        print("\n" + "="*70)
        logging.info(f"PROCESANDO: '{phrase}'")
        
        # --- PREDICCIÓN DE INTENCIÓN ---
        print("\n--- SALIDA MODELO 1 (INTENCIONES) ---")
        try:
            inputs = intent_tokenizer(phrase, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = intent_model(**inputs)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            intent_predictions = sorted(
                [{"intent": id_to_intent.get(i, f"unknown_{i}"), "confidence": prob.item()} for i, prob in enumerate(probabilities)],
                key=lambda x: x['confidence'],
                reverse=True
            )
            print(json.dumps(intent_predictions, indent=2, ensure_ascii=False))
        except Exception as e:
            logging.exception(f"Error prediciendo intención: {e}")

        # --- PREDICCIÓN DE ENTIDADES (NER) ---
        print("\n--- SALIDA MODELO 2 (ENTIDADES) ---")
        try:
            # --- USAMOS EL PUTO PIPELINE QUE SÍ FUNCIONA ---
            entities = ner_extractor(phrase)
            
            formatted_entities = [
                {'label': e['entity_group'], 'value': e['word'], 'score': float(e['score'])} 
                for e in entities
            ]
            
            print(json.dumps(formatted_entities, indent=2, ensure_ascii=False))
        except Exception as e:
            logging.exception(f"Error prediciendo entidades: {e}")

if __name__ == "__main__":
    
    INTENT_MODEL_PT_PATH = Path("small-intent-detector-cpu/output/intent_classifier_final.pt")
    NER_MODEL_PT_PATH = Path("ner_training/models/get_news_extractor/get_news_extractor.pt")

    if not INTENT_MODEL_PT_PATH.exists() or not NER_MODEL_PT_PATH.exists():
        logging.error("¡Error! No se encontraron los archivos de modelo .pt.")
    else:
        test_phrases = [
            "Deseo noticias sobre Arévalo del presente año",
            "Qué pasó ayer con el congreso",
            "Muéstrame lo último de la OEA",
            "Quiero pedir una pizza",
            "Cuál es el clima de hoy"
        ]
        debug_pipeline(INTENT_MODEL_PT_PATH, NER_MODEL_PT_PATH, test_phrases)

    print("\n" + "="*70)
    print("--- PRUEBA DE DEPURACIÓN FINALIZADA ---")