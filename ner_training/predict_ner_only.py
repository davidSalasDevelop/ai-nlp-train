# predict_ner_only.py
"""
Script de prueba directo para el modelo NER, cargado desde un archivo .pt.
Carga únicamente el modelo de extracción de entidades y realiza predicciones.
"""
import logging
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ner_model():
    print("\n--- INICIANDO PRUEBA DIRECTA DEL MODELO NER (desde .pt) ---")
    
    NER_MODEL_PT_PATH = Path("ner_training/models/get_news_extractor/get_news_extractor.pt")

    if not NER_MODEL_PT_PATH.exists():
        logging.error(f"¡Error! No se encontró el archivo del modelo NER en '{NER_MODEL_PT_PATH}'.")
        logging.error("Asegúrate de haber ejecutado 'ner_main.py' y que haya generado el .pt correctamente.")
        return

    try:
        logging.info(f"Cargando modelo NER desde: {NER_MODEL_PT_PATH}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(NER_MODEL_PT_PATH, map_location=device, weights_only=False)
        config = AutoConfig.from_pretrained(checkpoint['tokenizer_name'], **checkpoint['config'])
        model = AutoModelForTokenClassification.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        
        logging.info("✅ Modelo NER y Tokenizer cargados exitosamente.")
    except Exception as e:
        logging.exception(f"❌ Error al cargar el modelo NER desde el archivo .pt: {e}")
        return

    ner_extractor = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        aggregation_strategy="simple"
    )

    test_phrases = [
        "Deseo noticias sobre Arévalo del presente año",
        "Qué pasó ayer con el congreso",
        "Muéstrame lo último de la OEA",
        "Quiero pedir una pizza",
    ]

    for phrase in test_phrases:
        print("\n" + "="*60)
        logging.info(f"PROCESANDO: '{phrase}'")
        
        try:
            entities = ner_extractor(phrase)
            print("ENTIDADES EXTRAÍDAS:")
            
            # --- CORRECCIÓN DEFINITIVA AQUÍ ---
            # Convertimos explícitamente el score (que es numpy.float32) a un float de Python.
            formatted_entities = [
                {'label': e['entity_group'], 'value': e['word'], 'score': float(e['score'])} 
                for e in entities
            ]
            
            print(json.dumps(formatted_entities, indent=2, ensure_ascii=False))
        except Exception as e:
            logging.exception(f"Error durante la predicción para la frase: '{phrase}'")

    print("\n--- PRUEBA FINALIZADA ---")

if __name__ == "__main__":
    test_ner_model()