"""
PREDICCIÓN NER - VERSIÓN CORREGIDA
"""
import logging
import json
from pathlib import Path
import torch
from torch import nn 
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ner_model():
    print("\n--- INICIANDO PRUEBA DIRECTA DEL MODELO NER (desde .pt) ---")
    
    NER_MODEL_PT_PATH = Path("ner_training/models/get_news_extractor/get_news_extractor.pt")

    if not NER_MODEL_PT_PATH.exists():
        logging.error(f"¡Error! No se encontró el archivo del modelo NER en '{NER_MODEL_PT_PATH}'.")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Usando dispositivo: {device}")
        
        # 1. Cargar checkpoint
        checkpoint = torch.load(NER_MODEL_PT_PATH, map_location=device, weights_only=False)
        
        # 2. Información del checkpoint
        tokenizer_name = checkpoint['tokenizer_name']
        id2label = checkpoint['id2label']
        num_labels = len(id2label)
        
        logging.info(f"Tokenizer: {tokenizer_name}")
        logging.info(f"Número de etiquetas: {num_labels}")
        
        # 3. Analizar estructura REAL del modelo
        state_dict = checkpoint['model_state_dict']
        
        # Las claves nos dicen la arquitectura exacta:
        # classifier.1.weight = [256, 768]  -> Capa oculta: 256 neuronas
        # classifier.3.weight = [11, 256]   -> Capa de salida: 11 etiquetas
        # Esto significa: Dropout -> Linear(768->256) -> ReLU -> Linear(256->11)
        
        # 4. Crear modelo CON LA MISMA ARQUITECTURA
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()}
        )
        
        # Crear modelo base
        model = AutoModelForTokenClassification.from_config(config)
        
        # Reemplazar classifier con la arquitectura EXACTA
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout (capa 0)
            nn.Linear(768, 256),  # Capa oculta (capa 1)
            nn.ReLU(),  # Activación (capa 2)
            nn.Linear(256, num_labels)  # Capa de salida (capa 3)
        )
        
        # 5. Cargar pesos
        model.load_state_dict(state_dict)
        model.to(device).eval()
        
        # 6. Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        logging.info("✅ Modelo NER cargado exitosamente")
        logging.info("Arquitectura del classifier:")
        logging.info(model.classifier)
        
        # 7. Crear pipeline
        ner_extractor = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        
        # 8. Pruebas
        test_phrases = [
            "Deseo noticias sobre Bernardo Arévalo del presente año",
            "Qué pasó ayer con el Congreso de la República",
            "Información de la inflación de hace 3 meses",
            "Muéstrame lo último de la OEA en Washington",
        ]
        
        for phrase in test_phrases:
            print(f"\n{'='*60}")
            print(f"📝 Texto: '{phrase}'")
            
            entities = ner_extractor(phrase)
            
            if entities:
                for e in entities:
                    print(f"  🔹 {e['entity_group']}: '{e['word']}' (confianza: {e['score']:.2%})")
            else:
                print("  ℹ️  No se encontraron entidades")
        
        print(f"\n{'='*60}")
        print("✅ PRUEBA COMPLETADA")
        
    except Exception as e:
        logging.exception(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ner_model()