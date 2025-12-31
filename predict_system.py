# predict_system.py - SISTEMA COMPLETO DE PREDICCIÓN

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
from typing import Dict, List, Any

class NLUPredictor:
    """Sistema completo de predicción NLU"""
    
    def __init__(self, model_path: str = "nlu_complete_model.pt"):
        # Cargar checkpoint
        self.checkpoint = torch.load(model_path, map_location='cpu')
        
        # Cargar mapeos
        self.intent_to_id = self.checkpoint['intent_to_id']
        self.id_to_intent = self.checkpoint['id_to_intent']
        self.entity_to_id = self.checkpoint['entity_to_id']
        self.id_to_entity = self.checkpoint['id_to_entity']
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint.get('tokenizer_config', {}).get('pretrained_model_name_or_path',
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )
        
        # Configuración
        self.max_length = self.checkpoint.get('config', {}).get('max_length', 128)
        
        print(f"✅ Sistema NLU cargado")
        print(f"🎯 Intenciones: {list(self.intent_to_id.keys())}")
        print(f"🏷️  Entidades: {len(self.entity_to_id) - 1} tipos")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predice intención y extrae entidades"""
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # NOTA: Aquí necesitarías tu modelo cargado
        # Por ahora simulamos predicción
        intent_probs = self._simulate_intent_prediction(text)
        entities = self._extract_entities(text, encoding['offset_mapping'][0])
        
        # Obtener intención principal
        main_intent = max(intent_probs, key=intent_probs.get)
        confidence = intent_probs[main_intent]
        
        # Organizar entidades por tipo
        organized_entities = {}
        for entity in entities:
            entity_type = entity['type']
            if entity_type not in organized_entities:
                organized_entities[entity_type] = []
            organized_entities[entity_type].append(entity['text'])
        
        # Estructurar respuesta por intención
        result = self._structure_result(main_intent, organized_entities, confidence)
        
        return result
    
    def _simulate_intent_prediction(self, text: str) -> Dict[str, float]:
        """Simula predicción de intenciones (reemplazar con modelo real)"""
        # En producción, aquí iría tu modelo real
        intents = list(self.intent_to_id.keys())
        
        # Simulación simple basada en palabras clave
        text_lower = text.lower()
        
        scores = {}
        for intent in intents:
            score = 0.1  # Probabilidad base
            
            # Palabras clave por intención
            keywords = {
                "get_user_info": ["usuario", "perfil", "cuenta", "suscripción", "datos"],
                "get_news": ["noticias", "actualidad", "novedad", "información", "reportaje"],
                "get_date": ["fecha", "hora", "día", "tiempo", "calendario"],
                "get_business_information": ["empresa", "negocio", "compañía", "corporación"]
            }
            
            if intent in keywords:
                for keyword in keywords[intent]:
                    if keyword in text_lower:
                        score += 0.3
            
            scores[intent] = min(score, 0.99)
        
        # Normalizar
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _extract_entities(self, text: str, offset_mapping) -> List[Dict]:
        """Extrae entidades del texto (simulado)"""
        entities = []
        
        # Mapeo de palabras clave a tipos de entidad
        entity_patterns = {
            "SUBSCRIPTION": ["básica", "premium", "empresa", "gratuita", "anual", "mensual"],
            "TOPIC": ["tecnología", "deportes", "política", "economía", "salud", "entretenimiento"],
            "DATE_RANGE": ["hoy", "ayer", "semana", "mes", "días"],
            "DATE_TYPE": ["fecha", "hora", "día", "mes", "año"],
            "INFO_TYPE": ["contacto", "historia", "misión", "visión", "valores"]
        }
        
        text_lower = text.lower()
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    start = text_lower.find(pattern)
                    end = start + len(pattern)
                    
                    entities.append({
                        "type": entity_type,
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "confidence": 0.8
                    })
        
        return entities
    
    def _structure_result(self, intent: str, entities: Dict[str, List[str]], confidence: float) -> Dict[str, Any]:
        """Estructura el resultado según la intención"""
        
        result = {
            "text": "",
            "intent": intent,
            "confidence": f"{confidence:.1%}",
            "parameters": {},
            "entities": entities
        }
        
        # Estructurar parámetros según la intención
        if intent == "get_user_info":
            result["text"] = f"Información del usuario"
            result["parameters"] = {
                "subscription_type": entities.get("SUBSCRIPTION", ["No especificado"])[0] if entities.get("SUBSCRIPTION") else "No especificado",
                "date_range": entities.get("DATE_RANGE", []),
                "promotions": entities.get("PROMOTION", []),
                "payment_methods": entities.get("PAYMENT_METHOD", [])
            }
            
        elif intent == "get_news":
            result["text"] = f"Búsqueda de noticias"
            result["parameters"] = {
                "keywords": entities.get("TOPIC", []),
                "date_range": entities.get("DATE_RANGE", []),
                "tags": entities.get("TAG", []),
                "sources": entities.get("SOURCE", [])
            }
            
        elif intent == "get_date":
            result["text"] = f"Consulta de fecha/hora"
            result["parameters"] = {
                "date_type": entities.get("DATE_TYPE", ["fecha"])[0],
                "format": entities.get("FORMAT", ["DD/MM/YYYY"])[0] if entities.get("FORMAT") else "DD/MM/YYYY",
                "timezone": entities.get("TIMEZONE", ["local"])[0] if entities.get("TIMEZONE") else "local"
            }
            
        elif intent == "get_business_information":
            result["text"] = f"Información del negocio"
            result["parameters"] = {
                "information_type": entities.get("INFO_TYPE", ["general"])[0],
                "department": entities.get("DEPARTMENT", []),
                "documents": entities.get("DOCUMENT", [])
            }
        
        return result

# ==============================================================================
# INTERFAZ DE USUARIO
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de predicción NLU")
    parser.add_argument('--text', type=str, help='Texto a analizar')
    parser.add_argument('--file', type=str, help='Archivo con textos (uno por línea)')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    
    args = parser.parse_args()
    
    # Cargar predictor
    print("🔄 Cargando sistema NLU...")
    predictor = NLUPredictor()
    
    print("\n" + "="*60)
    print("🧠 SISTEMA NLU - PREDICCIÓN DE INTENCIONES")
    print("="*60)
    
    if args.interactive:
        # Modo interactivo
        print("\n📝 Modo interactivo (escribe 'salir' para terminar)")
        print("-" * 40)
        
        while True:
            text = input("\nTu mensaje: ").strip()
            
            if text.lower() in ['salir', 'exit', 'quit']:
                break
            
            if text:
                result = predictor.predict(text)
                
                print(f"\n🎯 Intención: {result['intent']} ({result['confidence']})")
                print(f"📝 Texto procesado: {result['text']}")
                
                if result['parameters']:
                    print(f"\n🔧 Parámetros extraídos:")
                    for param, value in result['parameters'].items():
                        print(f"   {param}: {value}")
                
                if result['entities']:
                    print(f"\n🏷️  Entidades detectadas:")
                    for entity_type, values in result['entities'].items():
                        print(f"   {entity_type}: {', '.join(values)}")
    
    elif args.file:
        # Procesar archivo
        print(f"\n📄 Procesando archivo: {args.file}")
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = []
        for text in texts:
            result = predictor.predict(text)
            results.append(result)
            
            print(f"\n📝 Texto: {text}")
            print(f"🎯 Intención: {result['intent']} ({result['confidence']})")
        
        # Guardar resultados
        with open('predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: predictions.json")
    
    elif args.text:
        # Procesar texto único
        print(f"\n📝 Texto: {args.text}")
        result = predictor.predict(args.text)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        # Ejemplos por defecto
        examples = [
            "Quiero ver mi información de suscripción premium",
            "Noticias sobre tecnología de esta semana",
            "¿Qué fecha es hoy en formato DD/MM/YYYY?",
            "Información de contacto de la empresa"
        ]
        
        print("\n🧪 Ejemplos de prueba:")
        for example in examples:
            print(f"\n📝 '{example}'")
            result = predictor.predict(example)
            print(f"   🎯 Intención: {result['intent']} ({result['confidence']})")
            if result['parameters']:
                params = list(result['parameters'].items())[:2]
                print(f"   🔧 Parámetros: {dict(params)}...")

if __name__ == "__main__":
    main()