# predict_fixed.py - USANDO LA MISMA ARQUITECTURA QUE ENTRENASTE
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
from typing import Dict, List
import sys

# ==============================================================================
# LA MISMA ARQUITECTURA QUE USÓ train_complete.py
# ==============================================================================

class SimpleIntentModel(nn.Module):
    """EXACTAMENTE la misma clase que usaste para entrenar"""
    def __init__(self, num_intents: int, num_entities: int):
        super().__init__()
        
        # Backbone ligero
        self.bert = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        bert_hidden_size = self.bert.config.hidden_size
        
        # Congelar la mayoría de capas para CPU (como en entrenamiento)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Descongelar última capa
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        # Clasificador simple - ¡MISMA ESTRUCTURA!
        self.intent_classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_intents)
        )
        
        # Clasificador de entidades (opcional)
        self.entity_classifier = nn.Linear(bert_hidden_size, num_entities) if num_entities > 0 else None
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Usar [CLS] para intención
        pooled_output = outputs.last_hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(pooled_output)
        
        # Entidades (opcional)
        entity_logits = None
        if self.entity_classifier:
            entity_logits = self.entity_classifier(outputs.last_hidden_state)
        
        return intent_logits, entity_logits

# ==============================================================================
# PREDICTOR COMPATIBLE
# ==============================================================================

class CompatiblePredictor:
    def __init__(self, model_path="nlu_model_cpu.pt"):
        print("🔧 Cargando modelo entrenado...")
        
        # Cargar checkpoint
        self.checkpoint = torch.load(model_path, map_location='cpu')
        
        # Verificar contenido
        print(f"📁 Checkpoint keys: {list(self.checkpoint.keys())}")
        
        # Mapeos
        self.intent_to_id = self.checkpoint['intent_to_id']
        self.id_to_intent = self.checkpoint['id_to_intent']
        
        print(f"✅ Modelo cargado:")
        print(f"   Intenciones: {list(self.intent_to_id.keys())}")
        
        # Tokenizer
        model_name = self.checkpoint.get('tokenizer_name', 
                      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configuración
        self.max_length = self.checkpoint.get('max_length', 128)
        
        # Crear modelo con arquitectura EXACTA
        num_entities = len(self.checkpoint.get('entity_to_id', {}))
        self.model = SimpleIntentModel(
            num_intents=len(self.intent_to_id),
            num_entities=num_entities
        )
        
        # Cargar pesos
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"🎯 Modelo preparado para {len(self.intent_to_id)} intenciones")
    
    def predict(self, text: str, top_k: int = 3):
        """Predice intención"""
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Predecir
        with torch.no_grad():
            intent_logits, _ = self.model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(intent_logits, dim=1)[0]
        
        # Obtener top-k
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.id_to_intent)))
        
        resultados = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            resultados.append({
                'intent': self.id_to_intent[idx],
                'confidence': f"{prob*100:.1f}%",
                'score': prob
            })
        
        return resultados
    
    def batch_predict(self, texts: List[str]):
        """Predice múltiples textos"""
        resultados = []
        
        for text in texts:
            preds = self.predict(text, top_k=1)
            resultados.append({
                'text': text,
                'intent': preds[0]['intent'],
                'confidence': preds[0]['confidence']
            })
        
        return resultados

# ==============================================================================
# INTERFAZ SIMPLE
# ==============================================================================

def test_model_quick():
    """Prueba rápida del modelo"""
    
    print("🧪 PRUEBA RÁPIDA DEL MODELO ENTRENADO")
    print("="*50)
    
    # Cargar predictor
    predictor = CompatiblePredictor("nlu_model_cpu.pt")
    
    # Textos de prueba
    test_cases = [
        ("es", "Quiero ver mi información de usuario", "get_user_info"),
        ("es", "Noticias sobre tecnología", "get_news"),
        ("es", "¿Qué fecha es hoy?", "get_date"),
        ("es", "Información de la empresa", "get_business_information"),
        ("en", "Show my user profile", "get_user_info"),
        ("en", "Latest news on sports", "get_news"),
        ("en", "What's the current date?", "get_date"),
        ("en", "Company business details", "get_business_information")
    ]
    
    print("\n📊 Resultados:")
    print("-" * 50)
    
    for lang, text, expected in test_cases:
        results = predictor.predict(text, top_k=1)
        predicted = results[0]['intent']
        confidence = results[0]['confidence']
        
        status = "✅" if predicted == expected else "❌"
        print(f"{status} [{lang}] '{text}'")
        print(f"   🎯 Predicción: {predicted} ({confidence})")
        print(f"   📍 Esperado: {expected}")
        print()

def interactive_mode():
    """Modo interactivo"""
    
    predictor = CompatiblePredictor("nlu_model_cpu.pt")
    
    print("\n💬 MODO INTERACTIVO")
    print("="*50)
    print("Escribe textos para analizar (o 'salir' para terminar)")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n📝 Tu texto: ").strip()
            
            if text.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if not text:
                continue
            
            results = predictor.predict(text, top_k=3)
            
            print(f"\n🔍 Resultados:")
            for i, res in enumerate(results):
                prefix = "🎯" if i == 0 else "  "
                print(f"{prefix} {res['intent']} - {res['confidence']}")
        
        except KeyboardInterrupt:
            print("\n👋 Interrumpido por usuario")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def export_predictions(texts_file: str, output_file: str = "predictions.json"):
    """Procesa un archivo de textos"""
    
    predictor = CompatiblePredictor("nlu_model_cpu.pt")
    
    print(f"📄 Procesando archivo: {texts_file}")
    
    # Leer textos
    with open(texts_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Predecir
    results = predictor.batch_predict(texts)
    
    # Guardar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Resultados guardados en: {output_file}")
    
    # Mostrar resumen
    print(f"\n📊 Resumen ({len(results)} textos):")
    intent_counts = {}
    for res in results:
        intent = res['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent, count in intent_counts.items():
        print(f"   {intent}: {count} textos")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predictor compatible con tu modelo entrenado")
    parser.add_argument('--test', action='store_true', help='Prueba rápida con ejemplos')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    parser.add_argument('--text', type=str, help='Texto a analizar')
    parser.add_argument('--file', type=str, help='Archivo con textos a procesar')
    parser.add_argument('--output', type=str, default='predictions.json', help='Archivo de salida')
    
    args = parser.parse_args()
    
    if args.test:
        test_model_quick()
    
    elif args.interactive:
        interactive_mode()
    
    elif args.text:
        predictor = CompatiblePredictor("nlu_model_cpu.pt")
        results = predictor.predict(args.text, top_k=3)
        
        print(f"\n📝 Texto: {args.text}")
        print("="*40)
        
        for i, res in enumerate(results):
            prefix = "🎯 PRINCIPAL" if i == 0 else f"📊 Alternativa {i}"
            print(f"{prefix}: {res['intent']} ({res['confidence']})")
    
    elif args.file:
        export_predictions(args.file, args.output)
    
    else:
        # Modo por defecto: prueba rápida
        test_model_quick()