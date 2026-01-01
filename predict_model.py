#!/usr/bin/env python3
"""
predict_model.py - Archivo simple para probar el modelo entrenado

Uso:
    python predict_model.py "Texto a predecir"
    python predict_model.py --file archivo.txt
    python predict_model.py --interactive
"""

import torch
import json
import argparse
from transformers import AutoTokenizer
import sys

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

MODEL_FILE = "intent_classifier_final.pt"
MAX_LENGTH = 64

# ==============================================================================
# CARGAR MODELO
# ==============================================================================

def load_model():
    """Cargar el modelo entrenado"""
    print(f"📥 Cargando modelo desde: {MODEL_FILE}")
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        
        # Extraer información del checkpoint
        model_config = checkpoint['config']
        intent_to_id = checkpoint['intent_to_id']
        id_to_intent = checkpoint['id_to_intent']
        tokenizer_name = checkpoint['tokenizer_name']
        max_length = checkpoint.get('max_length', MAX_LENGTH)
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   • Tokenizer: {tokenizer_name}")
        print(f"   • Intenciones: {len(id_to_intent)}")
        print(f"   • Accuracy durante entrenamiento: {checkpoint.get('val_accuracy', 'N/A')}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return {
            'checkpoint': checkpoint,
            'tokenizer': tokenizer,
            'id_to_intent': id_to_intent,
            'intent_to_id': intent_to_id,
            'max_length': max_length,
            'model_config': model_config
        }
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

# ==============================================================================
# PREDECIR
# ==============================================================================

def predict(text, model_info):
    """Predecir la intención de un texto"""
    tokenizer = model_info['tokenizer']
    checkpoint = model_info['checkpoint']
    id_to_intent = model_info['id_to_intent']
    max_length = model_info['max_length']
    
    # Tokenizar texto
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Crear modelo simple para inferencia
    from transformers import AutoModel
    import torch.nn as nn
    
    class InferenceModel(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            self.bert = AutoModel.from_pretrained(checkpoint['tokenizer_name'])
            hidden_size = self.bert.config.hidden_size
            self.classifier = nn.Linear(hidden_size, len(checkpoint['id_to_intent']))
            
            # Cargar pesos
            self.load_state_dict(checkpoint['model_state_dict'])
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            return self.classifier(pooled)
    
    # Crear y usar modelo
    model = InferenceModel(checkpoint)
    model.eval()
    
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(logits, dim=1)[0]
        
        # Obtener top 3 predicciones
        top_probs, top_indices = torch.topk(probs, 3)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            intent = id_to_intent[idx.item()]
            confidence = prob.item() * 100
            results.append({
                'intent': intent,
                'confidence': confidence
            })
    
    return results

# ==============================================================================
# FUNCIONES DE INTERFAZ
# ==============================================================================

def print_prediction(text, results):
    """Mostrar predicción de forma bonita"""
    print(f"\n📝 Texto: '{text}'")
    print(f"🔍 Longitud: {len(text)} caracteres")
    print(f"\n🎯 PREDICCIONES:")
    
    for i, result in enumerate(results, 1):
        confidence = result['confidence']
        intent = result['intent']
        
        # Barra de progreso simple
        bars = int(confidence / 5)  # 20 barras máx (100/5)
        bar = "█" * bars + "░" * (20 - bars)
        
        print(f"   {i}. {intent}")
        print(f"      {bar} {confidence:.1f}%")
    
    print()

def interactive_mode(model_info):
    """Modo interactivo"""
    print("\n" + "="*60)
    print("💬 MODO INTERACTIVO")
    print("="*60)
    print("Escribe textos para predecir su intención.")
    print("Escribe 'quit', 'exit' o 'q' para salir.")
    print("-" * 60)
    
    while True:
        try:
            text = input("\n📝 Ingresa texto: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            if not text:
                print("⚠️  Por favor ingresa algún texto")
                continue
            
            results = predict(text, model_info)
            print_prediction(text, results)
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_mode(file_path, model_info):
    """Procesar un archivo con múltiples textos"""
    print(f"\n📁 Procesando archivo: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"📄 {len(texts)} textos encontrados")
        print("-" * 60)
        
        all_results = []
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}] Procesando...")
            results = predict(text, model_info)
            print_prediction(text, results)
            all_results.append({'text': text, 'predictions': results})
        
        # Guardar resultados
        output_file = "predictions_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados en: {output_file}")
        
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🤖 PREDICTOR DE INTENCIONES")
    print("="*60)
    
    # Cargar modelo
    model_info = load_model()
    if not model_info:
        return
    
    # Mostrar información del modelo
    print(f"\n📊 INFORMACIÓN DEL MODELO:")
    print(f"   • Intenciones disponibles: {', '.join(model_info['id_to_intent'].values())}")
    print(f"   • Máxima longitud: {model_info['max_length']} tokens")
    print(f"   • Batch size usado: {model_info['model_config'].get('batch_size', 'N/A')}")
    print(f"   • Learning rate usado: {model_info['model_config'].get('learning_rate', 'N/A')}")
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Predecir intenciones de texto')
    parser.add_argument('text', nargs='?', help='Texto a predecir')
    parser.add_argument('--file', '-f', help='Archivo con textos a predecir')
    parser.add_argument('--interactive', '-i', action='store_true', help='Modo interactivo')
    
    args = parser.parse_args()
    
    # Determinar modo de operación
    if args.interactive:
        interactive_mode(model_info)
    
    elif args.file:
        batch_mode(args.file, model_info)
    
    elif args.text:
        results = predict(args.text, model_info)
        print_prediction(args.text, results)
    
    else:
        # Modo por defecto: pedir texto
        print("\n✍️  Modo simple - Ingresa un texto para predecir")
        print("-" * 40)
        
        while True:
            text = input("\n📝 Texto (o 'quit' para salir): ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 ¡Hasta luego!")
                break
            
            if not text:
                print("⚠️  Por favor ingresa algún texto")
                continue
            
            results = predict(text, model_info)
            print_prediction(text, results)

# ==============================================================================
# EJECUTAR
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")