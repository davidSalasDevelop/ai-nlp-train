# generate_dataset.py - GENERA 500+ EJEMPLOS PARA TUS 4 INTENCIONES

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ==============================================================================
# CONFIGURACIÓN DE INTENCIONES Y PARÁMETROS
# ==============================================================================

INTENT_CONFIG = {
    "get_user_info": {
        "español": [
            "Ver mi información de usuario",
            "Mostrar mi perfil",
            "Datos de mi cuenta",
            "Información de suscripción",
            "Estado de mi cuenta",
            "Ver mis datos personales",
            "Consulta mi perfil",
            "Mostrar información del usuario",
            "Detalles de mi cuenta",
            "Ver configuración de usuario"
        ],
        "english": [
            "Show my user information",
            "Display my profile",
            "My account details",
            "Subscription information",
            "Account status",
            "View my personal data",
            "Check my profile",
            "Show user information",
            "My account details",
            "View user settings"
        ],
        "parameters": {
            "subscription": ["básica", "premium", "empresa", "gratuita", "anual", "mensual"],
            "start_date": ["fecha inicio", "cuándo empecé", "desde cuándo", "inicio suscripción"],
            "end_date": ["fecha fin", "hasta cuándo", "cuándo termina", "vencimiento"],
            "promotions": ["promociones", "descuentos", "ofertas", "códigos promocionales"],
            "payment_method": ["tarjeta", "paypal", "transferencia", "efectivo"]
        }
    },
    
    "get_news": {
        "español": [
            "Noticias sobre {topic}",
            "Últimas noticias de {topic}",
            "Qué hay de nuevo en {topic}",
            "Actualidad sobre {topic}",
            "Novedades en {topic}",
            "Información sobre {topic}",
            "Reportajes de {topic}",
            "Tendencias en {topic}",
            "Lo último en {topic}",
            "Noticias recientes de {topic}"
        ],
        "english": [
            "News about {topic}",
            "Latest news on {topic}",
            "What's new in {topic}",
            "Updates about {topic}",
            "Recent news on {topic}",
            "Information about {topic}",
            "Reports on {topic}",
            "Trends in {topic}",
            "Latest in {topic}",
            "Recent updates on {topic}"
        ],
        "parameters": {
            "keywords": ["tecnología", "deportes", "política", "economía", "salud", "entretenimiento", 
                        "technology", "sports", "politics", "economy", "health", "entertainment"],
            "date_range": ["hoy", "ayer", "esta semana", "este mes", "últimos 7 días", "last week", "this month"],
            "tags": ["última hora", "breaking", "análisis", "exclusiva", "local", "internacional", 
                    "breaking news", "analysis", "exclusive", "local", "international"],
            "source": ["periódico", "revista", "blog", "redes sociales", "newspaper", "magazine", "blog"]
        }
    },
    
    "get_date": {
        "español": [
            "Qué fecha es hoy",
            "Cuál es la hora",
            "Fecha y hora actual",
            "Día de la semana",
            "Qué día es hoy",
            "Hora actual",
            "Fecha de hoy",
            "Cuál es la fecha",
            "Qué hora es",
            "Dime la fecha"
        ],
        "english": [
            "What date is today",
            "What time is it",
            "Current date and time",
            "Day of the week",
            "What day is today",
            "Current time",
            "Today's date",
            "What's the date",
            "What's the time",
            "Tell me the date"
        ],
        "parameters": {
            "date_type": ["fecha", "hora", "día", "mes", "año", "date", "time", "day", "month", "year"],
            "format": ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD", "12h", "24h", "full", "short"],
            "timezone": ["UTC", "local", "EST", "PST", "CET", "GMT"]
        }
    },
    
    "get_business_information": {
        "español": [
            "Información de la empresa",
            "Datos del negocio",
            "Información corporativa",
            "Detalles de la compañía",
            "Sobre la empresa",
            "Información comercial",
            "Datos empresariales",
            "Información de la organización",
            "Detalles del negocio",
            "Información institucional"
        ],
        "english": [
            "Company information",
            "Business details",
            "Corporate information",
            "Company details",
            "About the company",
            "Business information",
            "Enterprise data",
            "Organization information",
            "Business details",
            "Institutional information"
        ],
        "parameters": {
            "info_type": ["contacto", "historia", "misión", "visión", "valores", "empleados", "contact", 
                         "history", "mission", "vision", "values", "employees"],
            "department": ["ventas", "soporte", "marketing", "finanzas", "RRHH", "sales", "support", 
                          "marketing", "finance", "HR"],
            "document": ["informe anual", "estados financieros", "políticas", "annual report", 
                        "financial statements", "policies"]
        }
    }
}

# ==============================================================================
# GENERADOR DE EJEMPLOS
# ==============================================================================

def generate_example(intent_name: str, language: str = "es") -> Dict[str, Any]:
    """Genera un ejemplo para una intención específica"""
    config = INTENT_CONFIG[intent_name]
    
    # Seleccionar plantilla
    templates = config["español"] if language == "es" else config["english"]
    template = random.choice(templates)
    
    # Generar texto
    text = template
    entities = []
    
    # Añadir parámetros según la intención
    if intent_name == "get_user_info":
        # Añadir parámetros de usuario
        params = random.sample(list(config["parameters"].keys()), random.randint(1, 3))
        for param in params:
            value = random.choice(config["parameters"][param])
            if language == "es":
                text += f", {value}"
            else:
                text += f", {value}"
            # Simular entidad
            start = text.find(value)
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(value),
                    "label": param.upper()
                })
    
    elif intent_name == "get_news":
        # Añadir topic
        topic = random.choice(config["parameters"]["keywords"])
        text = text.format(topic=topic)
        
        # Entidad para topic
        start = text.find(topic)
        if start != -1:
            entities.append({
                "start": start,
                "end": start + len(topic),
                "label": "TOPIC"
            })
        
        # Añadir parámetros adicionales
        if random.random() > 0.5:
            date_range = random.choice(config["parameters"]["date_range"])
            text += f" {date_range}"
            start = text.find(date_range)
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(date_range),
                    "label": "DATE_RANGE"
                })
    
    elif intent_name == "get_date":
        # Añadir tipo de fecha/hora
        if random.random() > 0.3:
            date_type = random.choice(config["parameters"]["date_type"])
            text += f" en formato {date_type}"
            start = text.find(date_type)
            if start != -1:
                entities.append({
                    "start": start,
                    "end": start + len(date_type),
                    "label": "DATE_TYPE"
                })
    
    elif intent_name == "get_business_information":
        # Añadir tipo de información
        info_type = random.choice(config["parameters"]["info_type"])
        text += f" sobre {info_type}"
        
        start = text.find(info_type)
        if start != -1:
            entities.append({
                "start": start,
                "end": start + len(info_type),
                "label": "INFO_TYPE"
            })
    
    # Añadir variaciones lingüísticas
    variations = [
        ("Por favor, ", ""),
        ("Necesito ", ""),
        ("Podrías ", ""),
        ("Me gustaría ", ""),
        ("Quisiera ", ""),
        ("", "?"),
        ("", "."),
        ("", " por favor")
    ]
    
    prefix, suffix = random.choice(variations)
    text = prefix + text + suffix
    
    # Capitalizar
    if random.random() > 0.7:
        text = text.capitalize()
    
    return {
        "text": text.strip(),
        "language": language,
        "intent": intent_name,
        "entities": entities
    }

def generate_dataset(num_examples: int = 500) -> List[Dict[str, Any]]:
    """Genera dataset completo"""
    dataset = []
    
    # Distribuir ejemplos por intención
    intents = list(INTENT_CONFIG.keys())
    examples_per_intent = num_examples // len(intents)
    
    print(f"🔧 Generando dataset con {num_examples} ejemplos...")
    print(f"🎯 Intenciones: {intents}")
    
    for intent in intents:
        print(f"\n📝 Generando {examples_per_intent} ejemplos para '{intent}'...")
        
        for i in range(examples_per_intent):
            # Alternar idiomas
            language = "es" if i % 2 == 0 else "en"
            
            example = generate_example(intent, language)
            dataset.append(example)
            
            if i < 3:  # Mostrar primeros 3 ejemplos
                print(f"   {i+1}. {example['text']}")
    
    # Mezclar dataset
    random.shuffle(dataset)
    
    print(f"\n✅ Dataset generado: {len(dataset)} ejemplos")
    
    # Estadísticas
    stats = {}
    for example in dataset:
        intent = example["intent"]
        stats[intent] = stats.get(intent, 0) + 1
    
    print("\n📊 Estadísticas:")
    for intent, count in stats.items():
        print(f"   {intent}: {count} ejemplos")
    
    return dataset

# ==============================================================================
# FUNCIONES PARA AÑADIR NUEVAS INTENCIONES
# ==============================================================================

def add_new_intent(intent_name: str, config: Dict[str, Any]):
    """Añade una nueva intención al sistema"""
    if intent_name in INTENT_CONFIG:
        print(f"⚠️  La intención '{intent_name}' ya existe")
        return False
    
    INTENT_CONFIG[intent_name] = config
    print(f"✅ Intención '{intent_name}' añadida exitosamente")
    print(f"   Parámetros: {list(config['parameters'].keys())}")
    
    # Guardar configuración actualizada
    save_config()
    
    return True

def save_config():
    """Guarda la configuración actualizada"""
    with open("small-intent-detector-cpu/intent_config.json", "w", encoding="utf-8") as f:
        json.dump(INTENT_CONFIG, f, indent=2, ensure_ascii=False)
    print("💾 Configuración guardada en intent_config.json")

def load_config():
    """Carga la configuración desde archivo"""
    try:
        with open("small-intent-detector-cpu/intent_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        INTENT_CONFIG.update(config)
        print("📂 Configuración cargada desde intent_config.json")
    except FileNotFoundError:
        print("📝 Usando configuración por defecto")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador de dataset NLU")
    parser.add_argument("--size", type=int, default=500, help="Número de ejemplos a generar")
    parser.add_argument("--output", type=str, default="small-intent-detector/dataset_v2.json", help="Archivo de salida")
    parser.add_argument("--add-intent", action="store_true", help="Añadir nueva intención")
    
    args = parser.parse_args()
    
    # Cargar configuración existente
    load_config()
    
    if args.add_intent:
        # Interfaz para añadir nueva intención
        print("\n➕ AÑADIR NUEVA INTENCIÓN")
        intent_name = input("Nombre de la intención (ej: get_products): ").strip()
        
        if intent_name:
            print(f"\n📝 Configurando '{intent_name}'...")
            
            # Plantillas en español
            print("\n📌 Plantillas en español (separadas por '|'):")
            es_templates = input("Ej: 'Información sobre {producto}|Precio de {producto}': ").strip()
            es_templates = [t.strip() for t in es_templates.split("|") if t.strip()]
            
            # Plantillas en inglés
            print("\n📌 Plantillas en inglés (separadas por '|'):")
            en_templates = input("Ej: 'Information about {product}|Price of {product}': ").strip()
            en_templates = [t.strip() for t in en_templates.split("|") if t.strip()]
            
            # Parámetros
            print("\n🔧 Parámetros a extraer (separados por coma):")
            params_input = input("Ej: producto, precio, marca, categoría: ").strip()
            parameters = {}
            
            for param in params_input.split(","):
                param = param.strip()
                if param:
                    print(f"   Valores para '{param}' (separados por coma):")
                    values = input(f"   Ej: básico, premium, estándar: ").strip()
                    parameters[param] = [v.strip() for v in values.split(",") if v.strip()]
            
            # Crear configuración
            new_config = {
                "español": es_templates,
                "english": en_templates,
                "parameters": parameters
            }
            
            # Añadir intención
            add_new_intent(intent_name, new_config)
    
    else:
        # Generar dataset
        dataset = generate_dataset(args.size)
        
        # Guardar dataset
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Dataset guardado en: {args.output}")
        print(f"📏 Tamaño: {len(dataset)} ejemplos")
        print(f"🎯 Intenciones: {list(INTENT_CONFIG.keys())}")
        
        # Guardar configuración
        save_config()
        
        # Mostrar ejemplo
        print(f"\n📄 Ejemplo del dataset:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))