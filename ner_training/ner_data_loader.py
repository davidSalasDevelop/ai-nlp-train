import logging
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _reconstruct_generic(example, ner_tags_list, tags_column_name):
    """
    Reconstruye texto y entidades normalizando las etiquetas a un formato estándar.
    Mapea todo a: PER, ORG, LOC, MISC.
    """
    text = ""
    entities = []
    current_pos = 0
    tag_ids = example[tags_column_name]
    
    for i, token in enumerate(example['tokens']):
        tag_id = tag_ids[i]
        raw_tag = ner_tags_list[tag_id] # Ej: 'B-PER', 'B-Persona', 'B-Location'
        
        # --- Normalización de Etiquetas ---
        # Convertimos etiquetas específicas de datasets (ej: 'Person') a estándar ('PER')
        label_upper = raw_tag.upper()
        norm_label = None
        
        if 'PER' in label_upper or 'PERSON' in label_upper: norm_label = 'PER'
        elif 'ORG' in label_upper: norm_label = 'ORG'
        elif 'LOC' in label_upper or 'LUGAR' in label_upper: norm_label = 'LOC'
        elif 'MISC' in label_upper or 'OTHER' in label_upper: norm_label = 'MISC'
        elif 'DATE' in label_upper or 'FECHA' in label_upper: norm_label = 'MISC' # Guardamos fechas como MISC por ahora
        
        # Reconstrucción de texto
        if i > 0:
            text += " "
            current_pos += 1
        
        start_char = current_pos
        text += token
        current_pos += len(token)
        
        if raw_tag.startswith('B-') and norm_label:
            entity_start = start_char
            
            # Buscar final de entidad
            end_token_idx = i
            while (end_token_idx + 1 < len(example['tokens'])):
                next_raw_tag = ner_tags_list[tag_ids[end_token_idx + 1]]
                # Simplificación: si sigue siendo I-, asumimos que es la misma entidad
                if next_raw_tag.startswith('I-'):
                    end_token_idx += 1
                else:
                    break
            
            length_span = len(" ".join(example['tokens'][i : end_token_idx + 1]))
            entities.append({
                'label': norm_label,
                'start': entity_start,
                'end': entity_start + length_span
            })
            
    return {'text': text, 'entities': entities}

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label_list: list, max_length: int = 128):
    """
    Carga Wikiann y PlanTL-GOB-ES/lener-es de forma segura.
    """
    list_of_train = []
    list_of_test = []

    # --- 1. WIKIANN (La base segura) ---
    try:
        logging.info("📚 Cargando Dataset 1: 'wikiann' (Español)...")
        ds1 = load_dataset("wikiann", "es")
        tags1 = ds1['train'].features['ner_tags'].feature.names
        
        train1 = ds1['train'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['train'].column_names)
        test1 = ds1['validation'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['validation'].column_names)
        
        list_of_train.append(train1)
        list_of_test.append(test1)
        logging.info(f"   ✅ Wikiann cargado: {len(train1)} ejemplos.")
    except Exception as e:
        logging.error(f"❌ Error crítico cargando Wikiann: {e}")

    # --- 2. LENER-ES (Oficial Gobierno España - Legal/Noticias) ---
    try:
        logging.info("⚖️ Cargando Dataset 2: 'PlanTL-GOB-ES/lener-es'...")
        ds2 = load_dataset("PlanTL-GOB-ES/lener-es", "lener")
        tags2 = ds2['train'].features['ner_tags'].feature.names
        
        train2 = ds2['train'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags2, 'tags_column_name': 'ner_tags'}, remove_columns=ds2['train'].column_names)
        test2 = ds2['validation'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags2, 'tags_column_name': 'ner_tags'}, remove_columns=ds2['validation'].column_names)
        
        list_of_train.append(train2)
        list_of_test.append(test2)
        logging.info(f"   ✅ Lener-ES cargado: {len(train2)} ejemplos.")
    except Exception as e:
        logging.warning(f"⚠️ No se pudo cargar Lener-ES (continuando solo con Wikiann): {e}")

    if not list_of_train:
        raise RuntimeError("¡No se pudo cargar ningún dataset! Revisa tu conexión a internet.")

    # --- COMBINAR ---
    logging.info("🔗 Combinando datasets...")
    final_train = concatenate_datasets(list_of_train).shuffle(seed=42)
    final_test = concatenate_datasets(list_of_test).shuffle(seed=42)
    logging.info(f"🚀 DATASET TOTAL: {len(final_train)} entrenamiento | {len(final_test)} prueba")

    # --- TOKENIZAR ---
    label2id = {label: i for i, label in enumerate(label_list)}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=False, max_length=max_length, return_offsets_mapping=True)
        all_labels = []
        for i, entities in enumerate(examples["entities"]):
            offsets = tokenized_inputs["offset_mapping"][i]
            labels = [label2id["O"]] * len(offsets)
            for entity in entities:
                if f"B-{entity['label']}" not in label2id: continue
                token_start_index, token_end_index = None, None
                for idx, (start, end) in enumerate(offsets):
                    if start == end: continue
                    if max(start, entity['start']) < min(end, entity['end']):
                        if token_start_index is None: token_start_index = idx
                        token_end_index = idx
                if token_start_index is not None:
                    labels[token_start_index] = label2id[f"B-{entity['label']}"]
                    for j in range(token_start_index + 1, token_end_index + 1):
                        labels[j] = label2id[f"I-{entity['label']}"]
            all_labels.append(labels)
        tokenized_inputs["labels"] = all_labels
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    tokenized_train = final_train.map(tokenize_and_align_labels, batched=True)
    tokenized_test = final_test.map(tokenize_and_align_labels, batched=True)
    
    return {"train": tokenized_train, "test": tokenized_test}