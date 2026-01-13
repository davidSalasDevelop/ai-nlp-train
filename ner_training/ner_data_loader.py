import logging
import json
import os
from datasets import load_dataset, concatenate_datasets, Dataset
import ner_config 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _reconstruct_generic(example, ner_tags_list, tags_column_name):
    """
    Normaliza etiquetas de cualquier dataset a: PER, ORG, LOC, MISC, DATE.
    """
    text = ""
    entities = []
    current_pos = 0
    
    tag_ids = example[tags_column_name]
    
    for i, token in enumerate(example['tokens']):
        tag_id = tag_ids[i]
        raw_tag = ner_tags_list[tag_id]
        
        # --- LÓGICA DE MAPEO DE ETIQUETAS ---
        label_upper = raw_tag.upper()
        norm_label = None
        
        if 'PER' in label_upper: norm_label = 'PER'
        elif 'ORG' in label_upper: norm_label = 'ORG'
        elif 'LOC' in label_upper or 'LUGAR' in label_upper: norm_label = 'LOC'
        
        # Aquí capturamos explícitamente las FECHAS
        elif 'DATE' in label_upper or 'FECHA' in label_upper: norm_label = 'DATE'
        
        # Todo lo demás que sea una entidad, va a MISC
        elif 'MISC' in label_upper or 'OTHER' in label_upper: norm_label = 'MISC'
        
        # Reconstrucción del texto
        if i > 0:
            text += " "
            current_pos += 1
        
        start_char = current_pos
        text += token
        current_pos += len(token)
        
        if raw_tag.startswith('B-') and norm_label:
            entity_start = start_char
            
            end_token_idx = i
            while (end_token_idx + 1 < len(example['tokens'])):
                next_tag_id = tag_ids[end_token_idx + 1]
                next_raw_tag = ner_tags_list[next_tag_id]
                
                # Continuidad de la entidad
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
    list_of_train = []
    list_of_test = []

    # 1. WIKIANN (Base General - PER, ORG, LOC)
    try:
        logging.info("📚 Cargando 'wikiann' (Español)...")
        ds1 = load_dataset("wikiann", "es", cache_dir=ner_config.CACHE_DIR)
        tags1 = ds1['train'].features['ner_tags'].feature.names
        list_of_train.append(ds1['train'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['train'].column_names))
        list_of_test.append(ds1['validation'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['validation'].column_names))
        logging.info("   ✅ Wikiann cargado.")
    except Exception as e: logging.error(f"❌ Error Wikiann: {e}")

    # 2. LENER-ES (Base Legal - Incluye DATES/FECHAS)
    try:
        logging.info("⚖️ Cargando 'PlanTL-GOB-ES/lener-es'...")
        ds2 = load_dataset("PlanTL-GOB-ES/lener-es", "lener", cache_dir=ner_config.CACHE_DIR)
        tags2 = ds2['train'].features['ner_tags'].feature.names
        list_of_train.append(ds2['train'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags2, 'tags_column_name': 'ner_tags'}, remove_columns=ds2['train'].column_names))
        list_of_test.append(ds2['validation'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags2, 'tags_column_name': 'ner_tags'}, remove_columns=ds2['validation'].column_names))
        logging.info("   ✅ Lener-ES cargado (Aporta fechas).")
    except Exception as e: logging.warning(f"⚠️ Error Lener-ES: {e}")

    # 3. DATASET PROPIO (Tus fechas y sujetos específicos)
    if ner_config.INCLUDE_CUSTOM_DATASET:
        if os.path.exists(dataset_path):
            logging.info(f"💎 Cargando DATASET PROPIO desde: {dataset_path}")
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                custom_ds = Dataset.from_list(custom_data)
                custom_splits = custom_ds.train_test_split(test_size=ner_config.TEST_SIZE, seed=ner_config.SEED)
                list_of_train.append(custom_splits['train'])
                list_of_test.append(custom_splits['test'])
                logging.info(f"   ✅ Dataset Propio añadido.")
            except Exception as e: logging.error(f"❌ Error JSON: {e}")

    if not list_of_train: raise RuntimeError("¡No hay datos!")
        
    final_train = concatenate_datasets(list_of_train).shuffle(seed=ner_config.SEED)
    final_test = concatenate_datasets(list_of_test).shuffle(seed=ner_config.SEED)
    logging.info(f"🚀 TOTAL: {len(final_train)} train | {len(final_test)} test")

    label2id = {label: i for i, label in enumerate(label_list)}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=False, max_length=max_length, return_offsets_mapping=True)
        all_labels = []
        for i, entities in enumerate(examples["entities"]):
            offsets = tokenized_inputs["offset_mapping"][i]
            labels = [label2id["O"]] * len(offsets)
            for entity in entities:
                label_str = entity['label']
                if f"B-{label_str}" not in label2id: continue # Ignora etiquetas no configuradas
                
                token_start_index, token_end_index = None, None
                for idx, (start, end) in enumerate(offsets):
                    if start == end: continue
                    if max(start, entity['start']) < min(end, entity['end']):
                        if token_start_index is None: token_start_index = idx
                        token_end_index = idx
                if token_start_index is not None:
                    labels[token_start_index] = label2id[f"B-{label_str}"]
                    for j in range(token_start_index + 1, token_end_index + 1):
                        labels[j] = label2id[f"I-{label_str}"]
            all_labels.append(labels)
        tokenized_inputs["labels"] = all_labels
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    return {
        "train": final_train.map(tokenize_and_align_labels, batched=True),
        "test": final_test.map(tokenize_and_align_labels, batched=True)
    }