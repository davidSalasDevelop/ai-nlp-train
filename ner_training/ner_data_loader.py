import logging
import json
import os
from datasets import load_dataset, concatenate_datasets, Dataset
import ner_config 

logger = logging.getLogger(__name__)

def _reconstruct_generic(example, ner_tags_list, tags_column_name):
    """
    Normaliza etiquetas de cualquier dataset a: PER, ORG, LOC, MISC, DATE.
    """
    text = ""
    entities = []
    current_pos = 0
    
    # Manejo de columnas (algunos datasets usan 'tokens', otros 'sentences')
    tokens = example['tokens'] if 'tokens' in example else example['sentences']
    tag_ids = example[tags_column_name]
    
    for i, token in enumerate(tokens):
        tag_id = tag_ids[i]
        raw_tag = ner_tags_list[tag_id]
        
        label_upper = raw_tag.upper()
        norm_label = None
        
        # Mapeo estándar
        if 'PER' in label_upper: norm_label = 'PER'
        elif 'ORG' in label_upper: norm_label = 'ORG'
        elif 'LOC' in label_upper or 'LUGAR' in label_upper: norm_label = 'LOC'
        elif 'DATE' in label_upper or 'FECHA' in label_upper: norm_label = 'DATE'
        elif 'MISC' in label_upper or 'OTHER' in label_upper: norm_label = 'MISC'
        
        # Reconstrucción del texto (añadir espacio si no es el primer token)
        if i > 0:
            text += " "
            current_pos += 1
        
        start_char = current_pos
        text += token
        current_pos += len(token)
        
        if raw_tag.startswith('B-') and norm_label:
            entity_start = start_char
            end_token_idx = i
            # Buscar hasta dónde llega la entidad (I-...)
            while (end_token_idx + 1 < len(tokens)):
                next_tag_id = tag_ids[end_token_idx + 1]
                next_raw_tag = ner_tags_list[next_tag_id]
                if next_raw_tag.startswith('I-'):
                    end_token_idx += 1
                else:
                    break
            
            length_span = len(" ".join(tokens[i : end_token_idx + 1]))
            entities.append({
                'label': norm_label,
                'start': entity_start,
                'end': entity_start + length_span
            })
            
    return {'text': text, 'entities': entities}

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label_list: list, max_length: int = 128):
    list_of_train = []
    list_of_test = []

    # --- NUEVO DATASET PÚBLICO: CoNLL-2002 (Español) ---
    logger.info("   [1/2] Cargando dataset público 'CoNLL-2002' (Español)...")
    try:
        # CoNLL-2002 es el estándar académico para NER en español. Muy fiable.
        ds_conll = load_dataset("conll2002", "es", trust_remote_code=True, cache_dir=ner_config.CACHE_DIR)
        
        tags_conll = ds_conll['train'].features['ner_tags'].feature.names
        
        # Mapeamos train y validation
        list_of_train.append(ds_conll['train'].map(
            _reconstruct_generic, 
            fn_kwargs={'ner_tags_list': tags_conll, 'tags_column_name': 'ner_tags'}, 
            remove_columns=ds_conll['train'].column_names
        ))
        
        list_of_test.append(ds_conll['validation'].map(
            _reconstruct_generic, 
            fn_kwargs={'ner_tags_list': tags_conll, 'tags_column_name': 'ner_tags'}, 
            remove_columns=ds_conll['validation'].column_names
        ))
        logger.info(f"       ✅ CoNLL-2002 cargado ({len(ds_conll['train'])} ejemplos).")
        
    except Exception as e:
        logger.warning(f"       ⚠️ Error cargando CoNLL-2002: {e}")

    # --- DATASET PROPIO ---
    logger.info("   [2/2] Buscando Dataset Propio...")
    if ner_config.INCLUDE_CUSTOM_DATASET:
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                custom_ds = Dataset.from_list(custom_data)
                
                # Si el dataset propio es muy pequeño (menos de 10 ejemplos), lo duplicamos para evitar errores
                if len(custom_ds) < 10:
                     logger.info("       ℹ️ Dataset propio pequeño, no se dividirá en test.")
                     list_of_train.append(custom_ds)
                else:
                    custom_splits = custom_ds.train_test_split(test_size=ner_config.TEST_SIZE, seed=ner_config.SEED)
                    list_of_train.append(custom_splits['train'])
                    list_of_test.append(custom_splits['test'])
                
                logger.info(f"       ✅ Dataset Propio cargado: {dataset_path}")
            except Exception as e:
                logger.error(f"       ❌ Error en JSON propio: {e}")
        else:
            logger.warning(f"       ℹ️ No se encontró el archivo: {dataset_path}")

    # --- VERIFICACIÓN FINAL ---
    if not list_of_train:
        raise RuntimeError("¡ERROR FATAL! No hay datos. Revisa tu internet para descargar CoNLL-2002 o tu dataset local.")
        
    final_train = concatenate_datasets(list_of_train).shuffle(seed=ner_config.SEED)
    
    # Si no tenemos test (ej. solo con dataset propio pequeño), duplicamos train como test
    if not list_of_test:
        logger.warning("       ⚠️ No hay datos de prueba separados. Usando duplicado de entrenamiento.")
        final_test = final_train
    else:
        final_test = concatenate_datasets(list_of_test).shuffle(seed=ner_config.SEED)

    logger.info(f"   📊 DATOS FINALES: {len(final_train)} entrenamiento | {len(final_test)} prueba")

    label2id = {label: i for i, label in enumerate(label_list)}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=False, max_length=max_length, return_offsets_mapping=True)
        all_labels = []
        for i, entities in enumerate(examples["entities"]):
            offsets = tokenized_inputs["offset_mapping"][i]
            labels = [label2id["O"]] * len(offsets)
            for entity in entities:
                label_str = entity['label']
                if f"B-{label_str}" not in label2id: continue
                
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
        "train": final_train.map(tokenize_and_align_labels, batched=True, desc="Tokenizando"),
        "test": final_test.map(tokenize_and_align_labels, batched=True, desc="Tokenizando")
    }