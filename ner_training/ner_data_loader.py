#ner_data_loader.py
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
    
    # Manejo robusto de nombres de columnas
    tokens = example.get('tokens') or example.get('sentences')
    tag_ids = example.get(tags_column_name)
    
    # Validación básica
    if not tokens or not tag_ids:
        return {'text': "", 'entities': []}

    for i, token in enumerate(tokens):
        if i >= len(tag_ids): break
            
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
        
        if i > 0:
            text += " "
            current_pos += 1
        
        start_char = current_pos
        text += token
        current_pos += len(token)
        
        if raw_tag.startswith('B-') and norm_label:
            entity_start = start_char
            end_token_idx = i
            while (end_token_idx + 1 < len(tokens)):
                if end_token_idx + 1 >= len(tag_ids): break
                
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
    
    # -------------------------------------------------------------------------
    # INTENTO PRINCIPAL: Babelscape/wikineural (Moderno, Parquet, Público)
    # -------------------------------------------------------------------------
    logger.info("   [1/3] Cargando 'Babelscape/wikineural' (Español)...")
    try:
        # download_mode="force_redownload" evita errores de caché corrupto (404 previos)
        ds_wiki = load_dataset(
            "Babelscape/wikineural", 
            "es", 
            cache_dir=ner_config.CACHE_DIR,
            download_mode="force_redownload" 
        )
        
        tags_wiki = ds_wiki['train'].features['ner_tags'].feature.names
        
        # Usamos un subset pequeño (ej. 2000 ejemplos) para no saturar si es muy grande
        # O todo el dataset si tienes tiempo. Aquí limitamos a 5000 para velocidad.
        subset_train = ds_wiki['train'].select(range(min(5000, len(ds_wiki['train']))))
        subset_val = ds_wiki['val'].select(range(min(500, len(ds_wiki['val']))))

        list_of_train.append(subset_train.map(
            _reconstruct_generic, 
            fn_kwargs={'ner_tags_list': tags_wiki, 'tags_column_name': 'ner_tags'}, 
            remove_columns=ds_wiki['train'].column_names
        ))
        
        list_of_test.append(subset_val.map(
            _reconstruct_generic, 
            fn_kwargs={'ner_tags_list': tags_wiki, 'tags_column_name': 'ner_tags'}, 
            remove_columns=ds_wiki['val'].column_names
        ))
        logger.info(f"       ✅ WikiNeural cargado ({len(subset_train)} ejemplos).")
        
    except Exception as e:
        logger.warning(f"       ⚠️ Falló WikiNeural: {str(e).splitlines()[0]}")

    # -------------------------------------------------------------------------
    # INTENTO SECUNDARIO: Wikiann (Clásico, si WikiNeural falla)
    # -------------------------------------------------------------------------
    if not list_of_train:
        logger.info("   [2/3] Intentando fallback a 'wikiann'...")
        try:
            ds1 = load_dataset("wikiann", "es", cache_dir=ner_config.CACHE_DIR)
            tags1 = ds1['train'].features['ner_tags'].feature.names
            list_of_train.append(ds1['train'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['train'].column_names))
            list_of_test.append(ds1['validation'].map(_reconstruct_generic, fn_kwargs={'ner_tags_list': tags1, 'tags_column_name': 'ner_tags'}, remove_columns=ds1['validation'].column_names))
            logger.info("       ✅ Wikiann cargado.")
        except Exception:
            pass # Silencioso si falla

    # -------------------------------------------------------------------------
    # DATASET PROPIO
    # -------------------------------------------------------------------------
    logger.info("   [3/3] Buscando Dataset Propio...")
    if ner_config.INCLUDE_CUSTOM_DATASET:
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                custom_ds = Dataset.from_list(custom_data)
                
                if len(custom_ds) < 10:
                     logger.info("       ℹ️ Dataset propio pequeño, se añadirá a train.")
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

    # -------------------------------------------------------------------------
    # VERIFICACIÓN FINAL
    # -------------------------------------------------------------------------
    if not list_of_train:
        raise RuntimeError("¡ERROR FATAL! No se pudo cargar NINGÚN dataset (ni públicos ni local).")
        
    final_train = concatenate_datasets(list_of_train).shuffle(seed=ner_config.SEED)
    
    if not list_of_test:
        logger.warning("       ⚠️ No hay datos de prueba separados. Duplicando train.")
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