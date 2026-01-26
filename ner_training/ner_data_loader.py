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
    
    tokens = example.get('tokens') or example.get('sentences')
    tag_ids = example.get(tags_column_name)
    
    if not tokens or not tag_ids:
        return {'text': "", 'entities': []}

    for i, token in enumerate(tokens):
        if i >= len(tag_ids): 
            break
            
        tag_id = tag_ids[i]
        raw_tag = ner_tags_list[tag_id]
        
        label_upper = raw_tag.upper()
        norm_label = None
        
        # Mapeo estándar
        if 'PER' in label_upper: 
            norm_label = 'PER'
        elif 'ORG' in label_upper: 
            norm_label = 'ORG'
        elif 'LOC' in label_upper or 'LUGAR' in label_upper: 
            norm_label = 'LOC'
        elif 'DATE' in label_upper or 'FECHA' in label_upper: 
            norm_label = 'DATE'
        elif 'MISC' in label_upper or 'OTHER' in label_upper: 
            norm_label = 'MISC'
        
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
                if end_token_idx + 1 >= len(tag_ids): 
                    break
                
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
    """
    Carga SOLO datasets propios definidos en ner_config.CUSTOM_DATASET_FILES
    """
    list_of_train = []
    list_of_test = []
    
    logger.info("📂 Cargando datasets propios...")
    
    # Etiquetas estándar para nuestros datasets
    example_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "B-DATE", "I-DATE"]
    
    datasets_cargados = 0
    
    # Cargar CADA archivo de la lista en ner_config.py
    for dataset_file in ner_config.CUSTOM_DATASET_FILES:
        file_path = os.path.join(ner_config.CUSTOM_DATASET_DIR, dataset_file)
        
        logger.info(f"   → Intentando cargar: {dataset_file}")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                
                # Validar que sea una lista
                if not isinstance(custom_data, list):
                    logger.warning(f"      ⚠️ {dataset_file} no es una lista JSON. Saltando...")
                    continue
                
                # Verificar formato mínimo
                if len(custom_data) == 0:
                    logger.warning(f"      ⚠️ {dataset_file} está vacío. Saltando...")
                    continue
                
                if 'tokens' not in custom_data[0] or 'ner_tags' not in custom_data[0]:
                    logger.warning(f"      ⚠️ {dataset_file} no tiene formato 'tokens/ner_tags'. Saltando...")
                    continue
                
                # Crear Dataset de HuggingFace
                custom_ds = Dataset.from_list(custom_data)
                
                # Transformar al formato que necesita nuestro pipeline
                transformed_ds = custom_ds.map(
                    lambda x: _reconstruct_generic(x, example_tags, 'ner_tags'),
                    remove_columns=custom_ds.column_names,
                    desc=f"Procesando {dataset_file}"
                )
                
                # Filtrar ejemplos vacíos
                transformed_ds = transformed_ds.filter(lambda x: len(x['text']) > 0 and len(x['entities']) > 0)
                
                if len(transformed_ds) == 0:
                    logger.warning(f"      ⚠️ {dataset_file} no generó ejemplos válidos.")
                    continue
                
                # Dividir en train/test
                if len(transformed_ds) < 10:
                    # Dataset muy pequeño: usar TODO para train
                    logger.info(f"      ✅ {dataset_file} cargado ({len(transformed_ds)} ejemplos) → TODO para train")
                    list_of_train.append(transformed_ds)
                else:
                    # Dataset grande: dividir en train/test
                    splits = transformed_ds.train_test_split(
                        test_size=ner_config.TEST_SIZE, 
                        seed=ner_config.SEED
                    )
                    list_of_train.append(splits['train'])
                    list_of_test.append(splits['test'])
                    logger.info(f"      ✅ {dataset_file} cargado ({len(transformed_ds)} ejemplos) → {len(splits['train'])} train, {len(splits['test'])} test")
                
                datasets_cargados += 1
                
            except json.JSONDecodeError as e:
                logger.error(f"      ❌ Error en JSON {dataset_file}: {str(e)}")
            except Exception as e:
                logger.error(f"      ❌ Error cargando {dataset_file}: {str(e)}")
        else:
            logger.warning(f"      ⚠️ Archivo no encontrado: {dataset_file}")
    
    # VERIFICACIÓN FINAL
    if datasets_cargados == 0:
        logger.error("❌ No se pudo cargar NINGÚN dataset.")
        logger.info("💡 Crea al menos un archivo JSON con este formato:")
        logger.info("""
    [
      {
        "tokens": ["Juan", "Pérez", "vive", "en", "Madrid", "."],
        "ner_tags": [1, 2, 0, 0, 3, 0]
      }
    ]
        """)
        raise RuntimeError("¡No hay datos para entrenar!")
    
    logger.info(f"✅ {datasets_cargados} dataset(s) cargado(s) correctamente.")
    
    # COMBINAR TODOS LOS DATASETS
    if len(list_of_train) > 1:
        final_train = concatenate_datasets(list_of_train).shuffle(seed=ner_config.SEED)
    else:
        final_train = list_of_train[0].shuffle(seed=ner_config.SEED)
    
    # Manejar test
    if len(list_of_test) > 1:
        final_test = concatenate_datasets(list_of_test).shuffle(seed=ner_config.SEED)
    elif len(list_of_test) == 1:
        final_test = list_of_test[0].shuffle(seed=ner_config.SEED)
    else:
        # Si no hay test separados, dividir el train
        logger.warning("⚠️ No hay datos de test. Dividiendo train...")
        splits = final_train.train_test_split(test_size=ner_config.TEST_SIZE, seed=ner_config.SEED)
        final_train = splits['train']
        final_test = splits['test']
    
    logger.info(f"📊 DATOS FINALES: {len(final_train)} entrenamiento | {len(final_test)} prueba")
    
    # TOKENIZACIÓN (IGUAL QUE ANTES)
    label2id = {label: i for i, label in enumerate(label_list)}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["text"], 
            truncation=True, 
            is_split_into_words=False, 
            max_length=max_length, 
            return_offsets_mapping=True
        )
        all_labels = []
        for i, entities in enumerate(examples["entities"]):
            offsets = tokenized_inputs["offset_mapping"][i]
            labels = [label2id["O"]] * len(offsets)
            for entity in entities:
                label_str = entity['label']
                if f"B-{label_str}" not in label2id: 
                    continue
                
                token_start_index, token_end_index = None, None
                for idx, (start, end) in enumerate(offsets):
                    if start == end: 
                        continue
                    if max(start, entity['start']) < min(end, entity['end']):
                        if token_start_index is None: 
                            token_start_index = idx
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
        "train": final_train.map(tokenize_and_align_labels, batched=True, desc="Tokenizando train"),
        "test": final_test.map(tokenize_and_align_labels, batched=True, desc="Tokenizando test")
    }