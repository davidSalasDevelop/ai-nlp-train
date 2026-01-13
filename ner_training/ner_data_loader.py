# ner_data_loader.py
from datasets import Dataset
import logging
import json

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label_list: list, max_length: int = 128):
    """
    Carga el dataset y alinea las etiquetas de entidad con los tokens.
    VERSIÓN FINAL Y A PRUEBA DE BALAS.
    """
    logging.info(f"Cargando dataset crudo desde {dataset_path} usando carga manual de JSON.")
    
    # --- LA PUTA CORRECCIÓN DEFINITIVA ---
    # 1. Cargar el JSON manualmente para evitar los errores de pyarrow.
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Convertir la lista de diccionarios a un diccionario de listas.
    #    Ej: [{'text': 'a'}, {'text': 'b'}] -> {'text': ['a', 'b']}
    texts = [item['text'] for item in data]
    entities_list = [item['entities'] for item in data]
    
    # 3. Crear un Dataset de Hugging Face desde los datos en memoria.
    raw_dataset = Dataset.from_dict({'text': texts, 'entities': entities_list})
    
    label2id = {label: i for i, label in enumerate(label_list)}

    def tokenize_and_align_labels(examples):
        # Esta función ahora recibe un lote del Dataset en memoria
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
                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_label = entity["label"]

                token_start_index = None
                token_end_index = None

                for idx, (start, end) in enumerate(offsets):
                    if start == end: continue
                    
                    if max(start, entity_start) < min(end, entity_end):
                        if token_start_index is None:
                            token_start_index = idx
                        token_end_index = idx
                
                if token_start_index is not None and token_end_index is not None:
                    labels[token_start_index] = label2id[f"B-{entity_label}"]
                    for j in range(token_start_index + 1, token_end_index + 1):
                        labels[j] = label2id[f"I-{entity_label}"]

            all_labels.append(labels)

        tokenized_inputs["labels"] = all_labels
        tokenized_inputs.pop("offset_mapping")
        return tokenized_inputs

    tokenized_dataset = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    logging.info("Alineación de etiquetas completada.")
    return tokenized_dataset.train_test_split(test_size=0.2, seed=42)