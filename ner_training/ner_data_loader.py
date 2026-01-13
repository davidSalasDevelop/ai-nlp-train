# ner_data_loader.py
from datasets import load_dataset
import logging

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label2id: dict, max_length: int = 128):
    """
    Carga el dataset y alinea las etiquetas de entidad con los tokens de forma robusta.
    Versión corregida y simplificada.
    """
    raw_dataset = load_dataset('json', data_files=dataset_path, split='train')
    logging.info(f"Dataset crudo cargado con {len(raw_dataset)} ejemplos.")

    def align_labels_with_tokens(example):
        # 1. Tokenizar el texto para obtener los word_ids.
        tokenized_inputs = tokenizer(
            example["text"],
            truncation=True,
            is_split_into_words=False,
            max_length=max_length
        )
        word_ids = tokenized_inputs.word_ids()

        # 2. Crear una lista de etiquetas inicializada con -100.
        # -100 es una etiqueta especial que el Trainer ignora.
        labels = [-100] * len(word_ids)
        
        # 3. Mapear cada entidad a los tokens correspondientes.
        for entity in example["entities"]:
            entity_start = entity["start"]
            entity_end = entity["end"]
            label = entity["label"]

            # Encontrar el primer y último token que caen dentro de la entidad.
            # `char_to_token` nos da el índice del token para cualquier caracter del texto original.
            token_start_index = tokenized_inputs.char_to_token(entity_start)
            token_end_index = tokenized_inputs.char_to_token(entity_end - 1)

            # Si el mapeo es válido (a veces los offsets pueden estar al límite)...
            if token_start_index is not None and token_end_index is not None:
                # Marcar el primer token con B-
                labels[token_start_index] = label2id[f"B-{label}"]
                # Marcar todos los tokens intermedios (si los hay) con I-
                for i in range(token_start_index + 1, token_end_index + 1):
                    labels[i] = label2id[f"I-{label}"]

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Aplicar la función de alineación y dividir el dataset.
    # remove_columns es importante para que el Trainer no se confunda.
    tokenized_dataset = raw_dataset.map(align_labels_with_tokens, remove_columns=raw_dataset.column_names)
    logging.info("Alineación de etiquetas completada.")
    return tokenized_dataset.train_test_split(test_size=0.2, seed=42)