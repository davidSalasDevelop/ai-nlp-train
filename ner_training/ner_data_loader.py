# ner_data_loader.py
from datasets import load_dataset
import logging

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label2id: dict, max_length: int = 128):
    """
    Carga el dataset y alinea las etiquetas de entidad con los tokens de forma robusta.
    """
    raw_dataset = load_dataset('json', data_files=dataset_path, split='train')
    logging.info(f"Dataset crudo cargado con {len(raw_dataset)} ejemplos.")

    def align_labels_with_tokens(example):
        # 1. Tokenizar el texto para obtener los IDs de los tokens y los IDs de las palabras
        # Word IDs nos dice a qué palabra original pertenece cada sub-token.
        tokenized_inputs = tokenizer(
            example["text"], 
            truncation=True, 
            is_split_into_words=False, 
            max_length=max_length
        )
        
        # El tokenizer de Hugging Face puede devolver los word_ids directamente
        try:
            word_ids = tokenized_inputs.word_ids()
        except Exception:
            # Para algunos tokenizers más antiguos, necesitamos obtenerlos de otra manera
            tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
            word_ids = [None if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token] else i for i, token in enumerate(tokens) for _ in range(len(tokenizer.tokenize(example["text"].split()[i])) if i < len(example["text"].split()) else 0)]
            # Esto es una heurística y puede no ser perfecto, pero es una alternativa. La primera opción es la mejor.
            # Una forma más simple si falla:
            word_ids = [None] * len(tokenized_inputs["input_ids"]) # Fallback simple si todo falla
            logging.warning("No se pudieron generar word_ids, la alineación puede ser imprecisa.")


        # 2. Crear un array de etiquetas inicializado con -100
        # -100 es una etiqueta especial que el Trainer de HF ignora en el cálculo de la pérdida.
        # La asignamos a los tokens especiales ([CLS], [SEP]) y a los sub-tokens secundarios.
        labels = [-100] * len(word_ids)
        word_to_label = {}
        
        # 3. Mapear las entidades a nivel de palabra
        for entity in example["entities"]:
            # Esto es una simplificación: asumimos que las entidades no se solapan.
            # Buscamos la primera palabra que cae dentro del rango de la entidad.
            for i, word in enumerate(example["text"].split()):
                # Simplificación: encontrar el rango de cada palabra
                word_start = example["text"].find(word)
                word_end = word_start + len(word)
                if max(word_start, entity["start"]) < min(word_end, entity["end"]):
                    if word_to_label.get(i) is None: # Si no ha sido etiquetada
                        word_to_label[i] = f"B-{entity['label']}"
                    else: # Si ya es B-, los siguientes son I- (esto es raro con split)
                        word_to_label[i] = f"I-{entity['label']}"

        # 4. Asignar las etiquetas a los tokens usando los word_ids
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue # Ya está como -100
            
            # Si el word_id es el mismo que el anterior, es un sub-token.
            # La etiqueta debe ser la misma, pero en formato I- si no lo es ya.
            if word_idx == previous_word_idx:
                label = word_to_label.get(word_idx)
                if label and label.startswith("B-"):
                    labels[i] = label2id[f"I-{label.split('-')[1]}"]
                else:
                    labels[i] = label2id.get(label, label2id["O"])
            
            # Si es un nuevo word_id
            else:
                label = word_to_label.get(word_idx)
                labels[i] = label2id.get(label, label2id["O"])
            
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = raw_dataset.map(align_labels_with_tokens, remove_columns=raw_dataset.column_names)
    logging.info("Alineación de etiquetas completada.")
    return tokenized_dataset.train_test_split(test_size=0.2, seed=42)