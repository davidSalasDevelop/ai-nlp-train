# ner_data_loader.py
from datasets import load_dataset

def load_and_prepare_ner_data(dataset_path: str, tokenizer, label2id: dict):
    raw_dataset = load_dataset('json', data_files=dataset_path, split='train')

    def align_labels_with_tokens(example):
        tokenized_inputs = tokenizer(example["text"], truncation=True, is_split_into_words=False, return_offsets_mapping=True)
        entities = example["entities"]
        offsets = tokenized_inputs.pop("offset_mapping")
        token_labels = [label2id["O"]] * len(offsets)
        
        for entity in entities:
            entity_label = entity["label"]
            entity_start = entity["start"]
            entity_end = entity["end"]
            
            for idx, (start_char, end_char) in enumerate(offsets):
                if start_char == end_char: continue
                
                if max(start_char, entity_start) < min(end_char, entity_end):
                    is_begin = all(token_labels[i] == label2id["O"] for i in range(len(token_labels)) if max(offsets[i][0], entity_start) < min(offsets[i][1], entity_end))
                    if is_begin and start_char >= entity_start:
                        token_labels[idx] = label2id[f"B-{entity_label}"]
                    else:
                        token_labels[idx] = label2id[f"I-{entity_label}"]

        tokenized_inputs["labels"] = token_labels
        return tokenized_inputs

    tokenized_dataset = raw_dataset.map(align_labels_with_tokens, remove_columns=raw_dataset.column_names)
    return tokenized_dataset.train_test_split(test_size=0.2, seed=42)