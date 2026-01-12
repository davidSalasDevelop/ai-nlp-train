# data_loader.py
"""
Handles dataset loading and preprocessing using the Hugging Face datasets library.
"""
from datasets import load_dataset, ClassLabel, DatasetDict

def load_and_prepare_data(tokenizer, dataset_path):
    """
    Loads the JSON data, maps labels to integers, tokenizes, and splits the dataset.
    
    Returns:
        - tokenized_datasets (DatasetDict): The processed dataset ready for the Trainer.
        - id_to_intent (dict): A dictionary mapping label IDs back to intent names.
    """
    print("📂 Loading and preparing data with 'datasets' library...")

    # 1. Load the dataset from the JSON file
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # 2. Get unique intents and create mappings
    unique_intents = sorted(dataset.unique('intent'))
    intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
    id_to_intent = {i: intent for intent, i in enumerate(unique_intents)}

    # 3. Create a ClassLabel feature to encode intents as integers
    class_label = ClassLabel(names=unique_intents)

    # 4. Map the string labels to integer labels
    def map_labels(example):
        example['label'] = class_label.str2int(example['intent'])
        return example

    dataset = dataset.map(map_labels, batched=False)
    
    # 5. Define a tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

    # 6. Tokenize the entire dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 7. Split into training and validation sets (e.g., 80/20 split)
    split_datasets = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True, seed=42)

    print(f"   ✅ Data processing complete. Train: {len(split_datasets['train'])}, Test: {len(split_datasets['test'])}")
    
    return split_datasets, id_to_intent