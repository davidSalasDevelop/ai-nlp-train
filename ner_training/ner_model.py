# ner_model.py
from transformers import AutoModelForTokenClassification

def build_ner_model(model_name: str, id2label: dict, label2id: dict):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, id2label=id2label, label2id=label2id
    )
    return model