# model.py
"""
Defines the PyTorch model architecture.
"""
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TinyModel(nn.Module):
    """A simple classification model using a pre-trained BERT."""
    def __init__(self, model_name, num_labels):
        super().__init__()
        
        # --- FIX 1: Save the config as a class attribute ---
        # Instead of a local variable, 'config' is now 'self.config'
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        
        # Now use self.config to initialize the bert model
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # --- FIX 2: Corrected the typo from '--1' to '-1' ---
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits