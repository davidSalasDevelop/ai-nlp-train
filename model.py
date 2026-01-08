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
        # Store num_labels in the config, which the Trainer expects
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        # The Trainer requires the model to return a loss when labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits