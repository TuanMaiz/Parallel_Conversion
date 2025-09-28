"""
Standard BERT models for text classification (without QCFS)
For comparison with QCFS models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertForSequenceClassification(nn.Module):
    """Standard BERT for sequence classification"""
    def __init__(self, pretrained_name="bert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return logits


