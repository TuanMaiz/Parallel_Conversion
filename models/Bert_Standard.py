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


class DistilBertForSequenceClassification(nn.Module):
    """Standard DistilBERT for sequence classification"""
    def __init__(self, pretrained_name="distilbert-base-uncased", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled = hidden_state[:, 0]  # [CLS] token
        pooled = self.pre_classifier(pooled)
        pooled = nn.ReLU()(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return logits