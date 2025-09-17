#!/usr/bin/env python3

import torch
from transformers import BertModel, BertConfig
import sys
sys.path.append('.')
from models.Bert_QCFS import BertForSequenceClassificationQCFS

def debug_bert_forward():
    """Debug the BERT forward pass to understand where 2D input is coming from"""
    
    # Create model
    model = BertForSequenceClassificationQCFS(pretrained_name="bert-base-uncased", T=4)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input dim: {input_ids.dim()}")
    
    # Let's trace through the BERT model
    with torch.no_grad():
        try:
            # Check what bert.embeddings does
            embeddings = model.bert.embeddings(input_ids=input_ids)
            print(f"After embeddings shape: {embeddings.shape}")
            print(f"After embeddings dim: {embeddings.dim()}")
            
            # Now check the encoder
            print(f"Encoder type: {type(model.bert.encoder)}")
            print(f"First layer type: {type(model.bert.encoder.layer[0])}")
            
            # Try passing embeddings to first layer directly
            first_layer = model.bert.encoder.layer[0]
            output = first_layer(embeddings, attention_mask=attention_mask)
            print(f"First layer output shape: {output.shape}")
            
        except Exception as e:
            print(f"Error during debug: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_bert_forward()