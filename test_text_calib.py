#!/usr/bin/env python3
"""
Test script for text model calibration integration
"""

import torch
import torch.nn as nn
from utils import calib_text_one_epoch, set_calib_text_opt, set_calib_text_inf
from modules_text import DA_QCFS_Text
from models.Bert_QCFS import BertForSequenceClassificationQCFS

def test_text_calibration():
    """Test the text calibration process"""
    print("Testing text model calibration...")
    
    # Create a simple text model
    model = BertForSequenceClassificationQCFS(
        pretrained_name="bert-base-uncased",
        T=4,
        num_labels=4
    )
    
    # Create dummy text data
    batch_size = 8
    seq_len = 32
    hidden_size = 768
    
    # Create dummy dataloader
    dummy_data = []
    for _ in range(5):  # 5 batches
        input_ids = torch.randint(0, 30000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 4, (batch_size,))
        dummy_data.append(({'input_ids': input_ids, 'attention_mask': attention_mask}, labels))
    
    # Test calibration
    print("Before calibration:")
    model.eval()
    with torch.no_grad():
        test_input = dummy_data[0][0]
        output = model(**test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Run calibration
    print("\nRunning calibration...")
    acc = calib_text_one_epoch(model, dummy_data)
    print(f"Calibration accuracy: {acc:.4f}")
    
    print("\nAfter calibration:")
    model.eval()
    with torch.no_grad():
        test_input = dummy_data[0][0]
        output = model(**test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check if calibration flags are set
    print("\nChecking calibration flags:")
    for name, module in model.named_modules():
        if isinstance(module, DA_QCFS_Text):
            print(f"Module {name}: is_cab={module.is_cab}, calib_inf={module.calib_inf}")
            break
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_text_calibration()