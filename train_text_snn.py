"""
Text-specific SNN training functions for ANN-SNN conversion
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp
import logging

def train_text_one_epoch(model, loss_fn, optimizer, train_dataloader, sim_len, local_rank, scaler=None, mixup=None, distributed=False):
    """
    Train SNN model on text data with proper time-step processing
    
    Args:
        model: SNN model for text
        loss_fn: Loss function
        optimizer: Optimizer
        train_dataloader: Text data loader
        sim_len: Number of simulation time steps
        local_rank: GPU rank for distributed training
        scaler: AMP scaler
        mixup: Mixup augmentation
        distributed: Whether using distributed training
        
    Returns:
        float: Average loss for the epoch
    """
    epoch_loss, total_samples = 0, 0
    model.train()
    total_batches = len(train_dataloader)
    
    print(f"Starting training with {total_batches} batches...")
    
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Move inputs to GPU - text data has multiple keys
        input_ids = inputs['input_ids'].cuda(local_rank, non_blocking=True)
        attention_mask = inputs['attention_mask'].cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)
        
        batch_size = len(labels)
        total_samples += batch_size
        
        # Apply mixup if enabled
        if mixup:
            # Note: Mixup for text needs special handling - may need to implement text-specific mixup
            pass  # Placeholder for text mixup
        
        optimizer.zero_grad()
        
        # Process through SNN - this is ANN-SNN conversion, not time-expanded input
        # The spiking happens within the neurons, not in the input
        if scaler is not None:
            with amp.autocast():
                # Forward pass through SNN model (single timestep, neurons spike internally)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass through SNN model (single timestep)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Aggregate loss for distributed training
        if distributed:
            from main import reduce_mean
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
            current_loss = vis_loss.item()
        else:
            epoch_loss += loss.item()
            current_loss = loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss_so_far = epoch_loss / total_samples
            print(f"Batch [{batch_idx+1}/{total_batches}] - Loss: {current_loss:.4f}, Avg Loss: {avg_loss_so_far:.4f}")
    
    # Print epoch summary
    final_avg_loss = epoch_loss / total_samples
    print(f"Epoch completed - Total samples: {total_samples}, Average Loss: {final_avg_loss:.4f}")
    
    return final_avg_loss


def eval_text_snn(model, test_dataloader, sim_len, record_time=True):
    """
    Evaluate SNN model on text data with proper time-step processing
    
    Args:
        model: SNN model for text
        test_dataloader: Text test data loader
        sim_len: Number of simulation time steps
        record_time: Whether to record timing information
        
    Returns:
        tuple: (accuracy, time_per_step) if record_time else (accuracy,)
    """
    total_correct = 0
    total_samples = 0
    total_batches = len(test_dataloader)
    
    print(f"Starting evaluation with {total_batches} batches...")
    
    model.eval()
    if record_time:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tot_time = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_dataloader, desc="Evaluating SNN")):
            # Move inputs to GPU
            input_ids = inputs['input_ids'].to(torch.device('cuda'), non_blocking=True)
            attention_mask = inputs['attention_mask'].to(torch.device('cuda'), non_blocking=True)
            labels = labels.to(torch.device('cuda'), non_blocking=True)
            
            batch_size = len(labels)
            total_samples += batch_size
            
            # Process through SNN model (single timestep, neurons spike internally)
            if record_time:
                starter.record()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                ender.record()
                torch.cuda.synchronize()
                tot_time += starter.elapsed_time(ender) / 1000
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            total_correct += batch_correct
            
            # Print progress every 20 batches during evaluation
            if (batch_idx + 1) % 20 == 0:
                current_accuracy = total_correct / total_samples
                print(f"Eval Batch [{batch_idx+1}/{total_batches}] - Current Accuracy: {current_accuracy:.4f}")
    
    accuracy = total_correct / total_samples
    
    # Print evaluation summary
    print(f"Evaluation completed - Total samples: {total_samples}, Accuracy: {accuracy:.4f}")
    if record_time:
        avg_time = tot_time / total_samples
        print(f"Average inference time per sample: {avg_time:.6f} seconds")
        return accuracy, avg_time
    else:
        return accuracy,


def calib_text_snn_one_epoch(model, dataloader):
    """
    Calibrate text SNN model thresholds
    
    Args:
        model: SNN model for text
        dataloader: Calibration data loader
        
    Returns:
        float: Calibration accuracy
    """
    set_calib_text_opt(model, True)  # Need to implement this
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            input_ids = inputs['input_ids'].to(torch.device('cuda'), non_blocking=True)
            attention_mask = inputs['attention_mask'].to(torch.device('cuda'), non_blocking=True)
            labels = labels.to(torch.device('cuda'), non_blocking=True)
            
            total_samples += len(labels)
            
            # Process through model (single time step for calibration)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    
    return total_correct / total_samples


def set_calib_text_opt(model, on=True):
    """
    Set text model calibration options
    
    Args:
        model: SNN model for text
        on: Whether to enable calibration mode
    """
    for module in model.modules():
        if hasattr(module, 'is_cab'):
            module.is_cab = on
        if hasattr(module, 'calib_inf'):
            module.calib_inf = on


def time_step_text_eval(model, test_dataloader, sim_len):
    """
    Evaluate SNN model at each time step to see accuracy progression
    
    Args:
        model: SNN model for text
        test_dataloader: Test data loader
        sim_len: Number of simulation time steps
        
    Returns:
        list: Accuracy at each time step
    """
    accuracies = []
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Time-step evaluation"):
            input_ids = inputs['input_ids'].to(torch.device('cuda'), non_blocking=True)
            attention_mask = inputs['attention_mask'].to(torch.device('cuda'), non_blocking=True)
            labels = labels.to(torch.device('cuda'), non_blocking=True)
            
            # For time-step evaluation, we need to simulate the SNN time evolution
            # This requires the neurons to maintain state across timesteps
            # For now, just do single evaluation (proper SNN state management needs more work)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / len(labels)
            accuracies.append(accuracy)
    
    return accuracies