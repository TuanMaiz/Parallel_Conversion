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
    epoch_loss, lenth = 0, 0
    model.train()
    
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Move inputs to GPU - text data has multiple keys
        input_ids = inputs['input_ids'].cuda(local_rank, non_blocking=True)
        attention_mask = inputs['attention_mask'].cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)
        
        lenth += len(labels)
        
        # Apply mixup if enabled
        if mixup:
            # Note: Mixup for text needs special handling - may need to implement text-specific mixup
            pass  # Placeholder for text mixup
        
        # Time-step expansion for text inputs
        # input_ids: [B, S] -> [T*B, S] 
        # attention_mask: [B, S] -> [T*B, S]
        input_ids_expanded = input_ids.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
        attention_mask_expanded = attention_mask.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
        
        optimizer.zero_grad()
        
        # Process through SNN
        if scaler is not None:
            with amp.autocast():
                # Forward pass through SNN
                all_spikes = []
                for t in range(sim_len):
                    # Process single time step
                    spike_t = model(input_ids=input_ids_expanded[t:t+1], 
                                  attention_mask=attention_mask_expanded[t:t+1])
                    all_spikes.append(spike_t)
                
                # Mean over time steps for readout
                spikes = torch.stack(all_spikes).mean(dim=0)
                loss = loss_fn(spikes, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass through SNN
            all_spikes = []
            for t in range(sim_len):
                spike_t = model(input_ids=input_ids_expanded[t], 
                              attention_mask=attention_mask_expanded[t])
                all_spikes.append(spike_t)
            
            # Mean over time steps for readout
            spikes = torch.stack(all_spikes).mean(dim=0)
            loss = loss_fn(spikes, labels)
            loss.backward()
            optimizer.step()
        
        # Aggregate loss for distributed training
        if distributed:
            from main import reduce_mean
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
        else:
            epoch_loss += loss.item()
    
    return epoch_loss / lenth


def eval_text_snn(model, test_dataloader, sim_len, record_time=False):
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
    
    model.eval()
    if record_time:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tot_time = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Evaluating SNN"):
            # Move inputs to GPU
            input_ids = inputs['input_ids'].to(torch.device('cuda'), non_blocking=True)
            attention_mask = inputs['attention_mask'].to(torch.device('cuda'), non_blocking=True)
            labels = labels.to(torch.device('cuda'), non_blocking=True)
            
            total_samples += len(labels)
            
            # Time-step expansion
            input_ids_expanded = input_ids.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
            attention_mask_expanded = attention_mask.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
            
            # Process through SNN
            all_spikes = []
            for t in range(sim_len):
                if record_time:
                    starter.record()
                    spike_t = model(input_ids=input_ids_expanded[t:t+1], 
                                  attention_mask=attention_mask_expanded[t:t+1])
                    ender.record()
                    torch.cuda.synchronize()
                    tot_time += starter.elapsed_time(ender) / 1000
                else:
                    spike_t = model(input_ids=input_ids_expanded[t:t+1], 
                                  attention_mask=attention_mask_expanded[t:t+1])
                all_spikes.append(spike_t)
            
            # Mean over time steps for readout
            spikes = torch.stack(all_spikes).mean(dim=0)
            
            # Calculate accuracy
            _, predicted = torch.max(spikes, 1)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_samples
    
    if record_time:
        avg_time = tot_time / total_samples
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
            
            # Time-step expansion
            input_ids_expanded = input_ids.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
            attention_mask_expanded = attention_mask.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
            
            # Accumulate spikes over time
            accumulated_spikes = []
            for t in range(sim_len):
                spike_t = model(input_ids=input_ids_expanded[t:t+1], 
                              attention_mask=attention_mask_expanded[t:t+1])
                accumulated_spikes.append(spike_t)
                
                # Calculate accuracy up to current time step
                current_spikes = torch.stack(accumulated_spikes).mean(dim=0)
                _, predicted = torch.max(current_spikes, 1)
                accuracy = (predicted == labels).sum().item() / len(labels)
                accuracies.append(accuracy)
    
    return accuracies