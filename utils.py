from modules import *
import torch

def replace_qcfs_by_neuron(model, neuron_type):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_qcfs_by_neuron(module, neuron_type)
        if 'qcfs' in module.__class__.__name__.lower():
            if 'ParaInfNeuron_CW_ND' in neuron_type:
                if module.dim > 3:
                    model._modules[name] = ParaInfNeuron_CW_ND(module.t, module.up.item()*torch.ones_like(module.rec_th_mean.cpu()), module.rec_th_mean.cpu(), 0.5*module.up.item()+module.rec_in_mean.cpu()*module.t)
                else:
                    model._modules[name] = ParaInfNeuron(module.t, th=module.up.item(), init_mem=0.5, dim=module.dim)
            elif 'ParaInfNeuron' in neuron_type:
                model._modules[name] = ParaInfNeuron(module.t, th=module.up.item(), init_mem=0.5, dim=module.dim)
            elif 'IFNeuron' in neuron_type:
                model._modules[name] = IFNeuron(module.t, th=module.up.item(), init_mem=0.5)
                
    return model
    

def replace_relu_by_func(model, func_type, T=8):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_relu_by_func(module, func_type, T)
        if 'relu' in module.__class__.__name__.lower() or 'qcfs' in module.__class__.__name__.lower():
            if 'RecReLU' in func_type:
                model._modules[name] = RecReLU()
            elif 'QCFS' in func_type:
                model._modules[name] = QCFS(in_ch=module.up.shape[0],up=module.up.cpu(),t=T,is_cab=True,is_relu=True)            
            elif 'ParaInfNeuron_CW_ND' in func_type:
                model._modules[name] = ParaInfNeuron_CW_ND(module.t, module.up.cpu(), module.rec_th_mean.cpu(), 0.5*module.up.cpu()+module.rec_in_mean.cpu()*module.t)
            elif 'IFNeuron' in func_type:
                model._modules[name] = IFNeuron(T, th=module.up.cpu(), init_mem=0.5)

    return model


def set_calib_inf(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_calib_inf(module)
        if 'qcfs' in module.__class__.__name__.lower() and (module.dim > 3):
            module.cab_inf = True


def set_calib_opt(model, on=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_calib_opt(module, on)
        if 'qcfs' in module.__class__.__name__.lower() and (module.dim > 3):
            module.is_cab = on


# Text-specific calibration functions
def set_calib_text_opt(model, on=True):
    """Set text model calibration options"""
    for name, module in model.named_modules():
        if hasattr(module, "is_cab"):
            module.is_cab = on
        if hasattr(module, "calib_inf"):
            module.calib_inf = on


def set_calib_text_inf(model):
    """Set text model inference calibration mode"""
    for name, module in model.named_modules():
        if hasattr(module, "calib_inf"):
            module.calib_inf = True


def replace_text_qcfs_by_neuron(model, neuron_type):
    """Replace text QCFS layers with target neuron type"""
    from models.Bert_QCFS import BertIntermediateQCFS
    
    for name, module in model.named_modules():
        if isinstance(module, BertIntermediateQCFS):
            if 'ParaInfNeuron_Text' in neuron_type:
                module.spike_neuron = ParaInfNeuron_Text(module.da_qcfs.T)
            elif 'IFNeuron_Text' in neuron_type:
                # Need to implement IFNeuron_Text
                module.spike_neuron = ParaInfNeuron_Text(module.da_qcfs.T)  # Fallback for now
    return model


def calib_text_one_epoch(model, dataloader):
    """Calibrate text model for one epoch"""
    set_calib_text_opt(model, True)
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Handle text data format
            if hasattr(inputs, 'keys'):  # Dict format from HuggingFace
                input_ids = inputs['input_ids'].to(torch.device('cuda'), non_blocking=True)
                attention_mask = inputs['attention_mask'].to(torch.device('cuda'), non_blocking=True)
            else:  # Tensor format
                input_ids = inputs.to(torch.device('cuda'), non_blocking=True)
                attention_mask = None
            
            labels = labels.to(torch.device('cuda'), non_blocking=True)
            
            total_samples += len(labels)
            
            # Expand inputs in time dimension (like vision calibration)
            # Create 2 copies for calibration
            if attention_mask is not None:
                input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1).flatten(0, 1)
                attention_mask = attention_mask.unsqueeze(0).repeat(2, 1, 1).flatten(0, 1)
            else:
                input_ids = input_ids.unsqueeze(0).repeat(2, 1, 1).flatten(0, 1)
            
            # Forward pass with time expansion
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs.view(2, outputs.shape[0]//2, -1)[0]  # Take first copy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    
    set_calib_text_opt(model, False)
    set_calib_text_inf(model)
    return total_correct / total_samples 