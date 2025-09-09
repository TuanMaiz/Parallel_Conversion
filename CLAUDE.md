# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements "Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation" (ICML 2025). It researches parallel spiking neuron models for converting Artificial Neural Networks (ANNs) to Spiking Neural Networks (SNNs) with improved accuracy and speed.

## Architecture

The codebase follows a modular architecture for ANN-SNN conversion research:

### Core Components
- **`main.py`**: Main training/inference script supporting multiple datasets (CIFAR10/100, ImageNet, Text) and network architectures
- **`models/`**: Neural network implementations
  - `VGG_QCFS.py`: VGG with QCFS (Quantized Clipped Feature Scaling) layers
  - `ResNet_QCFS.py`: ResNet variants with QCFS layers
  - `ResNet_ReLU.py`: Standard ResNet models for baseline comparison
  - `Bert_QCFS.py`: BERT models for text classification tasks
- **`modules.py`**: Core neuron implementations (QCFS, ParaInfNeuron, IFNeuron, RecReLU)
- **`modules_text.py`**: Text-specific neuron implementations (ParaInfNeuron_Text, DA_QCFS_Text, RecReLU_Text)
- **`utils.py`**: Model conversion utilities (replace_relu_by_func, replace_qcfs_by_neuron)
- **`dataprocess.py`**: Dataset preprocessing for CIFAR10/100 and ImageNet
- **`dataprocess_text.py`**: Text data preprocessing and dataloaders

### Key Neuron Types
- **QCFS**: Quantized Clipped Feature Scaling
- **ParaInfNeuron**: Parallel Inference Neuron for faster spiking computation
- **ParaInfNeuron_CW_ND**: Enhanced version with clamped weights and non-deterministic behavior
- **IFNeuron**: Integrate-and-Fire neuron
- **RecReLU**: Recording ReLU for calibration

## Usage

### Training/Inference Command
```bash
python main.py --dataset ImageNet --datadir /path/to/datasets/ --savedir /path/to/save/ \
    --net_arch resnet34_qcfs --amp --batchsize 100 --dev 0 --time_step 8 \
    --neuron_type ParaInfNeuron --checkpoint_path /path/to/checkpoints \
    --pretrained_model --direct_inference
```

### Key Arguments
- `--dataset`: CIFAR10, CIFAR100, ImageNet, or TextCLS
- `--net_arch`: Network architecture (vgg16_qcfs, resnet20_qcfs, resnet34_qcfs, resnet18, etc.)
- `--neuron_type`: Neuron type for SNN conversion
- `--time_step`: Number of timesteps for SNN simulation
- `--amp`: Enable Automatic Mixed Precision training
- `--distributed_init_mode`: For multi-GPU training
- `--direct_inference`: Skip training, perform direct inference

## Development Environment

### Dependencies
- PyTorch
- torchvision
- timm (for some models and mixup)
- CUDA support required for GPU training

### Running Environment
The code supports both single-GPU and distributed multi-GPU training via PyTorch's DistributedDataParallel. Use environment variables RANK, WORLD_SIZE, and LOCAL_RANK for distributed training.

### Model Conversion Pipeline
1. Start with standard ANN (ResNet, VGG, BERT)
2. Convert ReLU layers to QCFS or other neuron types using `replace_relu_by_func()`
3. Calibrate threshold values using `calib_one_epoch()`
4. Convert to target neuron type for SNN inference
5. Evaluate accuracy and speed performance

### File Organization
- Models are modular and support both ANNs and SNNs
- Data preprocessing includes augmentation policies (AutoAugment, Cutout)
- Logging is handled via `get_logger()` function with timestamped output files
- Checkpoints save model state, optimizer, scheduler, and metrics

### Logging and Output
The `get_logger()` function in `main.py:22` creates timestamped log files with detailed formatting. Checkpoints include:
- Model state dict
- Optimizer state
- Learning rate scheduler state
- Training metrics and validation accuracy

### Common Commands
```bash
# Single GPU training
python main.py --dataset CIFAR10 --datadir /path/to/data --savedir /path/to/save \
    --net_arch resnet34_qcfs --batchsize 128 --dev 0 --time_step 8

# Direct inference (skip training)
python main.py --dataset ImageNet --datadir /path/to/data --savedir /path/to/save \
    --net_arch resnet34_qcfs --dev 0 --time_step 8 --direct_inference \
    --pretrained_model --checkpoint_path /path/to/checkpoints
```

## Common Tasks

### Adding New Datasets
Implement preprocessing functions in `dataprocess.py` or `dataprocess_text.py` following existing patterns.

### Adding New Network Architectures
Add model definitions in `models/` directory with consistent QCFS/layer integration.

### Adding New Neuron Types
Implement new neuron classes in `modules.py` or `modules_text.py`, then update conversion functions in `utils.py`.

### Environment Setup
The project uses Python 3.13 and requires PyTorch, torchvision, and timm. Dependencies should be installed in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
pip install torch torchvision timm numpy tqdm
```

### Running Experiments
Use environment variables for distributed training:
```bash
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
```

For multi-GPU training:
```bash
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0
python -m torch.distributed.launch --nproc_per_node=2 main.py [arguments]
```

### Debugging and Troubleshooting
- **CUDA Out of Memory**: Reduce batchsize or use `--amp` for mixed precision training
- **Distributed Training Issues**: Ensure RANK, WORLD_SIZE, and LOCAL_RANK are properly set
- **Import Errors**: Activate virtual environment before running scripts
- **Checkpoint Loading Issues**: Verify checkpoint path matches the expected model architecture