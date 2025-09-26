# Comprehensive Experiment Setup for ANN-SNN Conversion

## **Experiment Setup Proposal**

### **1. Neuron Type Comparisons**

**Available neuron types:**
- `ParaInfNeuron_Text` - Parallel inference (current focus)
- `IFNeuron_Text` - Integrate-and-Fire (sequential) 
- `RecReLU_Text` - Recording ReLU (calibration baseline)

**Experiments:**
```bash
# Compare parallel vs sequential approaches
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 2
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 4  
python main.py --dataset TextCLS --neuron_type IFNeuron_Text --time_step 8  # Sequential baseline
python main.py --dataset TextCLS --neuron_type IFNeuron_Text --time_step 16 # Sequential longer
```

### **2. Time Step Analysis**

**Time step variations:**
```bash
# For ParaInfNeuron_Text (parallel efficiency)
--time_step 1, 2, 3, 4, 5, 6, 7, 8

# For IFNeuron_Text (sequential comparison)  
--time_step 4, 8, 16, 32, 64

# Compare speedup: ParaInfNeuron_Text(T=4) vs IFNeuron_Text(T=16)
```

### **3. Threshold Variations**

**Test different threshold strategies:**
```bash
# Modify v_threshold in ParaInfNeuron_Text
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 2 --threshold 0.8
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 2 --threshold 1.0
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 2 --threshold 1.2
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 4 --threshold 0.8
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 4 --threshold 1.0
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 4 --threshold 1.2
```

### **4. Memory Efficiency Analysis**

**Compare memory usage:**
```bash
# Same models with different batch sizes
--batchsize 16, 32, 64, 128

# Measure memory footprint during inference
python -c "
import torch
from models.Bert_QCFS import BertForSequenceClassificationQCFS
model = BertForSequenceClassificationQCFS()
input_ids = torch.randint(0, 10000, (1, 128))
torch.cuda.max_memory_allocated()
"
```

### **5. Model Architecture Variations**

**Test different BERT variants:**
```bash
# Different model sizes
--model_name bert-base-uncased (110M params)
--model_name bert-large-uncased (340M params) 
--model_name distilbert-base-uncased (66M params)

# Layer-wise conversion depth
--convert_layers 1,4,8,12  # Convert first N layers only
```

### **6. Dataset Comparisons**

**Test on different text classification tasks:**
```bash
# Different datasets with varying complexity
python main.py --dataset TextCLS --subset imdb_sentiment
python main.py --dataset TextCLS --subset ag_news
python main.py --dataset TextCLS --subset dbpedia_14
python main.py --dataset TextCLS --subset yahoo_answers
```

### **7. Quantization Level Analysis**

**Test different quantization precision:**
```bash
# Modify T parameter for different precision levels
--time_step 2,4,8  # Lower T = lower precision, faster computation
```

### **8. Head-to-Head Comparison Matrix**

**Complete comparison setup:**
```bash
# Standard BERT (baseline)
python main.py --dataset TextCLS --neuron_type ReLU --time_step 1

# Parallel SNN (proposed method)
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 2
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --time_step 4

# Sequential SNN (baseline)
python main.py --dataset TextCLS --neuron_type IFNeuron_Text --time_step 8
python main.py --dataset TextCLS --neuron_type IFNeuron_Text --time_step 16

# Calibration variants
python main.py --dataset TextCLS --neuron_type RecReLU_Text --time_step 1
```

### **9. Performance Metrics**

**Measure comprehensive performance:**
```python
# Accuracy vs Time vs Memory trade-offs
{
    "model": "ParaInfNeuron_Text_T2",
    "accuracy": 0.923,
    "inference_time": 0.45,  # seconds
    "memory_usage": 2.1,     # GB
    "throughput": 222,       # samples/sec
    "flops": 1.2e10,         # FLOPs
    "params": 110e6          # Parameters
}
```

### **10. Ablation Studies**

**Component importance analysis:**
```bash
# Ablation 1: Remove distribution-aware calibration
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --no_calibration

# Ablation 2: Remove parallel processing  
python main.py --dataset TextCLS --neuron_type IFNeuron_Text --time_step 4

# Ablation 3: Combined baseline
python main.py --dataset TextCLS --neuron_type ParaInfNeuron_Text --no_calibration --time_step 4
```

## **Expected Outcomes**

### **1. Speedup Validation**
- **Target**: 2-4x speedup for ParaInfNeuron_Text vs IFNeuron_Text
- **Metrics**: Inference time, throughput, FLOPs per second

### **2. Accuracy Trade-offs**
- **Target**: >95% accuracy retention vs original BERT
- **Metrics**: Classification accuracy, F1-score, per-class performance

### **3. Memory Efficiency**
- **Target**: 20-40% memory reduction through quantization
- **Metrics**: Peak memory usage, memory footprint scaling

### **4. Optimal Configuration**
- **Target**: Identify best time_step/threshold combinations
- **Metrics**: Pareto optimal points in accuracy-speed-memory space

## **Implementation Notes**

### **Key Variables to Test**
1. **Time steps**: 1, 2, 3, 4, 5, 6, 7, 8 (parallel) vs 4, 8, 16, 32, 64 (sequential)
2. **Thresholds**: 0.8, 1.0, 1.2 for different sensitivity levels
3. **Batch sizes**: 16, 32, 64, 128 for memory scaling analysis
4. **Model sizes**: Base, large, distilled for architecture scaling

### **Critical Comparisons**
1. **ParaInfNeuron_Text(T=4) vs IFNeuron_Text(T=16)** - Direct speedup comparison
2. **ParaInfNeuron_Text(T=2) vs Standard BERT** - Minimal time step approach
3. **All neuron types with T=4** - Same complexity comparison
4. **Calibration vs No-calibration** - Ablation study importance

### **Data Collection Template**
```python
experiment_results = {
    "model_config": {
        "neuron_type": "ParaInfNeuron_Text",
        "time_step": 4,
        "threshold": 1.0,
        "model_name": "bert-base-uncased"
    },
    "performance": {
        "accuracy": 0.0,
        "inference_time_ms": 0.0,
        "throughput_samples_sec": 0.0,
        "memory_usage_gb": 0.0,
        "flops": 0.0
    },
    "dataset": {
        "name": "imdb_sentiment",
        "train_size": 0,
        "test_size": 0,
        "num_classes": 0
    }
}
```

This comprehensive setup will validate the proposed method's effectiveness and provide insights into optimal configurations for different deployment scenarios.