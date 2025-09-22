#!/usr/bin/env python3
"""
Comparison script for BERT vs BERT-QCFS
"""

# Example commands to run for comparison:

echo "=== Training Standard BERT ==="
python main.py \
    --dataset TextCLS \
    --net_arch distilbert_base \
    --text_dataset imdb \
    --text_max_len 256 \
    --trainsnn_epochs 5 \
    --batchsize 16 \
    --lr 0.00001 \
    --dev 0 \
    --direct_inference

echo -e "\n=== Training BERT-QCFS ==="
python main.py \
    --dataset TextCLS \
    --net_arch distilbert_base_qcfs \
    --neuron_type ParaInfNeuron \
    --text_dataset imdb \
    --text_max_len 256 \
    --time_step 4 \
    --trainsnn_epochs 5 \
    --batchsize 16 \
    --lr 0.00001 \
    --dev 0 \
    --direct_inference

echo -e "\n=== Training BERT-QCFS with Calibration ==="
python main.py \
    --dataset TextCLS \
    --net_arch distilbert_base_qcfs \
    --neuron_type ParaInfNeuron \
    --text_dataset imdb \
    --text_max_len 256 \
    --time_step 4 \
    --trainsnn_epochs 5 \
    --batchsize 16 \
    --lr 0.00001 \
    --dev 0 \
    --calibrate_th \
    --direct_inference

echo -e "\n=== Comparison Summary ==="
echo "1. Standard BERT: Regular BERT model without any quantization"
echo "2. BERT-QCFS: BERT with quantization but no calibration"
echo "3. BERT-QCFS + Calib: BERT with quantization AND calibration"
echo ""
echo "Expected Results:"
echo "- Standard BERT should give baseline accuracy"
echo "- BERT-QCFS may have slight accuracy drop due to quantization"
echo "- BERT-QCFS + Calib should recover most of the accuracy loss"