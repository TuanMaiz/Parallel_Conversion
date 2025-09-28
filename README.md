# Code implementation for applying Parallel Spiking Calculation in Natural Language Processing
## 👨‍💻 Quick Usage
```
!python Parallel_Conversion/main.py \
      --dataset TextCLS \
      --net_arch distilbert_base_qcfs \
      --savedir ./checkpoints \
      --neuron_type ParaInfNeuron_Text \
      --text_dataset imdb \
      --text_max_len 256 \
      --time_step 2 \
      --trainsnn_epochs 5 \
      --batchsize 16 \
      --lr 0.00001 \
      --measure_efficiency \
      --gpu_type A100 \
      --dev 0
```

## ✒️ Citation


```bibtex
@inproceedings{hao2025conversion,
  title={Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation},
  author={Hao, Zecheng and Ma, Qichao and Chen, Kang and Zhang, Yi and Yu, Zhaofei and Huang, Tiejun},
  year={2025},
  booktitle={International Conference on Machine Learning}
}
```

