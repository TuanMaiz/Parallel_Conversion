 Detailed Gap Analysis

  1. dataprocess_text.py:21 - Critical Bug

  Current Code:
  def __getitem__(self, idx):
      item = {key: torch.tensor(val[idx]) for key, val in
  self.encodings.items()}  # ❌ self.encodings undefined
      label = torch.tensor(self.labels[idx])  # ❌ self.labels 
  undefined
      return item, label

  Problem: The class never initializes self.encodings or
  self.labels. It should use the HuggingFace dataset directly.

  Correct Implementation:
  def __getitem__(self, idx):
      item = self.dataset[idx]  # This is a dict from HuggingFace 
  dataset
      text = item['text']
      label = item['label']

      # Tokenize text
      encoding = self.tokenizer(
          text,
          truncation=True,
          max_length=self.max_len,
          return_tensors='pt'
      )

      return {key: val.squeeze() for key, val in encoding.items()},
  torch.tensor(label)

● 2. Missing SNN Training Pipeline

  Problem: Your current implementation can only do ANN training, not
   SNN training with spiking neurons.

  Current Training Loop Issues:
  # main.py:76 - ANN forward pass
  spikes = model(img).mean(dim=0)  # ❌ Not spiking, just averaging

  Missing Components:

  a) No Spiking Forward Pass for Text:
  # Current: Bert model outputs logits directly
  outputs = self.bert(input_ids=input_ids,
  attention_mask=attention_mask)
  pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
  logits = self.classifier(pooled)
  return logits  # No spiking

  # Should be: Process through spiking neurons
  spike_outputs = model(input_ids, attention_mask, time_step=T)
  logits = spike_outputs.mean(dim=0)  # Mean over time steps

  b) No Time-Step Expansion:
  For SNNs, you need to expand inputs over time steps:
  # Current: Single forward pass
  input_ids: [B, S]

  # Should be: Time-expanded for SNN
  input_ids: [T, B, S]  # Repeat input over time steps

  c) Missing Text-Specific Neuron Reset:
  Vision neurons reset after each image, but text neurons need
  sequence-aware reset.

  d) No Spike Recording for Calibration:
  Missing mechanism to record spikes for threshold calibration like
  calib_one_epoch() in vision.

● 3. Missing Model Conversion Utilities

  Current utils.py limitations:
  # utils.py only has vision-specific conversions
  def replace_relu_by_func(model, func_type, T=8):
      # Replaces vision ReLU layers, not text layers

  def replace_qcfs_by_neuron(model, neuron_type):
      # Only works with vision QCFS modules

  Missing Text-Specific Functions:

  a) Text Model Conversion Function:
  def replace_text_qcfs_by_neuron(model, neuron_type):
      """Convert text QCFS layers to target neuron type"""
      for name, module in model.named_modules():
          if isinstance(module, BertIntermediateQCFS):
              if 'ParaInfNeuron_Text' in neuron_type:
                  module.spike_neuron =
  ParaInfNeuron_Text(module.da_qcfs.T)
              elif 'IFNeuron_Text' in neuron_type:
                  module.spike_neuron =
  IFNeuron_Text(module.da_qcfs.T)

  b) Text Layer Replacement:
  Your Bert_QCFS.py has hardcoded neuron types:
  class BertIntermediateQCFS(nn.Module):
      def __init__(self, config, T=4):
          self.da_qcfs = DA_QCFS_Text(config.intermediate_size, T,
  is_relu=True)
          self.spike_neuron = ParaInfNeuron_Text(T)  # ❌ Hardcoded

  Should be configurable:
  def __init__(self, config, T=4, neuron_type="ParaInfNeuron_Text"):
      self.da_qcfs = DA_QCFS_Text(config.intermediate_size, T,
  is_relu=True)
      if neuron_type == "ParaInfNeuron_Text":
          self.spike_neuron = ParaInfNeuron_Text(T)
      elif neuron_type == "IFNeuron_Text":
          self.spike_neuron = IFNeuron_Text(T)

  c) Missing Text Calibration Functions:
  def calib_text_model(model, dataloader, T=8):
      """Calibrate text neuron thresholds"""
      for module in model.modules():
          if hasattr(module, 'calibrate_threshold'):
              module.calibrate_threshold(dataloader, T)

● 4. Dataset Limitations

  Current Implementation Issues:

  a) Hardcoded Dataset:
  # dataprocess_text.py:12
  self.dataset = load_dataset(dataset_name, split=split)  # ❌ 
  Hardcoded to ag_news

  b) Missing Configuration in main.py:
  # main.py doesn't have text dataset arguments
  parser.add_argument('--text_dataset', type=str, default='ag_news',

                      choices=['ag_news', 'imdb', 'sst2', 'cola'])
  parser.add_argument('--max_seq_len', type=int, default=128)

  c) Inconsistent Dataset Formats:
  Different HuggingFace datasets have different field names:
  # ag_news: {'text': str, 'label': int}
  # imdb: {'text': str, 'label': int} 
  # sst2: {'sentence': str, 'label': int}  # ❌ Different field 
  names

  d) Missing Text Preprocessing:
  # Current: No text preprocessing (lowercasing, etc.)
  # Missing: Text cleaning, special token handling, padding

  e) No Support for Multiple Tokenizers:
  # Hardcoded to BERT tokenizer
  # Missing: Roberta, DistilBERT, T5 support

  Required Implementation:
  class TextCLSDataset(Dataset):
      def __init__(self, split="train", 
  tokenizer_name="bert-base-uncased",
                   max_len=128, dataset_name="ag_news", 
  text_field="text"):
          self.dataset = load_dataset(dataset_name, split=split)
          self.text_field = text_field  # Make configurable
          self.label_field = "label"

          # Handle different dataset formats
          if dataset_name == "sst2":
              self.text_field = "sentence"

● 5. Text-Specific Evaluation Gaps

  Current Training/Evaluation Loop Issues:

  a) No Time-Step Processing for Text:
  # main.py:71 - Vision approach
  img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1).flatten(0, 1)
  # Time-expand vision

  # Missing equivalent for text:
  input_ids = input_ids.unsqueeze(0).repeat(sim_len, 1,
  1).flatten(0, 1)  # Should do this
  attention_mask = attention_mask.unsqueeze(0).repeat(sim_len, 1,
  1).flatten(0, 1)

  b) Incorrect Loss Calculation:
  # main.py:77 - Current: Averages over time (not proper SNN loss)
  spikes = model(img).mean(dim=0)  # ❌ Wrong for text
  loss = loss_fn(spikes, label)

  Should be:
  # Proper SNN processing
  all_spikes = []
  for t in range(sim_len):
      spikes_t = model(input_ids_t, attention_mask_t)  # Single time
   step
      all_spikes.append(spikes_t)

  # Readout over time
  spikes = torch.stack(all_spikes).mean(dim=0)  # Mean over time
  loss = loss_fn(spikes, label)

  c) No Text-Specific Metric Tracking:
  Vision models track accuracy per timestep, but text needs
  sequence-aware metrics:
  # Missing: Sequence-level accuracy, per-timestep spike patterns
  # Missing: Text-specific timing metrics

  d) No Spiking Pattern Analysis:
  # Missing: Spike frequency analysis per token position
  # Missing: Attention pattern correlation with spike patterns

  e) Missing Direct Inference for Text:
  The --direct_inference flag in main.py works for vision but not
  for text because text models need special handling.

  Required Text-Specific Functions:
  def eval_text_snn(model, test_dataloader, sim_len, 
  record_time=False):
      """Evaluate SNN on text data with proper time-step 
  processing"""
      model.eval()
      with torch.no_grad():
          for input_ids, attention_mask, labels in test_dataloader:
              # Time-expand inputs
              input_ids_expanded =
  input_ids.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)
              attention_mask_expanded =
  attention_mask.unsqueeze(0).repeat(sim_len, 1, 1).flatten(0, 1)

              # Process through SNN
              all_spikes = []
              for t in range(sim_len):
                  spike_t = model(input_ids_expanded[t],
  attention_mask_expanded[t])
                  all_spikes.append(spike_t)

              # Readout and evaluate
              spikes = torch.stack(all_spikes).mean(dim=0)

● Summary of Critical Gaps

  Priority 1 (Critical - Implementation will fail):

  1. dataprocess_text.py:21 Bug - self.encodings and self.labels
  undefined
  2. No SNN Training Loop - Current code only does ANN training
  3. Missing Time-Step Expansion - Text inputs need temporal
  processing

  Priority 2 (High - Won't work properly):

  4. Missing Model Conversion Utilities - No
  replace_text_qcfs_by_neuron()
  5. Hardcoded Dataset - Only works with ag_news
  6. No Text-Specific Evaluation - Wrong loss calculation, no
  time-expansion

  Priority 3 (Medium - Limited functionality):

  7. No Configuration Support - Missing command line arguments for
  text
  8. No Spiking Pattern Analysis - Missing text-specific metrics
  9. Limited Neuron Types - Only ParaInfNeuron_Text, missing other
  variants

  Your implementation has a good foundation with proper neuron
  definitions for text data, but these gaps prevent it from actually
   working as an ANN-SNN conversion system for text. The most
  critical issues are the dataset bug and the missing SNN training
  pipeline.