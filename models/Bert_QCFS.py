import torch
from torch import nn
from transformers import BertModel, BertConfig
import sys 
sys.path.append("..") 
from modules_text import DA_QCFS_Text, ParaInfNeuron_Text   # <- from your modules.py

# === Intermediate + Output QCFS layers ===
class BertIntermediateQCFS(nn.Module):
    def __init__(self, config, T=4):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.da_qcfs = DA_QCFS_Text(config.intermediate_size, T, is_relu=True)
        self.spike_neuron = ParaInfNeuron_Text(T)

    def forward(self, hidden_states):
        # If hidden_states has time dimension [T, B, S, H], process across time
        if hidden_states.dim() == 4 and hidden_states.shape[0] == self.spike_neuron.T:
            # Time-expanded input: [T, B, S, H]
            x = self.dense(hidden_states)
            x = self.da_qcfs(x)
            x = self.spike_neuron(x)
            return x
        else:
            # Single timestep input: [B, S, H]
            x = self.dense(hidden_states)
            x = self.da_qcfs(x)
            # For single timestep, just apply threshold directly
            x = (x >= self.spike_neuron.v_threshold).float() * self.spike_neuron.v_threshold
            return x


class BertOutputQCFS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayerQCFS(nn.Module):
    def __init__(self, config, T=4):
        super().__init__()
        self.config = config
        from transformers.models.bert.modeling_bert import BertAttention
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediateQCFS(config, T)
        self.output = BertOutputQCFS(config)
        # Add embedding layer for fallback
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, *args, **kwargs):
        # Extract the first two arguments: hidden_states and attention_mask
        if len(args) >= 2:
            hidden_states = args[0]
            attention_mask = args[1]
            # All other args (3-5) are ignored as they' re not needed
        
        # Debug: Print tensor info
        print(f"DEBUG: hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}, dim: {hidden_states.dim()}")
        
        # Check if we got 2D input
        if hidden_states.dim() == 2:
            # Check if it's actually input_ids (int/long) or flattened hidden states (float)
            if hidden_states.dtype in [torch.long, torch.int]:
                # These are actual input_ids
                print("DEBUG: Received input_ids, embedding them...")
                input_ids = hidden_states
                batch_size, seq_length = input_ids.shape
                
                # Create position ids
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
                
                # Embed input_ids and position_ids
                inputs_embeds = self.word_embeddings(input_ids)
                position_embeds = self.position_embeddings(position_ids)
                
                # Combine embeddings
                hidden_states = inputs_embeds + position_embeds
                hidden_states = self.LayerNorm(hidden_states)
                hidden_states = self.dropout(hidden_states)
            else:
                # These are flattened hidden states, reshape to 3D
                print("DEBUG: Received flattened hidden states, reshaping...")
                # Assume the last dimension is hidden_size
                config = getattr(self, 'config', None)
                if config is None:
                    # Try to infer hidden_size
                    hidden_size = 768  # Default for bert-base
                else:
                    hidden_size = config.hidden_size
                
                batch_size = hidden_states.shape[0]
                seq_length = hidden_states.shape[1] // hidden_size
                hidden_states = hidden_states.view(batch_size, seq_length, hidden_size)
                print(f"DEBUG: Reshaped to: {hidden_states.shape}")
        
        # Now hidden_states should be 3D [B, S, H]
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# === Full model for classification ===
class BertForSequenceClassificationQCFS(nn.Module):
    def __init__(self, pretrained_name="bert-base-uncased", T=4, num_labels=2):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_name)
        config._attn_implementation = "eager"   # or "sdpa" if you want SDPA
        self.bert = BertModel.from_pretrained(pretrained_name)

        # Replace encoder layers
        self.bert.encoder.layer = nn.ModuleList(
            [BertLayerQCFS(config, T) for _ in range(config.num_hidden_layers)]
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(pooled)
        return logits
