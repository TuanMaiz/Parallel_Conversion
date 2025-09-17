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
        x = self.dense(hidden_states)
        x = self.da_qcfs(x)
        x = self.spike_neuron(x)
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
        from transformers.models.bert.modeling_bert import BertAttention
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediateQCFS(config, T)
        self.output = BertOutputQCFS(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, output_attentions=False, use_cache=False):
        # Pass all arguments to attention layer
        attention_output = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache
        )[0]
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
