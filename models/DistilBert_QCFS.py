from transformers import DistilBertConfig, DistilBertModel
import torch.nn as nn

class DistilBertIntermediateQCFS(nn.Module):
    def __init__(self, config, T=4):
        super().__init__()
        self.T = T
        self.dense = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.pdrop)
        self.activation = nn.GELU()
        
        # QCFS parameters for parallel inference
        self.TxT = nn.Parameter(torch.randn(T))
        self.bias = nn.Parameter(torch.randn(T))
        self.v_threshold = 1.0
        self.T = T
        
        # Initialize QCFS parameters
        nn.init.normal_(self.TxT, mean=0.0, std=0.1)
        nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, hidden_states):
        """
        Apply QCFS transformation to hidden states
        
        Args:
            hidden_states: Input hidden states [B, S, H]
            
        Returns:
            Output hidden states after QCFS transformation [B, S, H]
        """
        B, S, H = hidden_states.shape
        
        # Apply dense transformation
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        
        # Apply QCFS for parallel inference
        mean_over_time = hidden_states.mean(dim=1)  # [B, H]
        
        # Reshape for parallel computation
        mean_over_time = mean_over_time.unsqueeze(1)  # [B, 1, H]
        
        # Parallel computation across time steps
        # TxT: [T, 1, 1] -> [T, B, H] after broadcasting
        TxT_expanded = self.TxT.view(self.T, 1, 1).expand(-1, B, H)
        bias_expanded = self.bias.view(self.T, 1, 1).expand(-1, B, H)
        
        scaled = mean_over_time * TxT_expanded
        out = (scaled + bias_expanded) >= self.v_threshold
        
        # Average across time steps
        output = out.float().mean(dim=0)  # [B, H]
        output = output.unsqueeze(1).expand(-1, S, H)  # [B, S, H]
        
        return output


class DistilBertForSequenceClassificationQCFS(nn.Module):
    def __init__(self, pretrained_name="distilbert-base-uncased", T=4, num_labels=2):
        super().__init__()
        config = DistilBertConfig.from_pretrained(pretrained_name)
        self.distilbert = DistilBertModel.from_pretrained(pretrained_name)
        
        # Replace the transformer layer with QCFS
        # Note: DistilBERT has a different architecture than BERT
        # We need to handle this carefully
        
        self.classifier = nn.Linear(config.dim, num_labels)
        self.T = T

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get outputs from DistilBERT
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        
        return logits


# === Old implementation for reference - we may need to modify DistilBERT structure ===
class DistilBertLayerQCFS(nn.Module):
    def __init__(self, config, T=4):
        super().__init__()
        self.config = config
        self.T = T
        
        # Use original DistilBERT components
        self.attention = None  # DistilBERT uses a single attention layer
        self.intermediate_qcfs = DistilBertIntermediateQCFS(config, T)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        # DistilBERT has a simpler architecture than BERT
        # We'll need to adapt this based on DistilBERT's actual structure
        
        # For now, just apply QCFS transformation
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        # Apply QCFS transformation
        output = self.intermediate_qcfs(hidden_states)
        
        return output