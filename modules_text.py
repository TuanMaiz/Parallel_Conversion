import torch
from torch import nn

class ParaInfNeuron_Text(nn.Module):
    """
    Parallel inference neuron for text (Transformer hidden states).
    Works on [B, S, H] inputs (batch, seq_len, hidden_dim).
    """
    def __init__(self, T, th=1., init_mem=0.5):
        super().__init__()
        self.T = T
        self.v_threshold = th
        # Precompute scaling factors (same idea as vision version)
        self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1))    # [T,1]
        self.register_buffer('bias', (init_mem * th) / torch.arange(1, T+1).unsqueeze(-1))  # [T,1]

    def forward(self, x):
        # x: [B, S, H]
        B, S, H = x.shape

        # Reshape to [T, B*S, H] to apply over timesteps
        x = x.unsqueeze(0).expand(self.T, -1, -1, -1)   # [T, B, S, H]
        x = x.reshape(self.T, B*S, H)                   # [T, B*S, H]

        # Mean over time, then apply TxT scaling - apply scaling per timestep
        mean_over_time = x.mean(dim=0)  # [B*S, H]
        # Apply scaling for each timestep independently
        scaled = mean_over_time.unsqueeze(0) * self.TxT  # [T, B*S, H] scaling
        out = (scaled + self.bias) >= self.v_threshold
        out = out.float() * self.v_threshold

        # Reshape back to [B, S, H]
        return out.view(B, S, H)

class DA_QCFS_Text(nn.Module):
    """
    Distribution-Aware QCFS for text activations.
    Applies per-hidden-dim quantization and calibration.
    """
    def __init__(self, hidden_size, T, is_relu=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.T = T
        self.is_relu = is_relu

        # Per-hidden-dim learnable/calibrated params
        self.clip_min = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.clip_max = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.phi = nn.Parameter(torch.ones(hidden_size), requires_grad=False)

    def forward(self, x):  # x: [B, S, H]
        # Clamp per hidden dim
        x = torch.max(x, self.clip_min)
        x = torch.min(x, self.clip_max)

        # Distribution-aware affine correction
        x = (x + self.psi) * self.phi

        # Quantization (QCFS)
        x = torch.clamp(torch.floor(x * self.T + 0.5) / self.T, 0, 1)

        # Scale back
        return x * self.clip_max

class RecReLU_Text(nn.Module):
    """
    Record & clamp ReLU for text (per hidden dim).
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_up = False

    def forward(self, x):  # x: [B, S, H]
        # max over batch + sequence for each hidden dim
        max_th = x.reshape(-1, self.hidden_size).max(0)[0]  # [H]
        if not self.init_up:
            self.register_buffer('up', max_th)
            self.init_up = True
        else:
            self.up = torch.max(self.up, max_th)

        return torch.clamp(x, torch.zeros_like(self.up), self.up)
