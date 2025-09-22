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
        # mean_over_time: [B*S, H], TxT: [T, 1] -> expand both to [T, B*S, H]
        TxT_expanded = self.TxT.unsqueeze(1).expand(-1, B*S, -1)  # [T, B*S, 1]
        bias_expanded = self.bias.unsqueeze(1).expand(-1, B*S, -1)  # [T, B*S, 1]
        scaled = mean_over_time.unsqueeze(0) * TxT_expanded  
        out = (scaled + bias_expanded) >= self.v_threshold
        out = out.float() * self.v_threshold

        # Reshape back to [B, S, H] - out currently has shape [T, B*S, H]
        return out.view(self.T, B, S, H).mean(dim=0)  # Average over timesteps to get [B, S, H]

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
        self.is_cab = False
        self.calib_inf = False

        # Per-hidden-dim learnable/calibrated params
        self.clip_min = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.clip_max = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.phi = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        
        # Calibration parameters
        self.register_buffer('rec_in_mean', torch.zeros(hidden_size))
        self.register_buffer('rec_th_mean', torch.zeros(hidden_size))

    def forward(self, x):  # x: [B, S, H]
        if self.calib_inf:
            # In calibration mode, use recorded statistics
            return torch.clamp(torch.floor((x + self.rec_in_mean) * self.T / (self.clip_max + self.rec_th_mean) + 0.5) / self.T, 0, 1) * (self.clip_max + self.rec_th_mean)
        
        if self.is_cab:
            # Record statistics during calibration
            # Reshape to [B*S, H] for per-hidden-dim statistics
            x_flat = x.reshape(-1, self.hidden_size)
            self.rec_in_mean = 0.9 * self.rec_in_mean + 0.1 * x_flat.mean(0)
            self.rec_th_mean = 0.9 * self.rec_th_mean + 0.1 * (x_flat - self.rec_in_mean).abs().mean(0)
        
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
