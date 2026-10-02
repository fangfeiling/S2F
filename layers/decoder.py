import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionDecoder(nn.Module):
    """
    Prediction decoder that preserves time dimension.
    Operates on each time step independently.
    """
    def __init__(self, hidden_dim, pred_len=None, his_len=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.his_len = his_len
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        if pred_len is not None and his_len is not None and pred_len != his_len:
            self.temporal_proj = nn.Linear(his_len, pred_len)
        else:
            self.temporal_proj = None
    
    def forward(self, x):
        B, N, T, D = x.shape
        
        pred_per_timestep = self.decoder(x)
        
        if self.temporal_proj is None:
            pred = pred_per_timestep.squeeze(-1)
        else:
            pred_per_timestep = pred_per_timestep.transpose(2, 3).squeeze(-1)
            pred = self.temporal_proj(pred_per_timestep)
        
        return pred


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder for multi-step predictions.
    Optional alternative to direct decoding.
    """
    def __init__(self, hidden_dim, pred_len, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        B, N, T, D = x.shape
        
        h_last = x[:, :, -1, :]
        
        h_last = h_last.view(B * N, 1, D)
        
        predictions = []
        hidden = None
        
        for t in range(self.pred_len):
            out, hidden = self.rnn(h_last, hidden)
            pred_t = self.output_proj(out)
            predictions.append(pred_t.squeeze(-1))
            h_last = out
        
        predictions = torch.stack(predictions, dim=-1)
        predictions = predictions.squeeze(1).view(B, N, self.pred_len)
        
        return predictions