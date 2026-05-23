import math

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerPredictor(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_layers, output_dim, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])


class JointCompressorTransformer(nn.Module):
    def __init__(self, input_dim, compressor_cfg, transformer_cfg, output_dim):
        super().__init__()
        h1, h2 = compressor_cfg.get("hidden_dims", (512, 128))
        bottleneck_dim = compressor_cfg["bottleneck_dim"]
        dropout = compressor_cfg.get("dropout_rate", 0.1)

        self.compressor = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, bottleneck_dim),
        )
        self.predictor = TransformerPredictor(
            latent_dim=bottleneck_dim,
            d_model=transformer_cfg["d_model"],
            nhead=transformer_cfg["nhead"],
            num_layers=transformer_cfg["num_layers"],
            output_dim=output_dim,
            dropout_rate=transformer_cfg.get("dropout_rate", 0.1),
        )

    def forward(self, x):
        bsz, seq_len, input_dim = x.shape
        z = self.compressor(x.reshape(bsz * seq_len, input_dim))
        z = z.reshape(bsz, seq_len, -1)
        return self.predictor(z)

