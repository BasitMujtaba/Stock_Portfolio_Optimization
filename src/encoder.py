
"""
================================================================================
 File   : src/encoder.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Temporal encoder (LSTM / Transformer / Hybrid) that converts a
          window of (price_features + sentiment) into a fixed-size latent
          vector per ticker per timestep.

 Input  : Tensor  (batch, n_tickers, window, n_features)
 Output : Tensor  (batch, n_tickers, output_dim)

 Config keys used:
   encoder.window          — lookback window length (default 20)
   encoder.lstm_hidden     — LSTM hidden size (default 128)
   encoder.transformer_dim — Transformer d_model (default 64)
   encoder.n_heads         — Transformer attention heads (default 4)
   encoder.n_layers        — number of layers for both LSTM & Transformer
   data.n_stocks           — number of tickers (used in smoke test)
================================================================================
"""

import math
import logging
import os

import yaml
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Positional Encoding ────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal PE (Vaswani et al., 2017). Supports seq_len up to max_len."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── LSTM Encoder ───────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM.
    output_dim = lstm_hidden  (forward + backward halves = lstm_hidden total)
    """

    def __init__(self, input_dim: int, lstm_hidden: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        assert lstm_hidden % 2 == 0, "lstm_hidden must be even for bidirectional LSTM"
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = lstm_hidden // 2,   # //2 because bidirectional
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if n_layers > 1 else 0.0,
        )
        self.norm       = nn.LayerNorm(lstm_hidden)
        self.output_dim = lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers * 2, B, lstm_hidden//2)
        fwd = h_n[-2]                          # last layer, forward direction
        bwd = h_n[-1]                          # last layer, backward direction
        out = torch.cat([fwd, bwd], dim=-1)    # (B, lstm_hidden)
        return self.norm(out)


# ── Transformer Encoder ────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with learnable [CLS] token.
    output_dim = transformer_dim
    """

    def __init__(self, input_dim: int, transformer_dim: int, n_heads: int,
                 n_layers: int, dropout: float = 0.1, window: int = 20):
        super().__init__()
        assert transformer_dim % n_heads == 0, (
            f"transformer_dim ({transformer_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.input_proj  = nn.Linear(input_dim, transformer_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc     = SinusoidalPositionalEncoding(
            transformer_dim, max_len=window + 1, dropout=dropout
        )
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model         = transformer_dim,
            nhead           = n_heads,
            dim_feedforward = transformer_dim * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LN: stabler training under RL gradients
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(transformer_dim)
        self.output_dim  = transformer_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        B   = x.size(0)
        x   = self.input_proj(x)                         # (B, T, transformer_dim)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, transformer_dim)
        x   = torch.cat([cls, x], dim=1)                 # (B, T+1, transformer_dim)
        x   = self.pos_enc(x)
        x   = self.transformer(x)                        # (B, T+1, transformer_dim)
        return self.norm(x[:, 0])                        # CLS token → (B, transformer_dim)


# ── Hybrid Encoder ─────────────────────────────────────────────────────────────

class HybridEncoder(nn.Module):
    """
    LSTM captures local sequential patterns →
    Transformer attends globally across the window.
    output_dim = lstm_hidden  (LSTM output feeds Transformer of same width)
    """

    def __init__(self, input_dim: int, lstm_hidden: int, transformer_dim: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1, window: int = 20):
        super().__init__()
        # LSTM stage: unidirectional, full lstm_hidden width
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = lstm_hidden,
            num_layers  = max(1, n_layers // 2),
            batch_first = True,
            bidirectional = False,
            dropout = dropout if n_layers > 2 else 0.0,
        )
        # Project lstm_hidden → transformer_dim if they differ
        self.proj = (
            nn.Linear(lstm_hidden, transformer_dim)
            if lstm_hidden != transformer_dim else nn.Identity()
        )
        assert transformer_dim % n_heads == 0, (
            f"transformer_dim ({transformer_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc     = SinusoidalPositionalEncoding(
            transformer_dim, max_len=window + 1, dropout=dropout
        )
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model         = transformer_dim,
            nhead           = n_heads,
            dim_feedforward = transformer_dim * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, n_layers // 2),
            enable_nested_tensor=False
        )
        self.norm        = nn.LayerNorm(transformer_dim)
        self.output_dim  = transformer_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        B         = x.size(0)
        lstm_out, _ = self.lstm(x)                        # (B, T, lstm_hidden)
        lstm_out    = self.proj(lstm_out)                 # (B, T, transformer_dim)
        cls         = self.cls_token.expand(B, -1, -1)   # (B, 1, transformer_dim)
        seq         = torch.cat([cls, lstm_out], dim=1)  # (B, T+1, transformer_dim)
        seq         = self.pos_enc(seq)
        seq         = self.transformer(seq)              # (B, T+1, transformer_dim)
        return self.norm(seq[:, 0])                      # CLS → (B, transformer_dim)


# ── PortfolioEncoder ───────────────────────────────────────────────────────────

class PortfolioEncoder(nn.Module):
    """
    Shared-weight temporal encoder applied independently to each ticker.

    Input  : (batch, n_tickers, window, n_features)
    Output : (batch, n_tickers, output_dim)

    Weight sharing across tickers forces the encoder to learn
    market-agnostic temporal patterns and reduces parameter count.

    Config keys (all under encoder:):
      window          — sequence length
      lstm_hidden     — LSTM hidden size
      transformer_dim — Transformer d_model
      n_heads         — attention heads
      n_layers        — number of layers
    """

    MODES = ("lstm", "transformer", "hybrid")

    def __init__(self, cfg: dict, input_dim: int, mode: str = "hybrid",
                 dropout: float = 0.1):
        super().__init__()
        enc   = cfg["encoder"]
        w     = enc["window"]
        lh    = enc["lstm_hidden"]
        td    = enc["transformer_dim"]
        nh    = enc["n_heads"]
        nl    = enc["n_layers"]

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

        if mode == "lstm":
            self.encoder = LSTMEncoder(input_dim, lh, nl, dropout)
        elif mode == "transformer":
            self.encoder = TransformerEncoder(input_dim, td, nh, nl, dropout, w)
        else:  # hybrid
            self.encoder = HybridEncoder(input_dim, lh, td, nh, nl, dropout, w)

        self.output_dim = self.encoder.output_dim
        log.info(
            "PortfolioEncoder | mode=%-12s | input_dim=%d | output_dim=%d | window=%d",
            mode, input_dim, self.output_dim, w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, F)  →  (B, N, output_dim)"""
        B, N, T, F = x.shape
        flat    = x.reshape(B * N, T, F)              # (B*N, T, F)
        encoded = self.encoder(flat)                  # (B*N, output_dim)
        return encoded.reshape(B, N, self.output_dim) # (B, N, output_dim)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict = None, input_dim: int = 23,
                  mode: str = "hybrid", dropout: float = 0.1) -> PortfolioEncoder:
    """
    Convenience factory.

    Parameters
    ----------
    cfg       : loaded config dict (loads from config.yaml if None)
    input_dim : number of features per timestep per ticker
                (22 price features + 1 sentiment = 23 by default)
    mode      : 'lstm' | 'transformer' | 'hybrid'
    dropout   : dropout rate
    """
    if cfg is None:
        cfg = load_config()
    return PortfolioEncoder(cfg, input_dim=input_dim, mode=mode, dropout=dropout)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    cfg       = load_config()
    n_tickers = cfg["data"]["n_stocks"]        # 30
    window    = cfg["encoder"]["window"]       # 20
    input_dim = 23                             # 22 price cols + 1 sentiment

    for mode in ["lstm", "transformer", "hybrid"]:
        enc   = build_encoder(cfg, input_dim=input_dim, mode=mode)
        dummy = torch.randn(4, n_tickers, window, input_dim)
        out   = enc(dummy)
        print(f"[{mode:>11s}]  {tuple(dummy.shape)}  →  {tuple(out.shape)}")
