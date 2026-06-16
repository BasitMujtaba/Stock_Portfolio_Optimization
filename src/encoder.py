
"""
================================================================================
 File   : src/encoder.py
 Project: Stock Portfolio Optimization — PSX DRL Temporal Encoding
 Purpose: Temporal encoder (LSTM / Transformer) that converts a window of
          (price_features + sentiment) into a fixed-size latent vector per
          ticker per timestep.

 Input  : Tensor  (batch, n_tickers, window_len, n_features)
 Output : Tensor  (batch, n_tickers, hidden_dim)

 Modes  :
   lstm        — bidirectional LSTM, last hidden state
   transformer — positional-encoded Transformer encoder, CLS token output
   hybrid      — LSTM then Transformer (best of both)
================================================================================
"""

import math, logging
import yaml, os
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Sub-encoders ──────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bi-directional LSTM.
    Output: last hidden state of forward pass concatenated with last hidden
            state of backward pass → (batch, 2 * hidden_size) then projected
            to hidden_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,     # //2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden//2)
        fwd = h_n[-2]   # last layer, forward
        bwd = h_n[-1]   # last layer, backward
        out = torch.cat([fwd, bwd], dim=-1)   # (batch, hidden_dim)
        return self.norm(out)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with a learnable [CLS] token.
    Output: CLS token representation → (batch, hidden_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, dropout: float, max_len: int):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc     = SinusoidalPositionalEncoding(hidden_dim, max_len + 1, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch = x.size(0)
        x     = self.input_proj(x)                          # (B, T, hidden_dim)
        cls   = self.cls_token.expand(batch, -1, -1)        # (B, 1, hidden_dim)
        x     = torch.cat([cls, x], dim=1)                  # (B, T+1, hidden_dim)
        x     = self.pos_enc(x)
        x     = self.transformer(x)                         # (B, T+1, hidden_dim)
        return self.norm(x[:, 0])                           # CLS token → (B, hidden_dim)


class HybridEncoder(nn.Module):
    """
    LSTM captures local sequential patterns →
    Transformer captures global dependencies across the window.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 lstm_layers: int, tf_layers: int, ff_dim: int,
                 dropout: float, max_len: int):
        super().__init__()
        self.lstm        = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc     = SinusoidalPositionalEncoding(hidden_dim, max_len + 1, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers,
                                                  enable_nested_tensor=False)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        lstm_out, _  = self.lstm(x)                         # (B, T, hidden_dim)
        batch        = lstm_out.size(0)
        cls          = self.cls_token.expand(batch, -1, -1) # (B, 1, hidden_dim)
        seq          = torch.cat([cls, lstm_out], dim=1)    # (B, T+1, hidden_dim)
        seq          = self.pos_enc(seq)
        seq          = self.transformer(seq)                 # (B, T+1, hidden_dim)
        return self.norm(seq[:, 0])                         # CLS → (B, hidden_dim)


# ── Main PortfolioEncoder ─────────────────────────────────────────────────────

class PortfolioEncoder(nn.Module):
    """
    Wraps per-ticker encoding.

    Input  : (batch, n_tickers, window_len, n_features)
    Output : (batch, n_tickers, hidden_dim)

    Internally reshapes to (batch * n_tickers, window_len, n_features),
    runs the chosen encoder, then reshapes back.
    This is efficient and shares weights across tickers (weight tying).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        enc_cfg    = cfg["encoder"]
        mode       = enc_cfg["mode"]               # lstm | transformer | hybrid
        input_dim  = enc_cfg["input_dim"]
        hidden_dim = enc_cfg["hidden_dim"]
        num_heads  = enc_cfg.get("num_heads", 4)
        num_layers = enc_cfg.get("num_layers", 2)
        ff_dim     = enc_cfg.get("ff_dim", hidden_dim * 4)
        dropout    = enc_cfg.get("dropout", 0.1)
        max_len    = enc_cfg.get("window_size", 60)

        self.mode       = mode
        self.hidden_dim = hidden_dim

        if mode == "lstm":
            self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        elif mode == "transformer":
            self.encoder = TransformerEncoder(
                input_dim, hidden_dim, num_heads, num_layers,
                ff_dim, dropout, max_len
            )
        elif mode == "hybrid":
            lstm_layers = enc_cfg.get("lstm_layers", 1)
            tf_layers   = enc_cfg.get("tf_layers", num_layers)
            self.encoder = HybridEncoder(
                input_dim, hidden_dim, num_heads, lstm_layers,
                tf_layers, ff_dim, dropout, max_len
            )
        else:
            raise ValueError(f"Unknown encoder mode: {mode!r}. "
                             "Choose 'lstm', 'transformer', or 'hybrid'.")

        log.info("PortfolioEncoder | mode=%s | input_dim=%d | hidden_dim=%d",
                 mode, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_tickers, window_len, n_features)
        returns: (batch, n_tickers, hidden_dim)
        """
        B, N, T, F = x.shape
        x_flat  = x.reshape(B * N, T, F)             # (B*N, T, F)
        enc_out = self.encoder(x_flat)                # (B*N, hidden_dim)
        return enc_out.reshape(B, N, self.hidden_dim) # (B, N, hidden_dim)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_encoder(cfg: dict = None) -> PortfolioEncoder:
    if cfg is None:
        cfg = load_config()
    return PortfolioEncoder(cfg)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    cfg = load_config()

    for mode in ["lstm", "transformer", "hybrid"]:
        cfg["encoder"]["mode"] = mode
        enc = build_encoder(cfg)
        dummy = torch.randn(
            4,
            cfg["encoder"].get("n_tickers", 30),
            cfg["encoder"].get("window_size", 60),
            cfg["encoder"]["input_dim"],
        )
        out = enc(dummy)
        print(f"[{mode:>11s}]  input {tuple(dummy.shape)}  →  output {tuple(out.shape)}")
