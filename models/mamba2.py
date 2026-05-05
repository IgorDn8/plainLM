"""Mamba2 model, substituting Attention with Mamba2 from mamba_ssm"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional

from .components import RMSNorm, MLP, GLU, MLPReluSquared

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    # Mamba2 specific (with defaults)
    d_state: int = 64
    d_conv: int = 4
    mamba_expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}


class Mamba2Layer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        if Mamba2 is None:
            raise ImportError("mamba_ssm not installed. Please install it to use Mamba2 model.")
        
        self.mamba = Mamba2(
            d_model=cfg.dim,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            layer_idx=layer_idx,
        )
    
    def forward(self, x):
        # x: (bsz, seqlen, dim)
        return self.mamba(x)


class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.mamba = Mamba2Layer(cfg, layer_idx=layer_id)
        self.mamba_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.layer_id = layer_id
    
    def forward(self, x):
        # x: (bsz, seqlen, dim)
        x = x + self.mamba(self.mamba_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_layers = cfg.n_layers
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        # init all weights, scale residual branches
        self.apply(self._init_weights)
        self._scale_residual_branches()
        
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x) # (bsz, seqlen, dim)
        
        for layer in self.layers:
            x = layer(x)
        
        logits = self.lm_head(self.out_norm(x)) # (bsz, seqlen, vocab_size)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight'): # mlp/glu output layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
            # Mamba2's output projection is usually named 'out_proj' or handled inside Mamba2
            # but we follow the transformer's style of scaling if we can identify it.
            # In mamba_ssm.Mamba2, the output projection is self.out_proj
            if n.endswith('out_proj.weight'): 
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if not self.lm_head.weight is self.embed_tokens.weight:  # if no weight tying
                n_params -= self.lm_head.weight.numel()
        return n_params
