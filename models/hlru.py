"""HLRU model, substituting Attention with HLRU"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from .components import RMSNorm, MLP, GLU, MLPReluSquared, hopscan, hopscan_opt
from .embeddings import precompute_freqs_cis, apply_rotary_emb_complex_like

@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    hidden_dim: int
    window_dim: int
    implementation: str = 'orig'
    mlp: str = 'mlp'
    rmsorm_eps: float = 1e-6
    tie_embeddings: bool = False


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared
}

@torch.compile(mode="max-autotune", dynamic=False)
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.dim = cfg.dim
        self.window_dim = cfg.window_dim
        self.hidden_dim = cfg.hidden_dim
        self.implementation = cfg.implementation

        self.proj_gates = nn.Linear(self.dim, self.hidden_dim*(self.window_dim+1), bias=True)
        self.proj_v = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proj_out = nn.Linear(self.hidden_dim*self.window_dim, self.dim, bias=False)

        self.register_buffer("A_temp", torch.diag(torch.ones(self.window_dim-1), 1))
    
    def forward(self, hidden_states):

        B, T, _ = hidden_states.size()
         
        v = self.proj_v(hidden_states) # B T N

        gates = self.proj_gates(hidden_states) # B T N*(H+1)
        gates = gates.reshape(B,T,self.hidden_dim,self.window_dim+1)

        if self.implementation == "orig":
            #b h t c
            hidden_x = torch.zeros(B, self.hidden_dim, self.window_dim).to(hidden_states.device)      # B H       
            y = []

            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T N H 
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD

            for i in range(T):
                
                # check order
                hidden_x = torch.einsum('bci, bcji -> bcj',hidden_x,A_qk[:,i,:,:,:])+ nl[:,i,:,:]
                y.append(hidden_x) 

            y=torch.stack(y, dim=1).reshape(B, T, self.hidden_dim*self.window_dim) # B T N*H
            y=self.proj_out(y)

        elif self.implementation == "hopscan_opt":

            #b t n*h
            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = F.pad(A_qk[:,:,:,-1:]*v[:,:,:].unsqueeze(-1),(0,self.window_dim-1)) # B T N H 
            A_qk = self.A_temp + F.pad(A_qk[:,:,:,:-1].unsqueeze(-1),(0,self.window_dim-1)) # B T C WD WD


            y=hopscan_opt(nl, A_qk) # B T N H

            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        elif self.implementation == "hopscan_opt_chunk":
            chunk_num = 4
            if T%chunk_num != 0:
                raise ValueError(f"Chunked implementation is not supported for length {T}")
            chunk_size=T//chunk_num
            #b t n*h
            A_qk = torch.softmax(gates,-1) # B T N H H+1
            nl = A_qk[:,:,:,:,-1]*v[:,:,:,:] # B T N H 
            A_qk = A_qk[:,:,:,:,:-1] # B T N H H

            chunked_hidden=[]
            passed_hidden=torch.zeros(nl.shape[0],1,nl.shape[2],nl.shape[3], dtype=nl.dtype, device=nl.device)# B 1 N H
            passed_A_qk=torch.zeros(A_qk.shape[0],1,A_qk.shape[2],A_qk.shape[3],A_qk.shape[4], dtype=A_qk.dtype, device=A_qk.device)
            
            for i in range(chunk_num):

                chunked_nl = torch.cat([passed_hidden, nl[:,i*chunk_size:(i+1)*chunk_size,:,:]],dim=1)
                chunked_A_qk = torch.cat([passed_A_qk, A_qk[:,i*chunk_size:(i+1)*chunk_size,:,:]],dim=1)
                out=hopscan_opt(chunked_nl, chunked_A_qk) # B T N H
                
                chunked_hidden.append(out[:,1:,:,:])
                passed_hidden = out[:,-1:,:,:]

            y =  torch.cat(chunked_hidden, dim=1)
            # reshape back 
            y=y.reshape(B,T,self.hidden_dim*self.window_dim) # B T N*H

            y=self.proj_out(y)

        else: 
            raise ValueError(f"Parallel implementation {self.implementation} not supported")

        return y
        

class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.mlp = MLP_CLASSES[cfg.mlp](dim=cfg.dim, hidden_dim=int(cfg.expand * cfg.dim))
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.layer_id = layer_id
    
    def forward(self, x):
        # x: (bsz, seqlen, dim)
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_layers = cfg.n_layers
        #head_dim = cfg.dim // cfg.n_heads; assert cfg.dim % cfg.n_heads == 0
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        #self.freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0:cfg.seq_len]
        
        # init all weights, scale residual branches
        self.apply(self._init_weights)
        self._scale_residual_branches()
        
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x) # (bsz, seqlen, dim)
        #self.freqs_cis = self.freqs_cis.to(x.device)
        for layer in self.layers:
            x = layer(x) # (bsz, seqlen, dim)
        return self.lm_head(self.out_norm(x)) # (bsz, seqlen, vocab_size)

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
            if n.endswith('w_out.weight'): # attn output layer
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

