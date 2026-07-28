import math

import torch
import torch.nn as nn

class GatedMultiHeadAttention(nn.Module):
    """
    Gated multi-head attention module.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super(GatedMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.head_dim = int(embed_dim // num_heads)

        self.q_proj = nn.Linear(embed_dim, embed_dim * 2)  
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, q, k, v):
        """
        Apply gated attention to query, key, and value tokens.

        Args:
            q: Query tensor with shape ``[batch, query_tokens, embed_dim]``.
            k: Key tensor with shape ``[batch, key_tokens, embed_dim]``.
            v: Value tensor with shape ``[batch, key_tokens, embed_dim]``.

        Returns:
            Attention output and per-head attention weights.
        """
        
        query = self.q_proj(q)  
        key = self.k_proj(k)  
        value = self.v_proj(v)  

        
        batch_size, seq_len_k, embed_dim = key.shape
        seq_len_q = query.shape[1]
        query = query.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim * 2)
        query, gate_score = torch.split(query,[self.head_dim, self.head_dim], dim=-1)  

        
        query = query.transpose(1, 2)  
        key = key.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)  
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  
        attn_weights = self.dropout(attn_weights)

        
        attn_output = torch.matmul(attn_weights, value)  

        
        attn_output = attn_output.transpose(1, 2).contiguous()  
        attn_output = attn_output * torch.sigmoid(gate_score)  

        
        attn_output = attn_output.reshape(batch_size, seq_len_q, -1)  
        attn_output = self.o_proj(attn_output)  

        return attn_output, attn_weights







