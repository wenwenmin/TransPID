import math
import torch
import torch.nn as nn
from models.gmha import GatedMultiHeadAttention
from utils import append_history
from models.module.moe import FFN, MoE


class MAGCRBlock(nn.Module):
    """
    Fuse current tokens with representations stored by earlier layers.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        use_moe: Whether to use a mixture-of-experts feed-forward layer.
        num_experts: Number of experts.
        moe_top_k: Number of experts selected for each token.
        aux_loss_weight: Weight of the routing auxiliary loss.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.2,
        use_moe=False,
        num_experts=4,
        moe_top_k=1,
        aux_loss_weight=0.01,
    ):
        super().__init__()

        self.gate_attn = GatedMultiHeadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )

        self.attn_res_norm_q = nn.LayerNorm(embed_dim)
        self.attn_res_norm_k = nn.LayerNorm(embed_dim)
        self.attn_proj = nn.Linear(embed_dim, embed_dim)

        self.use_moe = use_moe
        self.aux_loss_weight = aux_loss_weight

        hidden_dim = embed_dim // 8
        if use_moe:
            self.ffn = MoE(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
            )
        else:
            self.ffn = FFN(embed_dim, hidden_dim, dropout=dropout)

        self.ffn_res_norm_q = nn.LayerNorm(embed_dim)
        self.ffn_res_norm_k = nn.LayerNorm(embed_dim)
        self.ffn_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _history_enhance_bucket(self, history_list, query, proj, norm_q, norm_k):
        if history_list is None or len(history_list) == 0:
            return torch.zeros_like(query)

        
        # Match each token with the same position in prior layers.
        v = torch.stack(history_list, dim=0)
        k = norm_k(v)  
        q = norm_q(proj(query))  

        
        
        logits = torch.einsum("b t d, h b t d -> h b t", q, k)
        logits = logits / math.sqrt(q.shape[-1])
        attn = torch.softmax(logits, dim=0)

        
        mem = torch.einsum("h b t, h b t d -> b t d", attn, v)
        return mem

    def history_enhance(self, history, query, proj, norm_q, norm_k):
        """
        Retrieve matching center and context features from layer history.

        Args:
            history: Dictionary of center and context feature histories.
            query: Current token features.
            proj: Query projection layer.
            norm_q: Query normalization layer.
            norm_k: History normalization layer.

        Returns:
            Historical features aligned with the current tokens.
        """
        query_center = query[:, :1, :]  
        center_history = history.get("center", [])
        mem_center = self._history_enhance_bucket(
            center_history, query_center, proj, norm_q, norm_k
        )
        if query.shape[1] == 1:
            return mem_center
        query_context = query[:, 1:, :]  
        context_history = history.get("context", [])
        mem_context = self._history_enhance_bucket(
            context_history, query_context, proj, norm_q, norm_k
        )

        mem = torch.cat([mem_center, mem_context], dim=1)
        return mem

    def forward(self, x, history, context=None):
        """
        Apply attention, historical retrieval, and feed-forward fusion.

        Args:
            x: Current token features.
            history: Center and context feature histories.
            context: Optional features from the other modality.

        Returns:
            Enhanced token features and the routing auxiliary loss.
        """
        x_norm = self.attn_norm(x)
        if context is None:
            kv = x_norm
        else:
            kv = self.context_norm(context)

        
        attn_out, _ = self.gate_attn(x_norm, kv, kv)
        attn_out = self.dropout(attn_out)

        attn_mem = self.history_enhance(
            history=history,
            query=attn_out,
            proj=self.attn_proj,
            norm_q=self.attn_res_norm_q,
            norm_k=self.attn_res_norm_k,
        )
        append_history(attn_out.detach(), history)

        h0 = x + attn_out + attn_mem
        h0_norm = self.ffn_norm(h0)

        
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(h0_norm)
            aux_loss = self.aux_loss_weight * aux_loss
        else:
            ffn_out = self.ffn(h0_norm)
            aux_loss = x.new_tensor(0.0)

        ffn_out = self.dropout(ffn_out)

        ffn_mem = self.history_enhance(
            history=history,
            query=ffn_out,
            proj=self.ffn_proj,
            norm_q=self.ffn_res_norm_q,
            norm_k=self.ffn_res_norm_k,
        )
        append_history(ffn_out.detach(), history)

        h1 = h0 + ffn_out + ffn_mem

        return h1, aux_loss
