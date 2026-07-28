import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.2
    ):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        return self.ffn(x)


class MoE(nn.Module):
    """
    Top-k mixture-of-experts feed-forward module.

    Args:
        embed_dim: Input and output feature dimension.
        hidden_dim: Hidden dimension of each expert.
        num_experts: Number of experts.
        top_k: Number of experts selected for each token.
        dropout: Expert dropout probability.
    """
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_experts=4,
        top_k=1,
        dropout=0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            FFN(embed_dim, hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_experts),
        )

    def _compute_aux_loss(self, expert_logits, expert_indices):
        expert_logits = expert_logits.float()
        router_prob_per_expert = F.softmax(expert_logits, dim=-1).mean(dim=0)

        T = expert_indices.shape[0]

        counts = torch.bincount(
            expert_indices.reshape(-1),
            minlength=self.num_experts
        ).float()

        dispatch_frac_per_expert = counts / (T * self.top_k)

        load_balance_loss = self.num_experts * (
                router_prob_per_expert * dispatch_frac_per_expert
        ).sum()

        router_z_loss = torch.logsumexp(expert_logits, dim=-1).square().mean()

        aux_loss = load_balance_loss + 0.001 * router_z_loss

        return aux_loss

    def forward(self, x):
        """
        Route tokens to experts and combine their outputs.

        Args:
            x: Token tensor with shape ``[batch, tokens, embed_dim]``.

        Returns:
            Fused token features and the load-balancing auxiliary loss.
        """
        B, N, D = x.shape
        all_token = x.reshape(-1, D)  

        
        # Route each token to its top-k experts.
        expert_logits = self.router(all_token)  
        expert_scores = F.softmax(expert_logits, dim=-1)  

        
        expert_weights, expert_indices = expert_scores.topk(
            self.top_k, dim=-1
        )  

        
        expert_weights = expert_weights / expert_weights.sum(
            dim=-1, keepdim=True
        )  

        
        expert_out = torch.zeros_like(all_token)  
        dtype = expert_out.dtype
        for e_idx in range(self.num_experts):
            
            token_idx, top_idx = torch.where(expert_indices == e_idx)

            if token_idx.numel() == 0:  
                continue
            e_input = all_token[token_idx]  

            
            e_output = self.experts[e_idx](e_input).to(dtype)  
            e_weight = expert_weights[token_idx, top_idx].unsqueeze(-1).to(dtype)  

            
            expert_out.index_add_(0, token_idx, e_weight * e_output)

        expert_out = expert_out.reshape(B, N, D)

        aux_loss = self._compute_aux_loss(expert_logits, expert_indices)

        return expert_out, aux_loss
