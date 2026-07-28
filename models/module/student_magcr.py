import torch.nn as nn
from models.module.magcr_block import MAGCRBlock
from utils import append_history


class StudentMAGCRLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.2,
        num_experts=4,
        moe_top_k=2,
        aux_loss_weight=0.01,
    ):
        super().__init__()

        self.patch_inter = MAGCRBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=True,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(self, patch, history_p):
        return self.patch_inter(patch, history_p)


class StudentMAGCR(nn.Module):
    """
    Stack patch-only MAGCR layers for the student model.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        num_layers: Number of MAGCR layers.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers=2,
        dropout=0.2,
        num_experts=4,
        moe_top_k=2,
        aux_loss_weight=0.01,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            StudentMAGCRLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                aux_loss_weight=aux_loss_weight,
            )
            for _ in range(num_layers)
        ])

    def forward(self, patch, return_history=False):
        """
        Fuse student patch features across layers.

        Args:
            patch: Patch token features.
            return_history: Whether to return the stored layer history.

        Returns:
            Patch features and auxiliary loss, optionally followed by history.
        """
        history_p = {
            "center": [],
            "context": [],
        }

        append_history(patch.detach(), history_p)

        total_aux_loss = patch.new_tensor(0.0)
        patch_out = patch

        for layer in self.layers:
            patch_out, aux_loss = layer(patch_out, history_p)
            total_aux_loss = total_aux_loss + aux_loss

        if return_history:
            return patch_out, total_aux_loss, history_p

        return patch_out, total_aux_loss
