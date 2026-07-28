import torch
import torch.nn as nn
from utils import append_history
from models.module.magcr_block import MAGCRBlock


class TeacherMAGCRLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.2,
        num_experts=4,
        moe_top_k=1,
        aux_loss_weight=0.01,
        use_moe=False,
    ):
        super().__init__()

        self.gene_inter = MAGCRBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=False,
        )
        self.patch_inter = MAGCRBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=False,
        )
        self.g_retrieval = MAGCRBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            aux_loss_weight=aux_loss_weight,
        )

        self.p_retrieval = MAGCRBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            aux_loss_weight=aux_loss_weight,
        )

    def forward(self, gene, patch, history_g, history_p):
        
        g1, _ = self.gene_inter(gene, history_g)
        p1, _ = self.patch_inter(patch, history_p)


        
        g2, aux_loss1 = self.g_retrieval(g1[:, [0], :], history_g, context=p1)
        p2, aux_loss2 = self.p_retrieval(p1[:, [0], :], history_p, context=g1)

        
        
        
        g1 = torch.cat(
            [g2, g1[:, 1:, :]],
            dim=1
        )

        p1 = torch.cat(
            [p2, p1[:, 1:, :]],
            dim=1
        )

        return g1, p1, aux_loss1 + aux_loss2

    def forward_patch(self, patch, history_p):
        p1, aux_loss = self.patch_inter(patch, history_p)
        return p1, aux_loss

class TeacherMAGCR(nn.Module):
    """
    Stack multimodal MAGCR layers for the teacher model.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        num_layers: Number of MAGCR layers.
        dropout: Dropout probability.
        num_experts: Number of experts in retrieval blocks.
        moe_top_k: Number of experts selected for each token.
        aux_loss_weight: Weight of the routing auxiliary loss.
        use_moe: Whether retrieval blocks use mixture-of-experts layers.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers=2,
        dropout=0.2,
        num_experts=4,
        moe_top_k=1,
        aux_loss_weight=0.01,
        use_moe=False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TeacherMAGCRLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                aux_loss_weight=aux_loss_weight,
                use_moe=use_moe,
            )
            for _ in range(num_layers)
        ])

    def forward(self, gene, patch, return_history=False):
        """
        Fuse gene and patch features across teacher layers.

        Args:
            gene: Gene token features.
            patch: Patch token features.
            return_history: Whether to return both modality histories.

        Returns:
            Gene features, patch features, and auxiliary loss, optionally with histories.
        """
        history_g = {
            "center": [],
            "context": [],
        }
        history_p = {
            "center": [],
            "context": [],
        }

        append_history(gene.detach(), history_g)
        append_history(patch.detach(), history_p)

        total_aux_loss = gene.new_tensor(0.0)

        gene_out = gene
        patch_out = patch

        for layer in self.layers:
            gene_out, patch_out, aux_loss = layer(
                gene_out, patch_out,
                history_g, history_p,
            )
            total_aux_loss = total_aux_loss + aux_loss

        if return_history:
            return (
                gene_out, patch_out, total_aux_loss,
                history_g, history_p
            )

        return gene_out, patch_out, total_aux_loss
