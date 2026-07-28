"""Inter-modal alignment and intra-modal supervised contrastive objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils import compute_intra_ctt_metrics


class I2SCLoss(torch.nn.Module):  
    """
    Inter-modal and intra-modal supervised contrastive objective.

    Args:
        tmp: Initial contrastive temperature.
        learn_tmp: Whether the temperature is learnable.
        inter_weight: Weight of the inter-modal loss.
        intra_weight: Weight of the intra-modal loss.
    """
    def __init__(
        self,
        tmp: float = 0.07,
        learn_tmp: bool = False,
        inter_weight: float = 0.5,
        intra_weight: float = 0.5,
    ):
        super().__init__()

        self.loss_weight = {
            "inter": inter_weight,
            "intra": intra_weight,
        }

        if learn_tmp:
            self._log_temperature = nn.Parameter(torch.tensor(float(tmp)).log())
        else:
            self.register_buffer("_log_temperature", torch.tensor(float(tmp)).log())

    @property
    def temperature(self):
        return self._log_temperature.exp()

    def _sup_ctt(
        self,
        z: torch.Tensor,      
        label: torch.Tensor,    
    ) -> torch.Tensor:

        num = z.shape[0]
        device = z.device
        label = label.reshape(-1)

        
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        
        # Exclude each sample from its own positive set.
        self_mask = torch.eye(num, device=device, dtype=torch.bool)

        
        pos_mask = label.unsqueeze(0).eq(label.unsqueeze(1))
        pos_mask = pos_mask & (~self_mask)


        exp_logits = torch.exp(logits) * (~self_mask).float()

        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        pos_count = pos_mask.sum(dim=1)  

        valid_anchor = pos_count > 0
        if valid_anchor.sum() == 0:
            return z.sum() * 0.0

        loss_per_anchor = -(
            log_prob * pos_mask.float()
        ).sum(dim=1) / pos_count.clamp_min(1)

        loss = loss_per_anchor[valid_anchor].mean()

        return loss

    def _intra_sup_ctt_loss(
        self,
        gene_norm: torch.Tensor,   
        patch_norm: torch.Tensor,  
        label: torch.Tensor,       
    ) -> torch.Tensor:
        gene_ctt_loss = self._sup_ctt(gene_norm, label)
        patch_ctt_loss = self._sup_ctt(patch_norm, label)

        return 0.5 * (gene_ctt_loss + patch_ctt_loss)

    def _inter_sample_contrastive_loss(
        self,
        gene_norm: torch.Tensor,   
        patch_norm: torch.Tensor,  
    ) -> torch.Tensor:
        logits = torch.matmul(gene_norm, patch_norm.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = torch.arange(gene_norm.shape[0], device=logits.device)

        g2p = F.cross_entropy(logits, labels)
        p2g = F.cross_entropy(logits.T, labels)

        return 0.5 * (g2p + p2g)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def forward(
        self,
        gene_emb: torch.Tensor,   
        patch_emb: torch.Tensor,  
        label: torch.Tensor,      
    ):
        """
        Compute contrastive losses and embedding metrics.

        Args:
            gene_emb: Gene embeddings with shape ``[batch, embed_dim]``.
            patch_emb: Patch embeddings with shape ``[batch, embed_dim]``.
            label: Binary class labels.

        Returns:
            Total loss, inter-modal loss, intra-modal loss, and modality metrics.
        """
        gene_norm = F.normalize(gene_emb, p=2, dim=-1)
        patch_norm = F.normalize(patch_emb, p=2, dim=-1)

        
        inter_ctt_loss = self._inter_sample_contrastive_loss(
            gene_norm,
            patch_norm
        )

        
        intra_ctt_loss = self._intra_sup_ctt_loss(
            gene_norm,
            patch_norm,
            label
        )

        gene_metrics = compute_intra_ctt_metrics(
            gene_emb,
            label,
            temperature=self.temperature.item()
        )

        patch_metrics = compute_intra_ctt_metrics(
            patch_emb,
            label,
            temperature=self.temperature.item()
        )

        loss = (
            self.loss_weight["inter"] * inter_ctt_loss
            + self.loss_weight["intra"] * intra_ctt_loss
        )

        return loss, inter_ctt_loss, intra_ctt_loss, gene_metrics, patch_metrics
