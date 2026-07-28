from torch import nn


class ModalityProjector(nn.Module):
    """
    Project gene and patch features into a shared space.

    Args:
        cls_cfg: Classification model configuration.
        patch_only: Whether to build only the patch encoder.
    """
    def __init__(self, cls_cfg, patch_only=False):
        super().__init__()
        self.patch_only = patch_only
        self.align_dim = int(cls_cfg.model.align_dim)
        gene_dim = int(cls_cfg.model.gene_dim)
        patch_dim = int(cls_cfg.model.patch_dim)
        dropout = float(cls_cfg.training.dropout)

        self.gene_encoder = None
        if not self.patch_only:
            self.gene_encoder = nn.Sequential(
                nn.Linear(gene_dim, self.align_dim * 2),
                nn.LayerNorm(self.align_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(self.align_dim * 2, self.align_dim),
                nn.LayerNorm(self.align_dim),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(self.align_dim, self.align_dim),
            )

            self.cls_encoder = nn.Sequential(
                nn.Linear(patch_dim, self.align_dim),
                nn.LayerNorm(self.align_dim),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(self.align_dim, self.align_dim),
            )
        else:
            self.cls_encoder = nn.Sequential(
                nn.Linear(patch_dim, self.align_dim *2),
                nn.LayerNorm(self.align_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(self.align_dim* 2, self.align_dim* 2),
            )

    def encode_gene(self, gene):
        batch_size, nb_num, gene_dim = gene.shape
        gene_emb = self.gene_encoder(gene.reshape(-1, gene_dim))
        return gene_emb.reshape(batch_size, nb_num, -1)

    def encode_patch(self, patch_embedding):
        batch_size, nb_num, patch_dim = patch_embedding.shape
        patch_emb = self.cls_encoder(patch_embedding.reshape(-1, patch_dim))
        return patch_emb.reshape(batch_size, nb_num, -1)

    def forward(self, batch):
        """
        Encode the modalities in one batch.

        Args:
            batch: Dictionary containing gene and patch tensors.

        Returns:
            Patch embeddings in patch-only mode, otherwise gene and patch embeddings.
        """
        if self.patch_only:
            return self.encode_patch(batch["patch"])
        else:
            return self.encode_gene(batch["gene"]), self.encode_patch(batch["patch"])
