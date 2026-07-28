import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import (
    AUROC,
    AveragePrecision,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from adan import Adan
from models.module.student_magcr import StudentMAGCR
from utils import feature_dropout, gene_feature_dropout


class TransPIDStudent(pl.LightningModule):
    """
    Patch-only TransPID student model.

    Args:
        project_model: Patch feature projector.
        cls_cfg: Student model and training configuration.
        distiller: Transcriptome-privileged distillation module.
        num_slides: Number of source slides.
    """
    def __init__(self, project_model, cls_cfg, distiller, num_slides=7):
        super().__init__()
        self.save_hyperparameters(ignore=["project_model", "distiller"])
        self.project_model = project_model
        self.cls_cfg = cls_cfg
        self.distiller = distiller

        model_cfg = cls_cfg.model
        training_cfg = cls_cfg.training
        align_dim = int(project_model.align_dim) * 2
        cross_fusion_cfg = model_cfg.cross_fusion

        self.num_train_slides = int(num_slides)
        self.cross_fusion = StudentMAGCR(
            embed_dim=align_dim,
            num_heads=int(model_cfg.num_heads),
            num_layers=int(model_cfg.num_layers),
            dropout=float(training_cfg.dropout),
            num_experts=int(cross_fusion_cfg.num_experts),
            moe_top_k=int(cross_fusion_cfg.moe_top_k),
            aux_loss_weight=float(cross_fusion_cfg.aux_loss_weight),
        )

        feature_dim = align_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, align_dim // 2),
            nn.GELU(),
            nn.Dropout(float(training_cfg.cls_dropout)),
            nn.Linear(align_dim // 2, 1),
        )
        self.slide_head = nn.Sequential(nn.Linear(feature_dim, self.num_train_slides))

        self.train_acc = BinaryAccuracy()
        self.train_auc = AUROC(task="binary")
        self.train_ap = AveragePrecision(task="binary")
        self.train_f1 = BinaryF1Score()
        self.train_sensitivity = BinaryRecall()
        self.train_specificity = BinarySpecificity()
        self.test_acc = BinaryAccuracy()
        self.test_auc = AUROC(task="binary")
        self.test_ap = AveragePrecision(task="binary")
        self.test_f1 = BinaryF1Score()
        self.test_sensitivity = BinaryRecall()
        self.test_specificity = BinarySpecificity()

        self.label_score = {}
        self.test_outputs = []
        self.result = {}

    @property
    def use_logit_distillation(self):
        return self.distiller is not None and self.distiller.use_logits

    def clear_distiller(self):
        self.distiller = None

    def forward(self, batch):
        """
        Predict cancer scores from patch features.

        Args:
            batch: Batch containing patch features.

        Returns:
            Classification logits, pooled features, and routing auxiliary loss.
        """
        aligned_patch = self.project_model.encode_patch(batch["patch"])
        p, aux_loss = self.cross_fusion(aligned_patch)
        pooled = p[:, 0, :]
        return self.classifier(pooled), pooled, aux_loss

    def training_step(self, batch, batch_idx):
        """
        Compute the student training objective for one batch.

        Args:
            batch: Training batch.
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar student training loss.
        """
        teacher_batch = batch
        batch = self._augment(batch)
        logits, pooled, aux_loss = self(batch)
        label = batch["label"].float()
        cls_loss = F.binary_cross_entropy_with_logits(logits, label)
        slide_terms = self._slide_terms(pooled, batch["slide"])
        distill_terms = self.distiller(teacher_batch, logits)
        distill_weight, cls_weight = self._training_loss_weights()
        loss = (
            cls_weight * cls_loss
            + aux_loss
            + slide_terms["probe_loss"]
            + 0.2 * slide_terms["confuse_loss"]
            + distill_weight * distill_terms["loss"]
        )

        self._log_train_common(loss, cls_loss, aux_loss, slide_terms, logits, label)
        if self.use_logit_distillation:
            self.log("distill_loss", distill_terms["logit_loss"], prog_bar=True)
        return loss

    def _training_loss_weights(self):
        return 0.6, 0.4

    def _slide_terms(self, pooled, slide):
        probe_logits = self.slide_head(pooled.detach())
        probe_loss = F.cross_entropy(probe_logits, slide)

        for param in self.slide_head.parameters():
            param.requires_grad_(False)
        confuse_logits = self.slide_head(pooled)
        log_probs = F.log_softmax(confuse_logits, dim=-1)
        for param in self.slide_head.parameters():
            param.requires_grad_(True)

        uniform_target = torch.full(
            size=log_probs.shape,
            fill_value=1.0 / self.num_train_slides,
            device=log_probs.device,
        )
        return {
            "probe_loss": probe_loss,
            "probe_acc": (probe_logits.argmax(dim=-1) == slide).float().mean(),
            "confuse_loss": F.kl_div(
                log_probs,
                uniform_target,
                reduction="batchmean",
                log_target=False,
            ),
        }

    def _log_train_common(self, loss, cls_loss, aux_loss, slide_terms, logits, label):
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.optimizers:
            self.log("lr", trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("cls_loss", cls_loss, prog_bar=True)
        self.log("aux_loss", aux_loss, prog_bar=True)
        self.log("slide_probe_acc", slide_terms["probe_acc"], prog_bar=True)
        self.log("slide_probe_loss", slide_terms["probe_loss"], prog_bar=True)
        self.log("slide_confuse_loss", slide_terms["confuse_loss"], prog_bar=True)
        self._log_stage_metrics("train", logits, label, prefix="t")

    def test_step(self, batch, batch_idx):
        logits, _, _ = self(batch)
        label = batch["label"].float()
        loss = F.binary_cross_entropy_with_logits(logits, label)
        self.test_outputs.append({
            "loss": loss.detach(),
            "logits": logits.detach(),
            "label": label.detach(),
        })

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        loss = torch.stack([item["loss"] for item in self.test_outputs]).mean()
        logits = torch.cat([item["logits"] for item in self.test_outputs], dim=0)
        label = torch.cat([item["label"] for item in self.test_outputs], dim=0)
        label_int = label.int()

        self.log("test_loss", loss, prog_bar=True)
        metric_values = {}
        for metric_name, metric in self._stage_metrics("test").items():
            metric_values[metric_name] = metric(logits, label_int)
            self.log(f"tt_{metric_name}", metric_values[metric_name], prog_bar=True)

        self.label_score = {"label": label_int, "score": torch.sigmoid(logits)}
        self.result = {
            metric_name: float(metric_value)
            for metric_name, metric_value in metric_values.items()
        }
        self.test_outputs.clear()
        self._reset_stage_metrics("test")

    def on_train_epoch_end(self):
        self._reset_stage_metrics("train")

    def _stage_metrics(self, stage):
        return {
            "acc": getattr(self, f"{stage}_acc"),
            "auc": getattr(self, f"{stage}_auc"),
            "ap": getattr(self, f"{stage}_ap"),
            "f1": getattr(self, f"{stage}_f1"),
            "sensitivity": getattr(self, f"{stage}_sensitivity"),
            "specificity": getattr(self, f"{stage}_specificity"),
        }

    def _log_stage_metrics(self, stage, logits, label, prefix):
        label_int = label.int()
        for metric_name, metric in self._stage_metrics(stage).items():
            self.log(f"{prefix}_{metric_name}", metric(logits, label_int), prog_bar=True)

    def _reset_stage_metrics(self, stage):
        for metric in self._stage_metrics(stage).values():
            metric.reset()

    def _augment(self, batch):
        return {
            **batch,
            "patch": feature_dropout(batch["patch"], self.cls_cfg.training.patch_mask),
            "gene": gene_feature_dropout(batch["gene"], self.cls_cfg.training.gene_mask),
        }

    def configure_optimizers(self):
        base_lr = self.cls_cfg.training.optimizer.lr
        weight_decay = self.cls_cfg.training.optimizer.weight_decay
        slide_head_param_ids = {id(param) for param in self.slide_head.parameters()}
        other_params = [
            param for param in self.parameters()
            if id(param) not in slide_head_param_ids and param.requires_grad
        ]
        slide_head_params = [
            param for param in self.slide_head.parameters()
            if param.requires_grad
        ]

        return {
            "optimizer": Adan(
                [
                    {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
                    {"params": slide_head_params, "lr": 2e-4, "weight_decay": 0},
                ],
                lr=base_lr,
                weight_decay=weight_decay,
                max_grad_norm=1,
            )
        }
