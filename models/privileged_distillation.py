import torch
from torch import nn
import torch.nn.functional as F


class TranscriptomePrivilegedDistiller(nn.Module):
    """
    Transfer privileged logits from a frozen teacher to a student.

    Args:
        teacher_model: Trained full-modality teacher.
        distill_cfg: Distillation configuration.
    """
    def __init__(self, teacher_model, distill_cfg):
        super().__init__()
        self.teacher_model = teacher_model
        self.temperature = float(distill_cfg.temperature)
        self.logit_weight = float(distill_cfg.loss_weight)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    @property
    def use_logits(self):
        return self.logit_weight > 0

    def forward(self, batch, student_logits):
        """
        Compute the logit distillation loss.

        Args:
            batch: Full-modality teacher batch.
            student_logits: Student classification logits.

        Returns:
            Dictionary containing total and logit losses.
        """
        zero = student_logits.new_tensor(0.0)
        terms = {
            "loss": zero,
            "logit_loss": zero,
        }
        if not self.use_logits:
            return terms

        teacher_logits = self.teacher_outputs(
            batch,
            student_logits.device,
        )
        terms["logit_loss"] = self._logit_loss(student_logits, teacher_logits)
        terms["loss"] = self.logit_weight * terms["logit_loss"]
        return terms

    def teacher_outputs(self, batch, device):
        self._move_teacher(device)
        teacher_batch = batch
        if "teacher_patch" in batch:
            teacher_batch = dict(batch)
            teacher_batch["patch"] = batch["teacher_patch"]

        self.teacher_model.eval()
        with torch.no_grad():
            logits, _, _, _, _, _, _, _ = self.teacher_model(teacher_batch)
        return logits.detach()

    def _move_teacher(self, device):
        for param in self.teacher_model.parameters():
            if param.device != device:
                self.teacher_model.to(device)
            return

    def _logit_loss(self, student_logits, teacher_logits):
        teacher_prob = torch.sigmoid(teacher_logits / self.temperature)
        return (
            F.binary_cross_entropy_with_logits(
                student_logits / self.temperature,
                teacher_prob,
            )
            * self.temperature
            * self.temperature
        )
