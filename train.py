from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from dataset.dataset import MineDataset, load_data
from models.modality_projector import ModalityProjector
from models.privileged_distillation import TranscriptomePrivilegedDistiller
from models.transpid_student import TransPIDStudent
from models.transpid_teacher import TransPIDTeacher
from ema import EMACallback


class TrainingRNGCompatibility(pl.Callback):

    """
    Preserve the historical random-number sequence during training.

    The callback consumes one random value after every ten epochs.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 10 == 0:
            torch.empty((), dtype=torch.int64).random_()


class TrainingManager:
    """
    Coordinate teacher training, student training, and checkpoint output.

    Args:
        train_names: Source slice names.
        cfg: Resolved experiment configuration.
        callbacks: Additional PyTorch Lightning callbacks.
        seed: Random seed.
        run_name: Name used for log and checkpoint subdirectories.
    """
    def __init__(self, train_names, cfg, callbacks=None, seed=None, run_name=None):
        self.train_names = train_names
        self.cfg = cfg
        self.dataset_cfg = self.cfg.experiment.dataset
        self.cls_cfg = self.cfg.experiment.cls
        self.distill_cfg = self.cls_cfg.distillation
        self.seed = seed
        self.callbacks = list(callbacks or [])
        self.num_train_slides = len(self.train_names)
        self.teacher_model = None
        self.student_model = None
        self.final_model = None
        self.run_name = str(run_name) if run_name is not None else "train"

        self.ema_cb = next(
            (cb for cb in self.callbacks if isinstance(cb, EMACallback)),
            None,
        )

        self.train_data = load_data(self.train_names, cfg)

    def _commit_ema(self, model):
        if self.ema_cb is None:
            return False
        return self.ema_cb.copy_ema_to_model(model)

    def _use_distillation(self):
        return self.distill_cfg.enabled

    def _copy_teacher_weights_to_student(self, cls_cfg=None):
        distill_cfg = self.distill_cfg if cls_cfg is None else cls_cfg.distillation
        return distill_cfg.copy_teacher_weights

    def _build_project_model(self, cls_cfg, patch_only=False):
        return ModalityProjector(
            cls_cfg,
            patch_only=patch_only,
        )

    def _build_teacher_model(self, cls_cfg):
        return TransPIDTeacher(
            self._build_project_model(cls_cfg),
            cls_cfg,
            num_slides=self.num_train_slides,
        )

    def _build_student_model(self, cls_cfg, distiller):
        return TransPIDStudent(
            self._build_project_model(cls_cfg, patch_only=True),
            cls_cfg,
            distiller=distiller,
            num_slides=self.num_train_slides,
        )

    def _student_checkpoint_root(self):
        checkpoint_dir = OmegaConf.select(
            self.distill_cfg,
            "student_checkpoint_dir",
            default="",
        )
        if checkpoint_dir in (None, ""):
            return None
        return Path(str(checkpoint_dir))

    def _save_student_checkpoint(self, student_model):
        root = self._student_checkpoint_root()
        if root is None or self.seed is None:
            return

        checkpoint_path = root / str(self.seed) / self.run_name / "checkpoint.ckpt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._cpu_state_dict(student_model)},
            checkpoint_path,
        )
        print(f"Student EMA model saved to: {checkpoint_path}")

    def _teacher_checkpoint_root(self):
        checkpoint_dir = OmegaConf.select(
            self.distill_cfg,
            "teacher_checkpoint_dir",
            default="",
        )
        if checkpoint_dir in (None, ""):
            return None
        return Path(str(checkpoint_dir))

    def _teacher_checkpoint_path(self):
        root = self._teacher_checkpoint_root()
        if root is None or self.seed is None:
            return None

        teacher_name = self.run_name
        ckpt_path = root / str(self.seed) / teacher_name / "checkpoint.ckpt"

        if ckpt_path.is_file():
            return ckpt_path

        return None

    def _load_checkpoint_state_dict(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        return checkpoint

    def _load_teacher_from_checkpoint(self, teacher_cls_cfg):
        checkpoint_path = self._teacher_checkpoint_path()
        if checkpoint_path is None:
            return None

        print(f"Loading teacher model from checkpoint: {checkpoint_path}")
        teacher_model = self._build_teacher_model(teacher_cls_cfg)
        state_dict = self._load_checkpoint_state_dict(checkpoint_path)
        teacher_model.load_state_dict(state_dict, strict=True)
        self._freeze_model(teacher_model)

        return teacher_model

    def _save_teacher_checkpoint(self, teacher_model):
        root = self._teacher_checkpoint_root()
        if root is None or self.seed is None:
            return

        checkpoint_path = root / str(self.seed) / self.run_name / "checkpoint.ckpt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._cpu_state_dict(teacher_model)}, checkpoint_path)
        print(f"Teacher model saved to: {checkpoint_path}")

    def _phase_cls_cfg(self, phase):
        phase_cfg = OmegaConf.select(self.cls_cfg, phase)
        if phase_cfg is None:
            return self.cls_cfg
        return OmegaConf.merge(self.cls_cfg, phase_cfg)

    @staticmethod
    def _freeze_model(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _cpu_state_dict(model):
        return {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }

    def _load_compatible_teacher_weights(self, student_model, teacher_model):
        teacher_state = self._cpu_state_dict(teacher_model)
        student_state = student_model.state_dict()

        def should_copy(name):
            return (
                name.startswith("project_model.cls_encoder.")
                or (
                    name.startswith("cross_fusion.")
                    and ".patch_inter." in name
                )
            )

        compatible_state = {
            name: tensor
            for name, tensor in teacher_state.items()
            if (
                should_copy(name)
                and name in student_state
                and student_state[name].shape == tensor.shape
            )
        }

        student_model.load_state_dict(compatible_state, strict=False)
        print(
            "Copied patch encoder and patch_inter teacher weights to student "
            f"({len(compatible_state)} tensors)."
        )

    def _get_log_dir(self):
        log_dir = Path.cwd() / "logs"
        if self.seed is not None:
            log_dir /= str(self.seed)
        log_dir /= self.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

    def get_train_dataloader(self, batch_size):
        dataset = MineDataset(self.train_data, cfg=self.cfg, is_train=True)
        num_workers = int(self.dataset_cfg.num_workers)
        persistent_workers = bool(self.cfg.runtime.persistent_workers) and num_workers > 0

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(self.cfg.runtime.pin_memory),
            persistent_workers=persistent_workers,
        )

    def _build_callbacks(self):
        callbacks = list(self.callbacks)
        callbacks.append(TQDMProgressBar(refresh_rate=1, leave=False))
        callbacks.append(TrainingRNGCompatibility())
        return callbacks

    def _build_trainer(self, epochs, callbacks):
        return pl.Trainer(
            max_epochs=int(epochs),
            precision=self.cfg.runtime.precision,
            accelerator=str(self.cfg.runtime.accelerator),
            devices=self.cfg.runtime.devices,
            default_root_dir=self._get_log_dir(),
            callbacks=callbacks,
            log_every_n_steps=int(self.cfg.runtime.log_every_n_steps),
            enable_progress_bar=bool(self.cfg.runtime.enable_progress_bar),
            enable_checkpointing=False,
        )

    def _train_model(self, model, train_dataloader, epochs):
        epochs = int(epochs)
        if epochs <= 0:
            if self.ema_cb is not None:
                self.ema_cb.clear()
            return False

        trainer = self._build_trainer(
            epochs=epochs,
            callbacks=self._build_callbacks(),
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
        )
        return True

    def _run_training_phase(self, model, batch_size, epochs):
        train_dataloader = self.get_train_dataloader(batch_size=batch_size)
        return self._train_model(
            model,
            train_dataloader,
            epochs,
        )

    def train_cls(self, cls_cfg=None, model=None):
        cls_cfg = self.cls_cfg if cls_cfg is None else cls_cfg
        return self._run_training_phase(
            model=self.final_model if model is None else model,
            batch_size=cls_cfg.training.batch_size,
            epochs=cls_cfg.training.epochs,
        )

    def train_teacher(self):
        pl.seed_everything(self.seed, workers=True)
        teacher_cls_cfg = self._phase_cls_cfg("teacher")
        teacher_model = self._load_teacher_from_checkpoint(teacher_cls_cfg)
        if teacher_model is not None:
            self.teacher_model = teacher_model
            return

        print("Training full-modality teacher model.")
        teacher_model = self._build_teacher_model(teacher_cls_cfg)
        trained = self.train_cls(teacher_cls_cfg, model=teacher_model)
        if trained and self._commit_ema(teacher_model):
            self._save_teacher_checkpoint(teacher_model)
        self._freeze_model(teacher_model)
        self.teacher_model = teacher_model


    def train_student(self):
        print("Training patch-only student model.")
        student_cls_cfg = self._phase_cls_cfg("student")
        distiller = TranscriptomePrivilegedDistiller(
            self.teacher_model,
            student_cls_cfg.distillation,
        )
        student_model = self._build_student_model(
            student_cls_cfg,
            distiller,
        )
        if self._copy_teacher_weights_to_student(student_cls_cfg):
            self._load_compatible_teacher_weights(student_model, self.teacher_model)

        self.student_model = student_model
        self.final_model = student_model
        trained = self.train_cls(student_cls_cfg, model=student_model)

        if trained and self._commit_ema(self.student_model):
            self.student_model.clear_distiller()
            self._save_student_checkpoint(self.student_model)
        else:
            self.student_model.clear_distiller()

    def train(self):
        """
        Run the configured teacher and student training workflow.
        """
        if self._use_distillation():
            self.train_teacher()
            self.train_student()
        else:
            self.final_model = self._build_teacher_model(self.cls_cfg)
            self.train_cls()
