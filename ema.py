import pytorch_lightning as pl
import torch

class EMACallback(pl.Callback):
    """
    Maintain exponential moving-average model weights after warm-up.

    Args:
        decay: Exponential moving-average decay factor.
    """
    def __init__(self, decay=0.0):
        super().__init__()
        self.decay = float(decay)
        self.ema_state_dict = {}
        self.backup_state_dict = {}
        self.global_step = 0

    @classmethod
    def create_ema(cls, cfg):
        ema_cfg = cfg.experiment.callbacks.ema
        if not ema_cfg.enabled:
            return None

        decay = float(ema_cfg.decay)
        return cls(decay=decay)

    @classmethod
    def test_callbacks_from(cls, callback):
        if callback is None:
            return None
        return callback.clone_for_test()

    @property
    def has_shadow_weights(self):
        return bool(self.ema_state_dict)

    @staticmethod
    def _named_trainable_parameters(pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                yield name, param

    @staticmethod
    def _clone_state_dict(state_dict):
        return {
            name: tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    def _capture_state(self, pl_module):
        return {
            name: param.detach().clone()
            for name, param in self._named_trainable_parameters(pl_module)
        }

    def _load_state(self, pl_module, state_dict):
        with torch.no_grad():
            for name, param in self._named_trainable_parameters(pl_module):
                if name in state_dict:
                    value = state_dict[name].to(device=param.device, dtype=param.dtype)
                    param.copy_(value)

    def clone_for_test(self):
        if not self.has_shadow_weights:
            return None

        callback = type(self)(decay=self.decay)
        callback.global_step = self.global_step
        callback.ema_state_dict = self._clone_state_dict(self.ema_state_dict)
        return callback

    def clear(self):
        self.ema_state_dict.clear()
        self.backup_state_dict.clear()
        self.global_step = 0

    def on_fit_start(self, trainer, pl_module):
        self.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._is_active(pl_module):
            return

        if not self.ema_state_dict:
            self._initialize(pl_module)

        with torch.no_grad():
            for name, param in self._named_trainable_parameters(pl_module):
                self.ema_state_dict[name].mul_(self.decay).add_(
                    param.detach(),
                    alpha=1.0 - self.decay,
                )

    def _swap_to_ema(self, pl_module):
        if not self.ema_state_dict:
            return

        self.backup_state_dict = self._capture_state(pl_module)
        self._load_state(pl_module, self.ema_state_dict)

    def _restore_original(self, pl_module):
        if not self.backup_state_dict:
            return

        self._load_state(pl_module, self.backup_state_dict)
        self.backup_state_dict = {}

    def _is_active(self, pl_module):
        start_epoch = int(
            getattr(pl_module.cls_cfg.training, "warm_up_epoch", 0)
        )
        return pl_module.current_epoch >= start_epoch

    def _initialize(self, pl_module):
        self.backup_state_dict = {}
        self.global_step = 0
        self.ema_state_dict = {
            name: param.detach().clone()
            for name, param in self._named_trainable_parameters(pl_module)
        }

    def copy_ema_to_model(self, pl_module):
        if not self.has_shadow_weights:
            return False

        params = dict(self._named_trainable_parameters(pl_module))

        missing = set(self.ema_state_dict) - set(params)
        if missing:
            raise RuntimeError(
                f"EMA parameters do not match current model: {sorted(missing)[:5]}"
            )

        for name, ema_param in self.ema_state_dict.items():
            if params[name].shape != ema_param.shape:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: "
                    f"{tuple(ema_param.shape)} != {tuple(params[name].shape)}"
                )

        self._load_state(pl_module, self.ema_state_dict)
        return True

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_state_dict"] = self._clone_state_dict(self.ema_state_dict)
        checkpoint["ema_global_step"] = self.global_step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.ema_state_dict = self._clone_state_dict(checkpoint.get("ema_state_dict", {}))
        self.global_step = int(checkpoint.get("ema_global_step", 0))
        self.backup_state_dict = {}

    def on_test_start(self, trainer, pl_module):
        self._swap_to_ema(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._restore_original(pl_module)
