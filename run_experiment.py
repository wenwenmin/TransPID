import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import torch
from torch.utils.data import DataLoader

from dataset.dataset import MineDataset, load_data
from ema import EMACallback
from train import TrainingManager
from utils import get_folds


def _build_test_dataloader(cfg, test_name):
    cls_cfg = cfg.experiment.cls
    test_data = load_data([test_name], cfg)
    test_dataset = MineDataset(test_data, cfg=cfg, is_train=False)
    num_workers = int(cfg.runtime.test_num_workers)
    persistent_workers = bool(cfg.runtime.persistent_workers) and num_workers > 0
    return DataLoader(
        test_dataset,
        batch_size=cls_cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.runtime.pin_memory),
        persistent_workers=persistent_workers,
    )


def _build_test_trainer(cfg, log_dir, callbacks):
    return pl.Trainer(
        default_root_dir=str(log_dir),
        precision=cfg.runtime.precision,
        accelerator=str(cfg.runtime.accelerator),
        devices=cfg.runtime.devices,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.runtime.log_every_n_steps),
        enable_progress_bar=bool(cfg.runtime.enable_progress_bar),
        enable_checkpointing=False,
    )


def _is_fixed_split(dataset_cfg):
    return str(dataset_cfg.split_mode).lower() == "fixed"


def _cleanup_runtime(cfg):
    if bool(cfg.runtime.cleanup.empty_cuda_cache) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if bool(cfg.runtime.cleanup.gc_collect):
        gc.collect()


def _cleanup_fold_resources(cfg, trainer, train_manager, ema_test_cb):
    if trainer is not None and bool(cfg.runtime.cleanup.teardown_strategy):
        trainer.strategy.teardown()

    if ema_test_cb is not None:
        ema_test_cb.ema_state_dict.clear()
        ema_test_cb.backup_state_dict.clear()

    if getattr(train_manager, "ema_cb", None) is not None:
        train_manager.ema_cb.ema_state_dict.clear()
        train_manager.ema_cb.backup_state_dict.clear()
        train_manager.ema_cb = None

    for attr in ("final_model", "student_model", "teacher_model"):
        model = getattr(train_manager, attr, None)
        if model is None:
            continue
        if hasattr(model, "clear_distiller"):
            model.clear_distiller()
        model.project_model = None
        setattr(train_manager, attr, None)

    train_manager.train_data = None
    _cleanup_runtime(cfg)


def _cleanup_test_resources(cfg, trainer, ema_test_cb):
    if trainer is not None and bool(cfg.runtime.cleanup.teardown_strategy):
        trainer.strategy.teardown()

    if ema_test_cb is not None:
        ema_test_cb.ema_state_dict.clear()
        ema_test_cb.backup_state_dict.clear()

    _cleanup_runtime(cfg)


def _save_results(results, model_type, cfg):
    save_dir = Path.cwd() / "results" / model_type
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = cfg.runtime.output.results_filename
    json_path = save_dir / filename
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    return json_path


def _write_score_csv(model, test_name, cfg, seed, model_type):
    """
    Write spot-level predictions for one model and test slice.

    Args:
        model: Tested teacher or student model.
        test_name: Test slice name.
        cfg: Experiment configuration.
        seed: Random seed.
        model_type: Output group name, such as ``teacher`` or ``student``.
    """
    if model is None:
        return
    label_score = getattr(model, "label_score", None)
    if label_score is None:
        print(f"  [WARN] No label_score on {model_type} model, skip score.csv for {test_name}")
        return

    labels = label_score["label"].cpu().numpy().ravel().astype(int)
    scores = label_score["score"].cpu().numpy().ravel()

    data_dir = Path(cfg.experiment.dataset.data_dir)
    h5ad_path = data_dir / f"{test_name}.h5ad"
    if h5ad_path.exists():
        adata = sc.read_h5ad(h5ad_path)
        if "raw_label" in adata.obs:
            raw_labels = adata.obs["raw_label"].values
        else:
            raw_labels = labels.astype(str)
        spot_ids = list(adata.obs_names)
        pixel_x = adata.obsm["spatial"][:, 0]
        pixel_y = adata.obsm["spatial"][:, 1]
    else:
        raw_labels = labels.astype(str)
        spot_ids = [str(i) for i in range(len(labels))]
        pixel_x = np.zeros(len(labels))
        pixel_y = np.zeros(len(labels))

    threshold = 0.5
    pred_labels = (scores >= threshold).astype(int)

    save_dir = Path.cwd() / "results" / model_type / str(seed) / test_name
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(labels)):
        rows.append({
            "slice_name": test_name,
            "spot_id": spot_ids[i],
            "label": raw_labels[i],
            "true_label": labels[i],
            "pred_label": pred_labels[i],
            "score": round(float(scores[i]), 6),
            "pixel_x": float(pixel_x[i]),
            "pixel_y": float(pixel_y[i]),
        })

    df = pd.DataFrame(rows)
    csv_path = save_dir / "score.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [{model_type}] score.csv saved to: {csv_path}")


def _test_target(train_manager, cfg, seed, test_name, ema, seed_results_s, seed_results_t):
    print(f"********************{test_name} is being tested********************")
    test_callbacks = []
    test_dataloader = _build_test_dataloader(cfg, test_name)
    test_ema = EMACallback.test_callbacks_from(ema)
    if test_ema is not None:
        test_callbacks.append(test_ema)

    log_dir = Path.cwd() / "logs" / str(seed) / test_name
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_test_trainer(cfg, log_dir, test_callbacks)

    trainer.test(train_manager.final_model, dataloaders=test_dataloader)
    _write_score_csv(train_manager.final_model, test_name, cfg, seed, "student")
    seed_results_s[test_name] = dict(train_manager.final_model.result)

    if train_manager.teacher_model is not None:
        trainer.test(train_manager.teacher_model, dataloaders=test_dataloader)
        _write_score_csv(train_manager.teacher_model, test_name, cfg, seed, "teacher")
        seed_results_t[test_name] = dict(train_manager.teacher_model.result)

    _cleanup_test_resources(cfg, trainer, test_ema)
    del test_dataloader
    del trainer
    del test_ema


def _run_fixed_split_seed(cfg, dataset_cfg, seed, seed_results_s, seed_results_t):
    train_names = list(dataset_cfg.ref_names)
    test_names = list(dataset_cfg.tgt_names)
    print("********************source slices are being trained********************")
    print(f"Test slices: {test_names}")
    pl.seed_everything(seed, workers=True)
    train_callbacks = []
    ema = EMACallback.create_ema(cfg)
    if ema is not None:
        train_callbacks.append(ema)

    run_name = "_".join(str(name) for name in test_names) if test_names else "targets"
    train_manager = TrainingManager(
        train_names,
        cfg,
        callbacks=train_callbacks,
        seed=seed,
        run_name=run_name,
    )
    train_manager.train()

    for test_name in test_names:
        _test_target(
            train_manager,
            cfg,
            seed,
            test_name,
            ema,
            seed_results_s,
            seed_results_t,
        )

    _cleanup_fold_resources(cfg, None, train_manager, None)
    del train_manager
    del ema


def _run_loo_seed(cfg, dataset_cfg, folds, seed, seed_results_s, seed_results_t):
    for train_names, test_name in folds:
        print(f"********************{test_name} is being tested********************")
        pl.seed_everything(seed, workers=True)
        train_callbacks, test_callbacks = [], []
        
        ema = EMACallback.create_ema(cfg)
        if ema is not None:
            train_callbacks.append(ema)
        train_manager = TrainingManager(
            train_names,
            cfg,
            callbacks=train_callbacks,
            seed=seed,
            run_name=test_name,
        )
        train_manager.train()


        test_dataloader = _build_test_dataloader(cfg, test_name)
        test_ema = EMACallback.test_callbacks_from(ema)
        if test_ema is not None:
            test_callbacks.append(test_ema)
        log_dir = Path.cwd() / "logs" / str(seed) / test_name
        log_dir.mkdir(parents=True, exist_ok=True)
        trainer = _build_test_trainer(cfg, log_dir, test_callbacks)


        trainer.test(train_manager.final_model, dataloaders=test_dataloader)
        _write_score_csv(train_manager.final_model, test_name, cfg, seed, "student")
        seed_results_s[test_name] = dict(train_manager.final_model.result)


        if train_manager.teacher_model is not None:
            trainer.test(train_manager.teacher_model, dataloaders=test_dataloader)
            _write_score_csv(train_manager.teacher_model, test_name, cfg, seed, "teacher")
            seed_results_t[test_name] = dict(train_manager.teacher_model.result)


        _cleanup_fold_resources(cfg, trainer, train_manager, test_ema)
        del train_manager
        del test_dataloader
        del trainer
        del test_ema
        del ema


def run_experiment(cfg):
    """
    Run all configured seeds and train-test splits.

    Args:
        cfg: Resolved experiment configuration.
    """
    dataset_cfg = cfg.experiment.dataset
    folds = get_folds(dataset_cfg)
    all_seed_results_t = {}
    all_seed_results_s = {}

    for seed in cfg.seeds:
        seed = int(seed)
        print(f"\n==================== Running seed = {seed} ====================")
        seed_results_t = {}
        seed_results_s = {}

        if _is_fixed_split(dataset_cfg):
            _run_fixed_split_seed(
                cfg,
                dataset_cfg,
                seed,
                seed_results_s,
                seed_results_t,
            )
        else:
            _run_loo_seed(
                cfg,
                dataset_cfg,
                folds,
                seed,
                seed_results_s,
                seed_results_t,
            )

        all_seed_results_s[seed] = seed_results_s
        json_path = _save_results(all_seed_results_s, "student", cfg)
        print(f"[student] results saved to: {json_path}")

        all_seed_results_t[seed] = seed_results_t
        json_path = _save_results(all_seed_results_t, "teacher", cfg)
        print(f"[teacher] results saved to: {json_path}")
