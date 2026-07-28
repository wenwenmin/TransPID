import torch
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def compute_intra_ctt_metrics(
    z: torch.Tensor,
    label: torch.Tensor,
    temperature: float = 0.07,
    ks=(1, 5, 20, 100),
):

    """
    Measure class separation within one embedding modality.

    Args:
        z: Embedding tensor with shape ``[batch, embed_dim]``.
        label: Class labels.
        temperature: Softmax temperature.
        ks: Neighborhood sizes used for retrieval accuracy.

    Returns:
        Dictionary of similarity, margin, probability-mass, and retrieval metrics.
    """
    z = F.normalize(z, p=2, dim=-1)
    label = label.reshape(-1)

    B = z.shape[0]
    device = z.device

    sim = torch.matmul(z, z.T)  

    self_mask = torch.eye(B, device=device, dtype=torch.bool)
    pos_mask = label.unsqueeze(0).eq(label.unsqueeze(1)) & (~self_mask)
    neg_mask = label.unsqueeze(0).ne(label.unsqueeze(1))

    pos_sim = sim[pos_mask]
    neg_sim = sim[neg_mask]

    metrics = {}

    metrics["pos_sim_mean"] = pos_sim.mean().item() if pos_sim.numel() > 0 else 0.0
    metrics["neg_sim_mean"] = neg_sim.mean().item() if neg_sim.numel() > 0 else 0.0
    metrics["sim_margin"] = metrics["pos_sim_mean"] - metrics["neg_sim_mean"]

    metrics["pos_sim_std"] = pos_sim.std().item() if pos_sim.numel() > 1 else 0.0
    metrics["neg_sim_std"] = neg_sim.std().item() if neg_sim.numel() > 1 else 0.0

    logits = sim / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * (~self_mask).float()
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)

    pos_mass = (exp_logits * pos_mask.float()).sum(dim=1) / denom
    valid_anchor = pos_mask.sum(dim=1) > 0

    if valid_anchor.sum() > 0:
        metrics["pos_softmax_mass"] = pos_mass[valid_anchor].mean().item()
    else:
        metrics["pos_softmax_mass"] = 0.0

    sim_no_self = sim.masked_fill(self_mask, float("-inf"))

    max_k = min(max(ks), B - 1)
    topk_idx = torch.topk(sim_no_self, k=max_k, dim=1).indices
    topk_label = label[topk_idx]
    same = topk_label.eq(label.unsqueeze(1))

    for k in ks:
        if k <= B - 1:
            metrics[f"top{k}_same_acc"] = same[:, :k].float().mean().item()

    return metrics

def _apply_masked_noise(x, std, mask):
    if std <= 0:
        return x

    noise = torch.randn_like(x) * std
    return x + noise * mask.to(x.dtype)


def add_noise(x, std, p=0.7):
    return _apply_masked_noise(x, std, torch.rand_like(x) < p)

def gene_add_noise(x, std, p=0.7):
    return _apply_masked_noise(x, std, (x != 0) & (torch.rand_like(x) < p))

def _apply_feature_mask(x, mask):
    return x * mask.to(x.dtype)

def feature_dropout(x, p):
    if p <= 0:
        return x
    return _apply_feature_mask(x, torch.rand_like(x) > p)

def gene_feature_dropout(x, p):
    if p <= 0:
        return x

    return _apply_feature_mask(x, (x != 0) & (torch.rand_like(x) > p))


def apply_feature_augmentation(x, noise_std, dropout_p):
    return feature_dropout(add_noise(x, noise_std), dropout_p)

def random_neighbor_sampling(indices, fixed_num, random_num):
    fixed = indices[:fixed_num]
    remain = indices[fixed_num:]

    if random_num > 0 and remain.size > 0:
        perm = np.random.permutation(remain.shape[0])[:random_num]
        sampled = remain[perm]
    else:
        sampled = np.empty((0,), dtype=indices.dtype)

    combined = np.concatenate([fixed, sampled], axis=0)
    if combined.size <= 1:
        return combined

    first = combined[:1]
    rest = combined[1:]
    rest = rest[np.random.permutation(rest.shape[0])]
    return np.concatenate([first, rest], axis=0)


def random_patch_sampling(samples, keep_num=64):
    sample_count = samples.shape[0]
    permutation = torch.randperm(sample_count, device=samples.device)
    if keep_num < sample_count:
        permutation = permutation[:keep_num]
    return samples[permutation]


def append_history(x, history):
    center = x[:, :1, :]
    history['center'].append(center)

    if x.shape[1] > 1:
        context = x[:, 1:, :]
        history['context'].append(context)

def get_folds(dataset_cfg):
    """
    Build train-test splits from the dataset configuration.

    Args:
        dataset_cfg: Dataset configuration with fixed or leave-one-out settings.

    Returns:
        List of ``(train_names, test_name)`` pairs.
    """
    split_mode = str(dataset_cfg.split_mode).lower()
    if split_mode == "loo":
        all_names = list(dataset_cfg.all_names)
        return [([name for name in all_names if name != test_name], test_name) for test_name in all_names]

    if split_mode == "fixed":
        train_names = list(dataset_cfg.ref_names)
        return [(train_names, test_name) for test_name in list(dataset_cfg.tgt_names)]

    raise ValueError("Cannot resolve dataset folds. Please set dataset.split_mode to 'loo' or 'fixed'.")


@torch.no_grad()
def compute_intra_ctt_metrics(
        z: torch.Tensor,
        label: torch.Tensor,
        temperature: float = 0.07,
        ks=(1, 5, 20, 100),
):

    """
    Measure class separation within one embedding modality.

    Args:
        z: Embedding tensor with shape ``[batch, embed_dim]``.
        label: Class labels.
        temperature: Softmax temperature.
        ks: Neighborhood sizes used for retrieval accuracy.

    Returns:
        Dictionary of similarity, margin, probability-mass, and retrieval metrics.
    """
    z = F.normalize(z, p=2, dim=-1)
    label = label.reshape(-1)

    B = z.shape[0]
    device = z.device

    sim = torch.matmul(z, z.T)  

    self_mask = torch.eye(B, device=device, dtype=torch.bool)
    pos_mask = label.unsqueeze(0).eq(label.unsqueeze(1)) & (~self_mask)
    neg_mask = label.unsqueeze(0).ne(label.unsqueeze(1))

    pos_sim = sim[pos_mask]
    neg_sim = sim[neg_mask]

    metrics = {}

    metrics["pos_sim_mean"] = pos_sim.mean().item() if pos_sim.numel() > 0 else 0.0
    metrics["neg_sim_mean"] = neg_sim.mean().item() if neg_sim.numel() > 0 else 0.0
    metrics["sim_margin"] = metrics["pos_sim_mean"] - metrics["neg_sim_mean"]

    metrics["pos_sim_std"] = pos_sim.std().item() if pos_sim.numel() > 1 else 0.0
    metrics["neg_sim_std"] = neg_sim.std().item() if neg_sim.numel() > 1 else 0.0

    logits = sim / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * (~self_mask).float()
    denom = exp_logits.sum(dim=1).clamp_min(1e-12)

    pos_mass = (exp_logits * pos_mask.float()).sum(dim=1) / denom
    valid_anchor = pos_mask.sum(dim=1) > 0

    if valid_anchor.sum() > 0:
        metrics["pos_softmax_mass"] = pos_mass[valid_anchor].mean().item()
    else:
        metrics["pos_softmax_mass"] = 0.0

    sim_no_self = sim.masked_fill(self_mask, float("-inf"))

    max_k = min(max(ks), B - 1)
    topk_idx = torch.topk(sim_no_self, k=max_k, dim=1).indices
    topk_label = label[topk_idx]
    same = topk_label.eq(label.unsqueeze(1))

    for k in ks:
        if k <= B - 1:
            metrics[f"top{k}_same_acc"] = same[:, :k].float().mean().item()

    return metrics
