from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from scipy.spatial import KDTree
from torch.utils.data import Dataset
import pandas as pd
from utils import random_neighbor_sampling


class MineDataset(Dataset):
    """
    Serve spot samples with their spatial-neighborhood features.

    Args:
        data: Combined arrays returned by ``load_data``.
        cfg: Experiment configuration.
        is_train: Whether the dataset is used for training.
    """
    def __init__(self, data, cfg, is_train=True):
        self.data = data
        self.is_train = is_train
        self.cfg = cfg

    def __len__(self):
        return int(self.data["gene"].shape[0])

    def __getitem__(self, idx):
        """
        Return one center spot and its neighborhood.

        Args:
            idx: Center-spot index.

        Returns:
            Dictionary containing patch and gene features, labels, and spot metadata.
        """
        indices = self.data['indices'][idx]
        ho1 = self.data["patch"][indices]
        return {
            "patch": torch.as_tensor(ho1, dtype=torch.float32),
            "gene": torch.from_numpy(self.data["gene"][indices]),
            "label": torch.as_tensor(self.data["cancer"][idx], dtype=torch.long),
            "slide": torch.as_tensor(self.data["slide"][idx], dtype=torch.long),
            "spot_id": self.data["spot_ids"][idx],
            "pixel_x": torch.as_tensor(self.data["pixel_x"][idx], dtype=torch.float32),
            "pixel_y": torch.as_tensor(self.data["pixel_y"][idx], dtype=torch.float32),
        }


def load_data(names, cfg):
    """
    Load and combine processed slices.

    Args:
        names: Slice names to load.
        cfg: Experiment configuration containing the processed-data directory.

    Returns:
        Dictionary of features, labels, neighbors, slide IDs, and spot metadata.
    """
    dataset_cfg = cfg.experiment.dataset
    patch_list, gene_list, cancer_list, index_list, slide_list = [], [], [], [], []
    spot_id_list, px_list, py_list = [], [], []
    offset = 0

    
    
    # Preserve slice order when assigning slide IDs.
    slide2id = {name: i for i, name in enumerate(names)}

    for name in names:
        save_path = Path(dataset_cfg.data_dir) / f"{name}.h5ad"
        adata = sc.read_h5ad(save_path)
        num_samples = adata.shape[0]

        
        patch_list.extend(adata.obsm["patch"])
        gene_list.extend(adata.X)
        cancer_list.extend(adata.obs["cancer"])

        
        slide_id = slide2id[name]
        slide_list.extend([slide_id] * num_samples)

        
        spot_id_list.extend(list(adata.obs_names))
        px_list.extend(adata.obsm["spatial"][:, 0])
        py_list.extend(adata.obsm["spatial"][:, 1])

        indices = get_neighbors(adata, dataset_cfg.k_neighbors) + offset
        index_list.extend(indices)
        offset += num_samples

    return {
        'patch': np.array(patch_list),
        'gene': np.asarray(gene_list, dtype=np.float32),
        'indices': np.asarray(index_list),
        'cancer': np.asarray(cancer_list).reshape(-1, 1),
        'slide': np.asarray(slide_list, dtype=np.int64),
        'slide2id': slide2id,
        'spot_ids': np.array(spot_id_list),
        'pixel_x': np.array(px_list, dtype=np.float32),
        'pixel_y': np.array(py_list, dtype=np.float32),
    }

def get_neighbors(adata, k=3):
    """
    Find the nearest spatial neighbors for every spot.

    Args:
        adata: Slice data with spatial coordinates.
        k: Number of neighbors excluding the center spot.

    Returns:
        Integer array containing each center spot followed by its neighbors.
    """
    
    spatial = adata.obsm["spatial"]

    tree = KDTree(spatial)
    _, indices = tree.query(spatial, k=k + 1)

    
    return indices
