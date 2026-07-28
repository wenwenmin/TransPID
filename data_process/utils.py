import scipy
from PIL import Image
from anndata import AnnData
from pretrained_model.scGPT_spatial_main import scgpt_spatial
Image.MAX_IMAGE_PIXELS = None
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import harmonypy as hm
from scipy import sparse
from pathlib import Path
from PIL import Image

def save_h5ad(adata, save_dir):
    """
    Save one processed slice.

    Args:
        adata: Processed slice data.
        save_dir: Output directory for the H5AD file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    name = adata.uns["name"]
    save_path = save_dir / f"{name}.h5ad"
    adata.write_h5ad(save_path)

    uni2_embed_shape = adata.obsm["uni2_embed"].shape if "uni2_embed" in adata.obsm else None
    ho1_embed_shape = adata.obsm["ho1_embed"].shape if "ho1_embed" in adata.obsm else None
    patch = adata.obsm["patch"].shape if "patch" in adata.obsm else None
    cancer_ratio = adata.obs["cancer"].mean() * 100 if "cancer" in adata.obs else None

    msg = f"Saved {name} shape: {adata.shape}"
    if patch is not None:
        msg += f", patch: {patch}"
    if uni2_embed_shape is not None:
        msg += f", uni2_embed_shape: {uni2_embed_shape}"
    if ho1_embed_shape is not None:
        msg += f", ho1_embed_shape: {ho1_embed_shape}"
    if cancer_ratio is not None:
        msg += f", cancer: {cancer_ratio:.1f}%"
    print(msg)

def _to_ndarray(x):
    if isinstance(x, np.ndarray):
        return x
    elif sp.issparse(x):
        return x.toarray()
    else:
        return np.asarray(x)

def _materialize_adata(adata):
    adata = adata.copy()

    adata.X = _to_ndarray(adata.X)

    for k in list(adata.layers.keys()):
        adata.layers[k] = _to_ndarray(adata.layers[k])

    return adata

def extract_expression(adata, max_length=3000):
    
    
    
    """
    Extract scGPT expression features for one slice.

    Args:
        adata: Slice data with gene expression values.
        max_length: Maximum number of genes passed to scGPT.

    Returns:
        AnnData object containing the expression embeddings.
    """
    model_dir = '../../pretrained_model/scGPT_spatial_v1'
    gene_col = 'index'

    new_adata = scgpt_spatial.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
        obs_to_save=list(adata.obs.columns),
        return_new_adata=True,
        max_length=max_length
    )

    new_adata.uns = adata.uns.copy()
    new_adata.obsm = {k: v.copy() for k, v in adata.obsm.items()}

    return new_adata

def gene_selection(
    adata_list,
    n_genes=3000,
    min_counts=10,
    min_cells=3,
    normalize_target_sum=1e4,
):

    """
    Select a shared set of highly variable genes across slices.

    Args:
        adata_list: Slices used to fit and apply gene selection.
        n_genes: Number of genes to retain.
        min_counts: Minimum total count per gene.
        min_cells: Minimum number of spots per gene.
        normalize_target_sum: Target library size after normalization.

    Returns:
        Processed slices with the same normalized gene features.
    """
    processed_list = []
    
    for i, adata in enumerate(adata_list):
        adata = adata.copy()
        adata.var_names_make_unique()

        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.filter_genes(adata, min_counts=min_counts)
        processed_list.append(adata)

    
    common = set(processed_list[0].var_names)
    for adata in processed_list[1:]:
        common &= set(adata.var_names)
    common = [g for g in processed_list[0].var_names if g in common]

    processed_list = [adata[:, common].copy() for adata in processed_list]

    for adata in processed_list:
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

    
    merged = sc.concat(
        processed_list,
        label="batch",
        keys=[adata.uns["name"] for adata in processed_list],
        join="inner",
    )

    sc.pp.highly_variable_genes(
        merged,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=n_genes,
        batch_key="batch",
        subset=False,
    )

    hvgs = merged.var_names[merged.var["highly_variable"]].tolist()

    
    new_list = []
    for adata in processed_list:
        adata = adata.copy()

        sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
        sc.pp.log1p(adata)

        adata = adata[:, hvgs].copy()
        adata = _materialize_adata(adata)
        new_list.append(adata)

    return new_list


def get_patch(image, adata, patch_size=224):
    """
    Crop one image patch around each spatial spot.

    Args:
        image: RGB histology image.
        adata: Slice data with spatial coordinates.
        patch_size: Output patch width and height in pixels.

    Returns:
        List of RGB image patches.
    """
    
    spatial_coords = adata.obsm['spatial']

    patches = []
    
    for coord in spatial_coords:
        
        pixel_x, pixel_y = coord

        
        half_patch = int((patch_size + 1) // 2)
        top_left_x = int(pixel_x) - half_patch
        top_left_y = int(pixel_y) - half_patch
        bottom_right_x = int(pixel_x) + half_patch
        bottom_right_y = int(pixel_y) + half_patch

        patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        
        if top_left_x < 0 or top_left_y < 0 or bottom_right_x > image.shape[1] or bottom_right_y > image.shape[0]:
            patch_resized = Image.fromarray(patch).resize((patch_size, patch_size), Image.ANTIALIAS)
            patch = np.array(patch_resized)


        
        patches.append(patch)

    
    return patches