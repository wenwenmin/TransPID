from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from anndata import AnnData

from data_process.utils import get_patch, gene_selection, extract_expression, save_h5ad
from pretrained_model.h_optimus_1 import HO1Extractor

Image.MAX_IMAGE_PIXELS = None

def process_STHBC(data_dir, name):
    
    """
    Load one STHBC slice.

    Args:
        data_dir: Directory containing the STHBC count and spot files.
        name: Slice name.

    Returns:
        AnnData object with expression values and spatial coordinates.
    """
    gene_path = data_dir / f"ST-cnts/{name}.tsv.gz"
    spatial_path = data_dir / f"ST-spotfiles/{name}_selection.tsv"

    
    gene = pd.read_csv(
        gene_path,
        sep="\t",
        header=0,
        index_col=0,
    )

    
    spatial = pd.read_csv(spatial_path, sep="\t")
    spatial = spatial.loc[spatial["selected"] == 1].copy()
    spatial.index = spatial["x"].astype(str) + "x" + spatial["y"].astype(str)

    
    common_index = gene.index[
        gene.index.isin(spatial.index)
    ]
    gene = gene.loc[common_index]
    spatial = spatial.loc[common_index]

    adata = AnnData(X=gene.values)
    adata.obs_names = common_index
    adata.var_names = gene.columns
    adata.obsm["spatial"] = spatial[["pixel_x", "pixel_y"]].values
    adata.uns["spot_diameter"] = 224
    adata.uns["name"] = name

    return adata

def process_visium(data_dir, name, res='fullres'):
    
    """
    Load one 10x Visium slice.

    Args:
        data_dir: Directory containing the Visium output files.
        name: Slice name.
        res: Spatial coordinate resolution.

    Returns:
        AnnData object with expression values and spatial coordinates.
    """
    matrix_path = data_dir / 'filtered_feature_bc_matrix.h5'
    scale_path = data_dir / 'spatial/scalefactors_json.json'
    position_list_path = data_dir / "spatial/tissue_positions_list.csv"
    position_csv_path = data_dir / "spatial/tissue_positions.csv"

    
    adata = sc.read_10x_h5(matrix_path)
    adata.var_names_make_unique()
    barcodes = adata.obs_names

    if position_list_path.exists():
        spatial = pd.read_csv(position_list_path, header=None)
        spatial.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        spatial = spatial.loc[spatial["in_tissue"] == 1].set_index("barcode").loc[barcodes]
    else:
        spatial = pd.read_csv(position_csv_path)
        spatial = spatial.loc[spatial["in_tissue"] == 1].set_index("barcode").loc[barcodes]

    
    with open(scale_path, 'r') as f:
        scalefactors = json.load(f)
    scale = scalefactors['tissue_hires_scalef']

    
    if res == 'fullres':
        adata.uns["spot_diameter"] = scalefactors['spot_diameter_fullres']
        adata.obsm['spatial'] = (spatial[['pxl_col_in_fullres', 'pxl_row_in_fullres']]).values
    elif res == 'hires':
        adata.uns["spot_diameter"] = scalefactors['spot_diameter_fullres'] * scale
        adata.obsm['spatial'] = (spatial[['pxl_col_in_fullres', 'pxl_row_in_fullres']] * scale).values
    adata.uns['name'] = name

    return adata

def extract_patch_features(adata, image_path, backbone):
    
    """
    Extract image features for all spots in a slice.

    Args:
        adata: Slice data with spatial coordinates.
        image_path: Path to the histology image.
        backbone: Image feature extractor.

    Returns:
        The input AnnData object with patch features in ``obsm["patch"]``.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    patches = get_patch(
        image,
        adata,
        patch_size=max(adata.uns["spot_diameter"], 28),
    )
    patch_embedding = backbone.extract(patches, normalize=False)
    adata.obsm["patch"] = patch_embedding
    return adata

def add_label_STHBC(adata, label_path):
    """
    Add binary cancer labels to an STHBC slice.

    Args:
        adata: STHBC slice data.
        label_path: Path to the labeled-coordinate file.

    Returns:
        Filtered AnnData object with binary and raw labels.
    """
    label = pd.read_csv(label_path, sep="\t").dropna(subset=["x", "y"]).copy()
    label.index = (
        label["x"].round(0).astype(int).astype(str)
        + "x"
        + label["y"].round(0).astype(int).astype(str)
    )

    valid = adata.obs_names[adata.obs_names.isin(label.index)]
    adata = adata[valid].copy()
    label = label.loc[valid]

    adata.obs["cancer"] = (
        label["label"].str.contains("cancer", case=False, na=False).astype(int).values
    )
    adata.obs["raw_label"] = label["label"].values
    return adata

def add_label_CRC(adata, label_path):
    """
    Add binary tumor labels to a CRC slice.

    Args:
        adata: CRC slice data.
        label_path: Path to the pathology annotation file.

    Returns:
        Filtered AnnData object with binary and raw labels.
    """
    label = pd.read_csv(label_path)
    label = label.set_index("Barcode")

    valid = adata.obs_names[adata.obs_names.isin(label.index)]
    adata = adata[valid].copy()
    label = label.loc[valid]

    adata.obs["cancer"] = label.iloc[:, 0].str.contains('tumor', case=False, na=False).astype(int).values
    adata.obs["raw_label"] = label.iloc[:, 0].values
    return adata

def add_label_ViHBC1(adata, label_path):
    """
    Add binary tumor labels to the ViHBC1 slice.

    Args:
        adata: ViHBC1 slice data.
        label_path: Path to the annotation file.

    Returns:
        Filtered AnnData object with binary and raw labels.
    """
    label = pd.read_csv(label_path, sep="\t", header=None, names=["Barcode", "Annotation"])
    label = label.set_index("Barcode")

    valid = adata.obs_names[adata.obs_names.isin(label.index)]
    adata = adata[valid].copy()
    label = label.loc[valid]

    tumor_annotations = [
      "IDC_8", "IDC_7", "IDC_6", "IDC_5", "IDC_4", "IDC_3", "IDC_2", "IDC_1",
      "DCIS/LCIS_5", "DCIS/LCIS_4", "DCIS/LCIS_2", "DCIS/LCIS_1",
    ]
    adata.obs["cancer"] = label["Annotation"].isin(tumor_annotations).astype(int).values
    adata.obs["raw_label"] = label["Annotation"].values
    return adata

def add_label_ViHBC2(adata, label_path):
    """
    Add binary tumor labels to the ViHBC2 slice.

    Args:
        adata: ViHBC2 slice data.
        label_path: Path to the binary label file.

    Returns:
        AnnData object with binary and raw labels.
    """
    barcodes = adata.obs_names
    label = pd.Series(list(map(int, label_path.read_text().strip().split())), index=barcodes)
    adata.obs['cancer'] = label.astype(int).values
    adata.obs["raw_label"] = label.astype(str).values
    return adata

def add_label_ViHBC3(adata, label_path):
    """
    Add binary tumor labels to the ViHBC3 slice.

    Args:
        adata: ViHBC3 slice data.
        label_path: Path to the annotation spreadsheet.

    Returns:
        Filtered AnnData object with binary and raw labels.
    """
    label = pd.read_excel(label_path)
    label = label.set_index("Barcode")

    valid = adata.obs_names[adata.obs_names.isin(label.index)]
    adata = adata[valid].copy()
    label = label.loc[valid]

    tumor_annotations = ['DCIS #1', 'DCIS #2', 'mixed/invasive', 'invasive']
    adata.obs["cancer"] = label["Annotation"].isin(tumor_annotations).astype(int).values
    adata.obs["raw_label"] = label["Annotation"].values

    return adata