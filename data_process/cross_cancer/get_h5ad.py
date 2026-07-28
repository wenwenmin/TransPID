from pathlib import Path

import scanpy as sc
from anndata import AnnData
from PIL import Image

from data_process.process import (
    add_label_CRC,
    add_label_STHBC,
    add_label_ViHBC1,
    add_label_ViHBC2,
    add_label_ViHBC3,
    extract_patch_features,
    process_STHBC,
    process_visium,
)
from data_process.utils import _materialize_adata, save_h5ad
from pretrained_model.h_optimus_1 import HO1Extractor


def gene_selection(
    adata_list,
    ref_names,
    n_genes,
    min_counts,
    min_cells,
    normalize_target_sum,
):
    
    """
    Select source-derived highly variable genes for cross-cancer transfer.

    Args:
        adata_list: Source and target slices.
        ref_names: Source slices used to fit highly variable genes.
        n_genes: Number of genes to retain.
        min_counts: Minimum total count per gene.
        min_cells: Minimum number of spots per gene.
        normalize_target_sum: Target library size after normalization.

    Returns:
        Processed slices with the same normalized gene features.
    """
    processed_list = []
    for adata in adata_list:
        adata = adata.copy()
        adata.var_names_make_unique()
        adata.layers["counts"] = adata.X.copy()
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.filter_genes(adata, min_counts=min_counts)
        processed_list.append(adata)

    
    common = set(processed_list[0].var_names)
    for adata in processed_list[1:]:
        common &= set(adata.var_names)
    common = [gene for gene in processed_list[0].var_names if gene in common]

    processed_list = [adata[:, common].copy() for adata in processed_list]
    adata_dict = {adata.uns["name"]: adata for adata in processed_list}

    
    # Fit HVGs on reference slices only.
    ref_list = []
    for name in ref_names:
        ref_adata = AnnData(X=adata_dict[name].layers["counts"].copy())
        ref_adata.var_names = common
        ref_list.append(ref_adata)

    merged = sc.concat(
        ref_list,
        label="slide",
        keys=ref_names,
        join="inner",
        index_unique="-",
    )
    sc.pp.highly_variable_genes(
        merged,
        flavor="seurat_v3",
        n_top_genes=n_genes,
        batch_key="slide",
        subset=False,
    )

    hvgs = merged.var_names[merged.var["highly_variable"]].tolist()
    hvg_stats = merged.var.loc[hvgs]

    
    new_list = []
    for adata in processed_list:
        sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
        sc.pp.log1p(adata)
        adata = adata[:, hvgs].copy()

        for column in (
            "highly_variable_rank",
            "highly_variable_nbatches",
            "means",
            "variances",
            "variances_norm",
        ):
            adata.var[column] = hvg_stats.loc[adata.var_names, column].to_numpy()
        adata.var["highly_variable"] = True
        adata.uns["gene_selection"] = {
            "mode": "fixed_source_only_hvg",
            "common_scope": "ref_and_target_filtered_genes",
            "expression_filtering": True,
            "min_counts": min_counts,
            "min_cells": min_cells,
            "hvg_fit_names": list(ref_names),
            "hvg_flavor": "seurat_v3",
            "n_common_genes": len(common),
            "n_hvgs": len(hvgs),
            "normalize_target_sum": normalize_target_sum,
        }
        new_list.append(_materialize_adata(adata))

    return new_list


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    
    # Configure data and output paths.
    raw_data_root = Path("/home/yfm/project/data")
    project_root = Path(__file__).resolve().parents[2]
    n_genes = 3000
    min_counts = 10
    min_cells = 3
    normalize_target_sum = 1e4

    sthbc_dir = raw_data_root / "STHBC"
    sthbc_label_dir = sthbc_dir / "ST-pat" / "lbl"
    sthbc_names = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]

    vihbc1_dir = raw_data_root / "ViHBC1"
    vihbc1_label = vihbc1_dir / "truth.txt"
    vihbc1_image_path = vihbc1_dir / "tissue_full_image.png"

    vihbc2_dir = raw_data_root / "ViHBC2"
    vihbc2_label = vihbc2_dir / "Visium_IDC_label.txt"
    vihbc2_image_path = (
        vihbc2_dir / "V1_Human_Invasive_Ductal_Carcinoma_image.png"
    )

    vihbc3_dir = raw_data_root / "ViHBC3"
    vihbc3_label = vihbc3_dir / "labels.xlsx"
    vihbc3_image_path = (
        vihbc3_dir / "CytAssist_FFPE_Human_Breast_Cancer_tissue_image.png"
    )

    vihbc_dataset = {
        "ViHBC1": {
            "data_dir": vihbc1_dir,
            "label_path": vihbc1_label,
            "image_path": vihbc1_image_path,
            "add_label": add_label_ViHBC1,
        },
        "ViHBC2": {
            "data_dir": vihbc2_dir,
            "label_path": vihbc2_label,
            "image_path": vihbc2_image_path,
            "add_label": add_label_ViHBC2,
        },
        "ViHBC3": {
            "data_dir": vihbc3_dir,
            "label_path": vihbc3_label,
            "image_path": vihbc3_image_path,
            "add_label": add_label_ViHBC3,
        },
    }
    vihbc_names = list(vihbc_dataset)
    hbc_names = sthbc_names + vihbc_names

    crc_dir = raw_data_root / "CRC"
    crc_label_dir = crc_dir / "Pathology_SpotAnnotations"
    crc_names = [
        "CRC_A1",
        "CRC_A2",
        "CRC_B1",
        "CRC_B2",
        "CRC_C1",
        "CRC_C2",
        "CRC_D1",
        "CRC_D2",
        "CRC_E1",
        "CRC_E2",
        "CRC_G1",
        "CRC_G2",
    ]

    experiments = {
        "CRC2HBC": {
            "ref_names": crc_names,
            "save_dir": project_root / "processed_data" / "CRC2HBC",
        },
        "HBC2CRC": {
            "ref_names": hbc_names,
            "save_dir": project_root / "processed_data" / "HBC2CRC",
        },
    }

    
    # Extract features once for both transfer directions.
    backbone = HO1Extractor()
    adata_list = []

    for name in sthbc_names:
        image_dir = sthbc_dir / "ST-imgs" / name[0] / name
        image_path = next(image_dir.glob("*.jpg"))
        label_path = sthbc_label_dir / f"{name}_labeled_coordinates.tsv"

        adata = process_STHBC(sthbc_dir, name)
        adata = add_label_STHBC(adata, label_path)
        adata = extract_patch_features(adata, image_path, backbone)
        adata_list.append(adata)

    for name, info in vihbc_dataset.items():
        adata = process_visium(info["data_dir"], name, res="fullres")
        adata = info["add_label"](adata, info["label_path"])
        adata = extract_patch_features(adata, info["image_path"], backbone)
        adata_list.append(adata)

    for name in crc_names:
        slice_dir = crc_dir / name
        image_path = slice_dir / "spatial/tissue_hires_image.png"
        label_path = crc_label_dir / f"{name}.csv"

        adata = process_visium(slice_dir, name, res="hires")
        adata = add_label_CRC(adata, label_path)
        adata = extract_patch_features(adata, image_path, backbone)
        adata_list.append(adata)

    
    # Fit source-specific genes and save each experiment.
    for experiment, info in experiments.items():
        new_list = gene_selection(
            adata_list,
            info["ref_names"],
            n_genes=n_genes,
            min_counts=min_counts,
            min_cells=min_cells,
            normalize_target_sum=normalize_target_sum,
        )
        for adata in new_list:
            save_h5ad(adata, info["save_dir"])

        print(
            f"Finished {experiment}: HVGs fitted on {info['ref_names']}, "
            f"output={info['save_dir']}"
        )
