from pathlib import Path

from data_process.process import (
    add_label_STHBC,
    extract_patch_features,
    process_STHBC,
)
from data_process.utils import gene_selection, save_h5ad
from pretrained_model.h_optimus_1 import HO1Extractor

if __name__ == "__main__":
    
    # Configure data and output paths.
    raw_data_root = Path("xxx")
    project_root = Path(__file__).resolve().parents[2]
    sthbc_dir = raw_data_root / "STHBC"
    sthbc_label_dir = sthbc_dir / "ST-pat" / "lbl"
    sthbc_names = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]

    
    save_dir = project_root / "processed_data" / "STHBC"
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # Extract features and save all slices.
    backbone = HO1Extractor()
    adata_list = []
    for name in sthbc_names:
        image_dir = sthbc_dir / "ST-imgs" / name[0] / name
        image_path = list(image_dir.glob("*.jpg"))[0]
        label_path = sthbc_label_dir / f"{name}_labeled_coordinates.tsv"
        adata = process_STHBC(sthbc_dir, name)
        adata = add_label_STHBC(adata.copy(), label_path)
        adata = extract_patch_features(adata.copy(), image_path, backbone)
        adata_list.append(adata)

    adata_list = gene_selection(adata_list, n_genes=3000)

    for adata in adata_list:
        save_h5ad(adata, save_dir)
