from pathlib import Path

from data_process.process import add_label_CRC, extract_patch_features, process_visium
from data_process.utils import gene_selection, save_h5ad
from pretrained_model.h_optimus_1 import HO1Extractor

if __name__ == "__main__":
    
    # Configure data and output paths.
    raw_data_root = Path("/home/yfm/project/data")
    project_root = Path(__file__).resolve().parents[2]
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

    
    save_dir = project_root / "processed_data" / "CRC"
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # Extract features and save all slices.
    backbone = HO1Extractor()
    adata_list = []
    for name in crc_names:
        slice_dir = crc_dir / name
        adata = process_visium(slice_dir, name, res="hires")
        image_path = crc_dir / name / "spatial" / "tissue_hires_image.png"
        label_path = crc_label_dir / f"{name}.csv"
        adata = add_label_CRC(adata.copy(), label_path)
        adata = extract_patch_features(adata.copy(), image_path, backbone)
        adata_list.append(adata)

    adata_list = gene_selection(adata_list, n_genes=3000)

    for adata in adata_list:
        save_h5ad(adata, save_dir)
