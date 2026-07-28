from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel


class GenBioPathFMDataset(Dataset):
    def __init__(self, patch_list, transform):
        super().__init__()
        self.patch_list = patch_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        image = Image.fromarray(self.patch_list[idx].astype("uint8")).convert("RGB")
        return self.transform(image)


class GenBioPathFMExtractor:
    def __init__(self, batch_size=16, device=None, model_dir=None, use_amp=True):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent / "genbio_pathfm")

        self.model = AutoModel.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.697, 0.575, 0.728),
                    std=(0.188, 0.240, 0.187),
                ),
            ]
        )

    def extract(self, patch, normalize=False, return_patch_tokens=False):
        self.model.to(self.device)
        self.model.eval()

        dataset = GenBioPathFMDataset(patch, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        cls_features = []
        patch_features = []
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_amp and torch.device(self.device).type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with amp_context:
                for images in dataloader:
                    images = images.to(self.device, non_blocking=True)
                    if return_patch_tokens:
                        cls_out, patch_out = self.model.forward_with_patches(images)
                        if normalize:
                            cls_out = F.normalize(cls_out, p=2, dim=-1)
                            patch_out = F.normalize(patch_out, p=2, dim=-1)
                        cls_features.append(cls_out.float().cpu())
                        patch_features.append(patch_out.float().cpu())
                    else:
                        output = self.model(images)
                        if normalize:
                            output = F.normalize(output, p=2, dim=-1)
                        cls_features.append(output.float().cpu())

        cls_features = torch.cat(cls_features, dim=0).numpy()
        if not return_patch_tokens:
            return cls_features

        patch_features = torch.cat(patch_features, dim=0).numpy()
        return cls_features, patch_features
