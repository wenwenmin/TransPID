import os
import timm
import torch
from PIL import Image
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

class HO1Dataset(Dataset):
    def __init__(self, patch_list, transform):
        super().__init__()
        self.patch_list = patch_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.patch_list[idx])
        image = self.transform(pil_image)
        return image


class HO1Extractor:
    """
    Extract H-Optimus-1 features from histology patches.

    Args:
        batch_size: Number of patches processed per batch.
        device: Torch device used for inference.
    """
    def __init__(self, batch_size=256, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_dir = '../../pretrained_model/HOptimus1/'

        self.model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2",
            pretrained=False,
            num_classes=0,
            global_pool="token",
            img_size=224,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

    def extract(self, patch, normalize=False):
        """
        Encode a sequence of histology patches.

        Args:
            patch: Sequence of RGB image arrays.
            normalize: Whether to L2-normalize the output features.

        Returns:
            NumPy array with one feature vector per patch.
        """
        self.model.to(self.device)
        self.model.eval()

        
        dataset = HO1Dataset(patch, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        features = []
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():  
                for images in dataloader:
                    images = images.to(self.device)
                    output = self.model(images)
                    if normalize:
                        output = F.normalize(output, p=2, dim=-1)
                    features.append(output.cpu())

        
        features = torch.cat(features, dim=0)
        features = features.cpu().numpy()
        
        return features


