import os
import timm
import torch
from PIL import Image
from huggingface_hub import login
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import json

class Virchow2Dataset(Dataset):
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


class Virchow2Extractor:
    def __init__(self, batch_size=256, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_dir = '../../pretrained_model/virchow2/'

        self.model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            global_pool="",
            dynamic_img_size=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pretrained_cfg = cfg["pretrained_cfg"]

        self.model.pretrained_cfg = pretrained_cfg

        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def extract(self, patch, normalize=False):
        self.model.to(self.device)
        self.model.eval()

        
        dataset = Virchow2Dataset(patch, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        features = []
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():  
                for images in dataloader:
                    images = images.to(self.device)
                    output = self.model(images)
                    class_token = output[:, 0]
                    patch_tokens = output[:, 5:]

                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                    if normalize:
                        class_token = F.normalize(class_token, p=2, dim=-1)
                    features.append(class_token.cpu())


        
        features = torch.cat(features, dim=0)
        features = features.cpu().numpy()
        
        return features


