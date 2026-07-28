import os
import timm
import torch
from PIL import Image
from huggingface_hub import login
from timm.data import create_transform, resolve_data_config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import json

class UNIDataset(Dataset):
    def __init__(self, patch_list, transform):
        super().__init__()
        self.patch_list = patch_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.patch_list[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


class UNI2Extractor:
    def __init__(self, batch_size=256, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_dir = '../../pretrained_model/UNI2/'
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True,
            'global_pool': 'token',
        }
        self.model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        self.model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pretrained_cfg = cfg["pretrained_cfg"]

        self.model.pretrained_cfg = pretrained_cfg

        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))


    def extract(self, patch, normalize=False):
        self.model.eval()  
        self.model.to(self.device)

        
        dataset = UNIDataset(patch, self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        features = []
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


