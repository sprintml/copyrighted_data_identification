import os
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import datasets
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import os
import einops
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from omegaconf import DictConfig

import sys

sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")


class MSCOCODatabase(Dataset):
    def __init__(self, img_path, emb_path, annFile, size=256):
        from pycocotools.coco import COCO

        self.img_path = img_path
        self.emb_path = emb_path
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def center_crop(self, width, height, img):
        resample = {"box": Image.BOX, "lanczos": Image.LANCZOS}["lanczos"]
        crop = np.min(img.shape[:2])
        img = img[
            (img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2,
            (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2,
        ]
        try:
            img = Image.fromarray(img, "RGB")
        except:
            img = Image.fromarray(img)
        img = img.resize((width, height), resample)

        return np.array(img).astype(np.uint8)

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.img_path, path)).convert("RGB")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = self.center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, "h w c -> c h w")

        k = np.random.randint(0, 4)
        c = np.load(os.path.join(self.emb_path, f"{index}_{k}.npy"))

        return image, c


def get_uvit_t2i_dataloaders(
    config: DictConfig, model_cfg: DictConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Retrun dataloaders matching UViT_t2i model
    """
    train_dataset = MSCOCODatabase(
        os.path.join(model_cfg.dataset_path, "train2014"),
        os.path.join(model_cfg.dataset_path, "train_text_emb"),
        os.path.join(model_cfg.dataset_path, "annotations/captions_train2014.json"),
        size=model_cfg.image_size,
    )
    test_dataset = MSCOCODatabase(
        os.path.join(model_cfg.dataset_path, "val2014"),
        os.path.join(model_cfg.dataset_path, "val_text_emb"),
        os.path.join(model_cfg.dataset_path, "annotations/captions_val2014.json"),
        size=model_cfg.image_size,
    )

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    members_loader = DataLoader(
        train_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    nonmembers_loader = DataLoader(
        test_dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    return members_loader, nonmembers_loader
