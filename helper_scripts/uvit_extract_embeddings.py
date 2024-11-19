import sys
import os

sys.path.append("./latent-diffusion")
sys.path.append("./U-ViT")

import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--resolution', default=256)
    args = parser.parse_args()
    print(args)

    

    if args.split == "train":
        datas = MSCOCODatabase(root='coco/train2014',
                             annFile='coco/annotations/captions_train2014.json',
                             size=args.resolution)
        save_dir = f'coco/train_text_emb'
    elif args.split == "val":
        datas = MSCOCODatabase(root='coco/val2014',
                             annFile='coco/annotations/captions_val2014.json',
                             size=arg.resolution)
        save_dir = f'coco/val_text_emb'
    else:
        raise ValueError("Chose split either 'train' or 'val'")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    clip = libs.clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            _, captions = data

            latent = clip.encode(captions)
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"{idx}_{i}.npy"), c)


if __name__ == "__main__":
    main()
