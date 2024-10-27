import torch
import glob
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .augmentations import get_transforms


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, masks_paths, transforms):
        self.img_paths = img_paths
        self.masks_paths = masks_paths
        self.transforms = transforms

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        mask_path = self.masks_paths[item]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(mask_path)

        transformed_data = self.transforms(image=img, mask=mask)
        img = transformed_data["image"]
        mask = transformed_data["mask"]

        return (
            torch.from_numpy(img).permute(2, 0, 1),
            torch.from_numpy(mask).permute(2, 0, 1).float(),
        )

    def __len__(self):
        return len(self.img_paths)


def get_dataloaders(batch_size: int = 1, base_dir=""):
    train_transforms, test_transforms = get_transforms()

    train_dataset = SegmentationDataset(
        sorted(glob.glob(f"{base_dir}data/train/images/*.jpg")),
        sorted(glob.glob(f"{base_dir}data/train/masks/*.npy")),
        train_transforms,
    )
    valid_dataset = SegmentationDataset(
        sorted(glob.glob(f"{base_dir}data/valid/images/*.jpg")),
        sorted(glob.glob(f"{base_dir}data/valid/masks/*.npy")),
        test_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    return train_loader, valid_loader
