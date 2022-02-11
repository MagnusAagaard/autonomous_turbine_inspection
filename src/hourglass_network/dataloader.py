import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize, CenterCrop
import json
import numpy as np
import preprocessing

class WindturbineDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = preprocessing.get_annotations(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img = self.annotations[idx]
        input_img, label_img = preprocessing.process_annotations(img, apply_augmentation=True)
        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)
        return input_img, label_img
     
def main():
    train_dataset = WindturbineDataset('./src/hourglass_network/data/annotations.json', './src/hourglass_network/data/all_data', transform=Compose([ToTensor(), CenterCrop((256, 256))]))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    next(iter(train_dataloader))

if __name__ == "__main__":
    main()