import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
import json
import numpy as np
import preprocessing

class WindturbineDataset(Dataset):
    def __init__(self, annotations_file, img_dir, train_transform=None, test_transform=None, train=True):
        self.annotations = preprocessing.get_annotations(annotations_file)
        self.img_dir = img_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train = train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img = self.annotations[idx]
        input_img, label_img = preprocessing.process_annotations(img, apply_augmentation=self.train)
        if self.train and self.train_transform:
            input_img = self.train_transform(input_img)
            label_img = self.train_transform(label_img)
        elif not self.train and self.test_transform:
            input_img = self.test_transform(input_img)
            label_img = self.test_transform(label_img)
            
        return input_img, label_img
     
def main():
    train_dataset = WindturbineDataset('./src/hourglass_network/data/annotations.json', './src/hourglass_network/data/all_data', transform=Compose([ToTensor(), CenterCrop((256, 256))]))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    next(iter(train_dataloader))

if __name__ == "__main__":
    main()