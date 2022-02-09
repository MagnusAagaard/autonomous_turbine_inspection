import os
from torch.utils.data import Dataset
import json
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
        input_img, label_img = preprocessing.process_annotations(img)
        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)
        return input_img, label_img