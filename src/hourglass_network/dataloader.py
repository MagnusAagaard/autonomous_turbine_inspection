import os
from torch.utils.data import Dataset
import json
import preprocessing

class WindturbineDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.annotations = preprocessing.get_annotations(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img = self.annotations[idx]
        input_img, label_img = preprocessing.process_annotations(img)
        #img_name = img.get('img').split('-')[1]
        #kps = []
        #for kp in img.get('kp-1'):
        #    width = float(kp.get('original_width')) / 100
        #    height = float(kp.get('original_height')) / 100
        #    x = float(kp.get('x')) * width
        #    y = float(kp.get('y')) * height
        #    kp_label = kp.get('keypointlabels')[0]
        #    kps.append([int(x), int(y), kp_label])
        #input_img, label_img = preprocessing.create_input_img(kps, img_name)
        
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return input_img, label_img