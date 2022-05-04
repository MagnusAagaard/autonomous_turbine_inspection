import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize, CenterCrop, RandomHorizontalFlip
from dataloader import WindturbineDataset
import numpy as np

BASE_DIR = './src/hourglass_network'

dataset = WindturbineDataset(f'{BASE_DIR}/data/annotations_125.json', f'{BASE_DIR}/data/all_data', train_transform=Compose([ToTensor(), CenterCrop(256), RandomHorizontalFlip(0.5)]), test_transform=Compose([ToTensor(), Resize(256), CenterCrop(256)]))
train_val_split = int(len(dataset)*0.8)
train_set, val_set = random_split(dataset, [train_val_split, len(dataset)-train_val_split], generator=torch.Generator().manual_seed(42))
print(f'Length of train dataset: {len(train_set)} \t Length of val dataset: {len(val_set)}')
train_dataloader = DataLoader(train_set, batch_size=4, num_workers=6, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=4, num_workers=2, shuffle=False)

for batch_idx, (inputs,labels) in enumerate(train_dataloader):
    inputs = inputs.numpy().transpose(0,2,3,1)
    for im in inputs:
        img = im[:,:,:3]
        plt.figure(1)
        plt.imshow(img[...,::-1])
        plt.figure(2)
        plt.imshow(im[:,:,4], cmap='gray', vmin=0, vmax=np.max(im[:,:,4]))
        plt.figure(3)
        plt.imshow(im[:,:,5], cmap='gray', vmin=0, vmax=np.max(im[:,:,5]))
        plt.figure(4)
        plt.imshow(im[:,:,6], cmap='gray', vmin=0, vmax=np.max(im[:,:,6]))
        plt.figure(5)
        plt.imshow(im[:,:,7], cmap='gray', vmin=0, vmax=np.max(im[:,:,7]))
        plt.figure(6)
        plt.imshow(im[:,:,8], cmap='gray', vmin=0, vmax=np.max(im[:,:,8]))
        plt.figure(7)
        plt.imshow(im[:,:,9], cmap='gray', vmin=0, vmax=np.max(im[:,:,9]))
        plt.figure(8)
        plt.imshow(im[:,:,3], cmap='gray', vmin=0, vmax=np.max(im[:,:,3]))
        plt.show()