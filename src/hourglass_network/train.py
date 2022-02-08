import os
import sys
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop
import numpy as np
import argparse
import shutil

from model import ConvEncoderDecoder
from dataloader import WindturbineDataset

def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--save', type=int, default=5, help='save checkpoint of model every x epoch')
        parser.add_argument('-e', '--epochs', type=int, default=100, help='max number of epochs')
        parser.add_argument('-r', '--resume', type=bool, default=False, help='whether to resume training from a checkpoint')
        parser.add_argument('-b', '--base_dir', type=str, 
                            default='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network',
                            help='base directory of model code')
        args = parser.parse_args()
        return args

class Trainer:
    def __init__(self, args):
        self.num_epochs = args.epochs
        self.save_interval = args.save
        self.resume = args.resume
        self.base_dir = args.base_dir
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvEncoderDecoder(10)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = BCELoss()
        self.epoch = 0
        if self.resume:
            # Reload checkpoint
            self.load_model()
        # Move model to GPU if available
        self.model.to(self.device)
    
    def load_model(self):
        # Load saved checkpoint
        checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            print(f'Checkpoint dir does not exist at {checkpoint_dir}, so can\'t resume training.')
            sys.exit(-1)
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print(f'No checkpoint file found at {checkpoint_file}')
            sys.exit(-1)
        print(f'Loading checkpoint {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        print('Loaded checkpoint!')
        
    def save_checkpoint(self, state, is_best, filename='checkpoint.pt'):
        """
        from pytorch/examples
        """
        basename = self.base_dir
        if not os.path.exists(basename):
            os.makedirs(basename)
        filename_loc = os.path.join(basename, filename)
        torch.save(state, filename_loc)
        if is_best:
            shutil.copyfile(filename_loc, 'model_best.pt')
            
    def save(self, is_best):
        print('Saving checkpoint..')
        self.save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch}, is_best)
        
    def train(self):
        # Run trainer
        train_dataset = WindturbineDataset('./src/hourglass_network/data/annotations.json', './src/hourglass_network/data/all_data', transform=Compose([ToTensor(), RandomCrop(256)]))
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        n_epochs = 10

        for epoch in range(1, n_epochs+1):
            # monitor training loss
            train_loss = 0.0

            #Training
            for data in tqdm(train_dataloader):
                input_images, label_images = data
                input_images = input_images.to(self.device)
                label_images = label_images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_images)
                loss = self.criterion(outputs, label_images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*input_images.size(0)
                
            train_loss = train_loss/len(train_dataloader)
            tqdm.write('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        self.save(is_best=True)

def main():
    args = parse_command_line()
    trainer = Trainer(args)
    trainer.train()
    

if __name__ == "__main__":
    main()