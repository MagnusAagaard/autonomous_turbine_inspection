import os
import sys
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize
import numpy as np
import argparse
import shutil

from model import ConvEncoderDecoder
from dataloader import WindturbineDataset

def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--save', type=int, default=5, help='save checkpoint of model every x epoch')
        parser.add_argument('-e', '--epochs', type=int, default=150, help='max number of epochs')
        parser.add_argument('-r', '--resume', type=int, default=0, help='whether to resume training from a checkpoint. Provide epoch number.')
        parser.add_argument('-b', '--base_dir', type=str, default='./src/hourglass_network', help='base directory of model code')
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
        # Move model to GPU if available
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = BCELoss()
        self.epoch = 0
        self.lowest_loss = 100
        if self.resume:
            # Reload checkpoint
            self.load_model(self.resume)
    
    def load_model(self, epoch):
        # Load saved checkpoint
        checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            print(f'Checkpoint dir does not exist at {checkpoint_dir}, so can\'t resume training.')
            sys.exit(-1)
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
        if not os.path.isfile(checkpoint_file):
            print(f'No checkpoint file found at {checkpoint_file}')
            sys.exit(-1)
        print(f'Loading checkpoint {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        print('Loaded checkpoint!')
        
    def save_checkpoint(self, state, is_best):
        """
        from pytorch/examples
        """
        basename = os.path.join(self.base_dir, 'checkpoints')
        if not os.path.exists(basename):
            os.makedirs(basename)
        filename_loc = os.path.join(basename, f'checkpoint_{self.epoch}.pt')
        torch.save(state, filename_loc)
        if is_best:
            best_filename_loc = os.path.join(basename, 'model_best.pt')
            shutil.copyfile(filename_loc, best_filename_loc)
            
    def save(self, is_best):
        print(f'Saving checkpoint for epoch {self.epoch}..')
        self.save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'loss': self.lowest_loss}, is_best)
        
    def train(self):
        # Run trainer
        train_dataset = WindturbineDataset(f'{self.base_dir}/data/annotations.json', f'{self.base_dir}/data/all_data', transform=Compose([ToTensor(), Resize((256, 256))]))
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        for self.epoch in tqdm(range(self.epoch+1, self.num_epochs+1)):
            # monitor training loss
            train_loss = 0.0
            best = False
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
            tqdm.write('Epoch: {} \tTraining Loss: {:.6f}'.format(self.epoch, train_loss))
            if self.epoch % self.save_interval == 0:
                if train_loss < self.lowest_loss:
                    self.lowest_loss = train_loss
                    best = True
                self.save(is_best=best)

def main():
    args = parse_command_line()
    trainer = Trainer(args)
    trainer.train()
    

if __name__ == "__main__":
    main()
