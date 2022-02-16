import os
import sys
from tqdm import tqdm
import numpy as np
import argparse
import shutil

import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize, CenterCrop
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from model import ConvEncoderDecoder
from dataloader import WindturbineDataset

def parse_command_line():
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epochs', type=int, default=1000, help='max number of epochs')
        parser.add_argument('-r', '--resume', type=bool, default=False, help='whether to resume training from a checkpoint (using model_best.pt)')
        parser.add_argument('-b', '--base_dir', type=str, default='./src/hourglass_network', help='base directory of model code')
        parser.add_argument('-v', '--validate', type=int, default=2, help='number of epochs between model validation. Also saves best model when validating.')
        args = parser.parse_args()
        return args

class Trainer:
    def __init__(self, args):
        self.num_epochs = args.epochs
        self.validate_interval = args.validate
        self.resume = args.resume
        self.base_dir = args.base_dir
        # TensorBoard writer
        self.writer = SummaryWriter(self.base_dir + '/runs/augmentation_experiment_2')
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
        checkpoint_file = os.path.join(checkpoint_dir, 'model_best.pt')
        if not os.path.isfile(checkpoint_file):
            print(f'No checkpoint file found at {checkpoint_file}')
            sys.exit(-1)
        print(f'Loading checkpoint {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.lowest_loss = checkpoint['loss']
        print(f'Loaded checkpoint from epoch {self.epoch} with a loss of {self.lowest_loss}!')
        
    def save_checkpoint(self, state, is_best):
        """
        from pytorch/examples
        """
        basename = os.path.join(self.base_dir, 'checkpoints')
        if not os.path.exists(basename):
            os.makedirs(basename)
        filename_loc = os.path.join(basename, f'checkpoint_{self.epoch}.pt')
        if is_best:
            #torch.save(state, filename_loc)
            best_filename_loc = os.path.join(basename, 'model_best.pt')
            torch.save(state, best_filename_loc)
            #shutil.copyfile(filename_loc, best_filename_loc)
            
    def save(self, is_best):
        #print(f'Saving checkpoint for epoch {self.epoch}..')
        self.save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'loss': self.lowest_loss}, is_best)
        
    # helper function to plot images to TensorBoard
    def plot_preds(self, images):
        outputs = self.model(images)
        fig = plt.figure(figsize=(15, 15))
        for idx in np.arange(images.shape[0]):
            # Remove expanded dim, move to cpu and numpyfi
            output = torch.squeeze(outputs[idx]).cpu().numpy().transpose(1,2,0)
            input_img_data = torch.squeeze(images[idx]).cpu().numpy().transpose(1,2,0)
            ax = fig.add_subplot(images.shape[0],4,4*idx+1,xticks=[],yticks=[])
            input_img = input_img_data[:,:,:3]
            plt.imshow(input_img[...,::-1])
            ax.set_title("Input")
            ax = fig.add_subplot(images.shape[0],4,4*idx+2,xticks=[],yticks=[])
            output_img = output[:,:,:3]
            plt.imshow(output_img[...,::-1])
            ax.set_title("Output")
            ax = fig.add_subplot(images.shape[0],4,4*idx+3,xticks=[],yticks=[])
            plt.imshow(np.sum(output[:,:,3:7], axis=2), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title("Points")
            ax = fig.add_subplot(images.shape[0],4,4*idx+4,xticks=[],yticks=[])
            plt.imshow(np.sum(output[:,:,7:], axis=2), cmap='gray', vmin=0, vmax=1.0)
            ax.set_title("Lines")
        return fig
    
    def evaluate(self):
        # Evaluate model and return validation loss
        val_loss = 0.0
        img_plotted = False
        self.model.eval()
        self.val_set.dataset.train = False
        with torch.no_grad():
            for data in self.val_dataloader:
                input_images, label_images = data
                input_images = input_images.to(self.device)
                label_images = label_images.to(self.device)
                outputs = self.model(input_images)
                loss = self.criterion(outputs, label_images)
                val_loss += loss.item()*input_images.size(0)
                if not img_plotted:
                    fig = self.plot_preds(input_images[:4])
                    self.writer.add_figure('validation figures', fig, self.epoch)
                    img_plotted = True
            
        val_loss = val_loss/len(self.val_dataloader)
        self.writer.add_scalar('validation loss', val_loss, self.epoch)
        tqdm.write('Epoch: {} \tValidation Loss: {:.6f}'.format(self.epoch, val_loss))
        return val_loss
        
    def train(self):
        # Run trainer
        dataset = WindturbineDataset(f'{self.base_dir}/data/annotations_125.json', f'{self.base_dir}/data/all_data', train_transform=Compose([ToTensor(), CenterCrop(256)]), test_transform=Compose([ToTensor(), Resize(256), CenterCrop(256)]))
        train_val_split = int(len(dataset)*0.8)
        self.train_set, self.val_set = random_split(dataset, [train_val_split, len(dataset)-train_val_split], generator=torch.Generator().manual_seed(42))
        print(f'Length of train dataset: {len(self.train_set)} \t Length of val dataset: {len(self.val_set)}')
        self.train_dataloader = DataLoader(self.train_set, batch_size=16, num_workers=6, shuffle=True)
        self.val_dataloader = DataLoader(self.val_set, batch_size=16, num_workers=2, shuffle=False)
        
        # Write model to TensorBoard
        input_to_model, _ = next(iter(self.train_set))
        input_to_model = torch.unsqueeze(input_to_model, 0).to(self.device)
        self.writer.add_graph(self.model, input_to_model)
        
        for self.epoch in tqdm(range(self.epoch+1, self.num_epochs+1)):
            # monitor training loss
            train_loss = 0.0
            self.model.train()
            self.train_set.dataset.train = True
            best = False
            #Training
            for data in self.train_dataloader:
                input_images, label_images = data
                input_images = input_images.to(self.device)
                label_images = label_images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_images)
                loss = self.criterion(outputs, label_images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()*input_images.size(0)
                
            train_loss = train_loss/len(self.train_dataloader)
            self.writer.add_scalar('training loss', train_loss, self.epoch)
            tqdm.write('Epoch: {} \tTraining Loss: {:.6f}'.format(self.epoch, train_loss))
            if self.epoch % self.validate_interval == 0:
                val_loss = self.evaluate()
                if val_loss < self.lowest_loss:
                    tqdm.write(f'New best model with loss: {val_loss} - Saving checkpoint for epoch {self.epoch}')
                    self.lowest_loss = val_loss
                    best = True
                    self.save(is_best=best)

def main():
    args = parse_command_line()
    trainer = Trainer(args)
    trainer.train()
    

if __name__ == "__main__":
    main()
