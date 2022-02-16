import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, RandomCrop
from torch.autograd import Variable

from model import ConvEncoderDecoder
import preprocessing


def show_output(self):
    '''
    Show the output from the network, RGB image overlayed with point and line data
    '''

def main():
    model = ConvEncoderDecoder(10)
    # Load model
    checkpoint = torch.load('./src/hourglass_network/checkpoints/run1/model_best.pt')
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model. Number of epochs: {}'.format(checkpoint['epoch']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # Get input image
    annotations = preprocessing.get_annotations('./src/hourglass_network/data/annotations_test.json')
    img_name = preprocessing.get_img_name(annotations[0])
    kps = preprocessing.get_kps(annotations[0])
    test_img = cv2.imread(f'./src/hourglass_network/data/test_data/{img_name}')
    preprocessing.show_keypoints_on_img(kps, test_img, show=True)
    input_img, label_img = preprocessing.process_annotations(annotations[0])
    img = input_img[:,:,:3]
    plt.figure(1)
    plt.imshow(np.sum(label_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    plt.figure(2)
    plt.imshow(np.sum(input_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    cv2.imshow('img',img)
    
    # Run inference
    with torch.no_grad():
        transform = Compose([ToTensor(), Resize(256), CenterCrop(256)])
        cropped_input_img = transform(input_img)
        # Expand dim such that shape is now (B, C, H, W) from (C, H, W)
        cropped_input_img = torch.unsqueeze(cropped_input_img,0)
        cropped_input_img = Variable(cropped_input_img.to(device))
        output = model(cropped_input_img)
        # Remove expanded dim, move to cpu and numpyfi
        output = torch.squeeze(output).cpu().numpy().transpose(1,2,0)
    cropped_img_data = torch.squeeze(cropped_input_img).cpu().numpy().transpose(1,2,0)
    cropped_img = cropped_img_data[:,:,:3]
    cv2.imshow('Cropped img', cropped_img)
    plt.figure(3)
    plt.imshow(np.sum(cropped_img_data[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    
    # Show output
    output_img = output[:,:,:3]
    cv2.imshow('Output_img', output_img)
    plt.figure(4)
    plt.imshow(np.sum(output[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=np.sum(output[:,:,3:].max()))
    
    plt.show()

if __name__ == "__main__":
    main()