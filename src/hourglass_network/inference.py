import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, RandomCrop
from torch.autograd import Variable

from model import ConvEncoderDecoder
import preprocessing

pt_threshold = 0.1

def get_wing_tips(img, original_image_dims):
    pts = []
    img_dim_y = original_image_dims[0]
    img_dim_x = original_image_dims[1]
    wing_tips = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, wing_tips = cv2.threshold(wing_tips, 10,255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=wing_tips, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours_to_keep = sorted(contours, key=cv2.contourArea)[-3:]
    for c in contours_to_keep:
        x1, y1, x2, y2 = cv2.boundingRect(c)
        #cv2.rectangle(img, (x1,y1),(x1+x2,y1+y2), 0.5, 2)
        _,max_val,_, pt = cv2.minMaxLoc(img[y1:y1+y2,x1:x1+x2])
        if max_val > pt_threshold:
            #cv2.imshow('Test', img[y1:y1+y2, x1:x1+x2])
            #cv2.waitKey(0)
            x = int((pt[0] + x1 + (img_dim_x*256/img_dim_y - 256)/2) * img_dim_y/256)
            y = int((pt[1] + y1) * img_dim_y/256)
            pts.append([x,y])
    #cv2.imshow('Wing', img)
    #cv2.waitKey(0)
    return pts

def get_pt_from_heatmap(img, original_image_dims):
    img_dim_y = original_image_dims[0]
    img_dim_x = original_image_dims[1]
    _,max_val,_,pt = cv2.minMaxLoc(img)
    #print(max_val, pt)
    if max_val > pt_threshold:
        x = int((pt[0]+ (img_dim_x*256/img_dim_y - 256)/2) * img_dim_y/256)
        y = int(pt[1] * img_dim_y/256)
        return [x,y]
    return None

def main():
    model = ConvEncoderDecoder(10)
    # Load model
    checkpoint = torch.load('./src/hourglass_network/checkpoints/run2/model_best.pt')
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model. Number of epochs: {}'.format(checkpoint['epoch']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # Get input image
    annotations = preprocessing.get_annotations('./src/hourglass_network/data/annotations_test.json')
    annotation_idx = 3
    img_name = preprocessing.get_img_name(annotations[annotation_idx])
    kps = preprocessing.get_kps(annotations[annotation_idx])
    test_img = cv2.imread(f'./src/hourglass_network/data/test_data/{img_name}')
    img_dim_x = test_img.shape[1]
    img_dim_y = test_img.shape[0]
    preprocessing.show_keypoints_on_img(kps, test_img, show=True)
    input_img, label_img = preprocessing.process_annotations(annotations[annotation_idx])
    img = input_img[:,:,:3].copy()
    plt.figure(1)
    plt.imshow(np.sum(label_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    plt.figure(2)
    plt.imshow(np.sum(input_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    
    
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
    cropped_img = cropped_img_data[:,:,:3].copy()
    
    # Show output
    plt.figure(3)
    plt.imshow(np.sum(output[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=np.sum(output[:,:,3:].max()))
    output_img = output[:,:,:3]
    # Points
    wing_tips = output[:,:,3]
    wing_center = output[:,:,4]
    tower_top = output[:,:,5]
    tower_bottom = output[:,:,6]
    tower_bottom_to_tower_top = output[:,:,7]
    tower_top_to_wing_center = output[:,:,8]
    wing_center_to_wing_tips = output[:,:,9]
    
    wing_tip_pts = get_wing_tips(wing_tips, original_image_dims=test_img.shape)
    wing_center_pt = get_pt_from_heatmap(wing_center, original_image_dims=test_img.shape)
    tower_top_pt = get_pt_from_heatmap(tower_top, original_image_dims=test_img.shape)
    tower_bottom_pt = get_pt_from_heatmap(tower_bottom, original_image_dims=test_img.shape)
    for pt in wing_tip_pts:
        cv2.circle(img, pt, 5, (0,1,0), -1)
    if wing_center_pt:
        cv2.circle(img, wing_center_pt, 5, (1,0,0), -1)
    if tower_top_pt:
        cv2.circle(img, tower_top_pt, 5, (0,0,1), -1)
    if tower_bottom_pt:
        cv2.circle(img, tower_bottom_pt, 5, (1,1,0), -1)
    plt.figure(4)
    plt.imshow(wing_tips, cmap='gray')
    plt.figure(5)
    plt.imshow(wing_center, cmap='gray')
    
    #plt.imshow(np.sum(cropped_img_data[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    cv2.imshow('img',img)
    cv2.imshow('Output_img', output_img)
    cv2.imshow('Cropped img', cropped_img)
    
    plt.show()

if __name__ == "__main__":
    main()