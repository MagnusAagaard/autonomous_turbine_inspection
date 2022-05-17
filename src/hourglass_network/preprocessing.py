import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from skimage import transform as tf

'''
Images are labelled using labelbox (cd labelbox && sudo docker-compose up)
Labelbox is then available at localhost:8080
'''

color_dict_bgr = {'wing_tip': (235, 52, 229),
                  'wing_center': (40, 133, 12),
                  'tower_top': (247, 104, 32),
                  'tower_bottom': (27, 11, 135)}

index_dict = {'wing_tip': 0,
              'wing_center': 1,
              'tower_top': 2,
              'tower_bottom': 3}

def get_translation_transform(trans_x, trans_y):
    '''
    Similarity transform (rotation, translation and scale in one).
    '''
    return np.array([
        [1, 0, trans_x],
        [0, 1, trans_y],
        [0, 0, 1]
    ])

def get_similarity_transform_with_offset(scale, angle, trans_x, trans_y, im_w, im_h):
    '''
    Similarity transform (rotation, translation and scale in one).
    inv(T) @ M @ T gives rotation around center of image instead
    '''
    T = np.array([
        [1, 0, -im_w/2],
        [0, 1, -im_h/2],
        [0, 0, 1]
    ])
    return np.linalg.inv(T) @ np.array([
        [scale*np.cos(angle), -np.sin(angle), trans_x],
        [np.sin(angle), scale*np.cos(angle), trans_y],
        [0, 0, 1]
    ]) @ T
    
def get_similarity_transform_no_offset(scale, angle, trans_x, trans_y):
    '''
    Similarity transform (rotation, translation and scale in one).
    '''
    return np.array([
        [scale*np.cos(angle), -np.sin(angle), trans_x],
        [np.sin(angle), scale*np.cos(angle), trans_y],
        [0, 0, 1]
    ])

def create_simple_input_img(kps, img, sigma):
    '''
    Takes an input image (full image) and keypoints and applies Gaussian kernel with sigma onto the image.
    Returns the 10-channel image used during inference
    Kps: wing_tips, wing_center, tower_top, tower_bottom
    
    '''
    kernel_size = 0    # From OpenCV formula. If set at 0, the kernel size is automatically calculated as 31 with sigma=5 based on sigma and vice versa if sigma = 0
    pt_data = [np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) for i in range(4)]
    line_data = [np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) for i in range(3)]
    input_img = np.zeros((img.shape[0], img.shape[1], 10), dtype=np.float32)
    input_img[:,:,:3] = np.copy(img.astype(np.float32)/255.0)
    
    for kp in kps[:3]:
        if kp[0] > 0 and kp[0] < img.shape[1] and kp[1] > 0 and kp[1] < img.shape[0]:
            pt_data[0][kp[1]-1,kp[0]-1] = 1.0
    
    for i, kp in enumerate(kps[3:]):
        if kp[0] > 0 and kp[0] < img.shape[1] and kp[1] > 0 and kp[1] < img.shape[0]:
            pt_data[i+1][kp[1]-1,kp[0]-1] = 1.0
    # Save labelled point data in numpy array
    for i, pts in enumerate(pt_data):
        input_img[:,:,3+i] = np.copy(pts)
    # Draw sorted keypoints on label image
    # tower_bottom --> tower_top
    cv2.line(line_data[0], kps[5][:2], kps[4][:2], 1.0, 1)
    # tower_top --> wing_center
    cv2.line(line_data[1], kps[4][:2], kps[3][:2], 1.0, 1)
    # wing_center --> wing_tips
    cv2.line(line_data[2], kps[3][:2], kps[2][:2], 1.0, 1)
    cv2.line(line_data[2], kps[3][:2], kps[1][:2], 1.0, 1)
    cv2.line(line_data[2], kps[3][:2], kps[0][:2], 1.0, 1)
    for i, lines in enumerate(line_data):
        input_img[:,:,3+len(pt_data)+i] = np.copy(lines)
    # Apply Gaussian blur on input image and renormalize values 0-1
    for i in range(3, input_img.shape[2]):
        if input_img[:,:,i].max() != 0.0:
            input_img[:,:,i] = cv2.GaussianBlur(input_img[:,:,i], (kernel_size, kernel_size), sigma)
            input_img[:,:,i] *= 1.0/input_img[:,:,i].max()
    return input_img

def create_input_img(kps, img_name, test=False, apply_augmentation=False, sigma_input=20, sigma_label=5):
    '''
    Takes as input the img_name and keypoints to draw on them. Sigma is used for Gaussian smoothing.
    Returns two numpy array of shape (img_shape[0], img_shape[1], 10).
    First is input_img, then label_img
    First three channels are BGR image.
    Next four channels are point data in order: wing_tips, wing_center, tower_top, tower_bottom
    Last three channels are line data in order: tower_bottom --> tower_top, tower_top --> wing_center, wing_center --> wing_tips
    '''
    img_path = './src/hourglass_network/data/all_data/' if not test else './src/hourglass_network/data/test_data/'
    kps.sort(key=lambda x: x[2])
    # Apply data augmentation if true
    if apply_augmentation:
        tmp_img = cv2.imread(img_path + img_name).astype(np.float32)/255.0
        s_range = 0.2
        a_range = np.deg2rad(20)
        #a_range = 0
        # Random vals to determine values within range
        random_vals = np.random.rand(4)
        #s = 1 + (random_vals[0] * s_range) - s_range/2
        s = 1 - random_vals[0] * s_range
        a = random_vals[1] * a_range - a_range/2
        trans_x = 0
        trans_y = 0
        similarity_transform = get_similarity_transform_with_offset(scale=s, angle=a, trans_x=trans_x, trans_y=trans_y, im_w=tmp_img.shape[1], im_h=tmp_img.shape[0])
        #img = tf.warp(tmp_img, similarity_transform)
        img = cv2.warpAffine(tmp_img, similarity_transform[:2,:], (int(tmp_img.shape[1]*s), int(tmp_img.shape[0]*s)))
        # Also transform keypoints..
        xs = [kp[0] for kp in kps]
        ys = [kp[1] for kp in kps]
        pts = np.array((xs, ys))
        pts = np.vstack((pts, np.ones(shape=(1,pts.shape[1]))))
        pts = similarity_transform @ pts
        for i, kp in enumerate(kps):
            kp[0] = round(pts[0,i])
            kp[1] = round(pts[1,i])
        # Now translate to center around kps.. + some deviation such that features always in view
        min_x = 1000
        min_y = 1000
        max_x = 0
        max_y = 0
        for x in xs:
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
        for y in ys:
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        if min_y < 0:
            min_y = 0
        if min_x < 0:
            min_x = 0
        if max_x > img.shape[1]:
            max_x = img.shape[1]
        if max_y > img.shape[0]:
            max_y = img.shape[0]
        trans_range_x = max_x - min_x
        trans_range_y = max_y - min_y
        xc = int((max_x + min_x)/2 + (random_vals[2] * trans_range_x - trans_range_x/2))
        yc = int((max_y + min_y)/2 + (random_vals[3] * trans_range_y - trans_range_y/2))
        
        #trans_range_x = int(img.shape[1]*0.1)
        #trans_range_y = int(img.shape[0]*0.1)
        #trans_range_x = int(img.shape[1])
        #trans_range_y = int(img.shape[0])
        #xc = int(sum([kp[0] for kp in kps[3:]])/3) #+ (random_vals[2] * trans_range_x - trans_range_x/2))
        #yc = int(sum([kp[1] for kp in kps[3:]])/3) #+ (random_vals[3] * trans_range_y - trans_range_y/2))
        #xc = int(kps[1][0] + (random_vals[2] * trans_range_x - trans_range_x/2))
        #yc = int(kps[1][1] + (random_vals[3] * trans_range_y - trans_range_y/2))
        if xc < 0:
            xc = 0
        if yc < 0:
            yc = 0
        if xc >= img.shape[1]:
            xc = img.shape[1] - 1
        if yc >= img.shape[0]:
            yc = img.shape[0] - 1
        #NOTE: change to crop size
        if xc-256/2 < 0:
            xc += 256/2 - xc
        elif xc+256/2 > img.shape[1]:
            xc -= xc+256/2 - img.shape[1]
        if yc-256/2 < 0:
            yc += 256/2 - yc
        elif yc+256/2 > img.shape[0]:
            yc -= yc+256/2 - img.shape[0]
        trans_x = xc - img.shape[1]/2
        trans_y = yc - img.shape[0]/2
        # Redo transform
        translation = get_translation_transform(-trans_x, -trans_y)
        #cv2.imshow('Imgbf', img)
        img = cv2.warpAffine(img, translation[:2,:], (img.shape[1], img.shape[0]))
        #cv2.imshow('imgaf', img)
        # Also transform keypoints..
        xs = [kp[0] for kp in kps]
        ys = [kp[1] for kp in kps]
        pts = np.array((xs, ys))
        pts = np.vstack((pts, np.ones(shape=(1,pts.shape[1]))))
        pts = translation @ pts
        for i, kp in enumerate(kps):
            kp[0] = round(pts[0,i])
            kp[1] = round(pts[1,i])
        #show_keypoints_on_img(kps, img, show=True)
    else:
        img = cv2.imread(img_path + img_name).astype(np.float32)/255.0
    #show_keypoints_on_img(kps, img, show=True)
    kernel_size = 0    # From OpenCV formula. If set at 0, the kernel size is automatically calculated as 31 with sigma=5 based on sigma and vice versa if sigma = 0
    # Random affine transform applied to input_img/prior (10 pixels max)
    #sx = 10./img.shape[1]
    #sy = 10./img.shape[0]
    if not test:
        sx = 0.1
        sy = 0.1
    else:
        sx = 0.0
        sy = 0.0
    transform = transforms.RandomAffine(degrees=2, translate=(sx, sy))
    
    # Data variables
    pt_data = [np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) for i in range(4)]
    line_data = [np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) for i in range(3)]
    line_data_input = [np.zeros((img.shape[0], img.shape[1]), dtype=np.float32) for i in range(3)]
    label_img = np.zeros((img.shape[0], img.shape[1], 10), dtype=np.float32)
    label_img[:,:,:3] = np.copy(img)
    input_img = np.copy(label_img)
    # Housekeeper variable for tmp keypoints
    out_of_bound_kps = []
    
    for kp in kps:
        if kp[2].find('tmp') == -1 and kp[0] > 0 and kp[0] < img.shape[1] and kp[1] > 0 and kp[1] < img.shape[0]:
            #print(kp)
            #print(index_dict.get(kp[2]))
            pt_data[index_dict.get(kp[2])][kp[1]-1,kp[0]-1] = 1.0
        else:
            out_of_bound_kps.append(kp)
    # Save labelled point data in numpy array
    for i, pts in enumerate(pt_data):
        label_img[:,:,3+i] = np.copy(pts)
    # Draw sorted keypoints on label image
    # tower_bottom --> tower_top
    cv2.line(line_data[0], kps[0][:2], kps[1][:2], 1.0, 1)
    # tower_top --> wing_center
    cv2.line(line_data[1], kps[1][:2], kps[2][:2], 1.0, 1)
    # wing_center --> wing_tips
    cv2.line(line_data[2], kps[2][:2], kps[3][:2], 1.0, 1)
    cv2.line(line_data[2], kps[2][:2], kps[4][:2], 1.0, 1)
    cv2.line(line_data[2], kps[2][:2], kps[5][:2], 1.0, 1)
    for i, lines in enumerate(line_data):
        label_img[:,:,3+len(pt_data)+i] = np.copy(lines)
    
    # Copy label image --> Input image and apply random affine transforms to landmark points
    landmarks = np.copy(label_img[:,:,3:]).transpose(2,0,1)
    tensor = torch.from_numpy(landmarks)
    tensor = transform(tensor)
    landmarks = tensor.numpy().transpose(1,2,0)
    input_img[:,:,3:] = np.copy(landmarks)
    
    # Draw lines on input image
    # Find indicies for where keypoints are transformed to
    #kp_input = np.argwhere(input_img[:,:,3:] == 1.0)
    #
    #if out_of_bound_kps:
    #    for pt in out_of_bound_kps:
    #        kp_input = np.vstack((kp_input, np.array([[pt[1], pt[0], index_dict.get(pt[2][:-4])]])))
    # Swap rows and columns for line drawing
    #kp_input[:, [1, 0]] = kp_input[:, [0, 1]]
    
    # Apply Gaussian blur on label image and renormalize values 0-1
    for i in range(3, label_img.shape[2]):
        if label_img[:,:,i].max() != 0.0:
            label_img[:,:,i] = cv2.GaussianBlur(label_img[:,:,i], (kernel_size, kernel_size), sigma_label)
            label_img[:,:,i] *= 1.0/label_img[:,:,i].max()
    # Apply Gaussian blur on input image and renormalize values 0-1
    for i in range(3, input_img.shape[2]):
        if input_img[:,:,i].max() != 0.0:
            input_img[:,:,i] = cv2.GaussianBlur(input_img[:,:,i], (kernel_size, kernel_size), sigma_input)
            input_img[:,:,i] *= 1.0/input_img[:,:,i].max()
    
    #plt.figure(1)
    #plt.imshow(np.sum(label_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    #plt.figure(2)
    #plt.imshow(np.sum(input_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
    #cv2.imshow('img',img)
    #plt.show()
    return input_img, label_img

def show_keypoints_on_img(kps, img, show=False):
    #img = cv2.imread(f'./src/hourglass_network/data/all_data/{img_name}')
    kps.sort(key=lambda x: x[2])
    for kp in kps:
        if kp[2].find('tmp') == -1:
            cv2.circle(img, kp[:2], 3, color_dict_bgr.get(kp[2]), -1)
        #else:
        #    show = True
    # Keypoints are sorted
    # tower_bottom --> tower_top
    cv2.line(img, kps[0][:2], kps[1][:2], (1,0,0), 2)
    # tower_top --> wing_center
    cv2.line(img, kps[1][:2], kps[2][:2], (0, 1, 0), 2)
    # wing_center --> wing_tips
    cv2.line(img, kps[2][:2], kps[3][:2], (0, 0, 1), 2)
    cv2.line(img, kps[2][:2], kps[4][:2], (0, 0, 1), 2)
    cv2.line(img, kps[2][:2], kps[5][:2], (0, 0, 1), 2)
    if show:
        print('Why are we here?')
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        
def get_annotations(path):
    '''
    Opens annotations and returns them.
    '''
    with open(path, 'rb') as f:
        annotations = json.load(f)
    return annotations

def process_annotations(img, apply_augmentation=False):
    '''
    Processes the annotations loaded with get_annotations().
    '''
    img_name = get_img_name(img)
    kps = get_kps(img)
    #show_keypoints_on_img(kps, img_name)
    test = False
    if img_name.find('test') != -1:
        test = True
    return create_input_img(kps, img_name, test=test, apply_augmentation=apply_augmentation)

def get_img_name(img):
    return img.get('img').split('-')[1]

def get_kps(img):
    kps = []
    for kp in img.get('kp-1'):
        width = float(kp.get('original_width')) / 100
        height = float(kp.get('original_height')) / 100
        x = float(kp.get('x')) * width
        y = float(kp.get('y')) * height
        kp_label = kp.get('keypointlabels')[0]
        kps.append([int(x), int(y), kp_label])
    return kps

def main():
    with open('./src/hourglass_network/data/annotations.json', 'rb') as f:
        annotations = json.load(f)
    for img in annotations:
        img_name = img.get('img').split('-')[1]
        kps = []
        for kp in img.get('kp-1'):
            width = float(kp.get('original_width')) / 100
            height = float(kp.get('original_height')) / 100
            x = float(kp.get('x')) * width
            y = float(kp.get('y')) * height
            kp_label = kp.get('keypointlabels')[0]
            kps.append([int(x), int(y), kp_label])
        #show_keypoints_on_img(kps, img_name)
        input_img, label_img = create_input_img(kps, img_name)
    
    #annotations = get_annotations()
    #for img in annotations:
    #    input_img, label_img = process_annotations(img)
    

if __name__ == "__main__":
    main()