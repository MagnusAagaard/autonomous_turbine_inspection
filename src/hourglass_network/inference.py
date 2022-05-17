import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, RandomCrop
from torch.autograd import Variable

from hourglass_network.model import ConvEncoderDecoder, ConvEncoderDecoderV2, ConvEncoderDecoderCor
from hourglass_network import preprocessing
from skeletal_turbine_model import SkeletalTurbineModel
import timeit

class Inference:
    def __init__(self, model_path, version='v1'):
        self.pt_threshold = 0.1 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device available for inference: {self.device}')
        self.version = version
        if version == 'v1':
            self.model = ConvEncoderDecoder(10, extra_layer=False)
        elif version == 'v1e':
            self.model = ConvEncoderDecoder(10, extra_layer=True)
        elif version == 'v1c':
            self.model = ConvEncoderDecoderCor(10, extra_layer=False)
        elif version == 'v1old':
            self.model = ConvEncoderDecoder(10, extra_layer=False, old_version=True)
        elif version == 'v2':
            self.model = ConvEncoderDecoderV2(10)
        else:
            print('Inferece model version invalid!')
        self.__load_model(model_path)
        
    def __load_model(self, model_path):
        # Load model
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print('Loaded model. Number of epochs: {}'.format(checkpoint['epoch']))
        
    def test_timing(self, annotation_idx):
        annotations = preprocessing.get_annotations('./src/hourglass_network/data/annotations_test.json')
        img_name = preprocessing.get_img_name(annotations[annotation_idx])
        kps = preprocessing.get_kps(annotations[annotation_idx])
        test_img = cv2.imread(f'./src/hourglass_network/data/test_data/{img_name}')
        print('inference\t\t', timeit.timeit(lambda: self.forward(test_img, kps), number=300) / 300)
        
    def run_test(self, annotation_idx):
        # Get annotations and load image + keypoints
        annotations = preprocessing.get_annotations('./src/hourglass_network/data/annotations_test.json')
        img_name = preprocessing.get_img_name(annotations[annotation_idx])
        kps = preprocessing.get_kps(annotations[annotation_idx])
        kps.sort(key=lambda x: x[2])
        test_img = cv2.imread(f'./src/hourglass_network/data/test_data/{img_name}')
        # Show keypoints on image
        #preprocessing.show_keypoints_on_img(kps, test_img, show=True)
        # Get input image for network
        input_img, label_img = preprocessing.process_annotations(annotations[annotation_idx])
        img = input_img[:,:,:3].copy()
        # Show label image points + lines
        #plt.figure(1)
        #plt.imshow(np.sum(label_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
        # Show input image points + lines (more Gaussian blur)
        #plt.figure(2)
        #plt.imshow(np.sum(input_img[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=1.0)
        # Run inference
        output, cropped_input_img = self.forward_test(input_img)
        
        cropped_img = cropped_input_img[:,:,:3].copy()
        # Show output
        plt.figure(3)
        #plt.imshow(np.sum(output[:,:,3:], axis=2), cmap='gray', vmin=0, vmax=np.sum(output[:,:,3:].max()))
        plt.imshow(np.sum(output, axis=2), cmap='gray', vmin=0, vmax=np.sum(output).max())
        output_img = output[:,:,:3]
        # Points
        if self.version == 'v1old':
            wing_tips = output[:,:,3]
            wing_center = output[:,:,4]
            tower_top = output[:,:,5]
            tower_bottom = output[:,:,6]
            # Lines
            tower_bottom_to_tower_top = output[:,:,7]
            tower_top_to_wing_center = output[:,:,8]
            wing_center_to_wing_tips = output[:,:,9]
        else:
            wing_tips = output[:,:,0]
            wing_center = output[:,:,1]
            tower_top = output[:,:,2]
            tower_bottom = output[:,:,3]
            # Lines
            tower_bottom_to_tower_top = output[:,:,4]
            tower_top_to_wing_center = output[:,:,5]
            wing_center_to_wing_tips = output[:,:,6]
        #TODO: Convert to PIL image and save to pdf?
        #cv2.imshow('label_lines', label_img[:,80:560,7:])
        cv2.imshow('label_lines', label_img[:,:,7:])
        #cv2.imshow('label_pts', label_img[:,:,::-1][:,80:560,4:7])
        cv2.imshow('label_pts', label_img[:,:,::-1][:,:,4:7])
        #cv2.imshow('input_lines', cropped_input_img[:,:,7:])
        #cv2.imshow('input_pts', cropped_input_img[:,:,::-1][:,:,4:7])
        cv2.imshow('input_lines', input_img[:,:,7:])
        cv2.imshow('input_pts', input_img[:,:,::-1][:,:,4:7])
        if self.version == 'v1old':
            cv2.imshow('output_lines', output[:,:,7:])
        else:
            cv2.imshow('output_lines', output[:,:,4:])
        cv2.imshow('output_pts', output[:,:,::-1][:,:,4:7])
        #template = Image.fromarray(vis_img)
        #template.save('/home/magnus/chamfer_matcher_dist_img.pdf')
        tmp_img = Image.fromarray(test_img[:,:,::-1])
        tmp_img.save('/home/magnus/test_img.pdf')
        tmp_label_lines = Image.fromarray((label_img[:,:,7:][:,:,::-1]*255/label_img[:,:,7:][:,:,::-1].max()).astype(np.uint8))
        tmp_label_lines.save('/home/magnus/label_lines.pdf')
        tmp_label_pts = Image.fromarray((label_img[:,:,::-1][:,:,4:7][:,:,::-1]*255/label_img[:,:,::-1][:,:,4:7][:,:,::-1].max()).astype(np.uint8))
        tmp_label_pts.save('/home/magnus/label_pts.pdf')
        tmp_input_lines = Image.fromarray((input_img[:,:,7:][:,:,::-1]*255/input_img[:,:,7:][:,:,::-1].max()).astype(np.uint8))
        tmp_input_lines.save('/home/magnus/input_lines.pdf')
        tmp_input_pts = Image.fromarray((input_img[:,:,::-1][:,:,4:7][:,:,::-1]*255/input_img[:,:,::-1][:,:,4:7][:,:,::-1].max()).astype(np.uint8))
        tmp_input_pts.save('/home/magnus/input_pts.pdf')
        tmp_output_pts = Image.fromarray((output[:,:,::-1][:,:,4:7][:,:,::-1]*255/output[:,:,::-1][:,:,4:7][:,:,::-1].max()).astype(np.uint8))
        tmp_output_pts.save('/home/magnus/output_pts.pdf')
        if self.version == 'v1old':
            tmp_output_lines = Image.fromarray((output[:,:,7:][:,:,::-1]*255/output[:,:,7:][:,:,::-1].max()).astype(np.uint8))
        else:
            tmp_output_lines = Image.fromarray((output[:,:,4:][:,:,::-1]*255/output[:,:,4:][:,:,::-1].max()).astype(np.uint8))
        tmp_output_lines.save('/home/magnus/output_lines.pdf')
        
        
        upscale = True
        vis_img = img.copy()
        vis_img2 = img.copy()
        # Project points to image
        #wing_tip_pts = self.get_wing_tips(wing_tips, original_image_dims=test_img.shape, upscale=upscale)
        #wing_center_pt = self.get_pt_from_heatmap(wing_center, original_image_dims=test_img.shape, upscale=upscale)
        #tower_top_pt = self.get_pt_from_heatmap(tower_top, original_image_dims=test_img.shape, upscale=upscale)
        #tower_bottom_pt = self.get_pt_from_heatmap(tower_bottom, original_image_dims=test_img.shape, upscale=upscale)
        wing_tip_pts = []
        wing_tip_pts.append(self.get_pt_from_heatmap_within_radius(wing_tips, self.downscale_pt(kps[5][:2],test_img.shape), 10, test_img.shape, threshold = 0.1, upscale=upscale))
        wing_tip_pts.append(self.get_pt_from_heatmap_within_radius(wing_tips, self.downscale_pt(kps[4][:2],test_img.shape), 10, test_img.shape, threshold = 0.1, upscale=upscale))
        wing_tip_pts.append(self.get_pt_from_heatmap_within_radius(wing_tips, self.downscale_pt(kps[3][:2],test_img.shape), 10, test_img.shape, threshold = 0.1, upscale=upscale))
        wing_center_pt = self.get_pt_from_heatmap_within_radius(wing_center, self.downscale_pt(kps[2][:2],test_img.shape), 10, test_img.shape, threshold = 0.2, upscale=upscale)
        tower_top_pt = self.get_pt_from_heatmap_within_radius(tower_top, self.downscale_pt(kps[1][:2],test_img.shape), 10, test_img.shape, threshold = 0.2, upscale=upscale)
        tower_bottom_pt = self.get_pt_from_heatmap_within_radius(tower_bottom, self.downscale_pt(kps[0][:2],test_img.shape), 10, test_img.shape, threshold = 0.2, upscale=upscale)
        for pt in wing_tip_pts:
            if pt[0] != -1 and pt[1] != -1:
                cv2.circle(vis_img, pt, 5, (0,1,0), -1)
        if wing_center_pt:
            if wing_center_pt[0] != -1 and wing_center_pt[1] != -1:
                cv2.circle(vis_img, wing_center_pt, 5, (1,0,0), -1)
        if tower_top_pt:
            if tower_top_pt[0] != -1 and tower_top_pt[1] != -1:
                cv2.circle(vis_img, tower_top_pt, 5, (0,0,1), -1)
        if tower_bottom_pt:
            if tower_bottom_pt[0] != -1 and tower_bottom_pt[1] != -1:
                cv2.circle(vis_img, tower_bottom_pt, 5, (1,1,0), -1)
        lines = self.project_lines_to_image(kps, vis_img2, tower_bottom_to_tower_top, tower_top_to_wing_center, wing_center_to_wing_tips)
        for line in lines:
            for pt in line:
                if pt[0] != -1 and pt[1] != -1:
                    cv2.circle(vis_img, (pt[0], pt[1]), 10, (1,0,1), 5)
                    
        # Input points to find radius used to detect points..
        #max_pts = self.get_wing_tips(wing_tips, original_image_dims=test_img.shape, upscale=upscale)
        #test_pts = self.get_wing_tips(wing_tips, original_image_dims=test_img.shape, upscale=False)
        #print(max_pts)
        
        #for pt in test_pts:
        #    r = 10 + int((300-100)/50)
        #    test_pt = self.get_pt_from_heatmap_within_radius(wing_tips, pt, r, test_img.shape, upscale=upscale)
        #    cv2.circle(vis_img, test_pt, 3, (1,0,1), 1)
        # Output wing tips and wing center 
        plt.figure(4)
        plt.imshow(wing_tips, cmap='gray')
        plt.figure(5)
        plt.imshow(wing_center, cmap='gray')
        # Show results
        tmp_kp_img = Image.fromarray((vis_img2[:,:,::-1]*255).astype(np.uint8))
        tmp_inference_img = Image.fromarray((vis_img[:,:,::-1]*255).astype(np.uint8))
        tmp_kp_img.save('/home/magnus/kp_img.pdf')
        tmp_inference_img.save('/home/magnus/inference_img.pdf')
        cv2.imshow('img', vis_img)
        cv2.imshow('img2', vis_img2)
        #cv2.imshow('Output_img', output_img)
        cv2.imshow('Cropped img', cropped_img)
        plt.show()
        
    def forward_test(self, input_img):
        # Run inference
        with torch.no_grad():
            transform = Compose([ToTensor(), Resize(256), CenterCrop(256)])
            cropped_input_img = transform(input_img)
            f_input = cropped_input_img.numpy().transpose(1,2,0)
            #cv2.imshow('input_img', f_input[:,:,:3])
            #cv2.imshow('input_pts', np.sum(f_input[:,:,3:7], axis=2))
            #cv2.imshow('input_lines', np.sum(f_input[:,:,7:], axis=2))
            #cv2.imshow('input_lines', f_input[:,:,7:])
            # Expand dim such that shape is now (B, C, H, W) from (C, H, W)
            cropped_input_img = torch.unsqueeze(cropped_input_img,0)
            cropped_input_img = Variable(cropped_input_img.to(self.device))
            output = self.model(cropped_input_img)
            # Remove expanded dim, move to cpu and numpyfi
            output = torch.squeeze(output).cpu().numpy().transpose(1,2,0)
            cropped_input_img = torch.squeeze(cropped_input_img).cpu().numpy().transpose(1,2,0)
            return output, cropped_input_img
        
    def forward(self, input_img, kps):
        # Transform input img with Gaussian + lines and kps
        input_img = preprocessing.create_simple_input_img(kps, input_img, sigma=20)
        #cv2.imshow('input_pts', np.sum(input_img[:,:,3:7], axis=2))
        #cv2.imshow('input_lines', np.sum(input_img[:,:,7:], axis=2))
        # Run inference
        with torch.no_grad():
            transform = Compose([ToTensor(), Resize(256), CenterCrop(256)])
            cropped_input_img = transform(input_img)
            f_input = cropped_input_img.numpy().transpose(1,2,0)
            #cv2.imshow('input_img', f_input[:,:,:3])
            #cv2.imshow('input_pts', np.sum(f_input[:,:,3:7], axis=2))
            #cv2.imshow('input_lines', np.sum(f_input[:,:,7:], axis=2))
            # Expand dim such that shape is now (B, C, H, W) from (C, H, W)
            cropped_input_img = torch.unsqueeze(cropped_input_img,0)
            cropped_input_img = Variable(cropped_input_img.to(self.device))
            output = self.model(cropped_input_img)
            # Remove expanded dim, move to cpu and numpyfi
            output = torch.squeeze(output).cpu().numpy().transpose(1,2,0)
            return output
        
    def upscale_pt(self, pt, original_image_dims):
        img_dim_y = original_image_dims[0]
        img_dim_x = original_image_dims[1]
        scale_factor = img_dim_y/256.
        trans_factor = (256.*img_dim_x/img_dim_y - 256)/2
        x = int((pt[0] + trans_factor)*scale_factor)
        y = int(pt[1]*scale_factor)
        return [x,y]
    
    def downscale_pt(self, pt, original_image_dims):
        img_dim_y = original_image_dims[0]
        img_dim_x = original_image_dims[1]
        scale_factor=256./img_dim_y
        trans_factor = (256.*img_dim_x/img_dim_y - 256)/2
        x = int((pt[0]*scale_factor) - trans_factor)
        y = int(pt[1]*scale_factor)
        return [x,y]

    def get_wing_tips(self, img, original_image_dims, upscale=True):
        pts = []
        wing_tips = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        _, wing_tips = cv2.threshold(wing_tips, int(self.pt_threshold*255), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=wing_tips, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contours_to_keep = sorted(contours, key=cv2.contourArea)[-3:]
        for c in contours_to_keep:
            x1, y1, x2, y2 = cv2.boundingRect(c)
            #cv2.rectangle(img, (x1,y1),(x1+x2,y1+y2), 0.5, 2)
            _,max_val,_, pt = cv2.minMaxLoc(img[y1:y1+y2,x1:x1+x2])
            if max_val > self.pt_threshold:
                #cv2.imshow('Test', img[y1:y1+y2, x1:x1+x2])
                #cv2.waitKey(0)
                x = pt[0] + x1
                y = pt[1] + y1
                if upscale:
                    pts.append(self.upscale_pt([x,y], original_image_dims))
                else:
                    pts.append([int(x),int(y)])
        #cv2.imshow('Wing', img)
        #cv2.waitKey(0)
        return pts

    def get_pt_from_heatmap(self, img, original_image_dims, upscale=True):
        _,max_val,_,pt = cv2.minMaxLoc(img)
        #print(max_val, pt)
        if max_val > self.pt_threshold:
            if upscale:
                return self.upscale_pt(pt, original_image_dims)
            return [int(pt[0]), int(pt[1])]
        return None

    def get_pt_from_heatmap_within_radius(self, img, pt, radius, original_image_dims, threshold = 0.5, upscale=True):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (pt[0], pt[1]), radius, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        if masked.max() >= threshold:
            max_pt = np.flip(np.argwhere(masked == masked.max())[0])
        else:
            max_pt = np.array([-1,-1])
        if upscale:
            return self.upscale_pt(max_pt, original_image_dims)
        return max_pt

    def get_line_from_heatmap(self, img, pt, perp_uvec, dist, original_image_dims, threshold = 0.1, upscale=True):
        '''
        Search for highest value along perpenducilar unit vector from pt in img.
        Dist indicates the distance to search for in each direction
        '''
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pt1 = (int(pt[0] - dist*perp_uvec[0]), int(pt[1] - dist*perp_uvec[1]))
        pt2 = (int(pt[0] + dist*perp_uvec[0]), int(pt[1] + dist*perp_uvec[1]))
        cv2.line(mask, pt1, pt2, 255, 1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        if masked.max() >= threshold:
            max_pt = np.flip(np.argwhere(masked == masked.max())[0])
        else:
            max_pt = np.array([-1,-1])
        if upscale:
            return self.upscale_pt(max_pt, original_image_dims)
        return max_pt
    
    def fit_line_to_pts(self, pts):
        A = []
        for pt in pts:
            if pt[0] > 0 and pt[0] < 640 and pt[1] > 0 and pt[1] < 480:
                A.append([pt[0], pt[1], 1])
        A = np.asarray(A)
        # Minimum 5 pts
        if not A.shape[0] > 4:
            return None
        u,s,vh = np.linalg.svd(A)
        # Take right hand collumn of V as it corresponds to best solution (smallest singular value s)
        # as we get V.T it is last row..
        return vh[-1,:]
    
    def project_lines_to_image(self, kps, vis_img, Pi_bottom_to_top, Pi_top_to_centre, Pi_centre_to_blades):
        '''
        Take label kps, create lines from them and calculate perpendicular uvec to these lines used to search for kps at in projection image
        '''
        # tower_bottom --> tower_top
        bottom = np.asarray(kps[0][:2])
        top = np.asarray(kps[1][:2])
        centre = np.asarray(kps[2][:2])
        wing1 = np.asarray(kps[3][:2])
        wing2 = np.asarray(kps[4][:2])
        wing3 = np.asarray(kps[5][:2])
        bot_to_top = top - bottom
        top_to_centre = centre - top
        centre_to_wing1 = wing1 - centre
        centre_to_wing2 = wing2 - centre
        centre_to_wing3 = wing3 - centre
        
        step_sizes = [1, 1, 1]
        h = 71.74-8
        r = 5.16
        b = 35.1
        tower_step = int(h / step_sizes[0])
        top_step = int(r / step_sizes[1])
        blade_step = int((b - step_sizes[2]) / step_sizes[2])
        
        est_D = 15
        lines = []
        lines.append(self.show_line_search_dist(bot_to_top, tower_step, bottom, vis_img, est_D, Pi_bottom_to_top))
        lines.append(self.show_line_search_dist(top_to_centre, top_step, top, vis_img, est_D, Pi_top_to_centre))
        lines.append(self.show_line_search_dist(centre_to_wing1, blade_step, centre, vis_img, est_D, Pi_centre_to_blades))
        lines.append(self.show_line_search_dist(centre_to_wing2, blade_step, centre, vis_img, est_D, Pi_centre_to_blades))
        lines.append(self.show_line_search_dist(centre_to_wing3, blade_step, centre, vis_img, est_D, Pi_centre_to_blades))
        preprocessing.show_keypoints_on_img(kps, vis_img, show=False)
        return lines
        
    def show_line_search_dist(self, v, steps, start_pt, vis_img, est_D, P_img, show=True):
        unit_v = v/np.linalg.norm(v)
        # Vector perpendicular to line
        perp_uvec = np.array([unit_v[1], -unit_v[0]])
        dist = 3*554.92/est_D
        dxy = v/steps
        if show:
            for i in range(steps+1):
                pt = start_pt + i*dxy
                nan_test1 = np.isnan(pt[0] - dist*perp_uvec[0])
                nan_test2 = np.isnan(pt[1] - dist*perp_uvec[1])
                if not nan_test1 and not nan_test2:
                    pt1 = (int(pt[0] - dist*perp_uvec[0]), int(pt[1] - dist*perp_uvec[1]))
                    pt2 = (int(pt[0] + dist*perp_uvec[0]), int(pt[1] + dist*perp_uvec[1]))
                    cv2.line(vis_img, pt1, pt2, (0,0,1), 1)
                #cv2.circle(vis_img, (int(pt[0]),int(pt[1])), 2, (1,1,0), -1)
        line_pts = []
        scaled_dist = np.ceil(0.54*dist).astype('int')
        for i in range(steps+1):
            pt = self.downscale_pt(start_pt + i*dxy, vis_img.shape)
            nan_test1 = np.isnan(pt[0] - scaled_dist*perp_uvec[0])
            nan_test2 = np.isnan(pt[1] - scaled_dist*perp_uvec[1])
            if not nan_test1 and not nan_test2:
                line_pts.append(self.get_line_from_heatmap(P_img, pt, perp_uvec, scaled_dist, vis_img.shape, upscale=True))
        return line_pts
        #for pt in line_pts:
        #    if pt[0] != -1 and pt[1] != -1:
        #        cv2.circle(vis_img, (pt[0], pt[1]), 2, (1,0,1), 1)
            #pt1 = (int(pt[0] - dist*perp_uvec[0]), int(pt[1] - dist*perp_uvec[1]))
            #pt2 = (int(pt[0] + dist*perp_uvec[0]), int(pt[1] + dist*perp_uvec[1]))
        

def main():
    # Get input image
    #inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run5/model_best_epoch744.pt', version='v2')
    #inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run11/model_best.pt', version='v1e')
    #inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run3/model_best_epoch704.pt', version='v1old')
    #inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run4/model_best.pt', version='v1old')
    #inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run13/model_best.pt')
    inferencer = Inference(model_path='./src/hourglass_network/checkpoints/run15/model_best.pt', version='v1c')
    # Use 3 and 4?
    # Get more images of wind turbine blades
    # Idx 2, 4, 9
    #annotation_idx = 9
    annotation_idx = 11
    #inferencer.test_timing(annotation_idx)
    inferencer.run_test(annotation_idx)

if __name__ == "__main__":
    main()