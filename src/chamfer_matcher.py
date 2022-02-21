import cv2
import numpy as np
import timeit
import matplotlib.pyplot as plt
from copy import copy

from renderer import Renderer
import utils

class ChamferMatcher:
    def __init__(self, img, renderer):
        self.img_color = copy(img)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_edges = cv2.Canny(self.img,5,50, apertureSize=3, L2gradient=True)
        self.dist_img = cv2.distanceTransform(255 - self.img_edges, cv2.DIST_L1, 3).astype(np.uint8)
        self.renderer = renderer
        self.template = cv2.cvtColor(self.renderer.offscreen_render([-100, 0, 65 + 8, 30, 45]), cv2.COLOR_BGR2GRAY)
        self.detect_edges()
    
    def render_new_template(self, estimate):
        '''
        Renders new template based on estimate parameter.
        Overwrites self.template doing this as well.
        '''
        self.template = cv2.cvtColor(self.renderer.offscreen_render(estimate), cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Template', self.template)
        #cv2.waitKey(0)
    
    def detect_edges(self):
        '''
        Detect edges in the template image and crop to minimum bounding box.
        '''
        self.template_edges = cv2.Canny(self.template,5,50, apertureSize=3, L2gradient=True)
        # Crop image by minimum bounding box
        self.template_edges = self.template_edges[~np.all(self.template_edges == 0, axis=1)]
        self.template_edges = self.template_edges[:, ~np.all(self.template_edges == 0, axis=0)]
        
    def match(self):
        '''
        Performs chamfer matching using distance image.
        '''
        # Apply 2D convolution
        #print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve(self.dist_img, self.template_edges), number=1))
        #print('convolved loop2\t\t', timeit.timeit(lambda: utils.convolve(self.dist_img, self.template_edges), number=1))
        #convolved_img = utils.convolve(self.dist_img, self.template_edges)
        #jitted_function = jit()(utils.convolve)
        #print('jitted loop\t\t', timeit.timeit(lambda: jitted_function(self.dist_img, self.template_edges), number=1))
        #print('jitted loop2\t\t', timeit.timeit(lambda: jitted_function(self.dist_img, self.template_edges), number=1))
        #print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve_mask(self.dist_img, self.template_edges), number=1))
        #print('convolved loop2\t\t', timeit.timeit(lambda: utils.convolve_mask(self.dist_img, self.template_edges), number=1))
        #print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve_mask(self.dist_img, self.template_edges), number=1))
        #convolved_img_mask = utils.convolve(self.dist_img, self.template_edges)
        #print('tm loop_ccorr\t\t', timeit.timeit(lambda: cv2.matchTemplate(self.dist_img, self.template_edges, cv2.TM_CCORR), number=1))
        res = cv2.matchTemplate(self.dist_img, self.template_edges, cv2.TM_CCORR_NORMED)
        #jitted_function2 = jit()(utils.convolve_mask)
        #print('jitted loop21\t\t', timeit.timeit(lambda: jitted_function2(self.dist_img, self.template_edges), number=1))
        #print('jitted loop22\t\t', timeit.timeit(lambda: jitted_function2(self.dist_img, self.template_edges), number=1))
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #print(max_val, min_val)
        w, h = self.template_edges.shape[::-1]
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(self.img, top_left, bottom_right, 255, 2)
        #plt.imshow(res)
        #plt.show()
        #cv2.imshow('Match result', res)
        #cv2.imshow('Image', self.img)
        #cv2.waitKey(5)
        return min_val, top_left, bottom_right
    
    def run_optimization(self, init_est, max_iterations, show_plots=True):
        '''
        Function to run optimization. Generates rendered models and optimizes on the pose estimation
        based on a score given from chamfer matching. After rough estimation has been performed,
        a refined estimate is found using a brute force method.
        Input: initial pose estimate of wind turbine and max iterations before returning
        '''
        # Define ranges to search for on each side of init estimate
        x_range = 5
        y_range = 1
        z_range = 0
        roll_range = 30
        yaw_range = 180
        
        # Start optimization
        opt_scores = []
        best_xi = copy(init_est)
        # First optimize roll and yaw without x,y,z..
        best_xi = self.optimize(opt_scores, best_xi, [x_range, y_range, z_range, roll_range, yaw_range], index=3)
        print(best_xi)
        # Run optimization. Stop if max_iterations is reached or change between parameters is small
        for i in range(max_iterations):
            new_best_xi = self.optimize(opt_scores, best_xi, [x_range, y_range, z_range, roll_range, yaw_range])
            print(new_best_xi)
            if np.linalg.norm(np.asarray(best_xi)-np.asarray(new_best_xi)) < 5:
                print(np.linalg.norm(np.asarray(best_xi)-np.asarray(new_best_xi)))
                best_xi = new_best_xi
                break
            best_xi = new_best_xi
        print(best_xi)
        # Refine estimation
        scores, best_xi = self.refine_optimization(best_xi)
        print(best_xi)
        top_left = scores[1]
        bottom_right = scores[2]
        cv2.rectangle(self.img_color, top_left, bottom_right, 255, 2)
        self.render_new_template(best_xi)
        self.detect_edges()
        edges = np.argwhere(self.template_edges == 255)
        for pt in edges:
            self.img_color[top_left[1]+pt[0], top_left[0]+pt[1],:] = np.array([0,0,255])
        # Show results
        #img_temp = copy(self.img_color)
        if show_plots:
            #img_temp[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.cvtColor(self.template_edges, cv2.COLOR_GRAY2RGB)
            
            #cv2.imshow('Image with template', img_temp)
            cv2.imshow('Template', self.template)
            cv2.imshow('Image', self.img_color)
            
            plt.plot([score[0] for score in opt_scores[0]], label='Roll1')
            plt.plot([score[0] for score in opt_scores[1]], label='Yaw1')
            plt.plot([score[0] for score in opt_scores[2]], label='X2')
            plt.plot([score[0] for score in opt_scores[3]], label='Y2')
            plt.plot([score[0] for score in opt_scores[4]], label='Z2')
            plt.plot([score[0] for score in opt_scores[5]], label='Roll2')
            plt.plot([score[0] for score in opt_scores[6]], label='Yaw2')
            plt.legend()
            #plt.show()
            
            #cv2.waitKey(0)
        return best_xi, self.img_color.copy()
    
    def optimize(self, opt_scores, init_est, est_range, index=0):
        scores = []
        step_size = 1 if index != 1 else 0.1
        current_min = 99999
        best_xi = copy(init_est)
        times_bigger_than_min = 0
        
        old_xi = copy(init_est)
        old_xi[index] = old_xi[index] - est_range[index]
        self.render_new_template(old_xi)
        self.detect_edges()
        f_old_xi = self.match()
        xi = copy(old_xi)
        xi[index] += step_size
        
        for est in np.arange(0, est_range[index]*2, step_size):
            self.render_new_template(xi)
            self.detect_edges()
            f_xi = self.match()
            scores.append(f_xi)
            dfx = (f_xi[0] - f_old_xi[0])
            old_xi = copy(xi)
            f_old_xi = f_xi
            xi[index] += step_size
            if f_xi[0] >= current_min:
                times_bigger_than_min += 1
            else:
                current_min = f_xi[0]
                best_xi = copy(xi)
                times_bigger_than_min = 0
            if times_bigger_than_min > 10 and index < 3:
                # Allow for roll and yaw to be fitted 
                print(f"Stopping for index: {index}")
                opt_scores.append(scores)
                if index < 4:
                    return self.optimize(opt_scores, best_xi, est_range, index=index+1)
                return best_xi
        opt_scores.append(scores)
        if index < 4:
            return self.optimize(opt_scores, best_xi, est_range, index=index+1)
        return best_xi
    
    def refine_optimization(self, params):
        '''
        Refines the estimated parameters of the wind turbine.
        Uses a solid estimate and brute force through ranges for the best estimate
        Input: params = estimated params for the wind turbine [x,y,z,roll,yaw]
        Output: scores (tuple of match score, top left, bottom right locations),
                estimates is refined estimate [x,y,z, roll, yaw]
        '''
        scores = []
        estimates = []
        print(params)
        for x in range(params[0]-3, params[0]+3, 1):
            for y in np.arange(params[1] - 0.2, params[1] + 0.2, 0.1):
                print(y)
                for z in range(params[2]-2, params[2]+2, 1):
                    for roll in range(params[3]-2, params[3]+2, 1):
                        for yaw in range(params[4]-2, params[4]+2, 1):
                            self.render_new_template([x,y,z,roll,yaw])
                            self.detect_edges()
                            scores.append(self.match())
                            estimates.append([x,y,z,roll,yaw])
        
        idx = np.argmin(scores, axis=0)[0]
        return scores[idx], estimates[idx]

def main():
    img = cv2.imread('./scripts/image_data/gazebo_100_45.png')
    render = Renderer(tower='./models/vestas_v52_rotation/meshes/vestas_v52_tower.stl', wings='./models/vestas_v52_rotation/meshes/vestas_v52_wings.stl')
    #print('render\t\t', timeit.timeit(lambda: render.offscreen_render([-100, 0, 65 + 8]), number=300) / 300)
    cm = ChamferMatcher(img, render)
    #cm.detect_edges()
    #cm.match()
    
    # Base estimates: UAV located at tower height.
    # Wind turbine located directly in front in the middle of the image with wings oriented
    init_x = -100
    init_y = 0
    init_z = 74
    init_roll = 20
    init_yaw = 0
    init_est = [init_x, init_y, init_z, init_roll, init_yaw]
    best_estimate = cm.run_optimization(init_est, 5)

if __name__ == "__main__":
    main()