import cv2
import numpy as np
import timeit
import matplotlib.pyplot as plt
from copy import copy
from PIL import Image

from renderer import Renderer
import utils

class ChamferMatcher:
    def __init__(self, img, render):
        self.img_color = copy(img)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img_edges = cv2.Canny(self.img,50,100, apertureSize=3, L2gradient=True)
        self.detect_and_remove_horizontal_lines(self.img_edges)
        #self.dist_img = cv2.distanceTransform(255 - self.img_edges, cv2.DIST_L1, 3).astype(np.uint8)
        self.dist_img = cv2.distanceTransform(255 - self.img_edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.dist_img *= 255/self.dist_img.max()
        self.dist_img = self.dist_img.astype(np.uint8)
        self.dist_img_y = cv2.distanceTransform(255 - self.img_edges[470:,:], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.dist_img_y *= 255/self.dist_img_y.max()
        self.dist_img_y = self.dist_img_y.astype(np.uint8)
        self.render = render
        self.template = cv2.cvtColor(self.render.offscreen_render([-100, 0, 66 + 8, 30, 46]), cv2.COLOR_BGR2GRAY)
        self.detect_edges()
        #self.match(init=True)
        #_, top_left, bottom_right = self.match()
        vis_img = cv2.cvtColor(self.dist_img_y.copy(), cv2.COLOR_GRAY2RGB)
        #edges = np.argwhere(self.template_edges == 255)
        #for pt in edges:
        #    vis_img[top_left[1]+pt[0], top_left[0]+pt[1],:] = np.array([255,0,0])
        template = Image.fromarray(vis_img)
        template.save('/home/magnus/chamfer_matcher_dist_img.pdf')
    
    def render_new_template(self, estimate):
        '''
        Renders new template based on estimate parameter.
        Overwrites self.template doing this as well.
        '''
        self.template = cv2.cvtColor(self.render.offscreen_render(estimate), cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Template', self.template)
        #cv2.waitKey(0)
    
    def detect_edges(self, fit_y=False):
        '''
        Detect edges in the template image and crop to minimum bounding box.
        '''
        self.template_edges = cv2.Canny(self.template,50,100, apertureSize=3, L2gradient=True)
        # Crop image by minimum bounding box
        if not fit_y:
            self.template_edges = self.template_edges[~np.all(self.template_edges == 0, axis=1)]
            self.template_edges = self.template_edges[:, ~np.all(self.template_edges == 0, axis=0)]
        else:
            self.template_edges[:470,:] = 0
        
    def detect_and_remove_horizontal_lines(self, img):
        '''
        Detects longest horizontal line (horizon) and removes it. (if it is there..)
        '''
        result = self.img_color.copy()
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60,1))
        detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            pt1 = np.array(c)[0].flatten()
            pt2 = np.array(c)[1].flatten()
            #print(np.linalg.norm(pt1-pt2))
            pt1[0] -= 1000
            pt2[0] += 1000
            #cv2.drawContours(result, (), -1, (36,255,12), 2)
            cv2.line(img, pt1, pt2, 0, 2)
        
    def match(self, init=False, fit_y=False):
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
        if not fit_y:
            res = cv2.matchTemplate(self.dist_img, self.template_edges, cv2.TM_CCORR_NORMED)
        else:
            res = cv2.matchTemplate(self.dist_img_y, self.template_edges, cv2.TM_CCORR_NORMED)
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
        if init:
            im1 = Image.fromarray(cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB))
            im1.save('/home/magnus/chamfer_matcher_color.pdf')
            im2 = Image.fromarray(self.img)
            im2.save('/home/magnus/chamfer_matcher_gray.pdf')
            im3 = Image.fromarray(self.img_edges)
            im3.save('/home/magnus/chamfer_matcher_edges.pdf')
            img5 = cv2.cvtColor(cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLOR_GRAY2RGB)
            cv2.circle(img5, min_loc, 5, (255,0,255), -1)
            cv2.rectangle(self.img_color, top_left, bottom_right, (0,255,0), 2)
            im5 = Image.fromarray(img5)
            im5.save('/home/magnus/chamfer_matcher_result.pdf')
            im6 = Image.fromarray(cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB))
            im6.save('/home/magnus/chamfer_matcher_color_result.pdf')
            
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
        y_range = 0
        z_range = 0
        roll_range = 60
        yaw_range = 180
        
        # Start optimization
        opt_scores = []
        best_xi = copy(init_est)
        # First optimize y to center in image..
        best_xi = self.optimize(opt_scores, best_xi, [x_range, 5, z_range, roll_range, yaw_range], index=1, fit_y=True)
        # First optimize roll and yaw without x,y,z..
        best_xi = self.optimize(opt_scores, best_xi, [x_range, y_range, z_range, roll_range, yaw_range], index=3)
        print(f'Before inverse rotation test: {best_xi}')
        best_xi = self.test_inverse_rotations(best_xi)
        print(f'After inverse rotation test: {best_xi}')
        # Run optimization. Stop if max_iterations is reached or change between parameters is small
        for i in range(max_iterations):
            new_best_xi = self.optimize(opt_scores, best_xi, [x_range, y_range, z_range, roll_range, yaw_range])
            print(f'Before inverse rotation test: {new_best_xi}')
            new_best_xi = self.test_inverse_rotations(new_best_xi)
            print(f'After inverse rotation test: {new_best_xi}')
            if np.linalg.norm(np.asarray(best_xi)-np.asarray(new_best_xi)) < 5:
                print(np.linalg.norm(np.asarray(best_xi)-np.asarray(new_best_xi)))
                best_xi = new_best_xi
                break
            best_xi = new_best_xi
        print(best_xi)
        # Refine estimation
        scores_final, best_xi_final = self.refine_optimization(best_xi)
        print(best_xi_final)
        top_left = scores_final[1]
        bottom_right = scores_final[2]
        cv2.rectangle(self.img_color, top_left, bottom_right, 255, 2)
        self.render_new_template(best_xi_final)
        self.detect_edges()
        edges = np.argwhere(self.template_edges == 255)
        for pt in edges:
            self.img_color[top_left[1]+pt[0], top_left[0]+pt[1],:] = np.array([0,0,255])
        _, _, _ = self.match(init=True)
        
        # Show results
        #img_temp = copy(self.img_color)
        if show_plots:
            #img_temp[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.cvtColor(self.template_edges, cv2.COLOR_GRAY2RGB)
            
            #cv2.imshow('Image with template', img_temp)
            cv2.imshow('Template', self.template)
            cv2.imshow('Image', self.img_color)
            
            plt.plot([score[0] for score in opt_scores[0]], label=r'$\phi$ iteration 1')
            plt.plot([score[0] for score in opt_scores[1]], label=r'$\omega$ iteration 1')
            #plt.plot([score[0] for score in opt_scores[2]], label='X2')
            #plt.plot([score[0] for score in opt_scores[3]], label='Y2')
            #plt.plot([score[0] for score in opt_scores[4]], label='Z2')
            plt.plot([score[0] for score in opt_scores[5]], label=r'$\phi$ iteration 2')
            plt.plot([score[0] for score in opt_scores[6]], label=r'$\omega$ iteration 2')
            plt.legend()
            plt.xlabel(r'Degrees [$\degree$]')
            plt.ylabel('Score')
            #plt.savefig('/home/magnus/optimization_scores.pdf', bbox_inches='tight')
            #plt.show()
            
            cv2.waitKey(0)
        
        return best_xi_final, self.img_color.copy()
    
    def optimize(self, opt_scores, init_est, est_range, index=0, fit_y=False):
        scores = []
        step_size = 1 if index != 1 else 0.1
        current_min = 99999
        best_xi = copy(init_est)
        times_bigger_than_min = 0
        
        old_xi = copy(init_est)
        old_xi[index] = old_xi[index] - est_range[index]
        self.render_new_template(old_xi)
        self.detect_edges(fit_y=fit_y)
        f_old_xi = self.match(fit_y=fit_y)
        xi = copy(old_xi)
        xi[index] += step_size
        
        for est in np.arange(0, est_range[index]*2, step_size):
            self.render_new_template(xi)
            self.detect_edges(fit_y=fit_y)
            f_xi = self.match(fit_y=fit_y)
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
            if times_bigger_than_min > 10 and (index == 0 or index == 2):
                # Allow for roll and yaw to be fitted 
                print(f"Stopping for index: {index}")
                opt_scores.append(scores)
                if index < 4 and not fit_y:
                    return self.optimize(opt_scores, best_xi, est_range, index=index+1)
                return best_xi
        opt_scores.append(scores)
        if index < 4 and not fit_y:
            return self.optimize(opt_scores, best_xi, est_range, index=index+1)
        return best_xi
    
    def test_inverse_rotations(self, xi):
        self.render_new_template(xi)
        self.detect_edges()
        f_xi = self.match()
        inv_xi = copy(xi)
        min_score = f_xi
        min_xi = copy(xi)
        # First test yaw with same roll
        for i in range(7):
            inv_xi[4] += 45
            for j in range(120):
                inv_xi[3] = j
                self.render_new_template(inv_xi)
                self.detect_edges()
                f_inv_xi = self.match()
                if f_inv_xi < min_score:
                    min_score = f_inv_xi
                    min_xi = copy(inv_xi)
        # Roll 180 degrees and test yaws again
        #inv_xi = copy(xi)
        #inv_xi[3] += 180
        #for i in range(3):
        #    inv_xi[4] += 90
        #    self.render_new_template(inv_xi)
        #    self.detect_edges()
        #    f_inv_xi = self.match()
        #    if f_inv_xi < min_score:
        #        min_score = f_inv_xi
        #        min_xi = copy(inv_xi)
        return min_xi
        
    
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
            y=params[1]
            #for y in np.arange(params[1] - 0.2, params[1] + 0.2, 0.1):
                #print(y)
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
    print('done')
    
    # Base estimates: UAV located at tower height.
    # Wind turbine located directly in front in the middle of the image with wings oriented
    init_x = -100
    init_y = 0
    init_z = 74
    init_roll = 80
    init_yaw = 0
    init_est = [init_x, init_y, init_z, init_roll, init_yaw]
    best_estimate = cm.run_optimization(init_est, 5)

if __name__ == "__main__":
    main()