import cv2
import numpy as np
import timeit
import matplotlib.pyplot as plt

from renderer import Renderer
import utils

class ChamferMatcher:
    def __init__(self, img, template):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def detect_edges(self):
        '''
        Detect edges in the two images. Crop template image to minimum bounding box.
        '''
        self.img_edges = cv2.Canny(self.img,5,50, apertureSize=3, L2gradient=True)
        self.template_edges = cv2.Canny(self.template,5,50, apertureSize=3, L2gradient=True)
        # Crop image by minimum bounding box
        self.template_edges = self.template_edges[~np.all(self.template_edges == 0, axis=1)]
        self.template_edges = self.template_edges[:, ~np.all(self.template_edges == 0, axis=0)]
    
    def match(self):
        '''
        Performs hamfer matching using distance image.
        '''
        # Get distance image
        dist_img = cv2.distanceTransform(255 - self.img_edges, cv2.DIST_L1, 3).astype(np.uint8)
        #cv2.imshow("Dist image", dist_img)
        #cv2.waitKey(0)
        # Apply 2D convolution
        print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve(dist_img, self.template_edges), number=1))
        #print('convolved loop2\t\t', timeit.timeit(lambda: utils.convolve(dist_img, self.template_edges), number=1))
        #convolved_img = utils.convolve(dist_img, self.template_edges)
        #jitted_function = jit()(utils.convolve)
        #print('jitted loop\t\t', timeit.timeit(lambda: jitted_function(dist_img, self.template_edges), number=1))
        #print('jitted loop2\t\t', timeit.timeit(lambda: jitted_function(dist_img, self.template_edges), number=1))
        print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve_mask(dist_img, self.template_edges), number=1))
        #print('convolved loop2\t\t', timeit.timeit(lambda: utils.convolve_mask(dist_img, self.template_edges), number=1))
        #print('convolved loop\t\t', timeit.timeit(lambda: utils.convolve_mask(dist_img, self.template_edges), number=1))
        #convolved_img_mask = utils.convolve(dist_img, self.template_edges)
        print('tm loop_ccorr\t\t', timeit.timeit(lambda: cv2.matchTemplate(dist_img, self.template_edges, cv2.TM_CCORR), number=1))
        res = cv2.matchTemplate(dist_img, self.template_edges, cv2.TM_CCORR_NORMED)
        #jitted_function2 = jit()(utils.convolve_mask)
        #print('jitted loop21\t\t', timeit.timeit(lambda: jitted_function2(dist_img, self.template_edges), number=1))
        #print('jitted loop22\t\t', timeit.timeit(lambda: jitted_function2(dist_img, self.template_edges), number=1))
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(max_val, min_val)
        w, h = self.template_edges.shape[::-1]
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(self.img, top_left, bottom_right, 255, 2)
        #plt.imshow(res)
        #plt.show()
        cv2.imshow('Match result', res)
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        return min_val
        

def main():
    img = cv2.imread('./scripts/image_data/gazebo_200.png')
    render = Renderer('/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/Vestas_V52/meshes/vestas_v52.stl')
    print('render\t\t', timeit.timeit(lambda: render.offscreen_render([-200, 0, 65 + 8]), number=300) / 300)
    template = render.offscreen_render([-200, 0, 65 + 8])
    cv2.imshow('Template', template)
    cv2.waitKey(0)
    cm = ChamferMatcher(img, template)
    cm.detect_edges()
    cm.match()
    '''
    min_x = -250
    max_x = -150
    min_z = 65
    max_z = 65 + 16
    scores = []
    estimates = []
    
    for x in range(min_x, max_x, 5):
        for z in range(min_z, max_z, 1):
            cm.template = render.offscreen_render([x, 0, z])
            cm.detect_edges()
            scores.append(cm.match_cm())
            t0 = timeit.default_timer()
            estimates.append([x,0,z])
            t1 = timeit.default_timer()
            print(f'Time: {t1-t0}')
    print(np.argmin(scores))
    print(estimates[np.argmin(scores)])
    '''
    

if __name__ == "__main__":
    main()