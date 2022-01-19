import cv2
import numpy as np
from renderer import Renderer

class ChamferMatcher:
    def __init__(self, img, template):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #self.detect_edges()
        #cv2.imshow("Image", self.img_edges)
        #cv2.imshow("Template", self.template_edges)
        #cv2.waitKey(0)
        #self.match()
    
    def detect_edges(self):
        '''
        Detect edges in the two images.
        '''
        self.img_edges = cv2.Canny(self.img,5,50, apertureSize=3, L2gradient=True)
        self.template_edges = cv2.Canny(self.template,5,50, apertureSize=3, L2gradient=True)
        # Crop image by minimum bounding box
        self.template_edges = self.template_edges[~np.all(self.template_edges == 0, axis=1)]
        self.template_edges = self.template_edges[:, ~np.all(self.template_edges == 0, axis=0)]
        
    def match(self):
        '''
        Performs chamfer matching by calculating distance from each pixel to nearest edge
        '''
        res = cv2.matchTemplate(self.img_edges,self.template_edges, cv2.TM_CCOEFF_NORMED)
        #w, h = self.template_edges.shape[::-1]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #print(max_val)
        #top_left = max_loc
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(self.img,top_left, bottom_right, 255, 2)
        #cv2.imshow('Match result', res)
        #cv2.imshow('Image', self.img)
        #cv2.waitKey(0)
        return max_val
        

def main():
    img = cv2.imread('/home/magnus/tmp_img_200.png')
    render = Renderer('/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/Vestas_V52/meshes/vestas_v52.stl')
    template = render.offscreen_render([-300, 0, 65 + 8])
    cm = ChamferMatcher(img, template)
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
            scores.append(cm.match())
            estimates.append([x,0,z])
    print(np.argmax(scores))
    print(estimates[np.argmax(scores)])
            
            
    

if __name__ == "__main__":
    main()