from blade_detector import BladeDetector
from renderer import Renderer
from chamfer_matcher import ChamferMatcher
from skeletal_turbine_model import SkeletalTurbineModel

import cv2
import numpy as np

def main():
    render = Renderer(tower='./models/vestas_v52_rotation/meshes/vestas_v52_tower.stl', wings='./models/vestas_v52_rotation/meshes/vestas_v52_wings.stl')
    img = cv2.imread('./scripts/image_data/gazebo_100_45.png')
    bd = BladeDetector(img_path='./scripts/image_data/gazebo_100_45.png', save_result=True)
    bd.get_mask('./scripts/image_data/offshore_wind_turbine.jpg', './scripts/image_data/annotated_wind_turbine.jpg')
    model_dict = bd.detect()
    mean_length = model_dict.get('mean_length')
    print(f'Mean length: {mean_length}')
    angles_between_lines = model_dict.get('angles_between_lines')
    print(f'Angle between blades: {angles_between_lines}')
    # D' = (known_width * focal_length / pixels) + length of turbine nacelle (we want distance to center of turbine)
    estimated_dist = (35.1 * 554.920125) / mean_length + (np.cos(0.875)*5.16)
    print(f'Estimated distance: {estimated_dist}')
    # Chamfer matcher needs an initial template..
    template = render.offscreen_render([-estimated_dist, 0, 71.74 - 8])
    cv2.imshow('Template', template)
    cv2.waitKey(0)
    cm = ChamferMatcher(img, template)
    
    min_x = int(estimated_dist) - 25
    max_x = int(estimated_dist) + 25
    min_z = 71 - 8
    max_z = 71 + 8
    scores = []
    estimates = []
    
    for x in range(min_x, max_x, 2):
        for z in range(min_z, max_z, 1):
            cm.template = render.offscreen_render([-x, 0, z])
            cm.detect_edges()
            scores.append(cm.match())
            estimates.append([-x,0,z])
    print(np.argmin(scores, axis=0)[0])
    print(estimates[np.argmin(scores, axis=0)[0]])
    top_left = scores[np.argmin(scores, axis=0)[0]][1]
    bottom_right = scores[np.argmin(scores, axis=0)[0]][2]
    template = render.offscreen_render(estimates[np.argmin(scores, axis=0)[0]])
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    cv2.imshow('Image', img)
    cv2.imshow('Template', template)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()