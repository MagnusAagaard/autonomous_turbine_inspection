from blade_detector import BladeDetector
from renderer import Renderer
from chamfer_matcher import ChamferMatcher
from skeletal_turbine_model import SkeletalTurbineModel

import cv2
import numpy as np

def main():
    render = Renderer(tower='./models/vestas_v52_rotation/meshes/vestas_v52_tower.stl', wings='./models/vestas_v52_rotation/meshes/vestas_v52_wings.stl')
    img = cv2.imread('./scripts/image_data/gazebo_100_45.png')
    #cv2.imshow('Image', img)
    bd = BladeDetector(img_path='./scripts/image_data/gazebo_100_45.png', save_result=True)
    bd.get_mask('./scripts/image_data/offshore_wind_turbine.jpg', './scripts/image_data/annotated_wind_turbine.jpg')
    model_dict = bd.detect()
    mean_length = model_dict.get('mean_length')
    print(f'Mean length: {mean_length}')
    #angles_between_lines = model_dict.get('angles_between_lines')
    #print(f'Angle between blades: {angles_between_lines}')
    # D' = (known_width * focal_length / pixels) + length of turbine nacelle (we want distance to center of turbine)
    #estimated_dist = (35.1 * 554.920125) / mean_length + 5.16
    estimated_dist = 100
    print(f'Estimated distance: {estimated_dist}')
    cm = ChamferMatcher(img, render)
    # Base estimates: UAV located at tower height.
    # Wind turbine located directly in front in the middle of the image with wings oriented
    init_x = -100
    init_y = 0
    init_z = 74
    init_roll = 20
    init_yaw = 0
    init_est = [init_x, init_y, init_z, init_roll, init_yaw]
    best_estimate = cm.run_optimization(init_est, 5, show_plots=True)
    x = -best_estimate[0]
    y = best_estimate[1]
    z = best_estimate[2]
    roll = best_estimate[3]
    yaw = best_estimate[4]
    stm = SkeletalTurbineModel(c=(x,y), h=z, omega=np.pi + np.deg2rad(yaw), phi=np.deg2rad(60 + roll))
    stm.plot_model()

if __name__ == "__main__":
    main()