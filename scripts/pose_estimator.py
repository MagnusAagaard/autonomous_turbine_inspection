#!/usr/bin/env python3

import rospy
from ros_numpy import numpify
import numpy as np
import cv2
from copy import copy

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from skeletal_turbine_model import SkeletalTurbineModel
from chamfer_matcher import ChamferMatcher
from renderer import Renderer
from hourglass_network.inference import Inference
from optimizer import Camera, Point, Observation, PoseGraphOptimization
import utils

from display import Display3D

np.set_printoptions(precision=4, suppress=True)

class PoseEstimator:
    def __init__(self):
        # Init
        self.stm = None
        self.img = None
        #TODO: Add this as a launch parameter
        self.img_shape = (480, 640)
        self._init_subscribers()
        self.pose = PoseStamped()
        self.est_pose = None
        #TODO: Add this as a launch parameter
        self.K = np.array([[554.920125, 0.000000, 320.077433], 
                     [0.000000, 554.921917, 239.661438], 
                     [0.000000, 0.000000, 1.000000]])
        self.stm = None
        self.trigger_save = False
        #TODO: Add this as a launch parameter
        self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run3/model_best_epoch704.pt')
        #TODO: Add this as a launch parameter
        self.render = Renderer(tower='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/vestas_v52_rotation/meshes/vestas_v52_tower.stl', 
                               wings='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/vestas_v52_rotation/meshes/vestas_v52_wings.stl')
        #self.stm = SkeletalTurbineModel(c=(360, 0), h=71.74-8, omega=np.pi+np.deg2rad(45), phi=np.pi/2)
        self.optimizer = PoseGraphOptimization(camera_matrix=self.K)
        self.three_dim_viewport = Display3D()
        self.last_optimization_time = rospy.Time.now()
        # Half a second
        #self.time_between_optimizations = rospy.Duration(secs=0, nsecs=500000000)
        self.time_between_optimizations = rospy.Duration(secs=1, nsecs=0)
        self._init_skeletal_model()

    def _init_subscribers(self):
        # Setup subscribers
        self.img_sub = rospy.Subscriber('/mono_cam/image_raw', Image, self._image_cb)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        self.trigger_sub = rospy.Subscriber('~trigger_image_save', Bool, self.__trigger_cb)
    
    def _init_skeletal_model(self):
        while self.img is None:
            rospy.sleep(0.1)
        
        print("Image recieved, estimating...")
        cam_pose = copy(self.pose)
        img = self.img.copy()
        estimated_dist = 100
        print(f'Estimated distance: {estimated_dist}')
        cm = ChamferMatcher(img, self.render)
        # Base estimates: UAV located at tower height.
        # Wind turbine located directly in front in the middle of the image with wings oriented
        init_x = -estimated_dist
        init_y = 0
        init_z = 74
        init_roll = 20
        init_yaw = 0
        init_est = [init_x, init_y, init_z, init_roll, init_yaw]
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        best_estimate, self.rst_img = cm.run_optimization(init_est, 5, show_plots=False)
        x += -best_estimate[0]
        y += best_estimate[1]
        # We know there is 8m from MSL to bottom/where turbine is located
        z = best_estimate[2] - 8 
        roll = np.deg2rad(60 + best_estimate[3])
        yaw = np.pi + np.deg2rad(best_estimate[4])
        print(f'Estimates: ({x},{y},{z},{roll},{yaw})')
        self.stm = SkeletalTurbineModel(c=(x,y), omega=yaw, phi=roll)
        self.init_optimizer(img, cam_pose)
        
    def init_optimizer(self, init_img, init_cam_pose):
        '''
        Initialize optimizer with first camera pose during skeletal model initiliaztion and keypoints.
        '''
        # First pose during STM initialization
        R,t = utils.get_pose_from_pose_msg(init_cam_pose)
        cam = Camera(R=R, t=t, camera_id=self.optimizer.increment_id(), fixed=True)
        self.optimizer.cameras.append(cam)
        points_3d = self.optimizer.point_model_to_points(self.stm.point_model)
        points_2d, lines_divided_2d = self.stm.project_model_to_image(img=init_img, K=self.K, cam_pose=np.column_stack((R,t)))
        self.optimizer.create_observations(points_3d, points_2d, cam.camera_id)
        self.stm.cam_pose_from_optimizer = self.optimizer.cameras[-1].pose()[:3,:]
        self.last_optimization_time = rospy.Time.now()
        self.three_dim_viewport.set_points_to_draw(self.optimizer.points, self.optimizer.cameras)
        
    def __trigger_cb(self, msg):
        self.trigger_save = msg.data

    def _image_cb(self, img_msg):
        # Image callback
            self.img = cv2.cvtColor(numpify(img_msg), cv2.COLOR_RGB2BGR)
            input_img = self.img.copy()
            drone_img = self.img.copy()
            if self.trigger_save:
                rospy.loginfo('Saving image..')
                cv2.imwrite('tmp_img.png', self.img)
                self.trigger_save = False
            if self.stm:
                #if self.est_pose is None:
                R,t = utils.get_pose_from_pose_msg(self.pose)
                cam_pose = np.column_stack((R,t))
                #else:
                #    R = self.est_pose[:,:3]
                #    t = self.est_pose[:,3]
                #    cam_pose = np.column_stack((R,t))
                #cam_pose = self.get_extrensic_parameters()
                kps, lines_divided_2d = self.stm.project_model_to_image(img=drone_img, K=self.K, cam_pose=cam_pose)
                output = self.inferencer.forward(input_img, kps)
                #TODO: Make a way to process it all and save a number of point correspondences
                # checking whether they are present in the current image or not
                # Also: Add function that removes current observations with high reprojection error to avoid drifting?
                new_kps = self.process_inference_output(kps, output)
                #points_3d = self.optimizer.point_model_to_points(self.stm.point_model)
                points_3d = None
                if (rospy.Time.now() - self.last_optimization_time) > self.time_between_optimizations:
                    cam = Camera(R=R, t=t, camera_id=self.optimizer.increment_id(), fixed=False)
                    self.optimizer.cameras.append(cam)
                    self.optimizer.create_observations(points_3d, new_kps, cam.camera_id)
                    #print(f'Point model: {self.stm.point_model}')
                    self.optimizer.optimize()
                    # Use current point estimate from optimizer?
                    self.stm.update_point_model_from_optimizer(self.optimizer.points)
                    # Use current pose estimate from optimzier
                    self.est_pose = self.optimizer.cameras[-1].pose()[:3,:]
                    self.stm.cam_pose_from_optimizer = self.optimizer.cameras[-1].pose()[:3,:]
                    self.last_optimization_time = rospy.Time.now()
                    self.three_dim_viewport.set_points_to_draw(self.optimizer.points, self.optimizer.cameras)
                    
                #cv2.imshow('Outputpt1', output[:,:,3])
                #cv2.imshow('Outputpt2', output[:,:,4])
                #cv2.imshow('Outputpt3', output[:,:,5])
                #cv2.imshow('Outputpt4', output[:,:,6])
                #cv2.imshow('Outputl1', output[:,:,7])
                #cv2.imshow('Outputl2', output[:,:,8])
                #cv2.imshow('Outputl3', output[:,:,9])
                search_radius = 14
                scaled_search_radius = np.ceil(0.4*search_radius).astype('int')   # As scale factor is 0.4 when downscaling
                # Wing tips
                for pt in kps[:3]:
                    pt = self.inferencer.downscale_pt(pt, input_img.shape)
                    test_pt = self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,3], pt, scaled_search_radius, input_img.shape, upscale=True)
                    cv2.circle(input_img, test_pt, 3, (0,0,255), 1)
                # Rest
                for i, pt in enumerate(kps[3:]):
                    pt = self.inferencer.downscale_pt(pt, input_img.shape)
                    test_pt = self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,4+i], pt, scaled_search_radius, input_img.shape, upscale=True)
                    cv2.circle(input_img, test_pt, 3, (0,0,255), 1)
                # Lines
                search_dist = 5
                scaled_search_dist = np.ceil(0.4*search_dist).astype('int')   # As scale factor is 0.4 when downscaling
                for i, line in enumerate(lines_divided_2d[:2]):
                    # Vector from first pt to last pt (line vector)
                    v = line[-1] - line[0]
                    # Unit vector
                    unit_v = v/np.linalg.norm(v)
                    # Vector perpendicular to line
                    unit_v_perp = np.array([unit_v[1], -unit_v[0]])
                    for pt in line:
                        #pt1 = (int(pt[0] - search_dist*unit_v_perp[0]), int(pt[1] - search_dist*unit_v_perp[1]))
                        #pt2 = (int(pt[0] + search_dist*unit_v_perp[0]), int(pt[1] + search_dist*unit_v_perp[1]))
                        #cv2.line(input_img, pt1, pt2, (255,0,0), 2)
                        pt = self.inferencer.downscale_pt(pt, input_img.shape)
                        test_pt = self.inferencer.get_line_from_heatmap(output[:,:,7+i], pt, unit_v_perp, scaled_search_dist, input_img.shape, upscale=True)
                        cv2.circle(input_img, test_pt, 3, (0,255,0), 1)
                for line in lines_divided_2d[2:]:
                    # Vector from first pt to last pt (line vector)
                    v = line[-1] - line[0]
                    # Unit vector
                    unit_v = v/np.linalg.norm(v)
                    # Vector perpendicular to line
                    unit_v_perp = np.array([unit_v[1], -unit_v[0]])
                    for pt in line:
                        #pt1 = (int(pt[0] - search_dist*unit_v_perp[0]), int(pt[1] - search_dist*unit_v_perp[1]))
                        #pt2 = (int(pt[0] + search_dist*unit_v_perp[0]), int(pt[1] + search_dist*unit_v_perp[1]))
                        #cv2.line(input_img, pt1, pt2, (255,0,0), 2)
                        pt = self.inferencer.downscale_pt(pt, input_img.shape)
                        test_pt = self.inferencer.get_line_from_heatmap(output[:,:,9].copy(), pt, unit_v_perp, scaled_search_dist, input_img.shape, upscale=True)
                        #cv2.circle(input_img, test_pt, 3, (0,0,255), 1)
                        
                #cv2.imshow('Result image', self.rst_img)
            
            cv2.imshow('Drone cam', drone_img)
            cv2.imshow('Pose cam', input_img)
            cv2.waitKey(1)

    def _pose_cb(self, pose_msg):
        # Pose callback
        self.pose = pose_msg
        
    def process_inference_output(self, kps, output):
        '''
        Takes the current kps estimate based on STM and 
        output from the neural network and processes it.
        Returns 2D kp locations in image.
        '''
        # Wing tips
        new_kps = []
        for pt in kps[:3]:
            pt = self.inferencer.downscale_pt(pt, self.img_shape)
            new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,3], pt, 14, self.img_shape, upscale=True))
        # Rest
        for i, pt in enumerate(kps[3:]):
            pt = self.inferencer.downscale_pt(pt, self.img_shape)
            new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,4+i], pt, 14, self.img_shape, upscale=True))
        # Lines
        #for i, line in enumerate(lines_divided_2d[:2]):
        #    # Vector from first pt to last pt (line vector)
        #    v = line[-1] - line[0]
        #    # Unit vector
        #    unit_v = v/np.linalg.norm(v)
        #    # Vector perpendicular to line
        #    unit_v_perp = np.array([unit_v[1], -unit_v[0]])
        #    for pt in line:
        #        #pt1 = (int(pt[0] - 5*unit_v_perp[0]), int(pt[1] - 5*unit_v_perp[1]))
        #        #pt2 = (int(pt[0] + 5*unit_v_perp[0]), int(pt[1] + 5*unit_v_perp[1]))
        #        #cv2.line(input_img, pt1, pt2, (255,0,0), 2)
        #        pt = self.inferencer.downscale_pt(pt, input_img.shape)
        #        test_pt = self.inferencer.get_line_from_heatmap(output[:,:,7+i], pt, unit_v_perp, 5, input_img.shape, upscale=True)
        #        cv2.circle(input_img, test_pt, 3, (0,255,0), 1)
        #for line in lines_divided_2d[2:]:
        #    # Vector from first pt to last pt (line vector)
        #    v = line[-1] - line[0]
        #    # Unit vector
        #    unit_v = v/np.linalg.norm(v)
        #    # Vector perpendicular to line
        #    unit_v_perp = np.array([unit_v[1], -unit_v[0]])
        #    for pt in line:
        #        #pt1 = (int(pt[0] - 5*unit_v_perp[0]), int(pt[1] - 5*unit_v_perp[1]))
        #        #pt2 = (int(pt[0] + 5*unit_v_perp[0]), int(pt[1] + 5*unit_v_perp[1]))
        #        #cv2.line(input_img, pt1, pt2, (255,0,0), 2)
        #        pt = self.inferencer.downscale_pt(pt, input_img.shape)
        #        test_pt = self.inferencer.get_line_from_heatmap(output[:,:,9], pt, unit_v_perp, 5, input_img.shape, upscale=True)
        #        cv2.circle(input_img, test_pt, 3, (0,255,0), 1)
        return new_kps

    def get_extrensic_parameters(self):
        # Calculate extrensic parameters
        # Get transform from world frame to camera frame
        Rex = utils.get_rotation_matrix_from_world_to_camera_frame()
        # Camera pose/extrinsic parameters [R|t]:
        R = utils.quarternion_to_rotation_matrix(q=self.pose.pose.orientation, inverse=True)
        t = np.array([self.pose.pose.position.x,self.pose.pose.position.y, self.pose.pose.position.z])
        # Cam pose is transformed from world frame to camera frame
        # and the camera pose itself is then applied
        cam_pose = np.column_stack((Rex @ R, -Rex @ R @ t))
        return cam_pose

def main():
    rospy.init_node('pose_estimator', anonymous=True)
    estimator = PoseEstimator()
    rospy.spin()

if __name__ == "__main__":
    main()