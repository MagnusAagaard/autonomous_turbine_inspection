#!/usr/bin/env python3

import rospy
from ros_numpy import numpify
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from skeletal_turbine_model import SkeletalTurbineModel
import utils

class PoseEstimator:
    def __init__(self):
        # Init
        self._init_subscribers()
        self.pose = PoseStamped()
        self.K = np.array([[277.191356, 0.0, 320.5], 
                     [0.0, 277.191356, 240.5], 
                     [0.0, 0.0, 1.0]])
        self.stm = SkeletalTurbineModel(c=(360, 0), h=65, omega=0, r=10, phi=np.pi/5, b=52/2)

    def _init_subscribers(self):
        # Setup subscribers
        self.img_sub = rospy.Subscriber('/mono_cam/image_raw', Image, self._image_cb)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)

    def _image_cb(self, img_msg):
        # Image callback
        self.img = numpify(img_msg)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        cam_pose = self.get_extrensic_parameters()
        self.stm.project_model_to_image(img=self.img, K=self.K, cam_pose=cam_pose)
        cv2.imshow('Drone cam', self.img)
        cv2.waitKey(3)

    def _pose_cb(self, pose_msg):
        # Pose callback
        self.pose = pose_msg

    def get_extrensic_parameters(self):
        # Calculate extrensic parameters
        # Get transform from world frame to camera frame
        Rex = utils.get_rotation_matrix_from_world_to_camera_frame()
        # Camera pose/extrinsic parameters [R|t]:
        R = utils.quarternion_to_rotation_matrix(self.pose.pose.orientation)
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