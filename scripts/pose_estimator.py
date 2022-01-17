#!/usr/bin/env python3

import rospy
from ros_numpy import numpify
import numpy as np
import cv2

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from skeletal_turbine_model import SkeletalTurbineModel
import utils

class PoseEstimator:
    def __init__(self):
        # Init
        self._init_subscribers()
        self.pose = PoseStamped()
        self.K = np.array([[554.920125, 0.000000, 320.077433], 
                     [0.000000, 554.921917, 239.661438], 
                     [0.000000, 0.000000, 1.000000]])
        self.stm = SkeletalTurbineModel(c=(360, 0), h=65, omega=np.pi+0.0, r=10, phi=np.pi/2, b=60/2)
        self.trigger_save = False

    def _init_subscribers(self):
        # Setup subscribers
        self.img_sub = rospy.Subscriber('/mono_cam/image_raw', Image, self._image_cb)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        self.trigger_sub = rospy.Subscriber('~trigger_image_save', Bool, self.__trigger_cb)
        
    def __trigger_cb(self, msg):
        self.trigger_save = msg.data

    def _image_cb(self, img_msg):
        # Image callback
        self.img = numpify(img_msg)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        if self.trigger_save:
            rospy.loginfo('Saving image..')
            cv2.imwrite('tmp_img.png', self.img)
            self.trigger_save = False
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