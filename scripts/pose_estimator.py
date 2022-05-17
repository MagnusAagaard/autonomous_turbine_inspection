#!/usr/bin/env python3

import rospy
from ros_numpy import numpify
import numpy as np
import cv2
from copy import copy

from std_msgs.msg import Bool, Float32, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import ModelStates
from skeletal_turbine_model import SkeletalTurbineModel
from chamfer_matcher import ChamferMatcher
from renderer import Renderer
from hourglass_network.inference import Inference
from optimizer import Camera, Point, Observation, PoseGraphOptimization
import utils
from itertools import chain

from display import Display3D

np.set_printoptions(precision=4, suppress=True)

class PoseEstimator:
    def __init__(self):
        # Init
        self.stm = None
        self.img = None
        self.img_num = 0
        #TODO: Add this as a launch parameter
        self.img_shape = (480, 640)
        # Half a second
        #self.time_between_optimizations = rospy.Duration(secs=0, nsecs=500000000)
        self.run_optimizer = False
        self.time_between_optimizations = rospy.Duration(secs=1, nsecs=0)
        self.time_before_running_optimization = rospy.Duration(secs=1, nsecs=0)
        # Number of frames added to pose graph before moving to next wp
        self.n_frames_for_pose_graph = 3
        self.n_frames_added = 0
        self.pose = PoseStamped()
        self.est_pose_offset = None
        #TODO: Add this as a launch parameter
        self.K = np.array([[554.920125, 0.000000, 320.077433], 
                     [0.000000, 554.921917, 239.661438], 
                     [0.000000, 0.000000, 1.000000]])
        self.stm = None
        self.trigger_save = False
        #TODO: Add this as a launch parameter
        #self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run3/model_best_epoch704.pt', version='v1old')
        #self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run4/model_best.pt', version='v1old')
        #self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run10/checkpoint_800.pt', version='v1e')
        #self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run11/model_best.pt', version='v1e')
        self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run13/model_best.pt')
        #self.inferencer = Inference(model_path='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/src/hourglass_network/checkpoints/run14/model_best.pt', version='v1c')
        #TODO: Add this as a launch parameter
        self.render = Renderer(tower='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/vestas_v52_rotation/meshes/vestas_v52_tower.stl', 
                               wings='/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/vestas_v52_rotation/meshes/vestas_v52_wings.stl')
        #self.stm = SkeletalTurbineModel(c=(360, 0), h=71.74-8, omega=np.pi+np.deg2rad(45), phi=np.pi/2)
        self.optimizer = PoseGraphOptimization(camera_matrix=self.K)
        #self.three_dim_viewport = Display3D()
        self.last_optimization_time = rospy.Time.now()
        
        self._init_subscribers()
        self._init_publishers()
        self._init_skeletal_model()

    def _init_subscribers(self):
        # Setup subscribers
        self.img_sub = rospy.Subscriber('/mono_cam/image_raw', Image, self._image_cb)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        self.true_pose_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._true_pose_cb)
        self.trigger_sub = rospy.Subscriber('~trigger_image_save', Bool, self.__trigger_cb)
        self.toggle_estimator_sub = rospy.Subscriber('/drone_control/toggle_pose_estimator', Bool, self._toggle_optimizer_cb)
        
    def _init_publishers(self):
        # Setup publishers
        self.next_wp_pub = rospy.Publisher('/pose_estimator/next_wp', Bool, queue_size=1)
        self.stm_params_pub = rospy.Publisher('/pose_estimator/turbine_params', Float32MultiArray, queue_size=1, latch=True)
        self.pose_offset_pub = rospy.Publisher('/pose_estimator/pose_offset', Pose, queue_size=1)
    
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
        #best_estimate, self.rst_img = cm.run_optimization(init_est, 5, show_plots=False)
        #print(f'BE: {best_estimate}')
        #x += -best_estimate[0]
        #y += best_estimate[1]
        # We know there is 8m from MSL to bottom/where turbine is located
        #z = best_estimate[2] - 8
        #roll = np.deg2rad(60 + best_estimate[3])
        #yaw = np.pi + np.deg2rad(best_estimate[4])
        #print(f'Estimates: ({x},{y},{z},{roll-np.deg2rad(60)},{yaw - np.pi})')
        #NOTE: Currently fixed estimate
        x = 110
        y = 0
        roll = np.deg2rad(60 + 30)
        yaw = np.pi + np.deg2rad(45)
        self.stm = SkeletalTurbineModel(c=(x,y), omega=yaw, phi=roll)
        self._publish_stm_params([x,y,yaw,roll])
        self.init_optimizer(img, cam_pose)
        
    def init_optimizer(self, init_img, init_cam_pose):
        '''
        Initialize optimizer with first camera pose during skeletal model initiliaztion and keypoints.
        '''
        # First pose during STM initialization
        R,t = utils.get_camera_pose_from_pose_msg(init_cam_pose)
        cam = Camera(R=R, t=t, fixed=True)
        cam = self.optimizer.add_camera(cam)
        # Add 3D points and lines
        self.optimizer.add_point_model_to_points(self.stm.point_model)
        #with open('/home/magnus/master_thesis/inspections/point_model.txt', 'a') as f:
        #    for pt in self.stm.point_model:
        #        f.write(f'{pt[0]},{pt[1]},{pt[2]}\n')
        self.optimizer.add_line_model_to_points(self.stm.subdivide_lines())
        # Project points to 2D and add observations
        points_2d, lines_divided_2d = self.stm.project_model_to_image(img=init_img, K=self.K, cam_pose=np.column_stack((R,t)), pose_in_world_frame=False)
        lines_divided_2d.insert(0,points_2d)
        list_of_2d_pts = list(chain.from_iterable(lines_divided_2d))
        self.optimizer.create_observations(list_of_2d_pts, cam.camera_id)
        self.last_optimization_time = rospy.Time.now()
        # Draw 3D views
        #self.three_dim_viewport.set_points_to_draw(self.optimizer.points, self.optimizer.cameras)
        self.launch_time = rospy.Time.now()
        
    def _publish_stm_params(self, params):
        # Publish lines to topic
        x = params[0]
        y = params[1]
        yaw = params[2]
        roll = params[3]
        arr = Float32MultiArray()
        arr.data.append(x)
        arr.data.append(y)
        arr.data.append(yaw)
        arr.data.append(roll)
        self.stm_params_pub.publish(arr)
        
    def _publish_pose_offset(self):
        if self.est_pose_offset is None:
            return
        pose = Pose()
        # Offset is in cam frame coords
        offset = self.est_pose_offset.copy()
        # World x = cam_z, y = -cam_x, z = -cam_y
        # Cam coords have "positive movement" as negative values, so reversed..
        pose.position.x = -offset[2,3]
        pose.position.y = offset[0,3]
        pose.position.z = offset[1,3]
        M = np.identity(4)
        M[:3, :3] = offset[:3,:3]
        q = utils.quaternion_from_matrix(M)
        # q[0] = -q.y(), q[1] = -q.z(), q[2] = q.x()
        pose.orientation.x = -q[2]
        pose.orientation.y = q[0]
        pose.orientation.z = q[1]
        pose.orientation.w = q[3]
        print(f'published trans pose: {pose}')
        self.pose_offset_pub.publish(pose)
        
    def __trigger_cb(self, msg):
        self.trigger_save = msg.data
        
    def _toggle_optimizer_cb(self, msg):
        self.run_optimizer = msg.data

    def _image_cb(self, img_msg):
        # Image callback
        clean_img = cv2.cvtColor(numpify(img_msg), cv2.COLOR_RGB2BGR)
        input_img = clean_img.copy()
        drone_img = clean_img.copy()
        if self.trigger_save:
            rospy.loginfo('Saving image..')
            cv2.imwrite('tmp_img.png', clean_img)
            self.trigger_save = False
        if self.run_optimizer:
            self.img = clean_img.copy()
            if self.stm:
                R,t = utils.get_camera_pose_from_pose_msg(self.pose)
                cam_pose = np.column_stack((R,t))
                pose_error = copy(self.pose)
                pose_error.pose.position.z -= 1
                Rerr,terr = utils.get_camera_pose_from_pose_msg(pose_error)
                cam_pose_with_error = np.column_stack((Rerr,terr))
                
                # pixels = known_width*focal_length/D'
                est_D = 100
                cam_pose_homo = np.eye(4)
                cam_pose_homo[:3, :3] = R
                cam_pose_homo[:3, 3] = t
                if np.linalg.inv(cam_pose_homo)[0,3] > 50:
                    est_D = 15
                search_radius = 4*554.92/est_D
                # Actual wing height is around 2.5m so 3m is good
                search_dist = 3*554.92/est_D
                #kps, lines_divided_2d = self.stm.project_model_to_image(img=drone_img, K=self.K, cam_pose=cam_pose, search_radius=search_radius, pose_offset=self.est_pose_offset, pose_in_world_frame=False)
                kps, lines_divided_2d = self.stm.project_model_to_image(img=drone_img, K=self.K, cam_pose=cam_pose, cam_pose_with_error=cam_pose_with_error, search_radius=search_radius, pose_offset=self.est_pose_offset, pose_in_world_frame=False)
                output = self.inferencer.forward(input_img, kps)
                new_kps = self.process_inference_output(kps, lines_divided_2d, output, search_radius, search_dist, input_img, use_line_fit=False)
                
                if (rospy.Time.now() - self.launch_time) > self.time_before_running_optimization and (rospy.Time.now() - self.last_optimization_time) > self.time_between_optimizations:
                    cam = Camera(R=Rerr, t=terr)
                    cam = self.optimizer.add_camera(cam)
                    self.optimizer.create_observations(new_kps, cam.camera_id)
                    #print(f'Point model: {self.stm.point_model}')
                    self.optimizer.optimize()
                    self.n_frames_added += 1
                    # Use current pose estimate offset from optimizer
                    self.est_pose_offset = self.optimizer.get_relative_pose_offset_new()
                    self.stm.cam_pose_from_optimizer = self.optimizer.cameras[-1].pose()[:3,:]
                    self.last_optimization_time = rospy.Time.now()
                    #self.three_dim_viewport.set_points_to_draw(self.optimizer.points, self.optimizer.cameras)
                for pt in new_kps:
                    cv2.circle(input_img, (int(pt[0]), int(pt[1])), 3, (0,0,255), 1)
        
        if self.n_frames_added >= self.n_frames_for_pose_graph:
            self._publish_pose_offset()
            self.run_optimizer = False
            # Save images during inspection
            cv2.imwrite(f'/home/magnus/master_thesis/inspections/{self.img_num}.png', clean_img)
            cv2.imwrite(f'/home/magnus/master_thesis/inspections/{self.img_num}_poses.png', drone_img)
            if self.img_num > 0:
                # Save true poses as txt file
                tx = self.true_pose.position.x
                ty = self.true_pose.position.y
                tz = self.true_pose.position.z
                tq0 = self.true_pose.orientation.x
                tq1 = self.true_pose.orientation.y
                tq2 = self.true_pose.orientation.z
                tq3 = self.true_pose.orientation.w
                tq = Quaternion()
                tq.x = tq0
                tq.y = tq1
                tq.z = tq2
                tq.w = tq3
                with open('/home/magnus/master_thesis/inspections/true_poses.txt', 'a') as f:
                    f.write(f'{tx},{ty},{tz}\n')
                # Save true poses but offset
                offset = self.est_pose_offset.copy()
                off_x = offset[2,3]
                off_y = -offset[0,3]
                off_z = -offset[1,3]
                M = np.identity(4)
                M[:3, :3] = offset[:3,:3]
                q = utils.quaternion_from_matrix(M)
                # q[0] = -q.y(), q[1] = -q.z(), q[2] = q.x()
                off_q = Quaternion()
                off_q.x = q[2]
                off_q.y = -q[0]
                off_q.z = -q[1]
                off_q.w = q[3]
                #est_offset = np.zeros((3,4))
                est_offset = np.identity(4)
                est_offset[:3,:3] = utils.quarternion_to_rotation_matrix(off_q)
                est_offset[0,3] = off_x
                est_offset[1,3] = off_y
                est_offset[2,3] = off_z
                #t_off = est_offset[:3,3]
                #R_off = est_offset[:3,:3]
                t = np.array([tx, ty, tz], dtype=np.float64)
                R = utils.quarternion_to_rotation_matrix(tq)
                #M = np.identity(4)
                Pbf = np.identity(4)
                Pbf[:3,:3] = R.copy()
                Pbf[:3,3] = t.copy()
                Poff = est_offset.copy()
                new_P = Pbf @ np.linalg.inv(Poff)
                new_x = new_P[0,3]
                new_y = new_P[1,3]
                new_z = new_P[2,3]
                # Now save..
                with open('/home/magnus/master_thesis/inspections/without_offset_poses.txt', 'a') as f:
                    f.write(f'{new_x},{new_y},{new_z}\n')
                # Save 3D wps
            self.next_wp_pub.publish(Bool(data=True))
            self.img_num += 1
            
        if not self.run_optimizer:
            self.n_frames_added = 0
        
        cv2.imshow('Drone cam', drone_img)
        cv2.imshow('Pose cam', input_img)
        cv2.waitKey(1)

    def _pose_cb(self, pose_msg):
        # Pose callback
        self.pose = pose_msg
        
    def _true_pose_cb(self, model_msg):
        # True pose callback
        self.true_pose = model_msg.pose[3]
        
    def process_inference_output(self, kps, lines_divided_2d, output, search_radius, search_dist, input_img, use_line_fit = False):
        '''
        Takes the current kps estimate based on STM and 
        output from the neural network and processes it.
        Returns 2D kp locations in image.
        '''
        scaled_search_radius = np.ceil(0.54*search_radius).astype('int')   # As scale factor is 0.54 when downscaling
        scaled_search_dist = np.ceil(0.54*search_dist).astype('int')   # As scale factor is 0.54 when downscaling
        #scaled_search_radius = np.ceil(search_radius).astype('int')
        #scaled_search_dist = np.ceil(search_dist).astype('int')
        # Wing tips
        new_kps = []
        for pt in kps[:3]:
            pt = self.inferencer.downscale_pt(pt, self.img_shape)
            if self.inferencer.version == 'v1old':
                new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,3], pt, scaled_search_radius, self.img_shape, upscale=True))
            else:
                new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,0], pt, scaled_search_radius, self.img_shape, upscale=True))
        # Rest
        #NOTE: Not using top/wing centre pts as they are too bad
        new_kps.append([-1,-1])
        new_kps.append([-1,-1])
        for i, pt in enumerate(kps[5:]):
            pt = self.inferencer.downscale_pt(pt, self.img_shape)
            if self.inferencer.version == 'v1old':
                new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,4+i+2], pt, scaled_search_radius, self.img_shape, upscale=True))
            else:
                new_kps.append(self.inferencer.get_pt_from_heatmap_within_radius(output[:,:,1+i+2], pt, scaled_search_radius, self.img_shape, upscale=True))
        #for line in lines_divided_2d:
        #   for pt in line:
        #       new_kps.append([int(pt[0]), int(pt[1])])
        # Lines
        tower_line = None
        top_line = None
        for i, line in enumerate(lines_divided_2d[:2]):
            # Vector from first pt to last pt (line vector)
            v = line[-1] - line[0]
            # Unit vector
            unit_v = v/np.linalg.norm(v)
            # Vector perpendicular to line
            unit_v_perp = np.array([unit_v[1], -unit_v[0]])
            line_pts = []
            for pt in line:
                pt = self.inferencer.downscale_pt(pt, self.img_shape)
                if self.inferencer.version == 'v1old':
                    line_pts.append(self.inferencer.get_line_from_heatmap(output[:,:,7+i], pt, unit_v_perp, scaled_search_dist, self.img_shape, upscale=True))
                else:
                    line_pts.append(self.inferencer.get_line_from_heatmap(output[:,:,4+i], pt, unit_v_perp, scaled_search_dist, self.img_shape, upscale=True))
            if use_line_fit:
                l1 = self.inferencer.fit_line_to_pts(line_pts)
            else:
                l1 = None
            if l1 is not None:
                if i == 0:
                    tower_line = l1.copy()
                elif i==1:
                    top_line = l1.copy()
                a1, b1, c1 = l1.ravel()
                if i == 0:
                    y1 = np.array([200, 400])
                    if a1 > 0 and a1 < 1e-6:
                        a1 = 1e-6
                    elif a1 < 0 and a1 > -1e-6:
                        a1 = -1e-6
                    x1 = -(b1*y1 + c1) / a1
                elif i == 1:
                    x1 = np.array([300, 370])
                    if b1 > 0 and b1 < 1e-6:
                        b1 = 1e-6
                    elif b1 < 0 and b1 > -1e-6:
                        b1 = -1e-6
                    y1 = -(a1*x1 + c1) / b1
                cv2.line(input_img, (int(x1[0]), int(y1[0])), (int(x1[1]), int(y1[1])), (255,0,255), 1)
                #print(f'l1: {l1}')
                for pt in line:
                    pt1 = np.array([pt[0] - search_dist*unit_v_perp[0], pt[1] - search_dist*unit_v_perp[1], 1])
                    pt2 = np.array([pt[0] + search_dist*unit_v_perp[0], pt[1] + search_dist*unit_v_perp[1], 1])
                    l2 = np.cross(pt1,pt2)
                    #a2, b2, c2 = l2.ravel()
                    #x2 = np.array([200, 400])
                    #y2 = -(a2*x2 + c2) / b2
                    #cv2.line(input_img, (int(x2[0]), int(y2[0])), (int(x2[1]), int(y2[1])), (255,255,0), 1)
                    #print(f'l2: {l2}')
                    intersection_pt = np.cross(l1,l2)
                    if intersection_pt[2] != 0:
                        intersection_pt /= intersection_pt[2]
                    #print(intersection_pt)
                    new_kps.append([int(intersection_pt[0]), int(intersection_pt[1])])
            else:
                for pt in line_pts:
                    new_kps.append([int(pt[0]), int(pt[1])])
        wing_line1 = None
        wing_line2 = None
        wing_line3 = None
        for i, line in enumerate(lines_divided_2d[2:]):
            # Vector from first pt to last pt (line vector)
            v = line[-1] - line[0]
            # Unit vector
            unit_v = v/np.linalg.norm(v)
            # Vector perpendicular to line
            unit_v_perp = np.array([unit_v[1], -unit_v[0]])
            line_pts = []
            for pt in line:
                pt = self.inferencer.downscale_pt(pt, self.img_shape)
                if self.inferencer.version == 'v1old':
                    line_pts.append(self.inferencer.get_line_from_heatmap(output[:,:,9], pt, unit_v_perp, scaled_search_dist, self.img_shape, upscale=True))
                else:
                    line_pts.append(self.inferencer.get_line_from_heatmap(output[:,:,6], pt, unit_v_perp, scaled_search_dist, self.img_shape, upscale=True))
            if use_line_fit:
                l1 = self.inferencer.fit_line_to_pts(line_pts)
            else:
                l1 = None
            if l1 is not None:
                #print(f'l1: {l1}')
                if i == 0:
                    wing_line1 = l1.copy()
                elif i == 1:
                    wing_line2 = l1.copy()
                elif i == 2:
                    wing_line3 = l1.copy()
                a1, b1, c1 = l1.ravel()
                if b1 > 0 and b1 < 1e-6:
                    b1 = 1e-6
                elif b1 < 0 and b1 > -1e-6:
                    b1 = -1e-6
                x1 = np.array([200, 400])
                y1 = -(a1*x1 + c1) / b1
                cv2.line(input_img, (int(x1[0]), int(y1[0])), (int(x1[1]), int(y1[1])), (255,0,255), 1)
                for pt in line:
                    pt1 = np.array([pt[0] - search_dist*unit_v_perp[0], pt[1] - search_dist*unit_v_perp[1], 1])
                    pt2 = np.array([pt[0] + search_dist*unit_v_perp[0], pt[1] + search_dist*unit_v_perp[1], 1])
                    l2 = np.cross(pt1,pt2)
                    #print(f'l2: {l2}')
                    intersection_pt = np.cross(l1,l2)
                    if intersection_pt[2] != 0:
                        intersection_pt /= intersection_pt[2]
                    #print(intersection_pt)
                    new_kps.append([int(intersection_pt[0]), int(intersection_pt[1])])
            else:
                for pt in line_pts:
                    new_kps.append([int(pt[0]), int(pt[1])])
        # Use intersections between blades to estimate wing centre kp
        # if tower_line is not None and top_line is not None:
        #    intersection_pt = np.cross(tower_line, top_line)
        #    if intersection_pt[2] != 0:
        #        intersection_pt /= intersection_pt[2]
        #    new_kps[4] = ([int(intersection_pt[0]), int(intersection_pt[1])])
        # if wing_line1 is not None and wing_line2 is not None and wing_line3 is not None:
        #     intersections = []
        #     intersections.append(np.cross(wing_line1, wing_line2))
        #     intersections.append(np.cross(wing_line2, wing_line3))
        #     intersections.append(np.cross(wing_line1, wing_line3))
        #     intersections = np.array(intersections)
        #     # Back to cartesian space (x,y)
        #     intersections = np.array([intersection[:2] / intersection[2] if intersection[2] != 0 else [np.NAN, np.NAN] for intersection in intersections])
        #     # Check for any NaNs (division by 0..)
        #     if not np.any(np.isnan(intersections).flatten()):
        #         # Calculate centroid of intersections (x,y) and round to int
        #         centroid = (np.sum(intersections, axis=0)/len(intersections)).astype(int)
        #         new_kps[3] = centroid.copy()
        #     if tower_line is not None:
        #         intersections = []
        #         intersections.append(np.cross(tower_line, wing_line1))
        #         intersections.append(np.cross(tower_line, wing_line2))
        #         intersections.append(np.cross(tower_line, wing_line3))
        #         intersections = np.array(intersections)
        #         # Back to cartesian space (x,y)
        #         intersections = np.array([intersection[:2] / intersection[2] if intersection[2] != 0 else [np.NAN, np.NAN] for intersection in intersections])
        #         # Check for any NaNs (division by 0..)
        #         if not np.any(np.isnan(intersections).flatten()):
        #             # Calculate centroid of intersections (x,y) and round to int
        #             centroid = (np.sum(intersections, axis=0)/len(intersections)).astype(int)
        #             new_kps[4] = centroid.copy()
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