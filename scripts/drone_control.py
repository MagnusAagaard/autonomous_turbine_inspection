#!/usr/bin/env python3

import rospy
import mavros
from std_msgs.msg import Header, Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from math import sqrt, pi, atan2
import numpy as np
from copy import copy

#from tf.transformations import quaternion_from_euler
from skeletal_turbine_model import SkeletalTurbineModel
import utils

from itertools import chain

class DroneControl:
    def __init__(self):
        self.hz = 20
        self.rate = rospy.Rate(self.hz)
        self.state = State()
        self.home_position = None
        self.current_position = PoseStamped()
        self.altitude = 120
        self.stm = None
        self.est_offset = None
        # Setup stuff
        self._init_publishers()
        self._init_subscribers()
        self._init_services()
        # Setup drone (arm and takeoff)
        self.setup_drone()
        self.go_to_next_wp = True
        
    def _init_publishers(self):
        # Setup publishers
        self.target_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        self.toggle_pose_estimator_pub = rospy.Publisher('/drone_control/toggle_pose_estimator', Bool, queue_size=1)

    def _init_subscribers(self):
        # Setup subscribers
        self.state_sub = rospy.Subscriber('/mavros/state', State, self._state_cb)
        self.pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._position_cb)
        self.next_wp_sub = rospy.Subscriber('/pose_estimator/next_wp', Bool, self._next_wp_cb)
        self.turbine_params = rospy.Subscriber('/pose_estimator/turbine_params', Float32MultiArray, self._stm_params_cb)
        self.offset_sub = rospy.Subscriber('/pose_estimator/pose_offset', Pose, self._pose_offset_cb)

    def _init_services(self):
        # Setup services
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.loginfo("/mavros/cmd/arming service ready!")
        rospy.wait_for_service("mavros/set_mode")
        rospy.loginfo("/mavros/set_mode service ready!")
        rospy.wait_for_service("mavros/cmd/land")
        rospy.loginfo("/mavros/cmd/land service ready!")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.land_client = rospy.ServiceProxy("/mavros/cmd/land", CommandTOL)
        self.takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

    # Callbacks
    def _state_cb(self, state):
        self.state = state

    def _position_cb(self, position):
        self.current_position = position
        # Save home position
        if self.home_position is None:
            self.home_position = position
    
    def _pose_offset_cb(self, pose_msg):
        offset = np.zeros((3,4))
        offset[:3,:3] = utils.quarternion_to_rotation_matrix(pose_msg.orientation)
        offset[0,3] = pose_msg.position.x
        offset[1,3] = pose_msg.position.y
        offset[2,3] = pose_msg.position.z
        self.est_offset = np.copy(offset)
        
    def _next_wp_cb(self, msg):
        self.go_to_next_wp = msg.data
        
    def _stm_params_cb(self, params):
        x = params.data[0]
        y = params.data[1]
        yaw = params.data[2]
        roll = params.data[3]
        self.stm = SkeletalTurbineModel(c=(x,y), omega=yaw, phi=roll)

    def setup_drone(self):
        rospy.loginfo("Waiting for FCU connection...")
        while not self.state.connected:
            self.rate.sleep()
        rospy.loginfo("FCU connected")

        rospy.loginfo("Waiting on position...")
        while self.home_position is None:
            self.rate.sleep()
        rospy.loginfo("Position received")

        self.take_off_position = PoseStamped()
        self.take_off_position.pose.position.x = self.home_position.pose.position.x
        self.take_off_position.pose.position.y = self.home_position.pose.position.y
        self.take_off_position.pose.position.z = self.altitude

        # Send a few takeoff commands before starting
        for i in range(20):
            self.target_pos_pub.publish(self.take_off_position)
            self.rate.sleep()

        rospy.loginfo("Waiting for change mode to offboard & arming rotorcraft...")
        while not self.state.mode == "OFFBOARD" and not self.state.armed:
            if not self.state.mode == "OFFBOARD":
                self.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
            if not self.state.armed:
                self.arming_client(True)
        rospy.loginfo("Offboard mode enabled and rotorcraft armed")

        # Reached takeoff position
        dist_to_takeoff_pos = self.altitude
        while(0.5 < dist_to_takeoff_pos):
            dist_to_takeoff_pos = self.distance_to_target(target_position=self.take_off_position, include_z=True)
            self.take_off_position.header = Header(stamp=rospy.Time.now())
            self.target_pos_pub.publish(self.take_off_position)
            self.rate.sleep()
        rospy.loginfo('Setup done.')

    def distance_to_target(self, target_position, include_z=False):
        x = target_position.pose.position.x - self.current_position.pose.position.x
        y = target_position.pose.position.y - self.current_position.pose.position.y
        if not include_z:
            return sqrt(x*x + y*y)
        z = target_position.pose.position.z - self.current_position.pose.position.z
        return sqrt(x*x + y*y + z*z)

    def create_pose_from_waypoint(self, wp, offset=None):
        # Unpacks waypoint [x,y,z,q1,q2,q3,q4] and returns pose
        pose = PoseStamped()
        # Create normal wp
        pose.pose.position.x = wp[0]
        pose.pose.position.y = wp[1]
        pose.pose.position.z = wp[2]
        pose.pose.orientation.x = wp[3]
        pose.pose.orientation.y = wp[4]
        pose.pose.orientation.z = wp[5]
        pose.pose.orientation.w = wp[6]
        if offset is not None:
            t_off = offset[:3,3]
            R_off = offset[:3,:3]
            t = np.array([wp[0], wp[1], wp[2]], dtype=np.float64)
            R = utils.quarternion_to_rotation_matrix(pose.pose.orientation)
            M = np.identity(4)
            Pbf = np.identity(4)
            Pbf[:3,:3] = R.copy()
            Pbf[:3,3] = t.copy()
            Poff = np.identity(4)
            Poff[:3,:3] = offset[:3,:3].copy()
            Poff[:3,3] = offset[:3,3].copy()
            # Add the two poses
            #new_P = Pbf @ Poff
            # "subtract" the two poses, ie. T2 "-" T1
            #new_P = np.linalg.inv(Poff) @ Pbf
            # Apply offset like addition, but opposite: P = Pbf @ Poff --> P = Pbf @ inv(Poff)
            new_P = Pbf @ np.linalg.inv(Poff)
            # In cam frame we do P_off @ inv(P)
            #print(f'Pose before offset: {Pbf}')
            #print(f'Pose after offset: {new_P}')
            #print(f'Offset: {Poff}')
            M[:3, :3] = new_P[:3,:3]
            new_q = utils.quaternion_from_matrix(M)
            #TODO: Figure out if turbine base parameter should be offset instead?
            pose.pose.position.x = new_P[0,3]
            pose.pose.position.y = new_P[1,3]
            pose.pose.position.z = new_P[2,3]
            pose.pose.orientation.x = new_q[0]
            pose.pose.orientation.y = new_q[1]
            pose.pose.orientation.z = new_q[2]
            pose.pose.orientation.w = new_q[3]
        return pose

    def fly_route(self, waypoints=[], hold_last_position=True, hold_first_position=False):
        if not len(waypoints) >= 1:
            rospy.loginfo('Tried to fly route, but no waypoints specified.')
            return -1
        rospy.loginfo("Flying route")
        wp_it = 0
        while (wp_it < len(waypoints) and self.state.armed):
            target_position = self.create_pose_from_waypoint(waypoints[wp_it])
            target_position.header = Header(stamp=rospy.Time.now())
            self.target_pos_pub.publish(target_position)
            self.rate.sleep()
            if(self.distance_to_target(target_position) < 0.50):
                if wp_it == 0 and hold_first_position:
                    rospy.loginfo('Holding first position for 30 sec..')
                    now = rospy.Time.now()
                    while (rospy.Time.now() - now) < rospy.Duration(secs=30):
                        self.target_pos_pub.publish(target_position)
                        self.rate.sleep()
                elif wp_it > 0:
                    rospy.loginfo('Holding position for 10 sec..')
                    now = rospy.Time.now()
                    while (rospy.Time.now() - now) < rospy.Duration(secs=10):
                        self.target_pos_pub.publish(target_position)
                        self.rate.sleep()
                rospy.loginfo('Next waypoint')
                print(target_position)
                wp_it += 1
        # Hold last position
        if hold_last_position:
            rospy.loginfo('Holding position')
            target_position = self.create_pose_from_waypoint(waypoints[-1])
            while True:
                target_position.header = Header(stamp=rospy.Time.now())
                self.target_pos_pub.publish(target_position)
                self.rate.sleep()
    
    def create_circular_waypoints(self, center, radius):
        center = [110, 0]
        xs = [center[0] + radius*np.sin(np.deg2rad(x-90)) for x in range(360)]
        ys = [center[1] - radius*np.cos(np.deg2rad(y-90)) for y in range(360)]
        angles = [atan2(center[1] - ys[i], center[0] - xs[i]) for i in range(360)]
        qs = [utils.quaternion_from_euler(0,0,angle) for angle in angles]
        wps = [[xs[i], ys[i], self.altitude, qs[i][0], qs[i][1], qs[i][2], qs[i][3]] for i in range(360)]
        return wps
    
    def create_square_waypoints(self):
        c1 = [0,0,0]
        #c2 = [10,-10,0]
        #c3 = [20,-10,0]
        #c4 = [20,0,0]
        c2 = [0, -10, np.pi/2]
        c3 = [10, -10, np.pi]
        c4 = [10, 0, 3*np.pi/2]
        corners = [c1,c2,c3,c4]
        wps = []
        wps_list = []
        for c in corners*10:
            qs = utils.quaternion_from_euler(0,0,c[2])
            wps.append([c[0], c[1], self.altitude, qs[0], qs[1], qs[2], qs[3]])
        wps_list.append(wps)
        return wps_list
    
    def create_static_waypoints(self):
        wps_list = []
        wps = []
        wp = [10, 0, self.altitude, 0, 0, 0, 0]
        for i in range(300):
            wps.append(wp)
        wps_list.append(wps)
        return wps_list
        
    
    def cross_lines(self, v1, v2):
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        uv12 = np.cross(uv1, uv2)
        uv12 /= np.linalg.norm(uv12)
        return uv12
    
    def get_circular_motion_around_wingtip(self, center, radius=15, step_size=10, inverse=False):
        if inverse:
            xs = [center[0] + radius*np.sin(self.stm.omega - np.deg2rad(x-90)) for x in range(0, 181, step_size)]
            ys = [center[1] - radius*np.cos(self.stm.omega - np.deg2rad(y-90)) for y in range(0, 181, step_size)]
        else:
            xs = [center[0] + radius*np.sin(self.stm.omega - np.deg2rad(x+90)) for x in range(0, 181, step_size)]
            ys = [center[1] - radius*np.cos(self.stm.omega - np.deg2rad(y+90)) for y in range(0, 181, step_size)]
        zs = [center[2] for z in range(0, 181, step_size)]
        angles = [atan2(center[1] - ys[i], center[0] - xs[i]) for i in range(len(xs))]
        qs = [utils.quaternion_from_euler(0,0,angle) for angle in angles]
        pts = [[xs[i], ys[i], zs[i], qs[i][0], qs[i][1], qs[i][2], qs[i][3]] for i in range(len(xs))]
        return pts
    
    def get_wps_from_model_lines(self, model_lines, dist=15):
        '''
        Calculates perpendicular line offset to points at certain distance.
        Model_lines are lines from wing_center --> wing tips
        '''
        # Perp vector is wing1 x wing2 == wing1 x wing3 == wing2 x wing3
        wing1_pts = model_lines[0]
        wing2_pts = model_lines[1]
        wing3_pts = model_lines[2]
        w1_line = np.asarray(wing1_pts[-1] - wing1_pts[0])
        w2_line = np.asarray(wing2_pts[-1] - wing2_pts[0])
        p_uv = self.cross_lines(w1_line, w2_line)
        wps = []
        # Omega paramter is estimated with wind turbine orientation 0 in negative x direction
        # ie. 180 degrees offset, so pi must be subtracted to get correct orientation
        q_front = utils.quaternion_from_euler(0,0,self.stm.omega-np.pi)
        # pi added to get 180 degree offset
        q_back = utils.quaternion_from_euler(0,0,self.stm.omega)
        # Always same direction around turbine tips as parameter estimation will be [0-60]
        for i, wing in enumerate(model_lines):
            if i % 2 == 0:
                wps.append([[w[0] + p_uv[0]*dist, w[1] + p_uv[1]*dist, w[2] + p_uv[2]*dist, q_front[0], q_front[1], q_front[2], q_front[3]] for w in wing])
                wps.append(self.get_circular_motion_around_wingtip(wing[-1], inverse=True))
                wps.append([[w[0] - p_uv[0]*dist, w[1] - p_uv[1]*dist, w[2] - p_uv[2]*dist, q_back[0], q_back[1], q_back[2], q_back[3]] for w in wing[::-1]])
            else:
                wps.append([[w[0] - p_uv[0]*dist, w[1] - p_uv[1]*dist, w[2] - p_uv[2]*dist, q_back[0], q_back[1], q_back[2], q_back[3]] for w in wing])
                wps.append(self.get_circular_motion_around_wingtip(wing[-1], inverse=False))
                wps.append([[w[0] + p_uv[0]*dist, w[1] + p_uv[1]*dist, w[2] + p_uv[2]*dist, q_front[0], q_front[1], q_front[2], q_front[3]] for w in wing[::-1]])
        return wps
    
    def run_inspection(self, init_pos):
        '''
        Creates waypoints around wind turbine, based on current pose estimates
        '''
        STATE = 'INIT'
        current_wp = self.create_pose_from_waypoint(init_pos)
        # Line iterator
        line_it = 0
        # Waypoint iterator (each pt in line)
        wp_it = 0
        while(STATE != 'DONE'):
            if STATE == 'INIT':
                if self.go_to_next_wp:
                    # Pause pose estimator before moving
                    self.pause_pose_estimator()
                    # Fly to waypoint and wait 2 sec.
                    self.fly_to_wp_and_wait(current_wp)
                    self.go_to_next_wp = False
                    # Start initial pose estimation
                    self.start_pose_estimator()
                    STATE = 'WAIT_FOR_INIT_POSE'
            elif STATE == 'WAIT_FOR_INIT_POSE':
                # Wait for initial pose estimation to finish
                if self.stm:
                    rospy.loginfo('Init pose obtained')
                    self.model_lines = self.stm.subdivide_lines(waypoints=True)
                    # Get perpendicular point at X distance
                    self.wps = self.get_wps_from_model_lines(self.model_lines[2:], dist=20)
                    #for i, step in enumerate(self.wps):
                    #    for pt in step:
                    #        if i % 3 != 1:
                    #            with open('/home/magnus/master_thesis/inspections/inspection_waypoints.txt', 'a') as f:
                    #                f.write(f'{pt[0]},{pt[1]},{pt[2]-1}\n')
                    #self.wps = self.create_square_waypoints()
                    #self.wps = self.create_static_waypoints()
                    STATE = 'WAIT_FOR_POSE_ESTIMATOR'
                else:
                    self.publish_wp_and_sleep(current_wp)
            elif STATE == 'WAIT_FOR_POSE_ESTIMATOR':
                if self.go_to_next_wp:
                    # Pause pose estimator before moving
                    self.pause_pose_estimator()
                    # Fly to waypoint and wait 2 sec.
                    if line_it > 0 or wp_it > 0:
                        current_wp = self.create_pose_from_waypoint(self.wps[line_it][wp_it], offset=self.est_offset)
                    else:
                        current_wp = self.create_pose_from_waypoint(self.wps[line_it][wp_it], offset=None)
                    if line_it % 3 == 1 and len(self.wps[line_it]) - line_it > 1:
                        self.fly_to_wp(current_wp)
                    else:
                        self.fly_to_wp_and_wait(current_wp)
                    wp_it += 1
                    if not line_it % 3 == 1:
                        self.go_to_next_wp = False
                        self.start_pose_estimator()
                    if len(self.wps[line_it]) - wp_it < 1:
                        line_it += 1
                        wp_it = 0
                    if len(self.wps) - line_it < 1:
                        STATE = 'TERMINATE'
                else:
                    self.publish_wp_and_sleep(current_wp)
            elif STATE == 'TERMINATE':
                if self.go_to_next_wp:
                    self.go_to_next_wp = False
                    self.start_pose_estimator
                    STATE = 'DONE'
                self.publish_wp_and_sleep(current_wp)
                
        self.pause_pose_estimator()
        rospy.loginfo('Done state reached.. Returning to home')
        current_wp.pose.position.z = self.stm.h + self.stm.b + 5
        while self.distance_to_target(current_wp, include_z=True) > 2.0:
            self.publish_wp_and_sleep(current_wp)
        home = copy(self.home_position)
        home.pose.position.x = 0.0
        home.pose.position.y = 0.0
        home.pose.position.z = self.stm.h + self.stm.b + 5
        self.fly_to_wp_and_wait(home)
        home.pose.position.z = -1.0
        rospy.loginfo('Landing..')
        while True:
            self.publish_wp_and_sleep(home)
                
    def pause_pose_estimator(self):
        self.toggle_pose_estimator_pub.publish(Bool(data=False))
    
    def start_pose_estimator(self):
        self.toggle_pose_estimator_pub.publish(Bool(data=True))
    
    def fly_to_wp_and_wait(self, wp):
        # Flies to waypoint and waits for signal to continue
        rospy.loginfo('New waypoint recieved.')
        target_position = wp
        target_position.header = Header(stamp=rospy.Time.now())
        while self.distance_to_target(target_position) > 0.50:
            self.publish_wp_and_sleep(target_position)
        # Waypoint within 0.5m, hold for 2 sec.
        rospy.loginfo('Waypoint within 0.5m, hold for 5 sec..')
        now = rospy.Time.now()
        while (rospy.Time.now() - now) < rospy.Duration(secs=5):
            self.publish_wp_and_sleep(target_position)
            
    def publish_wp_and_sleep(self, wp):
        target_position = wp
        target_position.header = Header(stamp=rospy.Time.now())
        self.target_pos_pub.publish(target_position)
        self.rate.sleep()
        
    def fly_to_wp(self, wp):
        rospy.loginfo('New waypoint recieved.')
        target_position = wp
        target_position.header = Header(stamp=rospy.Time.now())
        while self.distance_to_target(target_position) > 0.50:
            self.publish_wp_and_sleep(target_position)
        

def main():
    # Main loop
    rospy.init_node('drone_control', anonymous=True)
    drone = DroneControl()
    init_pos = [10,0,drone.altitude, 0, 0, 0, 0]
    drone.run_inspection(init_pos)
    #waypoints = [[x,y,z,q1,q2,q3,q4],...]
    #q = quaternion_from_euler(0,0,pi/8)
    #q = quaternion_from_euler(0,0,0)
    #waypoints = [[10,0,drone.altitude, 0, 0, 0, 0], [10, 0, drone.altitude, q[0],q[1],q[2],q[3]]]
    #waypoints = [[10, i*5, drone.altitude, 0, 0, 0, 0] for i in range(100)]
    #waypoints = [[10, 0, drone.altitude, 0, 0, 0, 0]]
    #cricle_points = drone.create_circular_waypoints(center=[110,0], radius=40)
    #for pt in cricle_points:
    #    waypoints.append(pt)
    #drone.fly_route(waypoints=waypoints, hold_first_position=True)
    #drone.shutdownDrone()
    rospy.spin()

if __name__ == "__main__":
    main()