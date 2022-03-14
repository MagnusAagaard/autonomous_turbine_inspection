#!/usr/bin/env python

import rospy
import mavros
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from math import sqrt, pi

from tf.transformations import quaternion_from_euler


class DroneControl:
    def __init__(self):
        self.hz = 20
        self.rate = rospy.Rate(self.hz)
        self.state = State()
        self.home_position = None
        self.current_position = PoseStamped()
        self.altitude = 65
        # Setup stuff
        self._init_publishers()
        self._init_subscribers()
        self._init_services()
        # Setup drone (arm and takeoff)
        self.setup_drone()

    def _init_publishers(self):
        # Setup publishers
        self.target_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)

    def _init_subscribers(self):
        # Setup subscribers
        self.state_sub = rospy.Subscriber('/mavros/state', State, self._state_cb)
        self.pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._position_cb)

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

    def create_pose_from_waypoint(self, wp):
        # Unpacks waypoint [x,y,z,q1,q2,q3,q4] and returns pose
        pose = PoseStamped()
        pose.pose.position.x = wp[0]
        pose.pose.position.y = wp[1]
        pose.pose.position.z = wp[2]
        pose.pose.orientation.x = wp[3]
        pose.pose.orientation.y = wp[4]
        pose.pose.orientation.z = wp[5]
        pose.pose.orientation.w = wp[6]
        return pose

    def fly_route(self, waypoints=[], hold_last_position=False):
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
                rospy.loginfo('Next waypoint')
                wp_it += 1
        # Hold last position
        if hold_last_position:
            rospy.loginfo('Holding position')
            target_position = self.create_pose_from_waypoint(waypoints[-1])
            while True:
                target_position.header = Header(stamp=rospy.Time.now())
                self.target_pos_pub.publish(target_position)
                self.rate.sleep()

def main():
    # Main loop
    rospy.init_node('drone_control', anonymous=True)
    drone = DroneControl()
    #waypoints = [[x,y,z,q1,q2,q3,q4],...]
    #q = quaternion_from_euler(0,0,pi/8)
    q = quaternion_from_euler(0,0,0)
    waypoints = [[10,0,drone.altitude, 0, 0, 0, 0], [10, 0, drone.altitude, q[0],q[1],q[2],q[3]]]
    drone.fly_route(waypoints=waypoints, hold_last_position=True)
    #drone.shutdownDrone()
    rospy.spin()

if __name__ == "__main__":
    main()