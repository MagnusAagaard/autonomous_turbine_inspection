import numpy as np
import cv2

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import get_rotation_matrix, get_rotation_matrix_from_world_to_camera_frame

class SkeletalTurbineModel:
    def __init__(self, c=(0,0), h=71.74, omega=np.pi, r=5.16, phi=0, b=35.1):
        # Init
        self.c = c          # (x,y) location of turbine tower base
        self.h = h          # Height of turbine tower (71.74m)
        self.omega = omega  # Heading of turbine relative to drone coordinate system
        self.r = r          # Length of turbine nacelle (5.16m)
        self.phi = phi      # Rotation angle of turbine blades
        self.b = b          # Length of turbine blades (35.1m)
        self.initiate_point_model()
        self.update_point_model()
        self.update_line_model()

    def initiate_point_model(self):
        # Initiate the default point model
        # Default model: x is straight ahead, y is left and z is up
        self.default_point_model = np.array(([0,0,0], 
                                     [0,0,1], 
                                     [1,0,1], 
                                     [1, 0, 2],
                                     [1, 0, 2],
                                     [1, 0 ,2]))

    def update_point_model(self):
        # Updates the current point model based on default model
        self.point_model = np.zeros((6,3))
        pc = np.append(np.array(self.c), 0)
        self.point_model[0] = pc + self.default_point_model[0]      # Turbine base
        Mh = np.diag((1,1,self.h))
        self.point_model[1] = pc + Mh @ self.default_point_model[1] # Top of turbine tower
        Mr = np.diag((self.r,1,1))
        R_omega = get_rotation_matrix(axis='z', angle=self.omega)
        self.point_model[2] = pc + R_omega @ Mr @ Mh @ self.default_point_model[2]    # Turbine hub (centre of blades)
        # Turbine blade tips
        Mb = np.diag((1,1,self.b))
        for i in range(3):
            phi = self.phi + i*2/3*np.pi
            R_phi = get_rotation_matrix(axis='x', angle=phi)
            self.point_model[i+3] = pc + R_omega @ Mr @ (R_phi @ Mb @ (self.default_point_model[i+3] - self.default_point_model[1]) + Mh @ self.default_point_model[1])

    def update_line_model(self):
        # Updates the line model based on point model
        # (line, end points, xyz)
        self.line_model = np.zeros((5,2,3))
        for i in range(3):
            self.line_model[i] = self.point_model[i:i+2]
        # Blades are different
        self.line_model[3,0] = self.point_model[2]
        self.line_model[3,1] = self.point_model[4]
        self.line_model[4,0] = self.point_model[2]
        self.line_model[4,1] = self.point_model[5]

    def plot_model(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=0, azim=0)
        #ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.scatter(self.point_model[:,0], self.point_model[:,1], self.point_model[:,2])
        for line in self.line_model:
            ax.plot(line[:,0], line[:,1], line[:,2])
        set_axes_equal(ax)
        plt.show()

    def project_model_to_image(self, img=None, K=None, cam_pose=None, show_img=False):
        # Project the model into image coordinate system (2D)
        if img is None:
            img_h = 480
            img_w = 640
            img = np.ones((img_h, img_w, 3), dtype=np.uint8)*255
        # Intrinsic parameters:
        if K is None:
            K = np.array([[554.920125, 0.000000, 320.077433], 
                     [0.000000, 554.921917, 239.661438], 
                     [0.000000, 0.000000, 1.000000]])
        # Extrensic parameters
        if cam_pose is None:
            # Get transform from world frame to camera frame
            Rex = get_rotation_matrix_from_world_to_camera_frame()
            # Camera pose/extrinsic parameters [R|t]:
            #R = np.array([[1,0,0],[0,1,0],[0,0,1]])
            R = get_rotation_matrix('y', np.pi/2)
            #t = np.array([2,2,1])
            t = np.array([0,0,0])
            cam_pose = np.column_stack((Rex @ R, -Rex @ R @ t))
            #print(cam_pose)
            #cam_pose = np.column_stack((R, t))
        # Projection matrix P = K [R|t]
        P = K @ cam_pose
        # Project to 2D
        img_pts = np.zeros((6,2))
        for i, pt in enumerate(self.point_model):
            # Transform point to homogenous coords
            point = np.copy(pt)
            point = np.append(point,1)
            #point_in_cam_coords = cam_pose @ point
            #print("Point in cam coords")
            #print(point_in_cam_coords)
            #point_in_cam_coords /= point_in_cam_coords[2]
            #print(point_in_cam_coords)
            #print("From cam coords")
            #print(K @ point_in_cam_coords)
            #print("From perspective")
            # Perspective transform
            point = P @ point
            # Re-scale homogenous point
            if point[2] > 0:
                point /= point[2]
            img_pts[i,:] = point[:2]
        # Show results
        for u, v in img_pts:
            cv2.circle(img, (int(u), int(v)), 5, (0,255,0), -1)
        for i in range(3):
            cv2.line(img, (int(img_pts[i,0]), int(img_pts[i,1])), (int(img_pts[i+1,0]), int(img_pts[i+1,1])), (255,0,0), 2)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[3,0]), int(img_pts[3,1])), (255,0,0), 2)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[4,0]), int(img_pts[4,1])), (255,0,0), 2)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[5,0]), int(img_pts[5,1])), (255,0,0), 2)
        if show_img:
            cv2.imshow('Projected point model', img)
            cv2.waitKey(0)
        

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        

def main():
    # Main
    stm = SkeletalTurbineModel()
    #stm.project_model_to_image(show_img=True)
    stm.plot_model()
    #utils.dist_between_points(stm.point_model[2],stm.point_model[3])

if __name__ == "__main__":
    main()