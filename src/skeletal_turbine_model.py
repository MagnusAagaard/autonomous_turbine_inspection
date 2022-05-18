import numpy as np
import cv2

# Plotting
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from utils import get_rotation_matrix, get_rotation_matrix_from_world_to_camera_frame, convert_pose_to_camera_frame

class SkeletalTurbineModel:
    def __init__(self, c=(0,0), h=71.74-8, omega=np.pi-np.pi/4, r=5.16, phi=0, b=35.1):
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
        self.init_point_model = np.copy(self.point_model)
        self.cam_pose_from_optimizer = None

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
        #ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=0, azim=180)
        ax.set_axis_off()
        #ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # Points
        #ax.scatter(self.point_model[:,0], self.point_model[:,1], self.point_model[:,2])
        ax.scatter(self.point_model[0,0], self.point_model[0,1], self.point_model[0,2], c='m')
        #ax.text(self.point_model[0,0], self.point_model[0,1], self.point_model[0,2], 'pb')
        ax.scatter(self.point_model[1,0], self.point_model[1,1], self.point_model[1,2], c='b')
        ax.scatter(self.point_model[2,0], self.point_model[2,1], self.point_model[2,2], c='g')
        ax.scatter(self.point_model[3,0], self.point_model[3,1], self.point_model[3,2], c='r')
        ax.scatter(self.point_model[4,0], self.point_model[4,1], self.point_model[4,2], c='r')
        ax.scatter(self.point_model[5,0], self.point_model[5,1], self.point_model[5,2], c='r')
        # Lines
        ax.plot(self.line_model[0,:,0], self.line_model[0,:,1], self.line_model[0,:,2], c='b')
        ax.plot(self.line_model[1,:,0], self.line_model[1,:,1], self.line_model[1,:,2], c='g')
        ax.plot(self.line_model[2,:,0], self.line_model[2,:,1], self.line_model[2,:,2], c='g')
        ax.plot(self.line_model[3,:,0], self.line_model[3,:,1], self.line_model[3,:,2], c='b')
        ax.plot(self.line_model[4,:,0], self.line_model[4,:,1], self.line_model[4,:,2], c='r')
        uv12 = self.cross_lines(self.line_model[2], self.line_model[3])
        ax.plot(self.line_model[2,:,0] + uv12[0]*15, self.line_model[2,:,1] + uv12[1]*15, self.line_model[2,:,2]+ uv12[2]*15, c='g', ls='--')
        ax.plot(self.line_model[3,:,0] + uv12[0]*15, self.line_model[3,:,1] + uv12[1]*15, self.line_model[3,:,2]+ uv12[2]*15, c='b', ls='--')
        ax.plot(self.line_model[4,:,0] + uv12[0]*15, self.line_model[4,:,1] + uv12[1]*15, self.line_model[4,:,2]+ uv12[2]*15, c='r', ls='--')
        circular_wps1 = self.get_circular_motion_around_wingtip(self.line_model[2,1,:], inverse=True)
        circular_wps2 = self.get_circular_motion_around_wingtip(self.line_model[3,1,:], inverse=False)
        circular_wps3 = self.get_circular_motion_around_wingtip(self.line_model[4,1,:], inverse=True)
        ax.plot(circular_wps1[0], circular_wps1[1], circular_wps1[2], c='g', ls='--')
        ax.plot(circular_wps2[0], circular_wps2[1], circular_wps2[2], c='b', ls='--')
        ax.plot(circular_wps3[0], circular_wps3[1], circular_wps3[2], c='r', ls='--')
        ax.plot(self.line_model[2,:,0] - uv12[0]*15, self.line_model[2,:,1] - uv12[1]*15, self.line_model[2,:,2]- uv12[2]*15, c='g', ls='--')
        ax.plot(self.line_model[3,:,0] - uv12[0]*15, self.line_model[3,:,1] - uv12[1]*15, self.line_model[3,:,2]- uv12[2]*15, c='b', ls='--')
        ax.plot(self.line_model[4,:,0] - uv12[0]*15, self.line_model[4,:,1] - uv12[1]*15, self.line_model[4,:,2]- uv12[2]*15, c='r', ls='--')
        # Get waypoints along inspection
        wps = self.subdivide_lines(waypoints=True)[2:]
        xs0 = [x[0] for x in wps[0]]
        xs1 = [x[0] for x in wps[1]]
        xs2 = [x[0] for x in wps[2]]
        ys0 = [y[1] for y in wps[0]]
        ys1 = [y[1] for y in wps[1]]
        ys2 = [y[1] for y in wps[2]]
        zs0 = [z[2] for z in wps[0]]
        zs1 = [z[2] for z in wps[1]]
        zs2 = [z[2] for z in wps[2]]
        ax.scatter(xs0+uv12[0]*15, ys0+uv12[1]*15, zs0+uv12[2]*15, marker='*', c='m', s=40)
        ax.scatter(xs0-uv12[0]*15, ys0-uv12[1]*15, zs0-uv12[2]*15, marker='*', c='m', s=40)
        ax.scatter(xs1+uv12[0]*15, ys1+uv12[1]*15, zs1+uv12[2]*15, marker='*', c='m', s=40)
        ax.scatter(xs1-uv12[0]*15, ys1-uv12[1]*15, zs1-uv12[2]*15, marker='*', c='m', s=40)
        ax.scatter(xs2+uv12[0]*15, ys2+uv12[1]*15, zs2+uv12[2]*15, marker='*', c='m', s=40)
        ax.scatter(xs2-uv12[0]*15, ys2-uv12[1]*15, zs2-uv12[2]*15, marker='*', c='m', s=40)
        marker = mlines.Line2D([], [], color='m', marker='*', linestyle='None',
                          markersize=10, label='Inspection waypoints')
        ax.legend(handles=[marker])
        #for line in self.line_model:
        #    ax.plot(line[:,0], line[:,1], line[:,2])
        set_axes_equal(ax)
        #fig.savefig('/home/magnus/instantiated_model.pdf', bbox_inches='tight')
        plt.show()
        fig.savefig('/home/magnus/inspection_path.pdf', bbox_inches='tight')
        
    def cross_lines(self, line1, line2):
        v1 = line1[1] - line1[0]
        v2 = line2[1] - line2[0]
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        uv12 = np.cross(uv1, uv2)
        uv12 /= np.linalg.norm(uv12)
        return uv12
    
    def get_circular_motion_around_wingtip(self, center, radius=15, step_size=10, inverse=False):
        if inverse:
            xs = [center[0] + radius*np.sin(self.omega - np.deg2rad(x-90)) for x in range(0, 181, step_size)]
            ys = [center[1] - radius*np.cos(self.omega - np.deg2rad(y-90)) for y in range(0, 181, step_size)]
        else:
            xs = [center[0] + radius*np.sin(self.omega - np.deg2rad(x+90)) for x in range(0, 181, step_size)]
            ys = [center[1] - radius*np.cos(self.omega - np.deg2rad(y+90)) for y in range(0, 181, step_size)]
        zs = [center[2] for z in range(0, 181, step_size)]
        return [xs, ys, zs]
    
    def get_line_steps_for_waypoints(self):
        '''
        Lines are subdivided into fixed amount of points based on the initial parameters.
        Step sizes are larger for waypoints than 3D points
        '''
        step_sizes = [1, 1, 5]
        tower_step = int(self.h / step_sizes[0])
        top_step = int(self.r / step_sizes[1])
        blade_step = int(self.b / step_sizes[2])
        return [tower_step, top_step, blade_step, blade_step, blade_step]
        
    def get_line_steps(self):
        '''
        Lines are subdivided into fixed amount of points based on the initial parameters.
        '''
        step_sizes = [1, 1, 1]
        tower_step = int(self.h / step_sizes[0])
        top_step = int(self.r / step_sizes[1])
        blade_step = int((self.b - step_sizes[2]) / step_sizes[2])
        return [tower_step, top_step, blade_step, blade_step, blade_step]
        
        
    def subdivide_lines(self, waypoints=False):
        '''
        Subdivides the line models into points along those lines. These points
        can be searched in a perpendicular direction to find correspondence
        with the output from the neural network.
        '''
        # Step sizes in [m] - tower-->top-->wing_center-->wings
        if waypoints:
            step_sizes = self.get_line_steps_for_waypoints()
        else:
            step_sizes = self.get_line_steps()
        lines_divided = []
        for i, line in enumerate(self.line_model):
            mag = np.linalg.norm(line[1]-line[0])
            #steps = int(mag / step_sizes[i])
            steps = step_sizes[i]
            dxyz = mag/steps
            #NOTE: For error test
            if waypoints:
                line[0][2] += 1
                line[1][2] += 1
            unit_vector = (line[1]-line[0])/mag
            line_divided = []
            for j in range(1,steps+1):
                sub_pt = line[0] + unit_vector*dxyz*j
                line_divided.append(sub_pt)
            lines_divided.append(line_divided)
            
        return lines_divided
            

    def project_model_to_image(self, img=None, K=None, cam_pose=None, cam_pose_with_error=None, search_radius=40, pose_in_world_frame=False, pose_offset=None, show_img=False):
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
        if pose_in_world_frame:
            cam_pose = convert_pose_to_camera_frame(cam_pose)
        
        # For error model model
        if cam_pose_with_error is not None:
            P = K @ cam_pose_with_error
            img_pts_error = np.zeros((6,2))
            for i, pt in enumerate(self.init_point_model):
                # Transform point to homogenous coords
                point = np.copy(pt)
                point = np.append(point,1)
                # Perspective transform
                point = P @ point
                # Re-scale homogenous point
                if point[2] > 0:
                    point /= point[2]
                img_pts_error[i,:] = point[:2]
            # Show results
            for u, v in img_pts_error:
                cv2.circle(img, (int(u), int(v)), 5, (0,0,255), 1)
                cv2.circle(img, (int(u), int(v)), int(search_radius), (0,0,255), 1)
            for i in range(2):
                cv2.line(img, (int(img_pts_error[i,0]), int(img_pts_error[i,1])), (int(img_pts_error[i+1,0]), int(img_pts_error[i+1,1])), (0,0,255), 1)
            cv2.line(img, (int(img_pts_error[2,0]), int(img_pts_error[2,1])), (int(img_pts_error[3,0]), int(img_pts_error[3,1])), (0,0,255), 1)
            cv2.line(img, (int(img_pts_error[2,0]), int(img_pts_error[2,1])), (int(img_pts_error[4,0]), int(img_pts_error[4,1])), (0,0,255), 1)
            cv2.line(img, (int(img_pts_error[2,0]), int(img_pts_error[2,1])), (int(img_pts_error[5,0]), int(img_pts_error[5,1])), (0,0,255), 1)
        
        # For init point model
        P = K @ cam_pose
        img_pts_init = np.zeros((6,2))
        for i, pt in enumerate(self.init_point_model):
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
            img_pts_init[i,:] = point[:2]
        # Show results
        #for u, v in img_pts_init:
        #    cv2.circle(img, (int(u), int(v)), 5, (255,0,0), 1)
        #    cv2.circle(img, (int(u), int(v)), int(search_radius), (255,0,0), 1)
        #for i in range(2):
        #    cv2.line(img, (int(img_pts_init[i,0]), int(img_pts_init[i,1])), (int(img_pts_init[i+1,0]), int(img_pts_init[i+1,1])), (255,0,0), 1)
        #cv2.line(img, (int(img_pts_init[2,0]), int(img_pts_init[2,1])), (int(img_pts_init[3,0]), int(img_pts_init[3,1])), (255,0,0), 1)
        #cv2.line(img, (int(img_pts_init[2,0]), int(img_pts_init[2,1])), (int(img_pts_init[4,0]), int(img_pts_init[4,1])), (255,0,0), 1)
        #cv2.line(img, (int(img_pts_init[2,0]), int(img_pts_init[2,1])), (int(img_pts_init[5,0]), int(img_pts_init[5,1])), (255,0,0), 1)
        
        
        '''
        if self.cam_pose_from_optimizer is not None:
            P = K @ self.cam_pose_from_optimizer
        # Project to 2D
        img_pts = np.zeros((6,2))
        for i, pt in enumerate(self.point_model):
            # Transform point to homogenous coords
            point = np.copy(pt)
            point = np.append(point,1)
            # Perspective transform
            point = P @ point
            # Re-scale homogenous point
            if point[2] > 0:
                point /= point[2]
            img_pts[i,:] = point[:2]
        # Show results
        for u, v in img_pts:
            cv2.circle(img, (int(u), int(v)), 5, (0,255,0), 1)
            cv2.circle(img, (int(u), int(v)), int(search_radius), (0,255,0), 1)
        for i in range(2):
            cv2.line(img, (int(img_pts[i,0]), int(img_pts[i,1])), (int(img_pts[i+1,0]), int(img_pts[i+1,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[3,0]), int(img_pts[3,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[4,0]), int(img_pts[4,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts[2,0]), int(img_pts[2,1])), (int(img_pts[5,0]), int(img_pts[5,1])), (0,255,0), 1)
        '''
        
        if pose_offset is not None and cam_pose_with_error is not None:
            P = np.identity(4)
            #P[:3,:] = cam_pose.copy()
            P[:3,:] = cam_pose_with_error.copy()
            inv_cam_pose = np.linalg.inv(P)
            # Offset in cam frame 
            P_off = pose_offset.copy()
            # Apply offset to cam pose: P_off @ inv(cam_pose)
            T = P_off @ P
            #print(f'Est. pose with offset: {T}')
            #print(f'Optimized pose: {self.cam_pose_from_optimizer}')
            P = K @ T[:3,:]
            
        # Project to 2D
        img_pts_offset = np.zeros((6,2))
        for i, pt in enumerate(self.point_model):
            # Transform point to homogenous coords
            point = np.copy(pt)
            point = np.append(point,1)
            # Perspective transform
            point = P @ point
            # Re-scale homogenous point
            if point[2] > 0:
                point /= point[2]
            img_pts_offset[i,:] = point[:2]
        # Show results
        for u, v in img_pts_offset:
            cv2.circle(img, (int(u), int(v)), 5, (0,255,0), 1)
            cv2.circle(img, (int(u), int(v)), int(search_radius), (0,255,0), 1)
        for i in range(2):
            cv2.line(img, (int(img_pts_offset[i,0]), int(img_pts_offset[i,1])), (int(img_pts_offset[i+1,0]), int(img_pts_offset[i+1,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts_offset[2,0]), int(img_pts_offset[2,1])), (int(img_pts_offset[3,0]), int(img_pts_offset[3,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts_offset[2,0]), int(img_pts_offset[2,1])), (int(img_pts_offset[4,0]), int(img_pts_offset[4,1])), (0,255,0), 1)
        cv2.line(img, (int(img_pts_offset[2,0]), int(img_pts_offset[2,1])), (int(img_pts_offset[5,0]), int(img_pts_offset[5,1])), (0,255,0), 1)
        
        if show_img:
            cv2.imshow('Projected point model', img)
            cv2.waitKey(0)
        rst_pts = []
        #NOTE: !
        # IF OLD MODEL NEEDS TO BE USED FOR MODEL ESTIMATION (LINES AND PTS LOCATIONS)
        # CHANGE FROM img_pts_offset to img_pts_init
        # AND UNCOMMENT THIS P AGAIN
        P = K @ cam_pose # NOW USES OLD MODEL AND NOT ESTIMATED FOR LINES AND PTS
        for _pt in img_pts_init:
            rst_pts.append([int(_pt[0]), int(_pt[1])])
        lines_divided_3d = self.subdivide_lines()
        lines_divided_2d = []
        for line in lines_divided_3d:
            line_divided_2d = []
            for pt in line:
                point = np.copy(pt)
                point = np.append(point,1)
                # Perspective transform
                point = P @ point
                # Re-scale homogenous point
                if point[2] > 0:
                    point /= point[2]
                line_divided_2d.append(point[:2])
            lines_divided_2d.append(line_divided_2d)
        #for line in lines_divided_2d:
        #    for pt in line:
        #        cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (255,0,0), -1)
        return rst_pts[::-1], lines_divided_2d
    
    def update_point_model_from_optimizer(self, points):
        '''
        Takes list of points and updates point model based on this.
        '''
        for i, point in enumerate(points[::-1]):
            self.point_model[i] = point.point
        self.update_line_model()

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
    #stm = SkeletalTurbineModel()
    stm = SkeletalTurbineModel(c=(0,0), h=71.74-8, omega=5/4*np.pi, r=5.16, phi=np.deg2rad(60+30), b=35.1)
    #stm = SkeletalTurbineModel(c=(0,0), h=1, omega=0, r=1, phi=0, b=1)
    #stm.project_model_to_image(show_img=True)
    stm.plot_model()
    #stm.subdivide_lines()
    #utils.dist_between_points(stm.point_model[2],stm.point_model[3])

if __name__ == "__main__":
    main()