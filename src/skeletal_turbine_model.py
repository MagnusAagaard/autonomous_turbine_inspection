import numpy as np

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SkeletalTurbineModel:
    def __init__(self, c=(0,0), h=1, omega=0, r=1, phi=0, b=1):
        # Init
        self.c = c          # (x,y) location of turbine tower base
        self.h = h          # Height of turbine tower
        self.omega = omega  # Heading of turbine relative to drone coordinate system
        self.r = r          # Length of turbine nacelle
        self.phi = phi      # Rotation angle of turbine blades
        self.b = b          # Length of turbine blades
        self.initiate_point_model()
        self.update_point_model()
        self.update_line_model()

    def initiate_point_model(self):
        # Initiate the default point model
        # Default model: x is straight ahead, y is left/right and z is up and down
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
        R_omega = self.get_rotation_matrix(axis='z', angle=self.omega)
        self.point_model[2] = pc + R_omega @ Mr @ Mh @ self.default_point_model[2]    # Turbine hub (centre of blades)
        # Turbine blade tips
        Mb = np.diag((1,1,self.b))
        for i in range(3):
            phi = self.phi + i*2/3*np.pi
            R_phi = self.get_rotation_matrix(axis='x', angle=phi)
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

    def get_rotation_matrix(self, axis, angle):
        # Directory containing indicies for cos, sin, -sin, cos
        indicies = {'x' : [[1,1], [2,1], [1,2], [2,2]],
                    'y' : [[0,0], [0,2], [2,0], [2,2]],
                    'z' : [[0,0], [1,0], [0,1], [1,1]]}
        if axis not in indicies.keys():
            print('Tried to get rotation matrix for axis: {}'.format(axis))
        R = np.diag((1,1,1)).astype('float32')
        inds = indicies.get(axis)
        R[inds[0][0], inds[0][1]] = np.cos(angle)
        R[inds[1][0], inds[1][1]] = np.sin(angle)
        R[inds[2][0], inds[2][1]] = -np.sin(angle)
        R[inds[3][0], inds[3][1]] = np.cos(angle)
        return R

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

    def dist_between_points(self, p1, p2):
        print(np.sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2])))
        

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
    stm.plot_model()
    #stm.dist_between_points(stm.point_model[2],stm.point_model[3])

if __name__ == "__main__":
    main()