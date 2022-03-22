import g2o
import numpy as np
import utils

point_colors = {1: np.array([1.0, 0.0, 0.0]),
                2: np.array([1.0, 0.0, 0.0]),
                3: np.array([1.0, 0.0, 0.0]),
                4: np.array([0.0, 1.0, 0.0]),
                5: np.array([0.0, 0.0, 1.0]),
                6: np.array([1.0, 1.0, 0.0])}

line_colors = {1: np.array([0.0, 0.0, 1.0]),
               2: np.array([0.0, 1.0, 0.0]),
               3: np.array([1.0, 0.0, 0.0]),
               4: np.array([1.0, 0.0, 1.0]),
               5: np.array([1.0, 0.0, 1.0])}

class Point:
    def __init__(self, point, point_id = None, color = None):
        self.point = point
        #self.descriptor = descriptor
        if color is None:
            self.color = point_colors.get(point_id)
        else:
            self.color = color
        #self.feature_id = feature_id
        self.point_id = point_id

    def __repr__(self):
        return repr('Point %5d (%8.2f %8.2f %8.2f)' % (
            self.point_id,
            self.point[0],
            self.point[1],
            self.point[2]))

class Camera:
    def __init__(self, R, t, camera_id = None, fixed = False):
        self.R = R
        self.t = t
        #self.frame_id = frame_id
        self.camera_id = camera_id
        self.fixed = fixed

    def pose(self):
        if self.t.shape == (3, 1):
            # TODO catch this earlier
            self.t = self.t.T[0]
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret

    def __repr__(self):
        return repr("Camera %d [%s] (%f %f %f) %s" % (self.camera_id,
            self.fixed,
            self.t[0],
            self.t[1],
            self.t[2],
            self.R))
        
class Observation:
    def __init__(self, point_id, camera_id, image_coordinates):
        self.point_id = point_id
        self.camera_id = camera_id
        self.image_coordinates = image_coordinates

    def __repr__(self):
        return repr("Observation - point %d - camera %d (%f %f)" % (
            self.point_id,
            self.camera_id,
            self.image_coordinates[0],
            self.image_coordinates[1]))

class PoseGraphOptimization:
    def __init__(self, camera_matrix):
        # Init
        self.camera_matrix = camera_matrix
        self.points = []
        self.cameras = []
        self.observations = []
        self.next_id_to_use = 0
        
    def increment_id(self):
        t = self.next_id_to_use
        self.next_id_to_use += 1
        return t
    
    def add_camera(self, camera):
        camera.camera_id = self.increment_id()
        self.cameras.append(camera)
        return camera
    
    def add_point_model_to_points(self, point_model):
        '''
        Takes point_model as input and adds to list of points as Point class (3D position and ID)
        Point_model point order: turbine_base, turbine_top, wing_center, wing_tips(1,2,3)
        '''
        for i, pt in enumerate(point_model[::-1]):
            point = Point(pt, point_id=self.increment_id())
            self.points.append(point)
        
    def add_line_model_to_points(self, lines_divided_3d):
        '''
        Takes line model divided into 3D lines and adds each point in each line to
        list of points as Point class (3D position and ID)
        lines_diveded_3d line order: tower --> top --> wing_center --> wings
        '''
        for i, line in enumerate(lines_divided_3d):
            for pt in line:
                point = Point(pt, point_id=self.increment_id(), color=line_colors.get(i+1))
                self.points.append(point)
    
    def create_observations(self, points_2d, cam_id):
        '''
        Creates observations for optimizer. Takes as input a sorted list of 2D image points.
        Points in self.points is sorted the same way is these 2D image points.
        '''
        for i, pt in enumerate(points_2d):
            if pt[0] > 0 and pt[0] < 640 and pt[1] > 0 and pt[1] < 480:
                obs = Observation(self.points[i].point_id, cam_id, pt)
                self.observations.append(obs)
        print(f'Number of cams: {len(self.cameras)}')
        print(f'Number of points: {len(self.points)} == {len(points_2d)}')
        print(f'Number of obs: {len(self.observations)}')
        
    def freeze_nonlast_cameras(self):
        for idx, camera in enumerate(self.cameras):
            self.cameras[idx].fixed = True
        self.cameras[-1].fixed = False
        if len(self.cameras) > 2:
            self.cameras[-2].fixed = False
        
    def optimize(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)
        focal_length = self.camera_matrix[0, 0]
        #principal_point = (320, 240)
        principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        baseline = 0
        cam = g2o.CameraParameters(focal_length, principal_point, baseline)
        cam.set_id(0)
        optimizer.add_parameter(cam)
        self.freeze_nonlast_cameras()
        camera_vertices = {}
        for camera in self.cameras:
            # Use the estimated pose of the second camera based on the 
            # essential matrix.
            pose = g2o.SE3Quat(camera.R, camera.t)

            # Set the poses that should be optimized.
            # 3D pose graph vertex (x,y,z,q.x,q.y,q.z) as q.w is given from sqrt(1-||q.x,q.y,q.z||)
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(camera.camera_id)
            v_se3.set_estimate(pose)
            v_se3.set_fixed(camera.fixed)
            optimizer.add_vertex(v_se3)
            camera_vertices[camera.camera_id] = v_se3

        point_vertices = {}
        for point in self.points:
            # Add 3d location of point to the graph
            vp = g2o.VertexPointXYZ()
            vp.set_id(point.point_id)
            vp.set_marginalized(True)
            # Use positions of 3D points from the triangulation
            point_temp = np.array(point.point, dtype=np.float64)
            vp.set_estimate(point_temp)
            optimizer.add_vertex(vp)
            point_vertices[point.point_id]= vp

        for observation in self.observations:
            # Add edge from first camera to the point
            edge = g2o.EdgeProjectXYZ2UV()

            # 3D point
            edge.set_vertex(0, point_vertices[observation.point_id]) 
            # Pose of camera
            edge.set_vertex(1, camera_vertices[observation.camera_id]) 
            
            edge.set_measurement(observation.image_coordinates)
            edge.set_information(np.identity(2))
            #if observation.point_id <= 6:
            #    edge.set_information(np.identity(2))
            #else:
            #    edge.set_information(np.array([[0.01, 0.0],[0.0, 1.0]]))
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            #edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))

        print('Performing full BA:')
        optimizer.initialize_optimization()
        optimizer.set_verbose(True)
        optimizer.optimize(40)
        optimizer.save("test.g2o")

        for idx, camera in enumerate(self.cameras):
            t = camera_vertices[camera.camera_id].estimate().translation()
            self.cameras[idx].t = t
            q = camera_vertices[camera.camera_id].estimate().rotation()
            self.cameras[idx].R = utils.quarternion_to_rotation_matrix_g2o(q, inverse=False)

        for idx, point in enumerate(self.points):
            p = point_vertices[point.point_id].estimate()
            # It is important to copy the point estimates.
            # Otherwise I end up with some memory issues.
            # self.points[idx].point = p
            self.points[idx].point = np.copy(p)

def main():
    # Main
    print("Hello world!")

if __name__ == "__main__":
    main()