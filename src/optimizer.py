import g2o
import numpy as np
import utils
from scipy.optimize import least_squares

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
        self.original_R = R.copy()
        self.original_t = t.copy()
        self.relative_pose = None

    def pose(self):
        if self.t.shape == (3, 1):
            # TODO catch this earlier
            self.t = self.t.T[0]
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret
    
    def original_pose(self):
        if self.original_t.shape == (3,1):
            self.original_t = self.t.T[0]
        #ret = np.eye(4)
        #ret[:3, :3] = self.original_R
        #ret[:3, 3] = self.original_t
        return self.original_R, self.original_t 

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
        if len(self.cameras) > 1:
            self.calculate_relative_pose(self.cameras[-2], self.cameras[-1])
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
        print(f'Number of cams: {len(self.cameras)}')
        print(f'Number of points: {len(self.points)} == {len(points_2d)}')
        for i, pt in enumerate(points_2d):
            if pt[0] > 0 and pt[0] < 640 and pt[1] > 0 and pt[1] < 480:
                obs = Observation(self.points[i].point_id, cam_id, pt)
                self.observations.append(obs)
        print(f'Number of obs: {len(self.observations)}')
        
    def freeze_nonlast_cameras(self, number_of_non_fixed_cameras = 5):
        for idx, camera in enumerate(self.cameras[::-1]):
            if idx >= number_of_non_fixed_cameras:
                self.cameras[idx].fixed = True
            else:
                self.cameras[idx].fixed = False
        #self.cameras[0].fixed = True
        
    def unfreeze_cameras(self, number_of_fixed_cameras = 5):
        for idx, camera in enumerate(self.cameras):
            if idx >= number_of_fixed_cameras:
                self.cameras[idx].fixed = False
            else:
                self.cameras[idx].fixed = True
                
    def limit_number_of_cameras(self, limit=20):
        if len(self.cameras) > limit:
            print('removing camera')
            cam_to_remove = self.cameras[0]
            # Remove observation associated with camera
            filtered_obs = []
            for obs in self.observations:
                if obs.camera_id != cam_to_remove.camera_id:
                    filtered_obs.append(obs)
            self.observations = filtered_obs
            self.cameras.remove(self.cameras[0])
                
    def remove_observations_with_reprojection_errors_above_threshold(self, threshold = 100):
        sqerror_array = []
        camera_dict = {}
        for camera in self.cameras:
            camera_dict[camera.camera_id] = camera
        point_dict = {}
        for point in self.points:
            point_dict[point.point_id] = point
        total_error = 0
        temp_observations = []
        for observation in self.observations:
            camera = camera_dict[observation.camera_id]
            point = point_dict[observation.point_id]
            point = np.array(point.point)
            point = np.hstack((point, 1))
            point = np.array([point]).T
            point_in_cam_coords = camera.pose() @ point
            t = self.camera_matrix @ point_in_cam_coords[0:3, :]
            t = t / t[2, 0]
            dx = t[0] - observation.image_coordinates[0]
            dy = t[1] - observation.image_coordinates[1]
            sqerror = np.abs(dx*dx) + np.abs(dy*dy)
            if sqerror < threshold or camera.fixed:
                temp_observations.append(observation)
                sqerror_array.append(sqerror)
                
        mean = np.mean(np.array(sqerror_array))
        std = np.std(np.array(sqerror_array))
        total = np.sum(np.array(sqerror_array))
        print("Total reprojection error: {}".format(total))
        print("Mean reprojection error: {}".format(mean))
        print("Std. dev reprojection error: {}".format(std))

        self.observations = temp_observations
        
    def get_relative_pose(self, R1, t1, R2, t2):
        '''
        Calculates relative pose between two camera poses
        Pose 1 in world frame
        Pose 2 in world frame
        Pose 2 wrt. pose 1 = T_ij =  T_1^-1*T2
        '''
        # Rex needs not to be removed prior to calculating relative pose, since they are in the same frame and it will cancel out
        # Rex @ t needs to be computed to get translation in camera frame
        # The R matrix is calculated by swapping quaternion values corresponding to transform from world to camera frame
        #Rex_inv = np.linalg.inv(utils.get_rotation_matrix_from_world_to_camera_frame())
        #pose1 = g2o.SE3Quat(Rex_inv @ R1, Rex_inv @ t1)
        #pose2 = g2o.SE3Quat(Rex_inv @ R2, Rex_inv @ t2)
        #pose1 = g2o.SE3Quat(R1, Rex_inv @ t1)
        #pose2 = g2o.SE3Quat(R2, Rex_inv @ t2)
        pose1 = g2o.SE3Quat(R1, t1)
        pose2 = g2o.SE3Quat(R2, t2)
        # Relative transformation between pose 1 and 2 in world frame
        Tij = pose1.inverse()*pose2
        Rex = utils.get_rotation_matrix_from_world_to_camera_frame()
        # Translation in camera frame
        t = Rex @ Tij.translation()
        # Rotation in camera frame
        R = utils.quarternion_to_rotation_matrix_g2o_cam_frame(Tij.rotation())
        pose = g2o.SE3Quat(R, t)
        return pose
    
    def calculate_relative_pose(self, cam1, cam2):
        pose1_R, pose1_t = cam1.original_pose()
        pose2_R, pose2_t = cam2.original_pose()
        cam1.relative_pose = self.get_relative_pose(pose1_R, pose1_t, pose2_R, pose2_t)
        #print('Relative pose SE3Quat: {}'.format(cam1.relative_pose.to_vector()))
        #print('Relative orientation euler: {}'.format(utils.euler_from_matrix(utils.quarternion_to_rotation_matrix_g2o(cam1.relative_pose.rotation()))))
        #print('Relative orientation R: {}'.format(utils.quarternion_to_rotation_matrix_g2o(cam1.relative_pose.rotation())))
        #Rex = utils.get_rotation_matrix_from_world_to_camera_frame()
        #print('Relative orientation test 1: {}'.format(utils.euler_from_matrix(Rex @ np.identity(3))))
        #print('Relative orientation 2: {}'.format(utils.euler_from_matrix(Rex @ utils.quarternion_to_rotation_matrix_g2o(cam1.relative_pose.rotation()))))
        #print('Cam 1 pose : {}'.format(cam1.original_pose()))
        #print('Cam 2 pose : {}'.format(cam2.original_pose()))
        
    def check_relative_pose(self, cam1, cam2):
        pose1_R, pose1_t = cam1.original_pose()
        pose2_R, pose2_t = cam2.original_pose()
        p1 = np.identity(4)
        p2 = np.identity(4)
        p1[:3,:3] = pose1_R
        p1[:3,3] = pose1_t
        p2[:3,:3] = pose2_R
        p2[:3,3] = pose2_t
        Tij = np.linalg.inv(p1) @ p2
        print(Tij)
        
    def get_relative_pose_offset(self):
        '''
        Calculates relative pose offset between optimized pose and original pose.
        Uses last camera as that is the current best estimate.
        '''
        opti_R = self.cameras[-1].R
        opti_t = self.cameras[-1].t
        ori_R, ori_t = self.cameras[-1].original_pose()
        # POSES OF CAMS: right is x, down is y, front is z - negative values corresponds to positive axis movement..
        #opti_t = np.copy(ori_t) + np.array([1,0,0])
        #opti_R = np.copy(ori_R)
        # Get transform from pose 1 --> pose 2 (ie. offset)
        pose1 = g2o.SE3Quat(ori_R, ori_t)
        pose2 = g2o.SE3Quat(opti_R, opti_t)
        # Relative transformation between pose 1 and 2 in world frame
        Tij = pose1.inverse()*pose2
        Rex = utils.get_rotation_matrix_from_world_to_camera_frame()
        # Translation in camera frame
        t = Rex @ Tij.translation()
        # Rotation in camera frame
        R = utils.quarternion_to_rotation_matrix_g2o_cam_frame(Tij.rotation())
        # Create pose to make sure that rotation is normalized
        pose = g2o.SE3Quat(R, t)
        ret = np.eye(4)
        ret[:3, :3] = utils.quarternion_to_rotation_matrix_g2o(pose.rotation())
        ret[:3, 3] = pose.translation()
        print('Estimated pose offset:')
        print(f'Translation: {ret[:3,3]}')
        print(f'Rotation: {utils.euler_from_matrix(ret[:3,:3])}')
        print(f'ret {ret}')
        return ret
        
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
        #self.freeze_nonlast_cameras(number_of_non_fixed_cameras=20)
        self.limit_number_of_cameras(limit=50)
        #self.unfreeze_cameras(number_of_fixed_cameras=0)
        
        #print(f'Obs before reprojection error adjustment: {len(self.observations)}')
        #self.remove_observations_with_reprojection_errors_above_threshold(10)
        #print(f'Obs after reprojection error adjustment: {len(self.observations)}')
        
        camera_vertices = {}
        for camera in self.cameras:
            # Use the estimated pose of the camera
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
            # Use positions of 3D points
            point_temp = np.array(point.point, dtype=np.float64)
            vp.set_estimate(point_temp)
            vp.set_fixed(True)
            optimizer.add_vertex(vp)
            point_vertices[point.point_id]= vp

        for observation in self.observations:
            # Add edge from first camera to the point
            edge = g2o.EdgeProjectXYZ2UV()

            # 3D point
            edge.set_vertex(0, point_vertices[observation.point_id]) 
            # Pose of camera
            edge.set_vertex(1, camera_vertices[observation.camera_id]) 
            # Image coordinate
            edge.set_measurement(observation.image_coordinates)
            edge.set_information(np.identity(2))
            # 0.01 and 0.01 to weight line correspondences lower than points
            #if observation.point_id >= 6:
            #    edge.set_information(np.array([[0.01, 0.0],[0.0, 0.01]]))
            #else:
            #    edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            #edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)
        
        for i, camera in enumerate(self.cameras[:-1]):
            # Add edge from camera to camera using internal exponential map
            edge = g2o.EdgeSE3Expmap()
            edge.set_vertex(0, camera_vertices[camera.camera_id])
            edge.set_vertex(1, camera_vertices[self.cameras[i+1].camera_id])
            # Measurement should be relative camera movement ie. pose 2 wrt. pose 1 in camera frame
            measurement = camera.relative_pose
            #print('Relative pose SE3Quat: {}'.format(measurement.to_vector()))
            #print('Relative orientation: {}'.format(utils.quarternion_to_rotation_matrix_g2o(measurement.rotation())))
            #print(f'P1: {camera_vertices[camera.camera_id].estimate().to_vector()}')
            #print(f'P2: {camera_vertices[self.cameras[i+1].camera_id].estimate().to_vector()}')
            edge.set_measurement(measurement)
            # Error in orientation weights high (meaning we are quite sure about our orientation from PX4)
            # Error in translation weights low (more room for translating the pose)
            #TODO: Try to weight position higher than points?
            information = np.identity(6)
            #information[0,0] = 0.5
            #information[1,1] = 0.5
            #information[2,2] = 0.5
            edge.set_information(information)
            edge.set_robust_kernel(g2o.RobustKernelHuber())
            edge.set_parameter_id(0,0)
            optimizer.add_edge(edge)
            

        #print('num vertices:', len(optimizer.vertices()))
        #print('num edges:', len(optimizer.edges()))

        print('Performing full BA:')
        optimizer.initialize_optimization()
        optimizer.set_verbose(False)
        optimizer.optimize(20)
        #optimizer.save("test.g2o")

        for idx, camera in enumerate(self.cameras):
            t = camera_vertices[camera.camera_id].estimate().translation()
            self.cameras[idx].t = t
            q = camera_vertices[camera.camera_id].estimate().rotation()
            self.cameras[idx].R = utils.quarternion_to_rotation_matrix_g2o(q, inverse=False)

        #for idx, point in enumerate(self.points):
        #    p = point_vertices[point.point_id].estimate()
            # It is important to copy the point estimates.
            # Otherwise I end up with some memory issues.
            # self.points[idx].point = p
        #    self.points[idx].point = np.copy(p)
            
class Vertex:
    def __init__(self, R, t, turbine_params):
        self.R = R
        self.t = t
        self.turbine_params = turbine_params
        
    def get_pose(self):
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret
            
class LeastSquareOptimizer:
    def __init__(self, init_params, K, stm):
        self.K = K
        self.stm = stm
        # Vertices contains estimated params
        self.vertices = []
        # Init_params contains init pose and turbine params [x,y,z,q1,q2,q3,q4,c,h,omega,r,phi,b]
        self.add_vertex(init_params)
        
    def add_vertex(self, params):
        self.vertices.append(params)
    
    def get_pose(self, params):
        ret = np.eye(4)
        ret[:3, :3] = utils.quarternion_to_rotation_matrix(params[3:7])
        ret[:3, 3] = params[:3]
        
    def cost_func(self, beta):
        '''
        Cost function to minimize.
        Beta are params to optimize
        '''
        # Use beta params to get
        pass
        
    def e1_pt(self, params):
        '''
        cost function E_unary(1) that calculates error between projected model points using
        current estimated pose and theta parameters and corresponding 2D point from network
        This is for pts only.
        '''
        for pt in params:
            pass
        
        
    def e1_line(self, params):
        '''
        Same as e1_pt but for lines. See comment there
        '''
        pass

def main():
    # Main
    print("Hello world!")

if __name__ == "__main__":
    main()