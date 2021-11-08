import numpy as np

def get_rotation_matrix(axis, angle):
    '''
    Returns rotation matrix around specific axis for specified angle
    '''
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

def get_rotation_matrix_from_world_to_camera_frame():
    '''
    Returns rotation matrix from world frame to camera frame
    '''
    # Rotation from world frame to camera frame
    #Rz = get_rotation_matrix(axis='z', angle=np.pi/2)
    #Rx = get_rotation_matrix(axis='x', angle=np.pi/2)
    #print(Rx @ Rz)
    #return Rx @ Ry
    # Equivalent: cam x is in world -y, cam y is in world -z, cam z is in world x
    R = np.array([[0, -1, 0], [0,0,-1], [1,0,0]])
    return R

def quarternion_to_rotation_matrix(q):
    """
    Returns rotation matrix given a quarternion as a PoseStamped.pose.orientation msg.
    The formula for converting from a quarternion to a rotation 
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q.w
    qx = q.x
    qy = q.y
    qz = q.z
    R11 = 1 - 2*qy**2 - 2*qz**2	
    R12 = 2*qx*qy - 2*qz*qw
    R13 = 2*qx*qz + 2*qy*qw
    R21 = 2*qx*qy + 2*qz*qw
    R22 = 1 - 2*qx**2 - 2*qz**2
    R23 = 2*qy*qz - 2*qx*qw
    R31 = 2*qx*qz - 2*qy*qw
    R32 = 2*qy*qz + 2*qx*qw 
    R33 = 1 - 2*qx**2 - 2*qy**2
    R = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    return R

def dist_between_points(self, p1, p2):
    '''
    Returns the distance between two points in 3D (x,y,z)
    '''
    return np.sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]))