import numpy as np
import cv2
import math

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

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

def get_camera_pose_from_pose_msg(pose_msg):
    '''
    Returns R,t in camera frame, given a PoseStamped.pose_msg in world frame
    '''
    x = pose_msg.pose.position.x
    y = pose_msg.pose.position.y
    z = pose_msg.pose.position.z
    t_cam = np.array([x, y, z])
    R_cam_inv = quarternion_to_rotation_matrix(pose_msg.pose.orientation, inverse=True)
    Rex = get_rotation_matrix_from_world_to_camera_frame()
    # Transform from world to camera frame (axis rotation)
    # Take rotation matrix given in world coordinates and transform to camera axis
    R = Rex @ R_cam_inv
    # Translate camera -tx
    t = -Rex @ R_cam_inv @ t_cam
    return R,t

def get_world_pose_from_pose_msg(pose_msg):
    '''
    Returns R, t in world frame, given a PoseStamped.pose msg in world frame.
    '''
    x = pose_msg.pose.position.x
    y = pose_msg.pose.position.y
    z = pose_msg.pose.position.z
    t_w = np.array([x, y, z])
    R_w = quarternion_to_rotation_matrix(pose_msg.pose.orientation, inverse=False)

    return R_w,t_w

def convert_pose_to_camera_frame(cam_pose_w):
    '''
    Returns cam_pose in camera frame given cam_pose in world frame
    '''
    cam_pose_c = np.ones((3,4))
    R = cam_pose_w[:3,:3]
    t = cam_pose_w[:3,3]
    Rex = get_rotation_matrix_from_world_to_camera_frame()
    cam_pose_c[:3,:3] = Rex @ np.linalg.inv(R)
    cam_pose_c[:3,3] = -Rex @ np.linalg.inv(R) @ t
    return cam_pose_c
    
    
def quarternion_to_rotation_matrix(q, inverse=False):
    """
    Returns rotation matrix given a quarternion as a PoseStamped.pose.orientation msg.
    The formula for converting from a quarternion to a rotation 
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q.w
    qx = q.x if not inverse else -q.x
    qy = q.y if not inverse else -q.y
    qz = q.z if not inverse else -q.z
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

def quarternion_to_rotation_matrix_g2o(q, inverse=False):
    """
    Converts quaternions q from g2o library into rotation matrix.
    The formula for converting from a quarternion to a rotation 
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q.w()
    qx = q.x() if not inverse else -q.x()
    qy = q.y() if not inverse else -q.y()
    qz = q.z() if not inverse else -q.z()
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

def quarternion_to_rotation_matrix_g2o_cam_frame(q, inverse=False):
    qw = q.w()
    qx = -q.y() if not inverse else q.y()
    qy = -q.z() if not inverse else q.z()
    qz = q.x() if not inverse else -q.x()
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

def quarternion_to_rotation_matrix_least_squares(q, inverse=False):
    """
    Converts quaternions q from g2o library into rotation matrix.
    The formula for converting from a quarternion to a rotation 
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q[0]
    qx = q[1] if not inverse else -q[1]
    qy = q[2] if not inverse else -q[2]
    qz = q[3] if not inverse else -q[3]
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

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def dist_between_points(self, p1, p2):
    '''
    Returns the distance between two points in 3D (x,y,z)
    '''
    return np.sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]))

def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    
    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size (let's say 3) to the current i
        added = i + kernel_size
        # It must be lower than the image size
        if added <= img_size:
            # Increment if so
            num_pixels += 1
            
    return num_pixels

def convolve(img: np.array, kernel: np.array) -> np.array:
    tgt_size = (img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1)
    # To simplify things
    r, c = kernel.shape
    
    # 2D array of zeros
    convolved_img = np.zeros(shape=tgt_size)
    
    # Iterate over the rows
    for i in range(tgt_size[0]):
        # Iterate over the columns
        for j in range(tgt_size[1]):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+r, j:j+c]
            
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
            
    return convolved_img

def convolve_mask(img: np.array, kernel: np.array) -> np.array:
    # Crop image
    tgt_size = (img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1)
    # To simplify things
    r, c = kernel.shape
    
    # 2D array of zeros
    convolved_img = np.zeros(shape=tgt_size)
    
    # Iterate over the rows
    for i in range(tgt_size[0]):
        # Iterate over the columns
        for j in range(tgt_size[1]):
            # img[i, j] = individual pixel value
            # Get the current matrix
            mat = img[i:i+r, j:j+c]
            
            # Apply the convolution - element-wise multiplication and summation of the result
            # Store the result to i-th row and j-th column of our convolved_img array
            convolved_img[i, j] = np.sum(cv2.bitwise_and(mat, mat, mask=kernel))
            
    return convolved_img

def main():
    q = [0.707, 0.707, 0, 0]
    print(quarternion_to_rotation_matrix_least_squares(q))

if __name__ == "__main__":
    main()