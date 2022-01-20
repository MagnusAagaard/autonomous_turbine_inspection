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
    # Assuming a rectangular image
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