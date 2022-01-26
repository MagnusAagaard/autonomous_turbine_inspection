import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import cv2
import numpy as np
import utils
import timeit

class Renderer:
    def __init__(self, tower, wings):
        self.__setup_renderer(tower, wings)
        
    def __setup_renderer(self, tower, wings):
        _trimesh_tower = trimesh.load(tower, file_type='stl')
        _trimesh_wings = trimesh.load(wings, file_type='stl')
        self.mesh_tower = pyrender.Mesh.from_trimesh(_trimesh_tower)
        self.mesh_wings = pyrender.Mesh.from_trimesh(_trimesh_wings)
        self.camera = pyrender.IntrinsicsCamera(fx=554.920125, fy=554.921917, cx=320.077433, cy=239.661438, znear=0.1, zfar=500)
        self.Rx = utils.get_rotation_matrix('x', np.pi/2)
        self.Rz = utils.get_rotation_matrix('z', -np.pi/2)
        self.camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]])
        self.r = pyrender.OffscreenRenderer(640, 480)

    def render_and_show(self):
        scene = pyrender.Scene()
        scene.add(self.mesh_tower)
        scene.add(self.mesh_wings)
        self.camera_pose[:3, 3] = [-30.0, 0.0, 65.0 + 8.043]
        self.camera_pose[:3,:3] = self.Rz @ self.Rx @ self.camera_pose[:3,:3]
        scene.add(self.camera, pose=self.camera_pose)
        pyrender.Viewer(scene, use_raymond_lighting=True)

    def offscreen_render(self, estimate):
        '''
        Renders CAD model to 2D image with the camera looking straight at the model
        with a position of x,y,z given from parameters 'estimate'.
        Returns the rendered 2D image.
        '''
        scene = pyrender.Scene()
        tower_pose = np.copy(self.camera_pose)
        wings_pose = np.copy(self.camera_pose)
        cam_pose = np.copy(self.camera_pose)
        # If tower heading is rotated (rotation around z) both wings and tower should rotate
        Rz_wings = utils.get_rotation_matrix('z', 0.785)
        Rz_tower = Rz_wings
        Rx_wings = utils.get_rotation_matrix('x', (30/180*np.pi))
        wings_pose[:3, 3] = Rz_wings @ [-6.1541, -0.170453, 71.7424]
        wings_pose[:3,:3] = Rz_wings @ Rx_wings @ wings_pose[:3,:3]
        tower_pose[:3,:3] = Rz_wings @ tower_pose[:3,:3]
        cam_pose[:3, 3] = estimate
        cam_pose[:3,:3] = self.Rz @ self.Rx @ self.camera_pose[:3,:3]
        scene.add(self.mesh_tower, name='tower', pose=tower_pose)
        scene.add(self.mesh_wings, name='wings', pose=wings_pose)
        scene.add(self.camera, name='camera', pose=cam_pose)
        #light = pyrender.DirectionalLight(intensity=3.0)
        #scene.add(light, pose=camera_pose)
        #r = pyrender.OffscreenRenderer(640, 480)
        #print('render\t\t', timeit.timeit(lambda: r.render(scene), number=10) / 10)
        
        color, depth = self.r.render(scene)
        return color

def main():
    renderer = Renderer(tower='./models/Vestas_V52/meshes/vestas_v52_tower.stl', wings='./models/Vestas_V52/meshes/vestas_v52_wings.stl')
    #renderer.render_and_show()
    template = renderer.offscreen_render([-200, 0, 65 + 8])
    cv2.imshow('Template', template)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()