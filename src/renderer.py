import trimesh
import pyrender
import cv2
import numpy as np
import utils

class Renderer:
    def __init__(self, file_path):
        self.file_path = file_path

    def render_and_show(self):
        _trimesh = trimesh.load(self.file_path, file_type='stl')
        mesh = pyrender.Mesh.from_trimesh(_trimesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        #camera = pyrender.PerspectiveCamera(yfov=1.047, aspectRatio=1.333)
        camera = pyrender.IntrinsicsCamera(fx=554.920125, fy=554.921917, cx=320.077433, cy=239.661438, znear=0.1, zfar=500)
        Rx = utils.get_rotation_matrix('x', np.pi/2)
        Rz = utils.get_rotation_matrix('z', -np.pi/2)
        camera_pose = np.array([[1.0, 0, 0, -30.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 65.0],
                                [0.0, 0.0, 0.0, 1.0]])
        camera_pose[:3,:3] = Rz @ Rx @ camera_pose[:3,:3]
        scene.add(camera, pose=camera_pose)
        pyrender.Viewer(scene, use_raymond_lighting=True)

    def offscreen_render(self):
        _trimesh = trimesh.load(self.file_path, file_type='stl')
        mesh = pyrender.Mesh.from_trimesh(_trimesh)
        scene = pyrender.Scene()
        scene.add(mesh)
        #camera = pyrender.PerspectiveCamera(yfov=1.047, aspectRatio=1.333)
        camera = pyrender.IntrinsicsCamera(fx=554.920125, fy=554.921917, cx=320.077433, cy=239.661438, znear=0.1, zfar=500)
        Rx = utils.get_rotation_matrix('x', np.pi/2)
        Rz = utils.get_rotation_matrix('z', -np.pi/2)
        camera_pose = np.array([[1.0, 0, 0, -300.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 65.0],
                                [0.0, 0.0, 0.0, 1.0]])
        camera_pose[:3,:3] = Rz @ Rx @ camera_pose[:3,:3]
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(intensity=3.0)
        scene.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(640, 480)
        color, depth = r.render(scene)
        
        cv2.imshow('Image', color)
        cv2.waitKey(0)

def main():
    renderer = Renderer('/home/magnus/master_thesis/catkin_ws/src/autonomous_turbine_inspection/models/Vestas_V52/meshes/vestas_v52.stl')
    #renderer.render_and_show()
    renderer.offscreen_render()

if __name__ == "__main__":
    main()