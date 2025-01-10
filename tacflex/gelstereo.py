import numpy as np
import cv2
import yaml
import trimesh
import joblib
import pyrender
import copy
from scipy.spatial.transform import Rotation as scipyR

from tacflex.sensor_base import BaseSensorSim


class GelStereoSim(BaseSensorSim):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.create_render_scene()


    def create_render_scene(self,):

        self.create_renderer()

        ## camera
        cam_position = self.cfg['camera']['position']
        cam_orientation = self.cfg['camera']['orientation']
        fx = self.cfg['camera']['fx']
        fy = self.cfg['camera']['fy']
        cx = self.cfg['camera']['cx']
        cy = self.cfg['camera']['cy']

        cam = pyrender.camera.IntrinsicsCamera(fx=fx, 
                                               fy=fy,
                                               cx=cx,
                                               cy=cy)  
        
        cam_pose = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0], 
                            [0, 0, 0, 1]])
        
        cam_pose[:3, 3] = cam_position
        _rot = scipyR.as_matrix(scipyR.from_euler('xyz', cam_orientation, degrees=True))
        cam_pose[:3,:3] = np.matmul(_rot, cam_pose[:3,:3])

        cam_node = pyrender.Node(camera=cam, matrix=cam_pose)
        self.scene.add_node(cam_node)

        ## lights
        assert self.cfg['lights']['type'] == 'Directional', "light type error!"

        light_positions = self.cfg['lights']['positions']  # list
        light_orientations = self.cfg['lights']['orientations']
        light_colors = self.cfg['lights']['colors']
        light_intensities = self.cfg['lights']['intensities']

        print(type(light_positions))

        for i in range(len(light_positions)):
            light_pose = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0.0],
                                [0, 0, 0, 1]])  # array
            light_pose[:3, 3] = np.array(light_positions[i])
            light_pose[:3,:3] = scipyR.as_matrix(scipyR.from_euler('xyz', light_orientations[i], degrees=True))

            light = pyrender.light.DirectionalLight(color=light_colors[i], intensity=light_intensities[i])

            light_node = pyrender.Node(light=light, matrix=light_pose)
            self.scene.add_node(light_node)


        



