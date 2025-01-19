import numpy as np
import cv2
import yaml
import trimesh
import joblib
import pyrender
import copy
from scipy.spatial.transform import Rotation as scipyR

from tacflex.sensor_base import BaseSensorSim


class GelSightMiniSim(BaseSensorSim):
    def __init__(self, config_path):
        super().__init__(config_path)

        ## surface mesh config
        _surface_mesh_filename = self.cfg['surface'].get('meshWoMarker', None)
        if _surface_mesh_filename is not None:
            self.surface_mesh_wo_marker = trimesh.load(_surface_mesh_filename)

        ## real background
        self.background_real = cv2.imread(self.cfg['backgroundReal'])

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
        assert self.cfg['lights']['type'] == 'Spot', "light type error!"

        light_positions = self.cfg['lights']['positions']  # list
        light_orientations = self.cfg['lights']['orientations']
        light_colors = self.cfg['lights']['colors']
        light_intensities = self.cfg['lights']['intensities']
        light_angles = self.cfg['lights']['angles']


        for i in range(len(light_positions)):
            light_pose = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0.0],
                                [0, 0, 0, 1]])  # array
            light_pose[:3, 3] = np.array(light_positions[i])
            light_pose[:3,:3] = scipyR.as_matrix(scipyR.from_euler('xyz', light_orientations[i], degrees=True))

            light = pyrender.light.SpotLight(color=light_colors[i], intensity=light_intensities[i], 
                                         innerConeAngle=0, outerConeAngle=light_angles[i]/180*np.pi)

            light_node = pyrender.Node(light=light, matrix=light_pose)
            self.scene.add_node(light_node)


    def render_img_wo_marker(self, nodes):
        _surface_mesh = copy.deepcopy(self.surface_mesh_wo_marker)

        nodes = np.array(nodes)

        # generate deform mesh for rendering
        for i in range(len(_surface_mesh.vertices)):
            _surface_mesh.vertices[i] = nodes[self.surface_idx[i]]  ## NOTE unit
        
        mesh = pyrender.Mesh.from_trimesh(_surface_mesh)
        nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))

        self.scene.add_node(nm)
        color1, depth1 = self.renderer.render(self.scene)
        self.scene.remove_node(nm)


        return cv2.cvtColor(color1, cv2.COLOR_RGB2BGR), depth1


    def post_process(self, color, _background_sim=None, scaling=1.0):

        # Simulated difference image, with scaling factor 1.0
        diff = (color.astype(np.float32) - _background_sim) * scaling

        # Add low-pass filter to match real readings
        # diff = cv2.GaussianBlur(diff, (7, 7), 0)

        # Combine the simulated difference image with real background image
        color = np.clip((diff[:, :, :3] + self.background_real), 0, 255).astype(np.uint8)

        return color
    

    def render_gelsight_img(self, nodes, _background_sim):
        rgb, depth = self.render_img(nodes)
        image = self.rectify_img(rgb, depth)
        image = self.post_process(image, _background_sim)
        return image
