
import numpy as np
import cv2
import yaml
import trimesh
import joblib
import pyrender
import copy
# from sklearn.preprocessing import PolynomialFeatures

class BaseSensorSim:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.cfg = yaml.safe_load(file)
        # self.params = self.config.get('cameras', {})

        ## surface mesh config
        _surface_mesh_filename = self.cfg['surface'].get('mesh', None)
        if _surface_mesh_filename is not None:
            self.surface_mesh = trimesh.load(_surface_mesh_filename)

        _surface_idx_filename = self.cfg['surface'].get('surfaceIdx', None)
        if _surface_idx_filename is not None:
            self.surface_idx = np.load(_surface_idx_filename)  # the surface vertices idx in volumetric mesh

        ## refraction config
        _refraction_rpp_filename = self.cfg['refraction'].get('rpp', None)
        if _refraction_rpp_filename is not None:
            self.refraction_rpp = np.load(_refraction_rpp_filename)

        _refraction_model_filename = self.cfg['refraction'].get('model', None)
        if _refraction_model_filename is not None:
            self.refraction_model = joblib.load(_refraction_model_filename)

        ## point cloud config
        _bary_coords_filename = self.cfg['pcd'].get('barycentricCoords', None)
        if _bary_coords_filename is not None:
            self.barycentric_coords = np.loadtxt(_bary_coords_filename)
            if self.barycentric_coords.ndim == 2 and self.barycentric_coords.shape[1] == 4:
                self.barycentric_coords = np.expand_dims(self.barycentric_coords, axis=2)  # shape (num_points,4,1)


        _bary_idx_filename = self.cfg['pcd'].get('verticesIdx', None)
        if _bary_idx_filename is not None:
            self.barycentric_vertices_idx = np.loadtxt(_bary_idx_filename).astype(np.int32)


        ## camera config
        self.cam_resolution = self.cfg['camera']['resolution']



    def create_renderer(self):
        self.renderer = pyrender.OffscreenRenderer(*self.cam_resolution)
        self.scene = pyrender.Scene()


    def delete_render(self):
        self.renderer.delete()


    def create_render_scene(self,):
        """  
        Should be implemented in an sensor class inherited from BaseSensorSim.
        """  
        pass


    def render_img(self, nodes):
        _surface_mesh = copy.deepcopy(self.surface_mesh)

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
    

    def rectify_img(self, img_wo_refract, depth):
        pass
    

    def render_rectified_img(self, ):
        pass


    def extract_3d_markers(self, nodes):
        '''
        extract markers 3D position based on barycentric coordinate
        nodes: all vertices coordinate of the elastomer; np.array shape (num_vertices, 3)
        return: marker position; np.array()  shape (num_marker, 3)
        '''

        # the vertices of cell corresponding to each marker
        bary_nodes = [nodes[self.barycentric_vertices_idx[k]].T.tolist() \
                                                    for k in range(self.barycentric_vertices_idx.shape[0])]
        bary_nodes = np.array(bary_nodes)  

        # compute the marker position on deformed gel in env coordinate system
        markers = np.matmul(bary_nodes, self.barycentric_coords)  

        return np.squeeze(markers)



