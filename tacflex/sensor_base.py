
import numpy as np
import cv2
import yaml
import trimesh
import joblib
import pyrender
import copy



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
    

    def init_rectify(self, u_min=0, u_max=640, v_min=0, v_max=480):

        u, v = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max))
        self.source_coords = np.vstack((u.flatten(), v.flatten())).T
        self.dist_center_wo_refract = np.linalg.norm(self.source_coords - self.refraction_rpp, axis=-1).astype(np.float32)
        self.center_source_uvec = (self.source_coords - self.refraction_rpp) / (np.linalg.norm(self.source_coords - self.refraction_rpp, axis=-1, keepdims=True) + 1e-20)
        self.rectify_poly = np.ones((self.dist_center_wo_refract.shape[0], 10), dtype=np.float32)
        self.rectify_poly[:, 1] = self.dist_center_wo_refract
        self.rectify_poly[:, 3] = self.dist_center_wo_refract ** 2
        self.rectify_poly[:, 6] = self.dist_center_wo_refract ** 3
        self.image_roi_range = [u_min, u_max, v_min, v_max]


    def rectify_img(self, img_wo_refract, depth):
        '''
        image transformation from image without refraction to image under multi-medium refraction 
        :param img_wo_refract: the tactile image without refrection effects, e.g., pyrender output
        :param depth: the depth image 
        :return: the tactile image under multi-medium refraction 
        '''
        u_min, u_max, v_min, v_max = self.image_roi_range[0], self.image_roi_range[1], self.image_roi_range[2], self.image_roi_range[3]
        target_img_roi = np.zeros((v_max-v_min, u_max-u_min, 3)).astype(np.uint8)

        depth[depth == 0] = depth.max()
        depth = depth[v_min:v_max, u_min:u_max].reshape(-1)

        self.rectify_poly[:, 2] = depth
        self.rectify_poly[:, 4] = self.dist_center_wo_refract * depth
        self.rectify_poly[:, 5] = depth ** 2
        self.rectify_poly[:, 7] = self.dist_center_wo_refract ** 2 * depth
        self.rectify_poly[:, 8] = self.dist_center_wo_refract * depth ** 2
        self.rectify_poly[:, 9] = depth ** 3

        diff_w_wo_refract = self.refraction_model.predict(self.rectify_poly)

        target_coords = self.source_coords + np.expand_dims(diff_w_wo_refract, axis=-1) * self.center_source_uvec

        _u = np.around(target_coords[:, 0]).astype(np.int32) - u_min
        _v = np.around(target_coords[:, 1]).astype(np.int32) - v_min

        _u = np.clip(_u, 0, u_max-u_min-1)
        _v = np.clip(_v, 0, v_max-v_min-1)

        target_img_roi[_v, _u] = img_wo_refract[self.source_coords[:,1], self.source_coords[:,0]]

        mask = (target_img_roi == 0).astype(np.uint8)
        kernel = np.ones((2,2), np.uint8)
        dil_img = cv2.dilate(target_img_roi, kernel, iterations=1)
        target_img_roi[mask == 1] = dil_img[mask == 1]
        target_img_blur = cv2.blur(target_img_roi, ksize=(3,3))
        target_img_roi[mask == 1] = target_img_blur[mask == 1]

        target_img = np.ones((self.cam_resolution[1], self.cam_resolution[0], 3)).astype(np.uint8) * 255
        target_img[v_min:v_max, u_min:u_max] = target_img_roi
        return target_img

  

    def render_rectified_img(self, nodes):
        rgb, depth = self.render_img(nodes)
        image = self.rectify_img(rgb, depth)
        return image


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



