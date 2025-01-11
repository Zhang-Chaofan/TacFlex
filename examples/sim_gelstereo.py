
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandpaeentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(0, grandpaeentdir) 

import argparse
# import copy
# import h5py
import numpy as np
np.set_printoptions(precision=6)
import os
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.spatial.transform import Rotation as R

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from tacflex.gelstereo import GelStereoSim


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='random', type=str, help="pattern on the sensor surface (marker, checker, random)")
    parser.add_argument('--indenter', default='sphere_2', type=str, help="indenter name")
    parser.add_argument('--elast_model', default='MAT_NEOHOOKEAN', type=str, help="Elastic model, 'MAT_NEOHOOKEAN' or 'MAT_COROTATIONAL'")
    parser.add_argument('--elast_mod', default=2.5e5, type=float, help="Elastic modulus of GelStereo [Pa]") # 1.0e4
    parser.add_argument('--poiss_ratio', default=0.47, type=float, help="Poisson's ratio of GelStereo")  # 0.4223
    parser.add_argument('--damping', default=0.0, type=float, help="damping of GelStereo")  # 1.0e-12
    parser.add_argument('--frict_coeff', default=0.2, type=float, help='Coefficient of friction between GelStereo and indenter')
    parser.add_argument('--err_tol', default=1.0e-5, type=float, help='Maximum acceptable error [m] between target and actual position of indenter')
    parser.add_argument('--sim_pcd', action='store_true', help='simulate tactile 3D point clouds')
    parser.add_argument('--render_img', action='store_true', help='rander tactile images')


    args = parser.parse_args()

    indenter = args.indenter
    sim_pcd_flag = args.sim_pcd
    render_img_flag = args.render_img

    init_indenter_pos = [[ 0. ,  0. , 0.034]]
    target_indenter_dof = np.arange(0.0001, 0.0028, step=0.0001).tolist()


    ## init tactile sensor
    if args.pattern == 'marker':
        sim_gelstereo = GelStereoSim(config_path='assets/gelstereo/conf/gelstereo.yaml')
        gelstereo_urdf_dir = "assets/gelstereo/urdf"   #
        u_min, u_max, v_min, v_max = 390, 950, 90, 650
    elif args.pattern == 'checker':
        sim_gelstereo = GelStereoSim(config_path='assets/gelstereo_checker/conf/gelstereo_checker.yaml')
        gelstereo_urdf_dir = "assets/gelstereo_checker/urdf"   # 
        if sim_pcd_flag:
            print('Warning!!! GelStereo 2.0 with checkerboard pattern does not have 3D point clouds. The VISUAL_MARKER is set to False.')
            sim_pcd_flag = False
        u_min, u_max, v_min, v_max = 390, 930, 110, 650
    elif args.pattern == 'random':
        sim_gelstereo = GelStereoSim(config_path='assets/gelstereo_random/conf/gelstereo_random.yaml')
        gelstereo_urdf_dir = "assets/gelstereo_random/urdf"   # 
        if sim_pcd_flag:
            print('Warning!!! GelStereo 2.0 with random color pattern does not have 3D point clouds. The VISUAL_MARKER is set to False.')
            sim_pcd_flag = False
        u_min, u_max, v_min, v_max = 360, 910, 120, 670
    else:
        print(args.pattern, 'is not available!')
        exit()

    gym = gymapi.acquire_gym()
    # Define sim parameters and create Sim object
    sim = create_sim(gym=gym, frict_coeff=args.frict_coeff)

    # Define and load assets
    asset_options = set_asset_options()
    indenter_urdf_dir = "assets/objects/urdf"
    asset_handles_gelstereo = load_assets(gym=gym,
                                       sim=sim,
                                       base_dir=gelstereo_urdf_dir,
                                       object='gelstereo',
                                       options=asset_options)
    asset_handles_indenters = load_assets(gym=gym,
                                          sim=sim,
                                          base_dir=indenter_urdf_dir,
                                          object=indenter,
                                          options=asset_options,
                                          thickness=0.0001)  # thickness=0.1 mm


    # Define and create scene
    scene_props = set_scene_props(num_envs=len(init_indenter_pos))
    env_handles, actor_handles_gelstereo, actor_handles_indenters = create_scene(gym=gym, 
                                                                               sim=sim, 
                                                                               props=scene_props,
                                                                               assets_gelstereo=asset_handles_gelstereo,
                                                                               assets_indenters=asset_handles_indenters,
                                                                               indenter_offset=init_indenter_pos,
                                                                               elast_model=args.elast_model,
                                                                               youngs=args.elast_mod,
                                                                               possions=args.poiss_ratio,
                                                                               damping=args.damping)    
    viewer, axes_geom = create_viewer(gym=gym, sim=sim)

    # Define controller for indenters
    set_ctrl_props(gym=gym,
                   envs=env_handles,
                   indenters=actor_handles_indenters)

    # # Run simulation loop
    for _ in range(20):
        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Visualize motion and deformation
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        gym.clear_lines(viewer)
   
    num_envs = len(env_handles)
    
    # Get particle state tensor and convert to PyTorch tensor
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    # gelstereo_state_init = copy.deepcopy(particle_state_tensor)

    num_particles_per_env = int(particle_state_tensor.shape[0] // num_envs)
    num_rigid_bodys = gym.get_env_rigid_body_count(env_handles[0])

    step_dist = 0.00002
    indenter_target_now = target_indenter_dof[0]
    indent_dist = target_indenter_dof[0] - 0.0      # target_indenter_dof = [0.0015 0.0017 0.0019 0.0021 0.0023 0.0025 0.0027 0.0029 0.0031 0.0033]
    indent_steps = round(indent_dist/step_dist)
    ctrl_target = 0.0
    ctrl_target = set_ctrl_target(gym=gym,
                                  envs=env_handles,
                                  indenters=actor_handles_indenters,
                                  ctrl_target=ctrl_target,
                                  indent_dist=indent_dist,
                                  indent_steps=indent_steps)


    indent_inc_flags = np.zeros(num_envs)
    indent_dist_flags = np.zeros(num_envs)
    step  = 0
    count_tar = 0  # count 

    if sim_pcd_flag:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        gs = o3d.io.read_triangle_mesh("assets/gelstereo/meshes/frame_v0.3.STL")
        gs.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(gs)

        marker_pcd = o3d.geometry.PointCloud()
        marker_pcd.points = o3d.utility.Vector3dVector([[0,0,28.7]])
        marker_pcd.paint_uniform_color([1, 0.3, 0])
        vis.add_geometry(marker_pcd)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        vis.add_geometry(mesh_frame)


    while not gym.query_viewer_has_closed(viewer):

        # Run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Visualize motion and deformation
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        gym.clear_lines(viewer)
        # print(ctrl_target)


        # Set indentation flags
        env_index = 0
        for _env, _indenter in zip(env_handles, actor_handles_indenters):
            indenter_dof_state = gym.get_actor_dof_states(_env, _indenter, gymapi.STATE_ALL)
            indenter_dof_pos = indenter_dof_state['pos']
            # If indenter joint position exceeds specified indentation increment, set flag
            if abs(ctrl_target - indenter_dof_pos) < args.err_tol:
                indent_inc_flags[env_index] = 1
            # If indenter joint position exceeds full indentation indentation distance, set flag
            if abs(indenter_target_now - indenter_dof_pos) < args.err_tol:
                indent_dist_flags[env_index] = 1
            env_index += 1

        # If all indenter joint positions have exceeded indentation increment targets, 
        # extract results and set new target
        if np.all(indent_inc_flags) and not np.all(indent_dist_flags):
            indent_inc_flags = np.zeros(num_envs)
            ctrl_target = set_ctrl_target(gym=gym,
                                          envs=env_handles,
                                          indenters=actor_handles_indenters,
                                          ctrl_target=ctrl_target,
                                          indent_dist=indent_dist,
                                          indent_steps=indent_steps)

        # If all indenter joint positions have exceeded full indentation distance, 
        # reset gelstereo and indenter states and target
        if np.all(indent_inc_flags) and np.all(indent_dist_flags):
            indent_inc_flags = np.zeros(num_envs)
            indent_dist_flags = np.zeros(num_envs)

            # node position
            nodal_coords = extract_nodal_coords(gym, sim, particle_states=particle_state_tensor)
            # nodal_coords_list.append(nodal_coords)


            if sim_pcd_flag:

                markers_all = []
                for i in range(num_envs):

                    markers = sim_gelstereo.extract_3d_markers(nodes=nodal_coords[i])
                    markers_all.append(markers)

                z_min = 26.5
                z_max = 28.7
                marker_pcd.points = o3d.utility.Vector3dVector(markers_all[0] * 1000)
                # marker_pcd.paint_uniform_color([1, 0.3, 0])
                ver_z = np.asarray(marker_pcd.points)[:, 2]
                z_colors = plt.get_cmap('viridis')(
                    (ver_z - z_min) / (z_max - z_min))
                z_colors = z_colors[:, :3]
                marker_pcd.colors = o3d.utility.Vector3dVector(z_colors)
                
                vis.update_geometry(marker_pcd)
                vis.poll_events()
                vis.update_renderer()

            ## image render
            if render_img_flag:
                for i in range(num_envs):

                    imgL, depthL = sim_gelstereo.render_img(nodes=nodal_coords[i]*1000)

                    # imgL_refract = sim_gelstereo.rectify_img(imgL, depthL)
                    cv2.imshow('tactile image', imgL[v_min:v_max, u_min:u_max]) 
                    cv2.waitKey(1)

            ## set next indenter target
            count_tar += 1
            if count_tar >= len(target_indenter_dof):
                break

            indent_dist = target_indenter_dof[count_tar] - target_indenter_dof[count_tar-1]     # target_indenter_dof = [0.0015 0.0017 0.0019 0.0021 0.0023 0.0025 0.0027 0.0029 0.0031 0.0033]
            indent_steps = round(indent_dist/step_dist)
            indenter_target_now = target_indenter_dof[count_tar]

            ctrl_target = set_ctrl_target(gym=gym,
                                          envs=env_handles,
                                          indenters=actor_handles_indenters,
                                          ctrl_target=ctrl_target,
                                          indent_dist=indent_dist,
                                          indent_steps=indent_steps)
        


    # Clean up
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sim_gelstereo.delete_render()



### --------------------------------------------------------

def create_scene(gym, sim, props, assets_gelstereo, assets_indenters, indenter_offset, elast_model, youngs, possions, damping):
    """Create a scene (i.e., ground plane, environments, gelstereo actors, and indenter actors)."""

    # create ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # create envs
    env_handles = []
    actor_handles_gelstereos = []
    actor_handles_indenters = []
    for i in range(props['num_envs']):
        env_handle = gym.create_env(sim, props['lower'], props['upper'], props['per_row'])
        env_handles.append(env_handle)

        pose = gymapi.Transform()
        collision_group = i
        collision_filter = 0

        pose.p = gymapi.Vec3(0.0, 0.044, 0.0288)  # 0.0288
        r = R.from_euler('XYZ', [90, 0, 180], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_gelstereo = gym.create_actor(env_handle, assets_gelstereo, pose, f"gelstereo_{i}", collision_group, collision_filter)
        actor_handles_gelstereos.append(actor_handle_gelstereo)

        ## set gel materials parameters (soft body simulation)
        # get asset soft body munber
        asset_soft_body_count = gym.get_asset_soft_body_count(assets_gelstereo)
        actor_soft_materials = gym.get_actor_soft_materials(env_handle, actor_handle_gelstereo)
        for j in range(asset_soft_body_count):
            if elast_model == "MAT_NEOHOOKEAN":
                actor_soft_materials[j].model = gymapi.SoftMaterialType.MAT_NEOHOOKEAN  # default = MAT_COROTATIONAL
            elif elast_model == "MAT_COROTATIONAL":
                actor_soft_materials[j].model = gymapi.SoftMaterialType.MAT_COROTATIONAL
            else:
                print('elastic model not supported!!')
                exit()
            actor_soft_materials[j].damping = damping  ## urdf 
            actor_soft_materials[j].youngs = youngs
            actor_soft_materials[j].poissons = possions
        gym.set_actor_soft_materials(env_handle, actor_handle_gelstereo, actor_soft_materials)
        
        pose.p = gymapi.Vec3(*indenter_offset[i])
        r = R.from_euler('xyz', [-90, 0, 90], degrees=True)
        quat = r.as_quat()
        pose.r = gymapi.Quat(*quat)
        actor_handle_indenter = gym.create_actor(env_handle, assets_indenters, pose, f"indenter_{i}", collision_group, collision_filter)
        actor_handles_indenters.append(actor_handle_indenter)

    return env_handles, actor_handles_gelstereos, actor_handles_indenters

def create_sim(gym, frict_coeff):
    """Set the simulation parameters and create a Sim object."""

    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0e-4  # Control frequency    
    sim_params.substeps = 2  # Physics simulation frequency (multiplier)

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    sim_params.stress_visualization = True  # von Mises stress
    sim_params.stress_visualization_min = 1.0e1
    sim_params.stress_visualization_max = 1.0e5

    sim_params.flex.solver_type = 5  # PCR (GPU, global)
    sim_params.flex.num_outer_iterations = 5  # 8
    sim_params.flex.num_inner_iterations = 50  # 40
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
    sim_params.flex.deterministic_mode = True

    # sim_params.flex.geometric_stiffness = 1.0
    sim_params.flex.shape_collision_distance = 0.00001  
    sim_params.flex.shape_collision_margin = 0.00001   
    sim_params.flex.contact_regularization = 1.0e-6
    sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
    sim_params.flex.dynamic_friction = frict_coeff
    sim_params.flex.static_friction = 1.0

    gpu_physics = 0
    gpu_render = 0
    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)

    return sim

def create_viewer(gym, sim):
    """Create viewer and axes objects."""

    camera_props = gymapi.CameraProperties()
    # camera_props.horizontal_fov = 5.0
    # camera_props.width = 1920
    # camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    camera_pos = gymapi.Vec3(0.1, 0.0, 0.1)
    camera_target = gymapi.Vec3(0.0, 0.0, 0.04)
    gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    axes_geom = gymutil.AxesGeometry(0.1)

    return viewer, axes_geom




def extract_net_forces(gym, sim, num_rigid_bodys):
    """Extract the net force vector on the gelstereo for each environment."""
    """Only support for single soft body in each env"""

    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    net_force_vecs = np.zeros((num_envs, 3))
    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // num_rigid_bodys  # rigid body number
        force_vec = contact_force_mag * contact_normal
        net_force_vecs[env_index] += force_vec
    net_force_vecs = -net_force_vecs
    
    return net_force_vecs

def extract_nodal_coords(gym, sim, particle_states):
    """Extract the nodal coordinates for the gelstereo from each environment."""

    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles /  num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env
        local_particle_index = global_particle_index % num_particles_per_env
        nodal_coords[env_index][local_particle_index] = pos.numpy()
    
    return nodal_coords


def load_assets(gym, sim, base_dir, object, options, fix=True, gravity=False, thickness=0.0):
    """Load assets from specified URDF files."""

    options.fix_base_link = True if fix else False
    options.disable_gravity = True if not gravity else False
    options.thickness = thickness
    handle = gym.load_asset(sim, base_dir, object + '.urdf', options)
    
    return handle



def set_asset_options():
    """Set asset options common to all assets."""

    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.armature = 0.01
    options.thickness = 0.0
    options.linear_damping = 1.0
    options.angular_damping = 0.0

    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    options.min_particle_mass = 1e-20

    return options


def set_ctrl_props(gym, envs, indenters, pd_gains=[1.0e9, 0.0]):
    """Set the properties for the indenter PD controllers."""

    for env, indenter in zip(envs, indenters):
        indenter_dof_props = gym.get_actor_dof_properties(env, indenter)
        indenter_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
        indenter_dof_props['stiffness'][0] = pd_gains[0]
        indenter_dof_props['damping'][0] = pd_gains[1]
        gym.set_actor_dof_properties(env, indenter, indenter_dof_props)

def set_ctrl_target(gym, envs, indenters, ctrl_target, indent_dist, indent_steps):
    """Set the controller targets for the next indentation increment."""

    ctrl_target += indent_dist / indent_steps
    for env, indenter in zip(envs, indenters):
        gym.set_actor_dof_position_targets(env, indenter, np.array([ctrl_target], dtype=np.float32))

    # print(ctrl_target)
    return ctrl_target


def set_scene_props(num_envs, env_dim=0.05):
    """Set the scene and environment properties."""

    envs_per_row = int(np.ceil(np.sqrt(num_envs)))
    env_lower = gymapi.Vec3(-env_dim, -env_dim, 0)
    env_upper = gymapi.Vec3(env_dim, env_dim, 2 * env_dim)
    scene_props = {'num_envs': num_envs,
                   'per_row': envs_per_row,
                   'lower': env_lower,
                   'upper': env_upper}

    return scene_props

if __name__ == "__main__":
    main()