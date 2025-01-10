import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandpaeentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(0, grandpaeentdir) 

import Sofa
import Sofa.Gui
import os

import Sofa.Core
import Sofa.constants.Key as Key

from typing import Union
from enum import Enum
from pathlib import Path
from typing import List
import numpy as np
import cv2

from tacflex.soft_bubble import SoftBubbleSim

LOADER_INFOS = {
    ".obj": "MeshObjLoader",
    ".stl": "MeshSTLLoader",
    ".STL": "MeshSTLLoader",
    ".vtk": "MeshVTKLoader",
    ".msh": "MeshGmshLoader",
    ".gidmsh": "GIDMeshLoader",
}

MATERIALS =  [
    "Corotated-large",
    "Corotated-small",
    "NeoHookean",
]

class Topology(Enum):
    TRIANGLE    = "Triangle"
    TETRAHEDRON = "Tetrahedron"


def add_loader(
    parent_node: Sofa.Core.Node, 
    filename: Union[Path, str],
    name: str = "loader",
    transformation: list = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1],
    scale: int = 1
    )-> Sofa.Core.Object:

    filepath = Path(filename)
    filetype = filepath.suffix

    loader = parent_node.addObject(LOADER_INFOS[filetype], 
                            filename=str(filepath),
                            name=name,
                            transformation=transformation,
                            scale = scale
                            )
    return loader

def add_topology(
    parent_node: Sofa.Core.Node, 
    mesh_loader: Sofa.Core.Object,
    topology: Topology
    ) -> Sofa.Core.Object:

    topology_container = parent_node.addObject(f"{topology.value}SetTopologyContainer", 
                                                name="topology", 
                                                src=mesh_loader.getLinkPath()
                                                )
    return topology_container

def add_collision_models(
    parent_node: Sofa.Core.Node, 
    contact_stiffness: float = 10,
    moving: int = 1,
    simulated: int = 1,
    color: list = [1, 0, 1, 1],
    triangles: bool = True,
    lines: bool = True,
    points: bool = True,
    spheres: bool = False,
    radius: float = 0.01,
    cuda_flag: bool = False
    ):

    if cuda_flag:
        template = 'CudaVec3f'
    else:
        template = 'Vec3d'

    if spheres:
        parent_node.addObject('SphereCollisionModel',
                                    template=template,
                                    radius=radius,
                                    moving=moving, 
                                    simulated=simulated, 
                                    contactStiffness=contact_stiffness, 
                                    color=color)
    else:
        if triangles:        
            parent_node.addObject('TriangleCollisionModel',
                                  template=template,
                                        moving=moving, 
                                        simulated=simulated, 
                                        contactStiffness=contact_stiffness, 
                                        color=color)
        if lines:
            parent_node.addObject('LineCollisionModel',
                                  template=template,
                                        moving=moving, 
                                        simulated=simulated, 
                                        contactStiffness=contact_stiffness, 
                                        color=color)
        if points:
            parent_node.addObject('PointCollisionModel',
                                  template=template,
                                        moving=moving,
                                        simulated=simulated,
                                        contactStiffness=contact_stiffness,
                                        color=color)

def add_visual_models(
    parent_node: Sofa.Core.Node, 
    color: list = [1, 0, 1, 1],
    ):
    parent_node.addObject('VisualStyle', displayFlags='showVisual ')  # showWireframe
    parent_node.addObject('OglModel', color=color)



class Controller(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.node = args[0]
        self.bubbleNode = self.node.getChild('bubble')
        self.pressureConstraint = self.bubbleNode.Cavity.getObject('SurfacePressureConstraint')

        self.cavity_state = self.bubbleNode.Cavity.getObject('MechCavity')

        ## ------- init object --------------
        init_pose = [0,0,170,0,0,0,1]
        collision_filename = 'assets/objects/meshes/sphere.stl'  # NOTE
        visual_filename = 'assets/objects/meshes/sphere.stl'
        target_position = [0,0,135]
        ## ----------------------------------

        ## init tactile sensor
        self.sim_soft_bubble = SoftBubbleSim(config_path='assets/soft_bubble/conf/soft_bubble.yaml')
        ## -------------

        probe_node = self.node.addChild('object')
        self.state = probe_node.addObject('MechanicalObject',
                                            name='probe_state',
                                            position=init_pose,
                                            template="Rigid3d",
                                            showObject=0,
                                            showObjectScale=1,  #
                                            listening=1
                                            )  # 'MechanicalObject',

        probe_collision = probe_node.addChild("objectCollision")


        collision_loader = add_loader( parent_node=probe_collision,
                                         filename=collision_filename,
                                         name='probe_collision_loader',
                                         scale= 1
                                        )  # ".stl": "MeshSTLLoader"

        add_topology( parent_node=probe_collision, 
                      mesh_loader=collision_loader,
                      topology=Topology.TRIANGLE
                    )   # Tri SetTopologyContainer

        self.collision_state = probe_collision.addObject('MechanicalObject',
                                name='probe_state',
                                template='Vec3d',
                                showObject=0,
                                showObjectScale=1,
                                listening=1
                                )  # 'MechanicalObject'

        add_collision_models( parent_node=probe_collision) 
        probe_collision.addObject('RigidMapping', name='CollisionMapping')

        # Add visual models ----------------------------------------
        visual_node = probe_node.addChild("VisualObject")
        visual_loader = add_loader( parent_node=visual_node,
                                    filename=visual_filename,
                                    name='probe_visual_loader',
                                    scale=1
                                    )  # ".stl": "MeshSTLLoader"

        add_topology( parent_node=visual_node, 
                      mesh_loader=visual_loader,
                      topology=Topology.TRIANGLE
                    ) # Tri SetTopologyContainer

        add_visual_models( parent_node = visual_node,
                            color = [0.957, 0.730, 0.582, 1]
                            # color = [0.957, 0.2, 0.2, 1]
                           ) # 'VisualStyle', displayFlags='showVisual showWireframe'
                           

        visual_node.addObject('RigidMapping', name='VisualMapping') #, input=, output=)


        self.motion_path = create_linear_motion(target_position=np.array(target_position), dt=0.1,
                                                     velocity=2, start_position=np.array(init_pose[:3]))

        self.steps = 0

    def onKeypressedEvent(self, e):
        pressureValue = self.pressureConstraint.value.value[0]

        if e["key"] == Key.plus:
            pressureValue += 0.00001
            if pressureValue > 0.0006:
                pressureValue = 0.0006

        if e["key"] == Key.minus:
            pressureValue -= 0.00001
            if pressureValue < 0:
                pressureValue = 0

        self.pressureConstraint.value = [pressureValue]

    def onAnimateBeginEvent(self, __):
        self.steps += 1
        if self.steps < 50:
            pass
        else:
            if len(self.motion_path):  # if there is a motion path, execute one step
                new_pose = np.append(self.motion_path.pop(0), [0,0,0,1])
                print(new_pose)
                self.state.position.value = [new_pose]
            else:
                self.node.animate = False

        color, depth = self.sim_soft_bubble.render_img(nodes=self.cavity_state.position.value)
        # color = self.sim_soft_bubble.rectify_img(img_wo_refract=color,depth=depth)

        cv2.imshow('rgb image', color)

        depth_min = 75
        depth_max = 140
        depth_norm = 255 - (depth - depth_min) / (depth_max - depth_min) * 255
        depth_norm = depth_norm.astype(np.uint8)
        cv2.imshow('depth image', depth_norm)
        cv2.waitKey(1)


def create_linear_motion(target_position: np.ndarray, dt: float, velocity: float, single_step: bool = False, start_position: np.ndarray = None) -> List[np.ndarray]:
    """
    Creates movement path to displace probe from its current position to the final position provided, at the given velocity.
    """

    # assert len(start_position) == 3

    displacement = target_position - start_position
    print(f"Now moving from {start_position} to {target_position} ")

    motion_path = []

    if single_step:
        motion_path = [target_position]
        motion_steps = 1
    else:
        displacement_per_step = velocity * dt
        motion_steps = int(np.ceil(np.linalg.norm(displacement) / displacement_per_step))

        progress = np.linspace(0.0, 1.0, motion_steps + 1)[1:]
        motion_path = start_position + displacement * progress[:, np.newaxis]

        motion_path[-1] = target_position

    return np.split(motion_path, motion_steps, axis=0)


def createScene(rootNode):
    rootNode.addObject('RequiredPlugin',
                       pluginName='SoftRobots SofaPython3')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')  # Needed to use components [FreeMotionAnimationLoop]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')  # Needed to use components [BVHNarrowPhase,BruteForceBroadPhase,CollisionPipeline]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')  # Needed to use components [LocalMinDistance]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')  # Needed to use components [LineCollisionModel,PointCollisionModel,TriangleCollisionModel]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')  # Needed to use components [CollisionResponse]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Correction')  # Needed to use components [GenericConstraintCorrection,UncoupledConstraintCorrection]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver')  # Needed to use components [GenericConstraintSolver]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select')  # Needed to use components [BoxROI]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh')  # Needed to use components [MeshOBJLoader,MeshSTLLoader,MeshVTKLoader]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct')  # Needed to use components [SparseLDLSolver]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')  # Needed to use components [CGLinearSolver]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear')  # Needed to use components [BarycentricMapping]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear')  # Needed to use components [RigidMapping]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')  # Needed to use components [UniformMass]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')  # Needed to use components [EulerImplicitSolver]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Setting')  # Needed to use components [BackgroundSetting]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')  # Needed to use components [TetrahedronFEMForceField]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')  # Needed to use components [RestShapeSpringsForceField]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')  # Needed to use components [MechanicalObject]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant')  # Needed to use components [MeshTopology]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')  # Needed to use components [TetrahedronSetTopologyContainer]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')  # Needed to use components [VisualStyle]  
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')  # Needed to use components [OglModel,OglSceneFrame]
    rootNode.addObject('RequiredPlugin', name='Sofa.GUI.Component')  # Needed to use components [AttachBodyButtonSetting] 
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.HyperElastic') 
    
    rootNode.addObject('VisualStyle',
                       displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                    'hideBoundingCollisionModels hideForceFields '
                                    'showInteractionForceFields hideWireframe')

    rootNode.gravity.value = [ 0, 0, 0]
    rootNode.addObject('AttachBodyButtonSetting', stiffness=10)
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', tolerance=1e-7, maxIterations=1000)

    rootNode.addObject('CollisionPipeline')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('CollisionResponse', response='FrictionContactConstraint', responseParams='mu=0.6')
    rootNode.addObject('LocalMinDistance', name='Proximity', alarmDistance=5, contactDistance=1)

    rootNode.addObject('BackgroundSetting', color=[0, 0.168627, 0.211765, 1.])
    rootNode.addObject('OglSceneFrame', style='Arrows', alignment='TopRight')

    ## -------------------- 
    bubble = rootNode.addChild('bubble')
    bubble.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    bubble.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixd')

    bubble.addObject('MeshVTKLoader', name='loader', filename='assets/soft_bubble/meshes/body_shell_4.vtk')
    bubble.addObject('MeshTopology', src='@loader', name='container')
    bubble.addObject('MechanicalObject', name='tetras', template='Vec3', showObject=False, showObjectScale=1)
    bubble.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.41,
                     youngModulus=1.0)
    
    bubble.addObject('UniformMass', totalMass=0.04)

    boxROI = bubble.addObject('BoxROI', name='boxROI', box=[-100, -100, -30, 100, 100, 96.9], drawBoxes=False)
    bubble.addObject('RestShapeSpringsForceField', points=boxROI.indices.linkpath, stiffness=1e12, angularStiffness=1e12)
    bubble.addObject('GenericConstraintCorrection')


    cavity = bubble.addChild('Cavity')
    cavity.addObject('MeshSTLLoader', name='cavityLoader', filename='assets/soft_bubble/meshes/cavity.stl')  # NOTE
    cavity.addObject('MeshTopology', src='@cavityLoader', name='cavityMesh')
    cavity.addObject('MechanicalObject', name='MechCavity', template='Vec3d')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.0002,
                     triangles='@cavityMesh.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # save_mesh_filename = 'outputs/sphere/2/'  # NOTE
    # os.makedirs(save_mesh_filename, exist_ok=True)
    # cavity.addObject('MeshExporter', filename=save_mesh_filename, format='mesh', listening="true",
	# 								edges="1", triangles="1", quads="0", tetras="1",
	# 								exportEveryNumberOfSteps = 5)

    # Collision
    collisionbubble = bubble.addChild('Collision')
    collisionbubble.addObject('MeshSTLLoader', name='loader', filename='assets/soft_bubble/meshes/body_shell_4_surface.stl')
    collisionbubble.addObject('MeshTopology', src='@loader', name='topo')
    collisionbubble.addObject('MechanicalObject')
    collisionbubble.addObject('TriangleCollisionModel')
    collisionbubble.addObject('LineCollisionModel')
    collisionbubble.addObject('PointCollisionModel')
    collisionbubble.addObject('BarycentricMapping')

    # Visualization
    modelVisu = bubble.addChild('Visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename='assets/soft_bubble/meshes/body_shell_4_surface.stl')
    modelVisu.addObject('OglModel', src='@loader', color=[0.7, 0.7, 0.7, 0.8])
    modelVisu.addObject('BarycentricMapping')


    rootNode.addObject(Controller(rootNode))


if __name__ == '__main__':
    root = Sofa.Core.Node("root")
    # Call 'createScene' function to create the scene graph
    createScene(root)
    Sofa.Simulation.init(root)

    use_gui = True

    if not use_gui:
        # Execute simulation in background, without GUI
        for iteration in range(1500):
            Sofa.Simulation.animate(root, root.dt.value)
            print(iteration+1)
    else:
        # Find out the supported GUIs
        print ("Supported GUIs are: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")

    print("Simulation is done.")