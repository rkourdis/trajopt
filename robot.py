import os
import hppfcl
import numpy as np

import pinocchio as pin
from pinocchio import casadi as cpin

class Solo12:
    def _load_robot(self):
        pkg_path = os.path.dirname(__file__)
        urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

        # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
        )

    def _create_visualizer(self, floor_z: float = 0.0):
        self.visualizer = pin.visualize.MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )

        self.robot.setVisualizer(self.visualizer)
        self.robot.initViewer()
        self.robot.loadViewerModel()

        # Add floor visual geometry:
        floor_obj = pin.GeometryObject("floor", 0, 0, hppfcl.Box(2, 2, 0.005), pin.SE3.Identity())
        self.visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))

        # Manually set the floor transform because the GeometryObject() constructor doesn't work:
        floor_obj_name = self.visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)
        self.visualizer.viewer[floor_obj_name].set_transform(
            pin.SE3(np.eye(3), np.array([0, 0, floor_z])).homogeneous
        )

        # Display an initial pose:
        self.robot.display(pin.neutral(self.robot.model))

    # Load Solo 12 robot and a floor at a given height, if visialize == True:
    def __init__(self, floor_z: float = -0.226274, visualize: bool = False):
        self._load_robot()
        
        if visualize:
            self._create_visualizer(floor_z)
        
        # Create autodiff robot model:
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata  = self.cmodel.createData()

        # Hold information about the robot's feet, actuated joints and environment:
        self.floor_z = floor_z

        # The order of forces will be as in this list:
        self.feet = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

        # Skip 'universe' and 'root_joint' as they're not actuated:
        self.actuated_joints = [j.id for j in self.robot.model.joints[2:]]

        # Coefficient of friction between the legs and the ground:
        self.μ = 0.7

        # Maximum absolute torque for all joints (N*m):
        self.τ_max = 2.2

        # Maximum L2 norm of the torque vector - this is to prevent
        # the power supply tripping:
        self.τ_norm_max = 8.0
    
    def q_off(self, joint: str):
        return 6 + list(self.robot.model.names)[2:].index(joint)