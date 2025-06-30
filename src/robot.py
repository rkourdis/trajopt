from abc import ABC, abstractmethod

import os
import hppfcl
import numpy as np
import casadi as ca

import pinocchio as pin
from pinocchio import casadi as cpin

class LeggedRobot(ABC):
    @property
    @abstractmethod
    def urdf_path() -> str:
        ...
    
    @property
    @abstractmethod
    def frames() -> dict[str, list[str]]:
        ...

    @property
    @abstractmethod
    def actuated_joints() -> list[int]:
        ...
    
    def _load_robot(self):
        pkg_path = os.path.join(os.path.dirname(__file__), "..")
        urdf_path = os.path.join(pkg_path, self.urdf_path)

        # Load full URDF. This creates a RobotWrapper containing the read-only model and runtime data:
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

    # Load robot and add floor if visualize == True:
    def __init__(self, floor_z: float = -0.226274, visualize: bool = False):
        self._load_robot()
        
        if visualize:
            self._create_visualizer(floor_z)
        
        # Create autodiff robot model:
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata  = self.cmodel.createData()

        # Hold information about the robot's feet, actuated joints and environment:
        self.floor_z = floor_z

        # Coefficient of friction between the legs and the ground:
        self.μ = 0
        # self.μ = ca.inf

        # Maximum absolute torque for all joints (N*m):
        self.τ_max = 3.25  # ~14 A for 0.026 Nm/A
        # self.τ_max = ca.inf

        # Maximum L2 norm of the torque vector - this is to prevent
        # the power supply tripping:
        # self.τ_norm_max = 8.0
        self.τ_norm_max = ca.inf

    def q_off(self, joint: str):
        # Get offset of joint in state vector (base orientation uses MRP):
        return 6 + list(self.robot.model.names)[2:].index(joint)


class Solo12(LeggedRobot):
    @property
    def urdf_path(self):
        return "example-robot-data/robots/solo_description/robots/solo12.urdf"
    
    @property
    def frames(self):
        return {
            # The order of GRFs will be as in these lists:
            "feet":      ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"],
            "shoulders": ["FR_SHOULDER", "FL_SHOULDER", "HR_SHOULDER", "HL_SHOULDER"], 
            "knees":     ["FR_LOWER_LEG", "FL_LOWER_LEG", "HR_LOWER_LEG", "HL_LOWER_LEG"], 
        }

    @property
    def actuated_joints(self):
        # Skip 'universe' and 'root_joint' as they're not actuated:
        return [j.id for j in self.robot.model.joints[2:]]

class Bolt(LeggedRobot):
    @property
    def urdf_path(self):
        return "example-robot-data/robots/bolt_description/robots/bolt.urdf"
    
    @property
    def frames(self):
        return {
            "feet": ["FR_FOOT", "FL_FOOT"],
        }

    @property
    def actuated_joints(self):
        # Skip 'universe' and 'root_joint' as they're not actuated:
        return [j.id for j in self.robot.model.joints[2:]]