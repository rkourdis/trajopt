from pathlib import Path
import coal
import numpy as np

import pinocchio as pin
from pinocchio import casadi as cpin

class Go2:
    def _load_robot(self):
        pkg_path = Path(__file__).resolve().parents[1] / "unitree_ros" / "robots"
        urdf_path = pkg_path / "go2_description" / "urdf" / "go2_description.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Go2 model missing at {urdf_path}.")

        # Load full URDF. This creates a RobotWrapper containing the read-only model and runtime data:
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path), package_dirs = [str(pkg_path)], root_joint = pin.JointModelFreeFlyer()
        )

    def _create_visualizer(self, floor_z: float = 0.0):
        from pinocchio.visualize import MeshcatVisualizer

        self.visualizer = MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )

        self.robot.setVisualizer(self.visualizer)
        self.robot.initViewer()
        self.robot.loadViewerModel()

        # Add floor visual geometry:
        floor_obj = pin.GeometryObject("floor", 0, 0, pin.SE3.Identity(), coal.Box(2, 2, 0.005))
        self.visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))

        floor_obj_name = self.visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)
        self.visualizer.viewer[floor_obj_name].set_transform(
            pin.SE3(np.eye(3), np.array([0, 0, floor_z - 0.0025])).homogeneous
        )

        # Display an initial pose:
        from poses import Pose, load_robot_pose
        from utilities import q_mrp_to_quat, ca_to_np
        q = load_robot_pose(Pose.STANDING, floor_z)[0]
        self.robot.display(ca_to_np(q_mrp_to_quat(q)))

    def __init__(self, floor_z: float = 0.0, visualize: bool = False):
        self._load_robot()
        
        # Create autodiff robot model:
        self.cmodel = cpin.Model(self.robot.model)
        self.cdata  = self.cmodel.createData()

        # Hold information about the robot's feet, actuated joints and environment:
        self.floor_z = floor_z
        self.foot_center_offset = np.array([-0.002, 0.0, 0.0])
        self.name = "go2"

        self.frames = {
            # The order of GRFs will be as in this list:
            "feet":      ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            "shoulders": ["FR_hip", "FL_hip", "RR_hip", "RL_hip"],
            "knees":     ["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
        }

        # Skip 'universe' and 'root_joint' as they're not actuated:
        self.actuated_joints = [j.id for j in self.robot.model.joints[2:]]

        # Coefficient of friction between the legs and the ground:
        self.μ = 0.7

        # Limits in model/actuator order, from Unitree's pinned URDF.
        # Hip/thigh: 23.7 Nm, 30.1 rad/s. Calf: 45.43 Nm, 15.7 rad/s.
        self.τ_max = self.robot.model.effortLimit[6:].copy()
        self.v_max = self.robot.model.velocityLimit[6:].copy()
        self.q_min = self.robot.model.lowerPositionLimit[7:].copy()
        self.q_max = self.robot.model.upperPositionLimit[7:].copy()
        self.joint_damping = self.robot.model.damping[6:].copy()

        if visualize:
            self._create_visualizer(floor_z)
    
    def q_off(self, joint: str):
        if not self.robot.model.existJointName(joint):
            raise ValueError(f"Unknown Go2 joint: {joint}")

        # MRP has one less coordinate than quaternion q:
        return self.robot.model.joints[self.robot.model.getJointId(joint)].idx_q - 1
