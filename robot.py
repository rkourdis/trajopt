import os
import hppfcl
import numpy as np
import pinocchio as pin

# Load Solo 12 robot and a floor at a given height.
# Returns robot and visualiser, if 'visualise' == True:
def load_solo12(floor_z = 0.0, visualise = False):
    pkg_path = os.path.dirname(__file__)
    urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

    # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
    )

    if not visualise:
        return robot, None

    visualizer = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )

    robot.setVisualizer(visualizer)
    robot.initViewer()
    robot.loadViewerModel()

    # Add floor visual geometry:
    floor_obj = pin.GeometryObject("floor", 0, 0, hppfcl.Box(2, 2, 0.005), pin.SE3.Identity())
    visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))
    
    floor_obj_name = visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)

    # Manually set the transform because the GeometryObject() constructor doesn't work:
    visualizer.viewer[floor_obj_name].set_transform(
        pin.SE3(np.eye(3), np.array([0, 0, floor_z])).homogeneous
    )

    robot.display(pin.neutral(robot.model))
    return robot, visualizer
