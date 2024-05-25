import os
import time
import hppfcl
import pickle
import argparse
import functools
from tqdm import tqdm
from itertools import chain
from typing import Optional
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin
import intervaltree as ivt
import matplotlib.pyplot as plt
from pinocchio import casadi as cpin

def load_solo12(floor_z = 0.0, visualize = False):
    pkg_path = os.path.dirname(__file__)
    urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

    # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
    )

    if not visualize:
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

if __name__ == "__main__":
    robot, viz = load_solo12(floor_z = -10, visualize = False)

    q = pin.neutral(robot.model)
    q[3:7] = pin.randomConfiguration(robot.model)[3:7]

    v = np.append(np.array([0, 0, 1]), np.zeros(15))
    a = np.zeros(robot.nv)

    qs = []
    HZ = 100
    for idx in range(HZ):
        q = pin.integrate(robot.model, q, v)
        qs.append(np.copy(q))
        print(q)
    # frame_xyzs, frame_vels = [], []
    # frame = "FR_FOOT"    
    # f_id = robot.model.getFrameId(frame)

    # for cur_q in qs:
    #     pin.forwardKinematics(robot.model, robot.data, cur_q, v, a)
    #     pin.updateFramePlacements(robot.model, robot.data)

    #     xyz = np.copy(robot.data.oMf[f_id].translation)
    #     frame_xyzs.append(xyz)

    #     fvel = pin.getFrameVelocity(robot.model, robot.data, f_id, pin.WORLD).linear
    #     frame_vels.append(fvel)
    #     print(fvel)
    