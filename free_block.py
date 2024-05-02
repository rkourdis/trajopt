import time
from tqdm import tqdm
import numpy as np
import pinocchio as pin

ROBOT = \
"""
<robot name="robot">
    <link name="block">
    <visual>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 1.570795 0.0" />
      <geometry>
        <cylinder length="0.1" radius="0.01"/>
      </geometry>
      <material name="gray">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 1.570795 0.0" />
      <geometry>
        <cylinder length="0.1" radius="0.01"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1"/>
      <inertia ixx="5e-6" ixy="0.0" ixz="0.0" iyy="8.583e-05" iyz="0.0" izz="8.583e-05"/>
    </inertial>
  </link>
</robot>
"""

def load_robot():
    URDF = "block.urdf"

    with open(URDF, "w") as wf:
        wf.write(ROBOT)
    
    robot = pin.RobotWrapper.BuildFromURDF(URDF, root_joint = pin.JointModelFreeFlyer())
    robot.model.gravity.setZero()

    robot.setVisualizer(
        pin.visualize.MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    )

    robot.initViewer()
    robot.loadViewerModel()
    robot.display(pin.neutral(robot.model))

    return robot

if __name__ == "__main__":
    NUDGE_N = 40
    NUDGE_POS_X_MM = -50    # [-50, 50]
    DELTA_T_SEC = 5e-4
    DURATION_SEC = 10

    robot = load_robot()

    force = pin.Force(np.array([0, NUDGE_N, 0]), np.zeros(3))
    appl_frame = pin.SE3(np.eye(3), np.array([NUDGE_POS_X_MM * 1e-3, 0, 0]))

    q = pin.neutral(robot.model)
    v = np.zeros(robot.nv)

    # Initial nudge:
    a = pin.aba(
        robot.model, robot.data, q, v, np.zeros(robot.nv), [pin.Force(), appl_frame.act(force)]
    )

    cur_t, q_hist = 0, []

    while cur_t < DURATION_SEC:
        # Integrate previous quantities:
        v += a * DELTA_T_SEC
        q = pin.integrate(robot.model, q, v * DELTA_T_SEC)
        q_hist.append(q.copy())

        # Calculate next accelerations, without external forces:
        a = pin.aba(robot.model, robot.data, q, v, np.zeros(robot.nv))

        # Advance time:
        cur_t += DELTA_T_SEC

    input("Press ENTER to start animation...")

    for state in tqdm(q_hist):
        robot.display(state)
        time.sleep(DELTA_T_SEC)
