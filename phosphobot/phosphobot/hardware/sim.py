"""
Setup the simulation environment for the robot
"""

import os
import subprocess
import time

import pybullet as p
from loguru import logger


def simulation_init():
    """
    Initialize the pybullet simulation environment based on the configuration.
    """
    from phosphobot.configs import config

    if config.SIM_MODE == "headless":
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        logger.debug("Simulation: headless mode enabled")

    elif config.SIM_MODE == "gui":
        # Spin up a new process for the simulation

        # Run a new python process
        # cd ./simulation/pybullet && uv run --python 3.8 main.py
        absolute_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "simulation",
                "pybullet",
            )
        )
        subprocess.Popen(["uv", "run", "--python", "3.8", "main.py"], cwd=absolute_path)
        # Wait for 1 second to allow the simulation to start
        time.sleep(1)
        p.connect(p.SHARED_MEMORY)
        logger.debug("Simulation: GUI mode enabled")

    else:
        raise ValueError("Invalid simulation mode")


def simulation_stop():
    """
    Cleanup the simulation environment.
    """
    from phosphobot.configs import config

    if p.isConnected():
        p.disconnect()
        logger.info("Simulation disconnected")

    if config.SIM_MODE == "gui":
        # Kill the simulation process: any instance of python 3.8
        subprocess.run(["pkill", "-f", "python3.8"])


def reset_simulation():
    """
    Reset the simulation environment.
    """
    if p.isConnected():
        p.resetSimulation()
        logger.info("Simulation reset")
    else:
        logger.warning("Simulation is not connected, cannot reset")


def step_simulation(steps=960):
    """
    Step the simulation environment.
    """
    if p.isConnected():
        for _ in range(steps):
            p.stepSimulation()
    else:
        logger.warning("Simulation is not connected, cannot step")


def set_joint_state(robot_id, joint_id: int, joint_position: float):
    """
    Set the joint state of a robot in the simulation.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        joint_id (int): The ID of the joint to set.
        joint_position (float): The position to set the joint to.
    """
    if p.isConnected():
        p.resetJointState(robot_id, joint_id, joint_position)
    else:
        logger.warning("Simulation is not connected, cannot set joint state")


def inverse_dynamics(
    robot_id,
    positions: list,
    velocities: list,
    accelerations: list,
):
    """
    Perform inverse kinematics to compute joint angles from end-effector pose.
    """
    if p.isConnected():
        # Call the PyBullet inverse kinematics function
        joint_angles = p.calculateInverseDynamics(
            robot_id, positions, velocities, accelerations
        )
        return joint_angles
    else:
        logger.warning("Simulation is not connected, cannot perform inverse kinematics")
        return None


def loadURDF(
    urdf_path: str,
    axis: list[float] | None,
    axis_orientation: list[int],
    use_fixed_base: bool,
):
    """
    Load a URDF file into the simulation.

    Args:
        urdf_path (str): The path to the URDF file.
        axis (list[float] | None): The axis of the robot.
        axis_orientation (list[int]): The orientation of the robot.
        use_fixed_base (bool): Whether to use a fixed base for the robot.

    Returns:
        int: The ID of the loaded robot in the simulation.
    """
    if p.isConnected():
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=axis,
            baseOrientation=axis_orientation,
            useFixedBase=use_fixed_base,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )
        num_joints = p.getNumJoints(robot_id)
        actuated_joints = []
        for i in range(num_joints):
            joint_type = get_joint_info(robot_id, i)[2]

            # Consider only revolute joints
            if joint_type in [p.JOINT_REVOLUTE]:
                actuated_joints.append(i)
        return robot_id, num_joints, actuated_joints
    else:
        logger.warning("Simulation is not connected, cannot load URDF")
        return None


def set_joints_states(robot_id, joint_indices, target_positions):
    """
    Set multiple joint states of a robot in the simulation.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        joint_indices (list[int]): The indices of the joints to set.
        target_positions (list[float]): The positions to set the joints to.
    """
    if p.isConnected():
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )
    else:
        logger.warning("Simulation is not connected, cannot set joint states")


def get_joint_state(robot_id, joint_index: int) -> list:
    """
    Get the state of a joint in the simulation.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        joint_index (int): The index of the joint to get.

    Returns:
        list: pybullet list describing the joit state.
    """
    if p.isConnected():
        joint_state = p.getJointState(robot_id, joint_index)
        return joint_state
    else:
        logger.warning("Simulation is not connected, cannot get joint state")
        return []


def inverse_kinematics(
    robot_id,
    end_effector_link_index: int,
    target_position,
    target_orientation,
    restPoses: list,
    jointDamping: list | None = None,
    lowerLimits: list | None = None,
    upperLimits: list | None = None,
    jointRanges: list | None = None,
    maxNumIterations: int = 200,
    residualThreshold: float = 1e-6,
) -> list:
    """
    Perform inverse kinematics to compute joint angles from end-effector pose.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        end_effector_link_index (int): The index of the end-effector link.
        target_position (list): The target position for the end-effector.
        target_orientation (list): The target orientation for the end-effector.
        jointDamping (list, optional): Damping for each joint.
        lowerLimits (list, optional): Lower limits for each joint.
        upperLimits (list, optional): Upper limits for each joint.
        jointRanges (list, optional): Joint ranges for each joint.
        maxNumIterations (int, optional): Maximum number of iterations for IK solver.
        residualThreshold (float, optional): Residual threshold for IK solver.

    Returns:
        list: Joint angles computed by inverse kinematics.
    """
    if p.isConnected():
        if jointDamping is None:
            return p.calculateInverseKinematics(
                robot_id,
                end_effector_link_index,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                restPoses=restPoses,
                lowerLimits=lowerLimits,
                upperLimits=upperLimits,
                jointRanges=jointRanges,
                maxNumIterations=maxNumIterations,
                residualThreshold=residualThreshold,
            )
        return p.calculateInverseKinematics(
            robot_id,
            end_effector_link_index,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            jointDamping=jointDamping,
            solver=p.IK_SDLS,
            restPoses=restPoses,
            lowerLimits=lowerLimits,
            upperLimits=upperLimits,
            jointRanges=jointRanges,
            maxNumIterations=maxNumIterations,
            residualThreshold=residualThreshold,
        )
    else:
        logger.warning("Simulation is not connected, cannot perform inverse kinematics")
        return []


def get_link_state(
    robot_id, link_index: int, compute_forward_kinematics: bool = False
) -> list:
    """
    Get the state of a link in the simulation.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        link_index (int): The index of the link to get.

    Returns:
        list: pybullet list describing the link state.
    """
    if p.isConnected():
        link_state = p.getLinkState(
            robot_id, link_index, computeForwardKinematics=compute_forward_kinematics
        )
        return link_state
    else:
        logger.warning("Simulation is not connected, cannot get link state")
        return []


def get_joint_info(robot_id, joint_index: int) -> list:
    """
    Get the information of a joint in the simulation.

    Args:
        robot_id (int): The ID of the robot in the simulation.
        joint_index (int): The index of the joint to get.

    Returns:
        list: pybullet list describing the joint info.
    """
    if p.isConnected():
        joint_info = p.getJointInfo(robot_id, joint_index)
        return joint_info
    else:
        logger.warning("Simulation is not connected, cannot get joint info")
        return []


def add_debug_text(
    text: str,
    text_position,
    text_color_RGB: list,
    life_time: int = 3,
):
    """
    Add debug text to the simulation.

    Args:
        text (str): The text to display.
        text_position (list): The position to display the text at.
        text_color_RGB (list): The color of the text in RGB format.
        life_time (int, optional): The lifetime of the debug text in seconds. Defaults to 3.
    """
    if p.isConnected():
        p.addUserDebugText(
            text=text,
            textPosition=text_position,
            textColorRGB=text_color_RGB,
            lifeTime=life_time,
        )
    else:
        logger.warning("Simulation is not connected, cannot add debug text")


def add_debug_points(
    point_positions: list,
    point_colors_RGB: list,
    point_size: int = 4,
    life_time: int = 3,
):
    if p.isConnected():
        p.addUserDebugPoints(
            pointPositions=point_positions,
            pointColorsRGB=point_colors_RGB,
            pointSize=point_size,
            lifeTime=life_time,
        )
    else:
        logger.warning("Simulation is not connected, cannot add debug points")


def add_debug_lines(
    line_from_XYZ,
    line_to_XYZ,
    line_color_RGB,
    line_width=4,
    life_time=3,
):
    if p.isConnected():
        p.addUserDebugLine(
            lineFromXYZ=line_from_XYZ,
            lineToXYZ=line_to_XYZ,
            lineColorRGB=line_color_RGB,
            lineWidth=line_width,
            lifeTime=life_time,
        )
    else:
        logger.warning("Simulation is not connected, cannot add debug lines")
