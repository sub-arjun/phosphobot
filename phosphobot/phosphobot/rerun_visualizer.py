import numpy as np
from typing import Optional, List
from loguru import logger
from phosphobot.models import Observation, Step
from phosphobot.hardware import BaseRobot
from phosphobot.camera import AllCameras
from phosphobot.utils import get_quaternion_from_euler

class RerunVisualizer:
    def __init__(self, enable: bool = True):
        self.enabled = enable and RERUN_AVAILABLE
        self.initialized = False
            
    def initialize(self, dataset_name: str, episode_index: int) -> None:
        if not self.enabled:
            return
            
        try:
            app_name = f"phosphobot_{dataset_name}_ep{episode_index}"
            rr.init(app_name, spawn=True)
            
            rr.log(
                "world",
                rr.ViewCoordinates.RIGHT_HAND_Y_UP,
                static=True,
            )
            
            self.initialized = True
            logger.info(f"Rerun initialized for {app_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Rerun: {e}")
            self.enabled = False

    def log_step(
        self,
        step: Step,
        robots: List[BaseRobot],
        cameras: AllCameras,
        step_index: int,
    ) -> None:
        if not self.enabled or not self.initialized:
            return
            
        try:
            observation = step.observation
            timestamp = observation.timestamp or 0.0
            
            rr.set_time("recording_time", timestamp=timestamp)
            rr.set_time("step", sequence=step_index)
            
            self._log_camera_data(observation)
            self._log_robot_data(observation, robots)
            self._log_joint_timeseries(observation, robots)
            
            if observation.language_instruction:
                rr.log(
                    "instruction",
                    rr.TextDocument(observation.language_instruction),
                )
                
        except Exception as e:
            logger.warning(f"Failed to log step to Rerun: {e}")

    def _log_camera_data(self, observation: Observation) -> None:
        if observation.main_image is not None and observation.main_image.size > 0:
            # Ensure image is in RGB format (rerun expects RGB)
            main_image = observation.main_image
            if len(main_image.shape) == 3 and main_image.shape[2] == 3:
                rr.log("world/cameras/main", rr.Image(main_image))
                
                # Log camera transform 
                rr.log(
                    "world/cameras/main",
                    rr.Transform3D(
                        translation=[0, 0, 1.5],  # 1.5m overhead
                        rotation=rr.Quaternion(xyzw=[0.5, -0.5, 0.5, 0.5])  # Looking down
                    )
                )
        
        #secondary cameras
        for i, secondary_image in enumerate(observation.secondary_images):
            if secondary_image is not None and secondary_image.size > 0:
                if len(secondary_image.shape) == 3 and secondary_image.shape[2] == 3:
                    rr.log(f"world/cameras/secondary_{i}", rr.Image(secondary_image))
                    
                    # Log secondary camera transforms (arranged in a circle)
                    angle = 2 * np.pi * i / max(len(observation.secondary_images), 1)
                    rr.log(
                        f"world/cameras/secondary_{i}",
                        rr.Transform3D(
                            translation=[np.cos(angle) * 1.2, np.sin(angle) * 1.2, 1.0]
                        )
                    )

    def _log_robot_data(self, observation: Observation, robots: List[BaseRobot]) -> None:        
        if observation.joints_position is not None and len(observation.joints_position) > 0:
            joints = observation.joints_position
            
            joint_positions = []
            joint_colors = []
            
            for i, joint_angle in enumerate(joints):
                x_pos = i * 0.1  # 10cm spacing
                y_pos = 0
                z_pos = 0.5  # 50cm height
                
                joint_positions.append([x_pos, y_pos, z_pos])
                
                normalized_angle = (joint_angle + np.pi) / (2 * np.pi)  # Normalize -π to π -> 0 to 1
                color_intensity = int(255 * normalized_angle)
                joint_colors.append([color_intensity, 100, 255 - color_intensity])
            
            if joint_positions:
                rr.log(
                    "world/robot/joints",
                    rr.Points3D(
                        positions=joint_positions,
                        colors=joint_colors,
                        radii=[0.015] * len(joint_positions)
                    )
                )

    def _log_joint_timeseries(self, observation: Observation, robots: List[BaseRobot]) -> None:
        if observation.joints_position is not None and len(observation.joints_position) > 0:
            joints = observation.joints_position
            
            # Get the number of actuated joints from the robot 
            num_actuated_joints = 6  # Default fallback
            if robots and hasattr(robots[0], 'num_actuated_joints'):
                num_actuated_joints = robots[0].num_actuated_joints
            
            for i, joint_angle in enumerate(joints):
                rr.log(
                    f"plots/joint_{i}",
                    rr.Scalars([joint_angle])
                )
            
            # Log gripper state if we have more joints than the actuated ones
            if len(joints) > num_actuated_joints:
                gripper_state = joints[num_actuated_joints]  # First joint after actuated ones
                rr.log("plots/gripper", rr.Scalars([gripper_state]))

    def finalize(self) -> None:
        if not self.enabled or not self.initialized:
            return
            
        try:
            logger.info("Rerun visualization completed")
        except Exception as e:
            logger.warning(f"Error finalizing Rerun: {e}") 