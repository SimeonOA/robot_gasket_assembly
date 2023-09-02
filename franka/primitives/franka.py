from r2d2.misc.server_interface import ServerInterface
import numpy as np
import time
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import math
import sys
from r2d2.misc.parameters import varied_camera_1_id, varied_camera_2_id, nuc_ip
from r2d2.robot_ik.robot_ik_solver import RobotIKSolver
from r2d2.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
from autolab_core import RigidTransform
from r2d2.franka.robot import FrankaRobot
import cowsay
import time
import pdb
import torch


class FrankaFranka:
    def __init__(self, ip_address=None) -> None:
        self.control_frequency = 100 # Hz
        if ip_address is None:
            self.robot = ServerInterface(ip_address=nuc_ip)
        else:
            self.robot = ServerInterface(ip_address)
    
    def go_home(self):
        self.go_to_ee_pose(np.array([0.2, 0, 0.4]), interpolation="linear", duration=2)

    def grab(self, position, rotation=None):
        if rotation is None:
            rotation = np.array([-np.pi, 0, 0])

        # Go to overhead position
        overhead = position + np.array([0, 0, 0.2])
        self.go_to_ee_pose(overhead, rotation, interpolation="linear",duration=2)
        # self.robot.update_pose(np.hstack([overhead, rotation]), False, False)
        self.open_grippers()
        time.sleep(0.3)

        # Descend
        self.delta_movement(delta=np.array([0, 0, -0.2]),delta_time=1, joint_space=False)
        time.sleep(1)

        # Close grippers
        self.close_grippers()
        time.sleep(1)

        # lift
        # self.go_to_ee_pose(overhead, np.array([-np.pi, 0, -rotation[2]]),interpolation="linear")
        self.delta_movement(delta=np.array([0, 0, 0.1]),delta_time=1,  joint_space=False)
    
    def move_to_channel(self, position, offset_position, rotation=None):
        if rotation is None:
            rotation = np.array([-np.pi, 0, 0])

        # Go to overhead position
        offset_position[0] -= 0.015
        # position[0] -= 0.01
        position[1] -= 0.02
        overhead = offset_position + np.array([0, 0, 0.3])
        self.go_to_ee_pose(overhead, rotation, interpolation="joint", duration=2)
        # self.robot.update_pose(np.hstack([overhead, rotation]), False, False)
        time.sleep(1)

        # Descend
        self.go_to_ee_pose(position, rotation, interpolation="linear", duration=2)
        # self.delta_movement(delta=np.array([0, 0, -0.18]), delta_time=1, joint_space=False)
        time.sleep(1)

        # lift
        # self.go_to_ee_pose(overhead, np.array([-np.pi, 0, -rotation[2]]),interpolation="linear")
        # self.delta_movement(delta=np.array([0, 0, 0.2]), delta_time=1, joint_space=False)
    
    def push_slide(self, position_start, position_end, rotation=None, start_offset=0.0, end_offset=0.0, duration = 3):
        position_start[0] += start_offset
        position_end[0] += end_offset
        # position_start[0] -= 0.005
        # position_end[0] -= 0.03

        if rotation is None:
            rotation = np.array([-np.pi, 0, 0])

        # Go to overhead position
        overhead = position_start + np.array([0, 0, 0.1])
        self.go_to_ee_pose(overhead, rotation, interpolation="linear", duration=duration)
        # self.robot.update_pose(np.hstack([overhead, rotation]), False, False)
        time.sleep(1)
        print("########################-1-########################")
        tmp = overhead + np.array([0.015, 0, 0])
        cowsay.cow("overhead" + str(tmp))
        cowsay.tux(str(self.robot.get_ee_pose()))

        # Descend
        # self.go_to_ee_pose(position, rotation, interpolation="joint")

        self.delta_movement(delta=np.array([0, 0, -0.09]), delta_time=1, joint_space=False)
        time.sleep(1)
        print("########################-2-########################")
        tmp = overhead + np.array([0, 0, -0.09]) + np.array([0.015, 0, 0])
        cowsay.cow("overhead" + str(tmp))
        cowsay.tux(str(self.robot.get_ee_pose()))

        # Slide
        # do some delta movement with our waypoint and the waypoint+1 to get a vector 
        # of the direction and then slide for some amount that we need to tune
        self.go_to_ee_pose(position_end, rotation, interpolation="linear", duration=5)
        # direction = position_end - position_start
        # print("the direction vector is: ", direction)
        # self.delta_movement(delta=direction, delta_time=1, joint_space=False)
        time.sleep(1)
        print("########################-3-########################")
        tmp = position_end + np.array([0.015, 0, 0])
        cowsay.cow("position_end" + str(tmp))
        cowsay.tux(str(self.robot.get_ee_pose()))

        # lift
        # self.go_to_ee_pose(overhead, np.array([-np.pi, 0, -rotation[2]]),interpolation="linear")
        self.delta_movement(delta=np.array([0, 0, 0.1]), delta_time=1, joint_space=False)

    def push(self, position, rotation=None):
        # Close grippers
        self.close_grippers()
        time.sleep(2)

        if rotation is None:
            rotation = np.array([-np.pi, 0, 0])

        # Go to overhead position
        overhead = position + np.array([0, 0, 0.1])
        self.go_to_ee_pose(overhead, rotation, interpolation="linear")
        # self.robot.update_pose(np.hstack([overhead, rotation]), False, False)
        time.sleep(1)

        # Descend
        # self.go_to_ee_pose(position, rotation, interpolation="joint")
        self.delta_movement(delta=np.array([0, 0, -0.2]),delta_time=1,  joint_space=False)
        time.sleep(2)

        # lift
        # self.go_to_ee_pose(overhead, np.array([-np.pi, 0, -rotation[2]]),interpolation="linear")
        self.delta_movement(delta=np.array([0, 0, 0.2]), delta_time=1, joint_space=False)


    def twist_eet(self, amount):
        delta_positions = np.zeros(7)
        delta_positions[-1] = amount
        self.delta_movement(delta_positions, delta_time=1, joint_space=True)

    def set_joint_angles(self, joints, duration=3):
        curr_joints = self.robot.get_joint_positions()
        print("curr_joints", curr_joints)
        self.robot.update_joints(curr_joints, velocity=False, blocking=False) # must keep this line
        time.sleep(0.5) # must keep this line

        print("target_joints", joints)
        # joint space interpolation
        waypoints = np.zeros((duration * self.control_frequency, 7))
        for i in range(7):
            waypoints[:, i] = np.linspace(curr_joints[i], joints[i], duration * self.control_frequency)

        for i in range(duration * self.control_frequency):
            self.robot.update_joints(waypoints[i], velocity=False, blocking=False)
            time.sleep(1 / self.control_frequency)
        """
        Don't use the below method even though it is functional because that uses a blocking call
        which loses control of the robot at the end, causing the robot to dip down
        """
        # self.robot.update_joints(joints, velocity=False, blocking=True)

    def go_to_ee_pose(self, position, rotation=None, interpolation="joint", duration=3):
        """
        interpolation: "joint" or "linear"
        duration is the time to reach in seconds
        """
        if rotation is None:
            rotation = np.array([np.pi, 0, 0])
        
        position = position.copy()
        position[0] += 0.015

        if interpolation == "joint":
            pos = torch.Tensor(position)
            quat = torch.Tensor(euler_to_quat(rotation))
            curr_joints = self.robot.get_joint_positions()
            print("curr_joints", curr_joints)
            self.robot.update_joints(curr_joints, velocity=False, blocking=False) # must keep this line
            time.sleep(0.5) # must keep this line

            desired_joints = self.robot.solve_inverse_kinematics(pos, quat, curr_joints)
            print("desired_joints", desired_joints)
            # joint space interpolation
            waypoints = np.zeros((duration * self.control_frequency, 7))
            for i in range(7):
                waypoints[:, i] = np.linspace(curr_joints[i], desired_joints[i], duration * self.control_frequency)

            for i in range(duration * self.control_frequency):
                self.robot.update_joints(waypoints[i], velocity=False, blocking=False)
                time.sleep(1 / self.control_frequency)
        elif interpolation == "linear":
            starting_pose = self.robot.get_ee_pose()
            print("starting_pose", starting_pose)
            self.robot.update_pose(starting_pose, velocity=False, blocking=False) # must keep this line
            time.sleep(0.5) # must keep this line

            target_pose = np.hstack([position, rotation])

            # for the rotation part of starting_pose and target_pose, mod 2pi if their difference is greater than pi
            for i in range(3, 6):
                if abs(starting_pose[i] - target_pose[i]) > np.pi:
                    if starting_pose[i] > target_pose[i]:
                        target_pose[i] += 2 * np.pi
                    else:
                        starting_pose[i] += 2 * np.pi
            
            print("target_pose", target_pose)
            waypoints = np.zeros((duration * self.control_frequency, 6))
            for i in range(6):
                waypoints[:, i] = np.linspace(starting_pose[i], target_pose[i], duration * self.control_frequency)
            for i in range(duration * self.control_frequency):
                self.robot.update_pose(waypoints[i], velocity=False, blocking=False)
                time.sleep(1 / self.control_frequency)

    def delta_movement(self, delta: np.ndarray, delta_time: float = 3, joint_space=False):
        if joint_space:
            curr_joints = self.robot.get_joint_positions()
            print("curr_joints", curr_joints)
            target_joints = curr_joints + delta
            print("target_joints", target_joints)
            self.set_joint_angles(target_joints, duration=delta_time)
        else:
            starting_pose = self.robot.get_ee_pose()
            print("starting_pose", starting_pose)
            if len(delta) == 3:
                delta = np.hstack([delta, np.array([0, 0, 0])])
            target_pose = add_poses(starting_pose, delta)
            print("target_pose", target_pose)
            self.go_to_ee_pose(target_pose[:3], target_pose[3:], interpolation="linear", duration=delta_time)
        """
        The code below is functional, but not as good as go_to_ee_pose() in the sense that there is a drop of control at the end.
        call go_to_ee_pose() instead
        """
        # if joint_space:
        #     self.robot.move_to_joint_positions(delta, time_to_go=delta_time, delta=True)
        # else:
        #     if len(delta) == 3:
        #         self.robot.move_to_ee_pose(delta, delta=True, time_to_go=delta_time, op_space_interp=False)
        #     elif len(delta) == 6:
        #         self.robot.move_to_ee_pose(delta[:3], delta[3:], delta=True, time_to_go=delta_time, op_space_interp=False)

        # current_pose = self.robot.get_ee_pose()
        # r.robot.update_pose(current_pose, velocity=False, blocking=False) # must keep this line to prevent the robot from falling
        # time.sleep(0.5) # must keep this line

    def open_grippers(self, velocity=False, blocking=False, force=0.01):
        self.robot.update_gripper(0, velocity, blocking)

    # def close_grippers(self, velocity=False, blocking=True, force=0.01):
    #     self.robot.update_gripper_analytic(1, velocity, blocking, force)

    def close_grippers(self, velocity=False, blocking=False, force=0.01):
        """
        position = 1 is fully closed
        position = 0 is fully open
        """
        self.robot.update_gripper_analytic(1, velocity, blocking, force)

    def get_robot(self):
        return self.robot


# r = FrankaFranka()
# # # breakpoint()
# # # r.close_grippers(0.1, velocity=False, blocking=False, force=0)
# # r.robot.update_pose(np.array([0.5, 0.4, 0.2, np.pi, 0, 0]), np.array([0.57521065,  0.12303228, -0.08589975]),velocity=False, blocking=True)
# r.go_to_ee_pose(np.array([0.5, -1, 0.2]), interpolation="joint")
# r.go_home()
# # 
#     # r.twist_eet(np.pi/2)
# # time.sleep(0.3)
# # r.delta_movement(np.array([0, -0.2, 0, 0, 0, 0, 0]), joint_space=True)
# # r.delta_movement(np.array([0, -0.5, 0]))
# # r.delta_movement(np.array([0, 0.5, 0]))
# # cowsay.cow("movement complete")

# # r.go_to_ee_pose(np.array([0.5, -0.3, 0.2]), interpolation="joint")

# # r.go_to_ee_pose(np.array([0.5606907,  0.0258932,  0.27876978]), np.array([np.pi, 0, 1.46975052]), interpolation="linear")
# r.grab(np.array([0.55505682, 0.05668623, 0.10147409]),np.array([np.pi, 0, 1.46975052]))
# # for i in range(180):
# #     r.robot.update_pose(np.hstack([np.array([0.5, 0.5 - 1/180 * i, 0.2]), np.array([-np.pi, 0, 0])]), velocity=False, blocking=False)
# #     time.sleep(1/60)


# # for i in range(5):
# #     try:
# #         r.go_to_ee_pose(np.array([0.5, 0.5 - 0.1 * i, 0.2]), velocity=True, blocking=False)
# #         time.sleep(0.3)
# #     except:
# #         pass
# cowsay.cow("movement complete")
