from ur5py.ur5 import UR5Robot
import time
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

class GasketRobot(UR5Robot):
    def __init__(self):
        super().__init__(gripper=2)
        self.close_gripper_time = 0.5
        self.open_gripper_time = 0.5
    
    def go_home(self):
        # pose that seems to produce a decent home position
        # [-0.06753579965968742, -0.27221476673273887, 0.2710406059524434, -0.11638054743562316, -3.1326272370313237, -0.1142659892269994]
        pose = [0.12344210595006844, -0.4612683741824295, 0.41821285455132917, 0, np.pi, 0]
        rot = R.from_euler("xyz", pose[3:]).as_matrix()
        trans = pose[:3]
        home_pose = RigidTransform(rotation=rot, translation=trans)
        self.move_pose(home_pose)

    def open_grippers(self):
        self.gripper.open()
          
    def close_grippers(self):
        self.gripper.close()
    
    def pick_and_place(self, pick_pose, place_pose):
        self.open_grippers()
        self.move_pose(pick_pose)
        self.close_grippers()
        # need to be slightly overhead first
        overhead_translation = copy.deepcopy(place_pose.translation)
        overhead_translation[2] += 0.017 # some offset, probably need to tune
        overhead_pose = RigidTransform(rotation=place_pose.rotation, translation=overhead_translation)
        breakpoint
        self.move_pose(overhead_pose)
        # go down to our final position
        self.move_pose(place_pose)
        self.open_grippers()

        # always want to push down on where we placed
        self.push(place_pose, is_place_pt=True)

    def push(self, pose, is_place_pt=False, force_ctrl=False):
        if is_place_pt:
            # decrease height so that we can push down and 
            pose.translation[2] -= 0.01
        if not force_ctrl:
            self.close_grippers()
            self.move_pose(pose)
        else:
            # robot = UR5Robot(gripper=2)
            self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,20,0,0,0],2,[3,3,3,3,3,3],damping=0.002)
            time.sleep(0.1)
            self.move_pose(pose, convert=False, interp="tcp", vel=0.1, acc=0.5)
            time.sleep(0.1)
            self.end_force_mode()  
