from ur5py.ur5 import UR5Robot
import time
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import copy
# from ur5_control import *

class GasketRobot(UR5Robot):
    def __init__(self):
        super().__init__(gripper=2)
        self.close_gripper_time = 0.5
        self.open_gripper_time = 0.5
    
    def go_home(self):
        # pose that seems to produce a decent home position
        # [-0.06753579965968742, -0.27221476673273887, 0.2710406059524434, -0.11638054743562316, -3.1326272370313237, -0.1142659892269994]
        pose = [0.12344210595006844, -0.4612683741824295, 0.41821285455132917, 0, np.pi, 0] #<== this is the home for the metal workspace
        # rot = R.from_euler("xyz", pose[3:]).as_matrix()
        # trans = pose[:3]
        # home_pose = RigidTransform(rotation=rot, translation=trans)
        
        # want to mvoe in jointspace cause that guarantees that the wrist doesn't overrotate
        home_joints = [-1.0757859388934534, -1.5192683378802698, -1.6562803427325647, -1.537062946950094, 1.5727322101593018, 0.49344709515571594]
        self.move_joint(home_joints, vel=0.5, interp="joint")

    def open_grippers(self):
        self.gripper.open()
          
    def close_grippers(self):
        self.gripper.close()
    
    def pick_and_place(self, pick_pose, place_pose, move_low=False):
        pick_overhead_trans= copy.deepcopy(pick_pose.translation)
        pick_overhead_trans[2] += 0.00 if move_low else 0.03 # some offset, probably need to tune
        pick_overhead_pose = RigidTransform(rotation=pick_pose.rotation, translation=pick_overhead_trans)
        self.gripper.set_pos(135)
        self.move_pose(pick_overhead_pose)
        self.move_pose(pick_pose, interp="tcp")
        self.close_grippers()
        time.sleep(0.5)

        self.move_pose(pick_overhead_pose, interp="tcp")
        # need to be slightly overhead first
        
        print('==========')
        print('MOVE TO PLACE POSE')
        print('==========')

        place_overhead_translation = copy.deepcopy(place_pose.translation)
        place_overhead_translation[2] += 0.00 if move_low else 0.03 # some offset, probably need to tune
        place_overhead_pose = RigidTransform(rotation=place_pose.rotation, translation=place_overhead_translation)
        place_pre_push_pose_trans = copy.deepcopy(place_pose.translation)
        place_pre_push_pose_trans[2] += 0.03
        place_pre_push_pose = RigidTransform(rotation=place_pose.rotation, translation=place_pre_push_pose_trans)
        self.move_pose(place_overhead_pose)

        place_release_translation = copy.deepcopy(place_pose.translation)
        place_release_translation[2] += 0.00 if move_low else 0.01
        place_release_pose = RigidTransform(rotation=place_pose.rotation, translation=place_release_translation)
        self.move_pose(place_release_pose, interp="tcp")
        self.open_grippers()
        time.sleep(0.5)
        

        self.move_pose(place_pre_push_pose, interp="tcp")
        self.close_grippers()
        time.sleep(0.5)

        # self.descend_to_pose(place_pose)
        # # self.gripper.set_pos(135)
        # time.sleep(0.5)

        # always want to push down on where we placed
        self.push(place_pose, is_place_pt=True, force_ctrl=True)
        # need to do this cause the overhead pose has a different rotation from the descend pose and we don't want to 
        #rotate the gripper while it's on the channel 
        self.rotate_pose90(place_overhead_pose)
        self.move_pose(place_overhead_pose, interp="tcp")

    def push(self, pose, is_place_pt=False, force_ctrl=True):
        if is_place_pt:
            # decrease height so that we can push down and 
            pose.translation[2] -= 0.02
            print(f"force control: {force_ctrl}")
        if not force_ctrl:
            self.close_grippers()
            self.move_pose(pose)
        else:
            # robot = UR5Robot(gripper=2)
            # self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,-10,0,0,0],2,[1,1,1,1,1,1],damping=0.002)
            # breakpoint()
            # self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,30,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
            # time.sleep(3)
            # self.move_pose(self.get_pose(convert=True), convert=True, interp="tcp", vel=0.1, acc=0.5)
            # self.end_force_mode()
            self.close_grippers()
            self.descend_to_pose(pose, convert=True)

    def descend_to_pose(self, pose, convert=True, force_limit=50):
        # breakpoint()
        if convert:
            pose.translation[2] += 0.03
        else:
            pose[2] += 0.03
        # pose.rotation = pose.rotation + R.from_euler("xyz",[0,0,-np.pi]).as_matrix()
        pose.rotation = R.from_euler("xyz",[0,0,np.pi/2]).as_matrix()@pose.rotation
        self.move_pose(pose, convert=convert, interp="tcp")
        time.sleep(0.5)
        prev_force = np.array(self.get_current_force()[:3])
        # self.force_mode(self.get_pose(False), [0,0,1,0,0,0], [0,0,10,0,0,0], 2, [1,1,1,1,1,1], damping=0.002)
        # self.force_mode(self.get_pose(convert=False),[0,1,1,0,0,0],[0,7,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
        # start moving downwards, then measure the baseline force.
        prev_force = np.linalg.norm(np.array(self.get_current_force()[:3]))
        for i in range(9):
            if convert:
                pose.translation[2] -= 0.0001
            else:
                pose[2] -= 0.0001
            self.servo_pose(pose, time=0.01, convert=convert)
            time.sleep(0.01)
            prev_force += np.linalg.norm(np.array(self.get_current_force()[:3]))
        prev_force /= 10

        current_force = []
        while True:
            # print(np.array(self.get_current_force()[:3]))
            print(f"Descending {np.linalg.norm(np.array(self.get_current_force()[:3]))}")
            if convert:
                pose.translation[2] -= 0.0001
            else:
                pose[2] -= 0.0001
            self.servo_pose(pose, time=0.01, convert=convert)
            time.sleep(0.01)
            current_force.append(np.linalg.norm(np.array(self.get_current_force()[:3])))
            if len(current_force) > 4:
                current_force = current_force[1:]
            if np.average(current_force) > force_limit:
                print("Over threshold, stopping push. Forces:")
                print(current_force)
                break
        self.stop_joint()

    def linear_push(self, start_pose, goal_pose):
        self.close_grippers()
        # want it rotated 90 degrees from the start pose 
        # want these to be the same cause we don't want the gripper rotating while it's sliding
        goal_pose.rotation = start_pose.rotation
        
        # want it rotated 90 degrees from the start pose and descend to pose does that for us for the start pose
        # but we need to explicitly set it for the goal pose
        self.rotate_pose90(goal_pose)
        # breakpoint()
        self.descend_to_pose(start_pose)
        self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,15,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05],damping=0.5)
        self.move_pose(goal_pose, interp='tcp')
        self.end_force_mode()
        self.stop_joint()
    
    def rotate_pose90(self,pose):
        pose.rotation = R.from_euler("xyz",[0,0,np.pi/2]).as_matrix()@pose.rotation


if __name__ == "__main__":


    # Note: linear insertion + slide

    ur = GasketRobot()    
    ur.set_playload(1)
    print(ur.get_joints())
    breakpoint()
    # ur.force_mode(ur.get_pose(convert=False),[0,1,1,0,0,0],[0,8,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
    # ur.force_mode(ur.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.002)
    # ur.force_mode(ur.get_pose(convert=False),[0,1,1,0,0,0],[0,7,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
    # angles = np.array([-30, -110, -102, -10, 121, 122])
    # ur.move_joint(angles * np.pi /180)
    # ur.forceMode(ur.getActualTCPPose(),[0,1,1,0,0,0],[0,8,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
    
    # tsfm = ur.get_pose()
    # tsfm.rotation = np.array([[1, 0, 0], [0, np.sqrt(2)/2, np.sqrt(2)/2], [0, -np.sqrt(2)/2, np.sqrt(2)/2]])@np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
    # # tsfm.translation[2] += 0.05
    # ur.descend_to_pose(tsfm)
    # ur.force_mode(ur.get_pose(convert=False),[0,0,1,0,0,0],[0,0,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05],damping=0.002)
    # tsfm.translation[1]=0.5
    # ur.move_pose(tsfm, interp='tcp')
    # ur.end_force_mode()



    # Recording
    # ur = GasketRobot()    
    # ur.set_playload(1)
    # ur.start_teach()
    # record_time = 20
    # poses = []
    # for i in tqdm(range(int(record_time/0.002))):
    #     last_record = time.time()
    #     poses.append([*ur.get_joints()])
    #     while time.time()-last_record < 0.002:
    #         pass 

    # np.savetxt("roy_recording.txt", poses)
    # ur.stop_teach()


    #playback
    # ur = GasketRobot()    
    # ur.set_playload(1)
    # poses = np.loadtxt("roy_recording.txt")
    # for p in tqdm(poses):
    #     last_record = time.time()
    #     ur.servo_joint(p)
    #     while time.time()-last_record < 0.002:
    #         pass 
