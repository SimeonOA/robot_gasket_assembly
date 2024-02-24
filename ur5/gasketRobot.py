from ur5py.ur5 import UR5Robot
import time
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
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
        pose = [0.12344210595006844, -0.4612683741824295, 0.41821285455132917, 0, np.pi, 0] #<== this is the home for the metal workspace
        # pose = [0.4860, -0.1003, 0.05539, 2.2289, 2.2339, 0.0000] #<== home for the table workspace
        rot = R.from_euler("xyz", pose[3:]).as_matrix()
        trans = pose[:3]
        home_pose = RigidTransform(rotation=rot, translation=trans)
        self.move_pose(home_pose)

    def open_grippers(self):
        self.gripper.open()
          
    def close_grippers(self):
        self.gripper.close()
    
    def pick_and_place(self, pick_pose, place_pose):
        pick_overhead_trans= copy.deepcopy(pick_pose.translation)
        pick_overhead_trans[2] += 0.03 # some offset, probably need to tune
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
        place_overhead_translation[2] += 0.03 # some offset, probably need to tune
        place_overhead_pose = RigidTransform(rotation=place_pose.rotation, translation=place_overhead_translation)
        self.move_pose(place_overhead_pose)

        place_release_translation = copy.deepcopy(place_pose.translation)
        place_release_translation[2] += 0.01
        place_release_pose = RigidTransform(rotation=place_pose.rotation, translation=place_release_translation)
        self.move_pose(place_release_pose, interp="tcp")
        self.open_grippers()
        time.sleep(0.5)
        
        self.move_pose(place_overhead_pose, interp="tcp")
        self.close_grippers()
        time.sleep(0.5)

        # self.descend_to_pose(place_pose)
        # # self.gripper.set_pos(135)
        # time.sleep(0.5)

        # always want to push down on where we placed
        self.push(place_pose, is_place_pt=True, force_ctrl=True)
        self.move_pose(place_overhead_pose, interp="tcp")

    def push(self, pose, is_place_pt=False, force_ctrl=False):
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
            curr_pose = self.get_pose(False)
            curr_pose[2] -= 0.05
            self.descend_to_pose(curr_pose, False)


    def descend_to_pose(self, pose, convert=True, force_limit=50):
        if convert:
            pose.translation[2] += 0.05
        else:
            pose[2] += 0.05
        
        self.move_pose(pose, convert=convert)
        time.sleep(0.5)
        prev_force = np.array(self.get_current_force()[:3])
        # self.force_mode(self.get_pose(False), [0,0,1,0,0,0], [0,0,10,0,0,0], 2, [1,1,1,1,1,1], damping=0.002)
        # self.force_mode(self.get_pose(convert=False),[0,1,1,0,0,0],[0,7,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
        while True:
            # print(np.array(self.get_current_force()[:3]))
            if convert:
                pose.translation[2] -= 0.0001
            else:
                pose[2] -= 0.0001
            self.servo_pose(pose, time=0.01, convert=convert)
            time.sleep(0.01)
            diff = np.linalg.norm(np.array(self.get_current_force()[:3]) - prev_force)
            if diff > force_limit:
                break
        self.stop_joint()


if __name__ == "__main__":


    # Note: linear insertion + slide

    ur = GasketRobot()    
    ur.set_playload(1)
    # ur.force_mode(ur.get_pose(convert=False),[0,1,1,0,0,0],[0,8,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
    # ur.force_mode(ur.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.002)
    ur.force_mode(ur.get_pose(convert=False),[0,1,1,0,0,0],[0,7,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05])
    angles = np.array([-30, -110, -102, -10, 121, 122])
    ur.move_joint(angles * np.pi /180)
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

    positions = []
    for x in range(560, 756,2):
        y = (0.07*(x-560.13))**2 - 513
        z = 171
        # rx = 0.00478*x - 1.958
        # ry = 0.00192*x - 4.1765
        rx = 0
        ry = np.pi
        rz = 0
        rot = R.from_euler("xyz", [rx,ry,rz]).as_matrix()
        trans = [x/1000,y/1000,z/1000]
        print("trans is this: ", trans)
        print("rot is this: ", rot)
        pose = RigidTransform(rotation=rot, translation=trans)
        # pose = [x,y,z,rx,ry,rz]
        positions.append(pose)

    # breakpoint()
    for pos in positions:
        # print(ur.get_pose())
        # ur.move_pose(pos)
        last_record = time.time()
        ur.move_pose(pos)
        while time.time()-last_record < 0.002:
            pass


# 0.78, -3.03, 0.13
# 0.18 -3.06 1.9