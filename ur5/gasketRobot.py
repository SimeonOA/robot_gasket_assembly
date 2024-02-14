from ur5py import UR5Robot
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
            pose.translation[2] -= 0.005
        if not force_ctrl:
            self.close_grippers()
            self.move_pose(pose)
        else:
            # robot = UR5Robot(gripper=2)
            self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,-10,0,0,0],2,[1,1,1,1,1,1],damping=0.002)
            time.sleep(0.1)
            self.move_pose(pose, convert=False, interp="tcp", vel=0.1, acc=0.5)
            time.sleep(0.1)
            self.end_force_mode()  


    def descend_to_pose(self, pose, convert=True, force_limit=69):
        if convert:
            pose.translation[2] += 0.05
            
        else:
            pose[2] += 0.05

        self.move_pose(pose, convert=convert)
        time.sleep(0.5)
        prev_force = np.array(self.get_current_force()[:3])
        while True:
            print(np.array(self.get_current_force()[:3]))
            if convert:
                pose.translation[2] -= 0.001
            else:
                pose[2] -= 0.001
            self.servo_pose(pose, time=0.1, convert=convert)
            time.sleep(0.1)
            diff = np.linalg.norm(np.array(self.get_current_force()[:3]) - prev_force)
            if diff > force_limit:
                break
        self.stop_joint()


if __name__ == "__main__":


    # Note: linear insertion + slide

    ur = GasketRobot()    
    ur.set_playload(1)
    tsfm = ur.get_pose()
    tsfm.rotation = np.array([[1, 0, 0], [0, np.sqrt(2)/2, np.sqrt(2)/2], [0, -np.sqrt(2)/2, np.sqrt(2)/2]])@np.array([[0,-1,0],[-1,0,0],[0,0,-1]])
    # tsfm.translation[2] += 0.05
    ur.descend_to_pose(tsfm)
    ur.force_mode(ur.get_pose(convert=False),[0,0,1,0,0,0],[0,0,10,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05],damping=0.002)
    tsfm.translation[1]=0.5
    ur.move_pose(tsfm, interp='tcp')
    ur.end_force_mode()



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





