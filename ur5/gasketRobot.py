from ur5py import UR5Robot
import time

class GasketRobot(UR5Robot):
    def __init__(self):
        super().__init__()
    
    def pick_and_place(self, pick_pose, place_pose):
        self.move_pose(pick_pose)
        self.close_grippers()
        # need to be slightly overhead first 
        overhead_pose = pick_pose
        overhead_pose[2] += 0.017 # some offset, probably need to tune
        self.move_pose(overhead_pose)
        self.move_pose(place_pose)
        self.open_grippers()
    def push(self, pose, force_ctrl=False):
        if not force_ctrl:
            self.close_grippers()
            self.move_pose(pose)
        else:
            # figure out a way to set this in code!!!
            # robot = UR5Robot(gripper=2)
            self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,20,0,0,0],2,[3,3,3,3,3,3],damping=0.002)
            time.sleep(0.1)
            self.move_pose(pose, convert=False, interp="tcp", vel=0.1, acc=0.5)
            time.sleep(0.1)
            self.end_force_mode()  
