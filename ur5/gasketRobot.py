from ur5py.ur5 import UR5Robot
import time
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import copy
import math
from calibration.image_robot import ImageRobot
from utils import get_rotation, find_nth_nearest_point
from resources import TEMPLATE_HEIGHT
class GasketRobot(UR5Robot, ImageRobot):
    def __init__(self):
        UR5Robot.__init__(gripper=2)
        # loads calibration model for camera to real world
        ImageRobot.__init__()
        self.close_gripper_time = 0.5
        self.open_gripper_time = 0.5
    
    def go_home(self):
        # TODO: fill these in with your values!
        home_joints = ...
        self.move_joint(home_joints, vel=0.8, interp="joint")

    def open_grippers(self):
        self.gripper.open()
          
    def close_grippers(self):
        self.gripper.close()
    
    def pick_and_place(self, pick_pose, place_pose, move_low=False):
        pick_overhead_trans= copy.deepcopy(pick_pose.translation)
        pick_overhead_trans[2] += 0.00 if move_low else 0.03
        pick_overhead_pose = RigidTransform(rotation=pick_pose.rotation, translation=pick_overhead_trans)
        self.gripper.set_pos(135)
        self.move_pose(pick_overhead_pose)
        self.move_pose(pick_pose, interp="tcp")
        self.close_grippers()
        time.sleep(0.5)

        self.move_pose(pick_overhead_pose, interp="tcp")
        print('==========')
        print('MOVE TO PLACE POSE')
        print('==========')

        place_overhead_translation = copy.deepcopy(place_pose.translation)
        # some offset, probably need to tune
        place_overhead_translation[2] += 0.00 if move_low else 0.03
        place_overhead_pose = RigidTransform(rotation=place_pose.rotation, translation=place_overhead_translation)
        place_pre_press_pose_trans = copy.deepcopy(place_pose.translation)
        place_pre_press_pose_trans[2] += 0.03
        place_pre_press_pose = RigidTransform(rotation=place_pose.rotation, translation=place_pre_press_pose_trans)
        self.move_pose(place_overhead_pose)

        place_release_translation = copy.deepcopy(place_pose.translation)
        place_release_translation[2] += 0.00 if move_low else 0.01
        place_release_pose = RigidTransform(rotation=place_pose.rotation, translation=place_release_translation)
        self.move_pose(place_release_pose, interp="tcp")
        self.open_grippers()
        time.sleep(0.5)
        
        self.move_pose(place_pre_press_pose, interp="tcp")
        self.close_grippers()
        time.sleep(0.5)

        # always want to press down on where we placed
        self.press(place_pose, is_place_pt=True, force_ctrl=True)
        # need to do this cause the overhead pose has a different rotation from the descend pose and we don't want to 
        # rotate the gripper while it's on the channel 
        self.rotate_pose90(place_overhead_pose)
        self.move_pose(place_overhead_pose, interp="tcp")

    def press(self, pose, is_place_pt=False, force_ctrl=True):
        if is_place_pt:
            pose.translation[2] -= 0.02
        if not force_ctrl:
            self.close_grippers()
            self.move_pose(pose)
        else:
            self.close_grippers()
            self.descend_to_pose(pose, convert=True)

    def descend_to_pose(self, pose, convert=True, force_limit=50):
        if convert:
            pose.translation[2] += 0.025
        else:
            pose[2] += 0.025
        pose.rotation = R.from_euler("xyz",[0,0,np.pi/2]).as_matrix()@pose.rotation
        self.move_pose(pose, convert=convert, interp="tcp")
        time.sleep(0.5)
        prev_force = np.array(self.get_current_force()[:3])
        # start moving downwards, then measure the baseline force.
        prev_force = np.linalg.norm(np.array(self.get_current_force()[:3]))
        for _ in range(9):
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
            # print(f"Descending {np.linalg.norm(np.array(self.get_current_force()[:3]))}")
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
                print(f"Over threshold, stopping press. Forces: {current_force}")
                break
        self.stop_joint()

    def slide_linear(self, start_pose, goal_pose):
        self.close_grippers()
        goal_pose.rotation = start_pose.rotation
        self.rotate_pose90(goal_pose)
        self.descend_to_pose(start_pose)
        self.force_mode(self.get_pose(convert=False),[0,0,1,0,0,0],[0,0,15,0,0,0],2,[0.05,1,0.2,0.05,0.05,0.05],damping=0.5)
        self.move_pose(goal_pose, interp='tcp')
        self.end_force_mode()
        self.stop_joint()
    
    def slide_curved(self, swapped_sorted_channel_pts):
        # want to be slightly elevated first before going down to slide
        start_overhead = self.get_rw_pose(swapped_sorted_channel_pts[0], swapped_sorted_channel_pts,15, matched_template='curved', is_channel_pt=True)
        start_overhead.translation[2] += 0.03
        self.move_pose(start_overhead)

        poses = []
        self.close_grippers()
        for idx, pt in enumerate(swapped_sorted_channel_pts):
            if idx % 10 != 0:
                continue
            z = -1/1000
            transformed_pose = self.get_rw_pose(pt, swapped_sorted_channel_pts,15, matched_template='curved', is_channel_pt=True)
            transformed_pose.translation[2] = z
            self.rotate_pose90(transformed_pose)
            # pose = [x,y,z,rx,ry,rz]
            poses.append(transformed_pose)

        for pose in poses:
            last_record = time.time()
            self.move_pose(pose)
            while time.time()-last_record < 0.002:
                pass 

    def rotate_pose90(self,pose):
        pose.rotation = R.from_euler("xyz",[0,0,np.pi/2]).as_matrix()@pose.rotation

    def press_down(self, sorted_press_idx, sorted_channel_pts):
        self.go_home()
        for idx in sorted_press_idx:
            idx = math.floor(idx*len(sorted_channel_pts))
            self.press_idx(sorted_channel_pts, idx)

    def get_rw_pose(self, orig_pt, sorted_pixels, n, is_channel_pt, matched_template, use_depth = False):
        behind_idx, infront_idx = find_nth_nearest_point(orig_pt, sorted_pixels, n)
        behind_pt = sorted_pixels[behind_idx]
        infront_pt = sorted_pixels[infront_idx]
        # needs to be done since the point was relative to the entire view of the camera but our model is trained on points defined only in the cropped frame of the image
        orig_pt = np.array(orig_pt) 
        orig_rw_xy = self.image_pt_to_rw_pt(orig_pt) 
        behind_rw_xy = self.image_pt_to_rw_pt(behind_pt)
        infront_rw_xy = self.image_pt_to_rw_pt(infront_pt)
        rot = get_rotation(behind_rw_xy, infront_rw_xy)
        # converting values to meters
        orig_rw_xy = orig_rw_xy / 1000

        # want this z height to have the gripper when closed be just barely above the table
        if not use_depth:
            # TODO: fill these in with your values!
            # make sure to have value in meters
            z_pos = ... 
            # if we want a point on the channel need to account for the height of the template
            if is_channel_pt:
                z_pos += TEMPLATE_HEIGHT[matched_template]
            orig_rw = np.array([orig_rw_xy[0], orig_rw_xy[1], z_pos])
        else:
            raise NotImplementedError
        # converting pose to rigid transform to use with ur5py library
        orig_rt_pose = RigidTransform(rotation=rot, translation=orig_rw)
        return orig_rt_pose 

if __name__ == "__main__":
    ur = GasketRobot()    
    ur.set_playload(1)
    print(ur.get_joints())
