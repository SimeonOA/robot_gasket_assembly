import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from shape_match import detect_cable, detect_channel
from trapezoid_channel import trapezoid_actuation
from zed_cam import Zed
from utils import *
from gasketRobot import GasketRobot
from resources import *
import time

robot = GasketRobot()
gripper_weight = 1
robot.set_playload(gripper_weight)
# TODO: fill these in with your values!
overhead_cam_id = ...
overhead_cam = Zed(overhead_cam_id)
# TODO: fill these in with your values!
front_cam_id = ...
front_cam = Zed(front_cam_id)
time.sleep(1)

def get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton)

    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] < 555:
            sorted_channel_pts = sorted_channel_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] >= 555:
            sorted_channel_pts = sorted_channel_pts[::-1]

    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    return swapped_sorted_channel_pts

def no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, matched_template, rgb_img=None, viz=False):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton)
    pick_pt = sorted_cable_pts[0] 
    place_pt = sorted_channel_pts[0]
    if viz:
        plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        plt.imshow(rgb_img)
        plt.title("Pick and Place Pts")
        plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose = robot.get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, matched_template, is_channel_pt=False)
    place_pose = robot.get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, matched_template, is_channel_pt=True)

    robot.pick_and_place(pick_pose, place_pose)
    return swapped_sorted_cable_pts, swapped_sorted_channel_pts, place_pose

def one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, matched_template, second_endpt_side, rgb_img=None, viz=False):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton)

    # now we want the last endpoints to pick and place
    if second_endpt_side == 'right':
        if sorted_cable_pts[-1][1] >= 555:
            pick_pt = sorted_cable_pts[-1]
        else:
            pick_pt = sorted_cable_pts[0]
        if sorted_channel_pts[-1][1] >= 555:
            place_pt = sorted_channel_pts[-1]
        else:
            place_pt = sorted_channel_pts[0]
    else:
        if sorted_cable_pts[-1][1] < 555:
            pick_pt = sorted_cable_pts[-1]
        else:
            pick_pt = sorted_cable_pts[0]
        if sorted_channel_pts[-1][1] < 555:
            place_pt = sorted_channel_pts[-1]
        else:
            place_pt = sorted_channel_pts[0]
    if viz:
        pick_pt = sorted_cable_pts[-1] 
        place_pt = sorted_channel_pts[-1]
        plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        plt.imshow(rgb_img)
        plt.show()

    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose = robot.get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 20, matched_template, False)
    place_pose = robot.get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, matched_template, True)
    robot.pick_and_place(pick_pose, place_pose)
    # this is the pose of the goal position
    return place_pose

def get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton)

    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] < 555:
            sorted_channel_pts = sorted_channel_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] >= 555:
            sorted_channel_pts = sorted_channel_pts[::-1]
    place_pt1 = sorted_channel_pts[0]
    place_pt2 = sorted_channel_pts[-1]
    
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    place_pose_start = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts, 20, True)
    place_pose_stop = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts, 20, True)
    return place_pose_start, place_pose_stop

def pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, ratio, channel_mask=None, is_trapezoid=False, pick_closest_endpoint=False, rgb_img=None, viz=False):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton, is_trapezoid, pick_closest_endpoint)
    
    # doesn't make sense to do this swapping stuff in the context of a trapezoid since the beginning and end points are right next to each other 
    if not is_trapezoid:
        if START_SIDE == 'left':
            if sorted_cable_pts[-1][1] < 555:
                sorted_cable_pts = sorted_cable_pts[::-1]
            if sorted_channel_pts[-1][1] < 555:
                sorted_channel_pts = sorted_channel_pts[::-1]
        else:
            if sorted_cable_pts[-1][1] >= 555:
                sorted_cable_pts = sorted_cable_pts[::-1]
            if sorted_channel_pts[-1][1] >= 555:
                sorted_channel_pts = sorted_channel_pts[::-1]
    else:
        if START_SIDE == 'left':
            if sorted_cable_pts[-1][1] < 555:
                sorted_cable_pts = sorted_cable_pts[::-1]
        else:
            if sorted_cable_pts[-1][1] >= 555:
                sorted_cable_pts = sorted_cable_pts[::-1]
    cable_idx = math.floor(len(sorted_cable_pts)*ratio)
    channel_idx = math.floor(len(sorted_channel_pts)*ratio)

    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]
    if viz:
        plt.title("pick and place pts")
        plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        plt.imshow(rgb_img)
        plt.show()
    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    
    if channel_mask is not None and channel_mask[pick_pt[0]][pick_pt[1]][0] != 0:
        on_channel = True
    else:
        on_channel = False
    pick_pose = robot.get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, is_channel_pt=on_channel)
    place_pose = robot.get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, is_channel_pt=True)
    robot.pick_and_place(pick_pose, place_pose, on_channel)
    return place_pose

class PointSelector:
    def __init__(self, image):
        self.image = image
        self.points = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                self.points.append((x, y))
                print("Clicked at (x={}, y={}) - RGB: {}".format(x, y, self.image[y, x]))
                rw_pt = robot.image_pt_to_rw_pt([x,	y])
                rot = R.from_euler("xyz", [0,np.pi,0]).as_matrix()
                # TODO: fill these in with your values!
                # remember to account for the length of the gripper 
                z_pos = ...
                trans = [rw_pt[0]/1000, rw_pt[1]/1000, z_pos]
                rw_pose = RigidTransform(rotation=rot, translation=trans)
                robot.move_pose(rw_pose)

def save_eval_imgs():
    f_name = f'evaluation_images/trapezoid/overhead_{N}_{PICK_MODE}.png'
    overhead_img = overhead_cam.get_zed_img()
    overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
    plt.imsave(f_name, overhead_img)
    f_name = f'evaluation_images/trapezoid/front_{N}_{PICK_MODE}.png'
    front_img = front_cam.get_zed_img()
    front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
    plt.imsave(f_name, front_img)

def main():
    if not TEST_MODE:
        robot.end_force_mode()
        robot.force_mode(robot.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.1)
        robot.go_home()
        rgb_img = overhead_cam.get_zed_img()
    else:
        rgb_img = cv2.imread(TEST_IMG_PATH)

    #### Calibration Testing ####
    # you can check your calibration by clicking a point on the image 
    # and the robot's end effector should travel to it in the real world
    # have your hand on the e-stop!
    # rgb_img = overhead_cam.get_zed_img()
    # PointSelector(rgb_img)
    # plt.imshow(rgb_img)
    # plt.show()

    if PICK_MODE == "binary" or PICK_MODE == 'hybrid':
        sorted_search_idx = get_binary_search_idx(NUM_PTS)
    elif PICK_MODE == "uni":
        sorted_search_idx = [(1/NUM_PTS)*i for i in range(NUM_PTS)]

    #### PICK AND PLACE #####
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
    
    channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt = detect_channel(rgb_img, VIZ)

    if matched_template != "trapezoid":
        slide_start_pose, slide_end_pose = get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton)
        slide_mid_pose = None
        swapped_sorted_channel_pts = get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton)
        for i in range(len(sorted_search_idx)):
            robot.go_home()
            rgb_img = overhead_cam.get_zed_img()
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            ratio = sorted_search_idx[i]
            if i == 0:
                slide_mid_pose = pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, ratio, channel_cnt_mask, rgb_img=rgb_img, viz=VIZ)
            else:
                pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, ratio, channel_cnt_mask, rgb_img=rgb_img, viz=VIZ)
            robot.gripper.open()

            if PICK_MODE == 'hybrid' and i == 0:
                robot.go_home()
                rgb_img = overhead_cam.get_zed_img()
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                swapped_sorted_cable_pts, swapped_sorted_channel_pts, slide_start_pose = no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, channel_skeleton, matched_template)
                existing_pick_pt = swapped_sorted_cable_pts[0]
                second_endpt_side = 'right' if existing_pick_pt[0] < 555 else 'left'
                robot.go_home()
                rgb_img = overhead_cam.get_zed_img()
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                slide_end_pose = one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, matched_template, second_endpt_side)
    elif matched_template == "trapezoid":
        trapezoid_actuation(channel_cnt, channel_cnt_mask, channel_skeleton, VIZ, overhead_cam, args, PICK_MODE, robot, matched_template)
    
    #### SLIDING/PRESSING ####
    if matched_template != "trapezoid":
        robot.go_home()
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        
        if PRESS_MODE == "binary":
            sorted_press_idx = get_binary_search_idx()
        elif PRESS_MODE == "uni" or PRESS_MODE=='hybrid':
            sorted_press_idx = [(1/NUM_PTS_PRESS)*i for i in range(NUM_PTS_PRESS+1)]

        if matched_template == "curved":
            if args.use_slide:
                if PRESS_BEFORE_SLIDE:
                    robot.press_down(sorted_press_idx)
                if PRESS_MODE == 'uni':
                    robot.slide_curved(swapped_sorted_channel_pts)
                elif PRESS_MODE == 'binary' or PRESS_MODE == "hybrid" or PRESS_MODE == 'golden':
                    midpt_idx = len(swapped_sorted_channel_pts) // 2
                    mid_pose = RigidTransform()
                    mid_pose.rotation = slide_mid_pose.rotation.copy()
                    mid_pose.translation = slide_mid_pose.translation.copy()
                    mid_pose.translation[2] += 0.03

                    robot.move_pose(mid_pose)
                    robot.slide_curved(swapped_sorted_channel_pts[midpt_idx:])
                    robot.go_home()
                    robot.move_pose(mid_pose)
                    robot.slide_curved(swapped_sorted_channel_pts[:midpt_idx][::-1])
            else:
                robot.press_down(sorted_press_idx)
        elif matched_template == "straight":
            if args.use_slide:
                if PRESS_BEFORE_SLIDE:
                    robot.press_down(sorted_press_idx)
                if PRESS_MODE == 'uni':
                    robot.slide_linear(slide_end_pose, slide_start_pose)
                else:
                    robot.rotate_pose90(slide_mid_pose)
                    smp_copy = RigidTransform()
                    smp_copy.rotation = slide_mid_pose.rotation.copy()
                    smp_copy.translation = slide_mid_pose.translation.copy()
                    mid_pose = RigidTransform()
                    mid_pose.rotation = slide_mid_pose.rotation.copy()
                    mid_pose.translation = slide_mid_pose.translation.copy()

                    robot.slide_linear(slide_mid_pose, slide_end_pose)
                    robot.go_home()
                    mid_pose.translation[2] += 0.03
                    robot.move_pose(mid_pose)
                    mid_pose.translation[2] -= 0.03
                    robot.slide_linear(mid_pose, slide_start_pose)
            else:
                robot.press_down(sorted_press_idx)
    robot.go_home()
    save_eval_imgs()
    print("Done with Experiment!")

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    # options for experiment type, either 'no_ends' or 'one_end'
    argparser.add_argument('--exp', type=str, default='no_ends_attached')
    argparser.add_argument('--num_points', type=int, default=8)
    argparser.add_argument('--use_slide', action='store_true', default=False)
    # options for where to press on channel: 'uni', 'binary', 'slide'
    argparser.add_argument('--press_mode', type=str, default='uni')
    # options for where to pick the cable at: 'uni', 'binary'
    argparser.add_argument('--pick_mode', type=str, default='uni')
    argparser.add_argument('--blur_radius', type=int, default=5)
    argparser.add_argument('--sigma', type=int, default=0)
    argparser.add_argument('--dilate_size_channel', type=int, default=2)
    argparser.add_argument('--dilate_size_rope', type=int, default=20)
    argparser.add_argument('--canny_threshold_channel', type=tuple, default=(100,255))
    argparser.add_argument('--canny_threshold_rope', type=tuple, default=(0,255))
    argparser.add_argument('--visualize', default=False, action='store_true')
    argparser.add_argument('--press_before_slide', default=False, action='store_true', help="if True, will press down")
    argparser.add_argument('--exp_num', type=int, default=0, help="used for indicating which experiment you're on")
    args = argparser.parse_args()
    PRESS_MODE = args.press_mode
    PICK_MODE = args.pick_mode
    EXP_MODE = args.exp
    PRESS_BEFORE_SLIDE = args.press_before_slide
    N = args.exp_num
    # how many points along the channel we want to move and press to
    NUM_PTS = args.num_points
    NUM_PTS_PRESS = args.num_points
    TEST_IMG_PATH = args.test_img_path
    VIZ = args.visualize
    main()
