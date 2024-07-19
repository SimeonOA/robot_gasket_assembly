import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from autolab_core import RigidTransform
from scipy.spatial.transform import Rotation as R
from calibration.image_robot import ImageRobot
from shape_match import align_channel, get_cable, get_channel
from cam_utils import *
from utils import *
from gasketRobot import GasketRobot
from resources import *
import time

argparser = argparse.ArgumentParser()
# options for experiment type, either 'no_ends' or 'one_end'
argparser.add_argument('--exp', type=str, default='no_ends_attached')
argparser.add_argument('--num_points', type=int, default=8)
argparser.add_argument('--use_slide', action='store_true', default=False)
# options for where to push on channel: 'uni', 'binary', 'slide'
argparser.add_argument('--push_mode', type=str, default='uni')
# options for where to pick the cable at: 'uni', 'binary'
argparser.add_argument('--pick_mode', type=str, default='uni')
argparser.add_argument('--blur_radius', type=int, default=5)
argparser.add_argument('--sigma', type=int, default=0)
argparser.add_argument('--dilate_size_channel', type=int, default=2)
argparser.add_argument('--dilate_size_rope', type=int, default=20)
argparser.add_argument('--canny_threshold_channel', type=tuple, default=(100,255))
argparser.add_argument('--canny_threshold_rope', type=tuple, default=(0,255))
argparser.add_argument('--visualize', default=False, action='store_true')
argparser.add_argument('--push_before_slide', default=False, action='store_true')
argparser.add_argument('--exp_num', type=int, default=0)

def push_down(sorted_push_idx):
    robot.go_home()
    for idx in sorted_push_idx:
        idx = math.floor(idx*len(sorted_channel_pts))
        push_idx(sorted_channel_pts, idx) 

def slide_curved(swapped_sorted_channel_pts, camCal, robot):
    # want to be slightly elevated first before going down to slide
    start_overhead = get_rw_pose(swapped_sorted_channel_pts[0], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
    start_overhead.translation[2] += 0.03
    robot.move_pose(start_overhead)

    poses = []
    robot.close_grippers()
    for idx, pt in enumerate(swapped_sorted_channel_pts):
        if idx % 10 != 0:
            continue
        z = -1/1000
        transformed_pose = get_rw_pose(pt, swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
        transformed_pose.translation[2] = z
        robot.rotate_pose90(transformed_pose)
        # pose = [x,y,z,rx,ry,rz]
        poses.append(transformed_pose)

    for pose in poses:
        last_record = time.time()
        robot.move_pose(pose)
        while time.time()-last_record < 0.002:
            pass 

def detect_cable(rgb_img, args):
    cable_cnt, cable_mask_hollow  = get_cable(img = rgb_img, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize)
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv2.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
    cable_mask_binary = (cable_mask_binary.sum(axis=2)/255).astype('uint8')
    cable_mask_binary = cv2.morphologyEx(cable_mask_binary,cv2.MORPH_CLOSE,np.ones((5,5), np.uint8))
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    assert len(cable_endpoints) == 2
    return cable_skeleton, cable_length, cable_endpoints, cable_mask_binary

def detect_channel(rgb_img):
    matched_template, matched_results, channel_cnt = get_channel(rgb_img)
    if matched_template == 'curved':
        template_mask = curved_template_mask_align
    elif matched_template == 'straight':
        template_mask = straight_template_mask_align
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask
    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    channel_cnt_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    _ = cv2.drawContours(channel_cnt_mask, [channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])],-1, [255,255,255], -1)
    channel_skeleton = skeletonize(aligned_channel_mask)
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    return channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt

def get_sorted_cable_pts(cable_endpoints, cable_skeleton, not_uni=False):
    cable_endpoint_in = cable_endpoints[0]
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

    # TODO trapezoid skeleton is smooth, don't want to delete parts of it. 
    if not_uni:
        sorted_cable_pts = sorted_cable_pts[3:-3]
    else:
        sorted_cable_pts = sorted_cable_pts[START_IDX:END_IDX]
    return sorted_cable_pts

def find_nth_nearest_point(point, sorted_points, given_n):
    idx = sorted_points.index(point)
    behind_idx = np.clip(idx - given_n, 0, len(sorted_points)-1)
    infront_idx = np.clip(idx + given_n, 0, len(sorted_points)-1)
    return behind_idx, infront_idx

def get_rotation(point1, point2):
    direction = np.array(point2) - np.array(point1)
    direction = direction / np.linalg.norm(direction) #unit vector

    reference_direction = np.array([1,0])
    cos_theta = np.dot(direction, reference_direction)
    sin_theta = np.sqrt(1 - cos_theta**2)
    dz = np.arctan2(direction[1], direction[0])
    euler = np.array([-np.pi, 0, dz])
    rot_matrix = R.from_euler("xyz", euler).as_matrix()
    return rot_matrix

def get_rw_pose(orig_pt, sorted_pixels, n, ratio, camCal, is_channel_pt, use_depth = False):
    behind_idx, infront_idx = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    behind_pt = sorted_pixels[behind_idx]
    infront_pt = sorted_pixels[infront_idx]
    # needs to be done since the point was relative to the entire view of the camera but our model is trained on points defined only in the cropped frame of the image
    orig_pt = np.array(orig_pt) 
    orig_rw_xy = camCal.image_pt_to_rw_pt(orig_pt) 
    behind_rw_xy =    camCal.image_pt_to_rw_pt(behind_pt)
    infront_rw_xy = camCal.image_pt_to_rw_pt(infront_pt)
    rot = get_rotation(behind_rw_xy, infront_rw_xy)
    # converting values to meters
    orig_rw_xy = orig_rw_xy / 1000

    # want this z height to have the gripper when closed be just barely above the table
    if not use_depth:
        # make sure to have value in meters
        z_pos = ... 
        # if we want a point on the channel need to account for the height of the template
        if is_channel_pt:
            z_pos += TEMPLATE_HEIGHT[matched_template]
        orig_rw = np.array([orig_rw_xy[0], orig_rw_xy[1],z_pos])
    # converting pose to rigid transform to use with ur5py library
    orig_rt_pose = RigidTransform(rotation=rot, translation=orig_rw)
    return orig_rt_pose

def get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints):
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

def no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, use_depth = False):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton)
    pick_pt = sorted_cable_pts[0] 
    place_pt = sorted_channel_pts[0]
    # Visualization:
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.title("Pick and Place Pts")
    # plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
    place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose, place_pose)
    return swapped_sorted_cable_pts, swapped_sorted_channel_pts, place_pose

def place_halfway(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio):
    pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio)

def one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side):
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
    # Visualization:
    # pick_pt = sorted_cable_pts[-1] 
    # place_pt = sorted_channel_pts[-1]
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.show()

    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 20, 0.1, camCal, False)
    place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)
    robot.pick_and_place(pick_pose, place_pose)
    # this is the pose of the goal position
    return place_pose

def get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, camCal):
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
    place_pose_start = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)
    place_pose_stop = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)
    return place_pose_start, place_pose_stop

def push_idx(sorted_channel_pts, idx, trap=False):
    if idx >= len(sorted_channel_pts):
        idx = len(sorted_channel_pts) - 1
    push_pt = sorted_channel_pts[idx]
    # Visualization:
    # plt.scatter(x=push_pt[1], y=push_pt[0], c='r')
    # plt.title("Push Points")
    # plt.imshow(rgb_img)

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    if trap:
        push_pose_swap = get_rw_pose((push_pt[1], push_pt[0]), swapped_sorted_channel_pts[::-1], 15, 0.1, camCal, is_channel_pt=True)
    else:
        push_pose_swap = get_rw_pose((push_pt[1], push_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)
    push_above_translation = push_pose_swap.translation
    push_above_translation[2] += 0.02
    push_above_pose = RigidTransform(push_pose_swap.rotation, push_above_translation)
    robot.rotate_pose90(push_above_pose)
    robot.push(push_pose_swap)
    robot.move_pose(push_above_pose, interp="tcp")

def pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio, channel_mask=None, is_trapezoid=False, pick_closest_endpoint=False):
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
    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    
    if channel_mask is not None and channel_mask[pick_pt[0]][pick_pt[1]][0] != 0:
        on_channel = True
    else:
        on_channel = False
    pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=on_channel)
    place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)
    robot.pick_and_place(pick_pose, place_pose, on_channel)
    return place_pose

def pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac, idx, num_points, channel_mask=None):
    curr_cable_end = None
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts = get_sorted_cable_pts(cable_endpoints, cable_skeleton)
    # doesn't make sense to do this swapping stuff in the context of a trapezoid since the beginning and end points are right next to each other 
    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
        place_pts = np.linspace(pair[0], pair[1], num_points).astype(int)
    else:
        place_pts = np.linspace(pair[0], len(sorted_channel_pts)-1-pair[1], num_points).astype(int)
    prev_idx = max(0, idx-1)
    curr_frac = np.abs(place_pts[0] - place_pts[idx])/len(sorted_channel_pts)

    cable_idx = int((prev_frac + curr_frac) * len(sorted_cable_pts))
    channel_idx = place_pts[idx]
    print("my channel idx is", channel_idx)
    print("my cable idx is", cable_idx)
    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]
    # Visualization:
    # plt.title("Pick and Place Pts")
    # plt.scatter(x=long_corner0[1], y=long_corner0[0], c='m')
    # plt.scatter(x=long_corner1[1], y=long_corner1[0], c='y')
    # plt.scatter(x=med_corner0[1], y=med_corner0[0], c='c')
    # plt.scatter(x=med_corner1[1], y=med_corner1[0], c='k')
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.show()

    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    
    if channel_mask is not None and channel_mask[pick_pt[0]][pick_pt[1]][0] != 0:
        on_channel = True
    else:
        on_channel = False

    if idx == 0:
        side_len = np.abs(pair[0]-pair[1])
        pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, side_len, 0.1, camCal, is_channel_pt=on_channel)
        place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, side_len, 0.1, camCal, is_channel_pt=True)
    else:
        pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=on_channel)
        place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose, place_pose, on_channel)
    if idx == num_points - 1:
        return prev_frac + curr_frac
    else:
        return prev_frac


def pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac, idx, num_points, channel_mask=None, last=False, 
                               descending=False, channel_idx1_used=False, channel_idx2_used=False, last_trap_side=False, hybrid=False):
    curr_cable_end = None
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts = get_sorted_cable_pts(cable_endpoints, cable_skeleton)
    
    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
        corner1 = pair[0]
        corner2 = pair[1]
    else:
        corner1 = pair[0]
        corner2 = len(sorted_channel_pts)-1-pair[1]

    curr_dist = int(np.abs(corner1-corner2)*idx/num_points)
    if last_trap_side:
        if idx == 0:
            if prev_frac < 0.5:
                cable_idx = 0
            else:
                cable_idx = -1
        elif idx == 1:
            if prev_frac < 0.5:
                cable_idx = int(prev_frac * len(sorted_cable_pts))//2
            else:
                cable_idx = len(sorted_cable_pts) - int((1-prev_frac) * len(sorted_cable_pts))//2
    else:
        if descending:
            cable_idx = int(prev_frac*len(sorted_cable_pts)) - curr_dist
        else:
            cable_idx = int(prev_frac*len(sorted_cable_pts)) + curr_dist

    if idx/num_points != 0.5:
        channel_idx1 = (corner1+corner2)//2 - curr_dist
        channel_idx2 = (corner1+corner2)//2 + curr_dist
        if last_trap_side:
            if np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx1])) < np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx2])):
                channel_idx = channel_idx1
            else:
                channel_idx = channel_idx2
        else:
            if not channel_idx1_used and not channel_idx2_used:
                if np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx1])) < np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx2])):
                    channel_idx = channel_idx1
                    channel_idx1_used = True
                    channel_idx2_used = False
                else:
                    channel_idx = channel_idx2
                    channel_idx1_used = False
                    channel_idx2_used = True
            else:
                assert (channel_idx1_used or channel_idx2_used) and not (channel_idx1_used and channel_idx2_used)
                if channel_idx1_used:
                    channel_idx = channel_idx2
                    channel_idx1_used = True
                    channel_idx2_used = True
                else:
                    channel_idx = channel_idx1
                    channel_idx1_used = True
                    channel_idx2_used = True
    else:
        if descending:
            channel_idx = corner1
        else:
            channel_idx = corner2
    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]

    # Visualization:
    # plt.scatter(x=long_corner0[1], y=long_corner0[0], c='m')
    # plt.scatter(x=long_corner1[1], y=long_corner1[0], c='y')
    # plt.scatter(x=med_corner0[1], y=med_corner0[0], c='c')
    # plt.scatter(x=med_corner1[1], y=med_corner1[0], c='k')
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.show()
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    
    if channel_mask is not None and channel_mask[pick_pt[0]][pick_pt[1]][0] != 0:
        on_channel = True
    else:
        on_channel = False
    pick_pose = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=on_channel)
    if hybrid:
        place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)
    else:
        place_pose = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)
    robot.pick_and_place(pick_pose, place_pose, on_channel)
    robot.go_home()
    if last:
        return prev_frac - np.abs(corner1 - corner2)/(2*len(sorted_channel_pts)) if descending else prev_frac + np.abs(corner1 - corner2)/(2 *len(sorted_channel_pts)), channel_idx1_used, channel_idx2_used
    else:
        return prev_frac, channel_idx1_used, channel_idx2_used

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
                rw_pt = camCal.image_pt_to_rw_pt([x,	y])
                rot = R.from_euler("xyz", [0,np.pi,0]).as_matrix()
                # TODO remember to account for the length of the gripper 
                z_pos = ...
                trans = [rw_pt[0]/1000, rw_pt[1]/1000, z_pos]
                rw_pose = RigidTransform(rotation=rot, translation=trans)
                robot.move_pose(rw_pose)

def save_eval_imgs():
    f_name = f'evaluation_images/trapezoid/overhead_{N}_{PICK_MODE}.png'
    overhead_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
    plt.imsave(f_name, overhead_img)
    f_name = f'evaluation_images/trapezoid/front_{N}_{PICK_MODE}.png'
    front_img = get_zed_img(front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth)
    front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
    plt.imsave(f_name, front_img)

if __name__=='__main__':
    args = argparser.parse_args()
    PUSH_MODE = args.push_mode
    PICK_MODE = args.pick_mode
    EXP_MODE = args.exp
    PUSH_BEFORE_SLIDE = args.push_before_slide
    N = args.exp_num
    # how many points along the channel we want to move and push to
    NUM_PTS = args.num_points
    NUM_PTS_PUSH = args.num_points

    robot = GasketRobot()
    robot.set_playload(1)
    robot.end_force_mode()
    robot.force_mode(robot.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.1)
    robot.go_home()

    # Sets up the zed camera and gets us an image
    overhead_cam_id = ... # overhead camera
    front_eval_cam_id = ... # front eval camera
    side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id)
    front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth = setup_zed_camera(front_eval_cam_id)
    time.sleep(1)
    rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)

    # loads calibration model for camera to real world
    camCal = ImageRobot()

    #### Calibration Testing #########
    # rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    # PointSelector(rgb_img)
    # plt.imshow(rgb_img)
    # plt.show()

    if PICK_MODE == "binary" or PICK_MODE == 'hybrid':
        sorted_search_idx = get_binary_search_idx(NUM_PTS)
    elif PICK_MODE == "uni":
        sorted_search_idx = [(1/NUM_PTS)*i for i in range(NUM_PTS)]

    ### PICK AND PLACE ########
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
    
    channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt = detect_channel(rgb_img, cable_mask_binary, args)

    if matched_template != "trapezoid":
        slide_start_pose, slide_end_pose = get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
        slide_mid_pose = None
        swapped_sorted_channel_pts = get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints)
        for i in range(len(sorted_search_idx)):
            robot.go_home()
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            ratio = sorted_search_idx[i]
            if i == 0:
                slide_mid_pose = pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio, channel_cnt_mask)
            else:
                pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio, channel_cnt_mask)
            robot.gripper.open()

            if PICK_MODE == 'hybrid' and i == 0:
                robot.go_home()
                rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                swapped_sorted_cable_pts, swapped_sorted_channel_pts, slide_start_pose = no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
                existing_pick_pt = swapped_sorted_cable_pts[0]
                second_endpt_side = 'right' if existing_pick_pt[0] < 555 else 'left'
                robot.go_home()
                rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                slide_end_pose = one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side)
    elif matched_template == "trapezoid":
        corners = get_corners(channel_cnt)
        channel_skeleton_corners = match_corners_to_skeleton(corners, channel_skeleton)
        long_corner0, long_corner1, med_corner0, med_corner1 = classify_corners(channel_skeleton_corners)
        channel_start_pt = long_corner0
        # Visualization:
        # plt.title("correct long and med corner pts")
        # plt.imshow(channel_skeleton)
        # plt.scatter(long_corner0[1], long_corner0[0], c='m')
        # plt.scatter(long_corner1[1], long_corner1[0], c='y')
        # plt.scatter(med_corner0[1], med_corner0[0], c='c')
        # plt.scatter(med_corner1[1], med_corner1[0], c='k')
        # plt.scatter(channel_start_pt[1], channel_start_pt[0], c='r')
        # plt.show()
        rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_start_pt, cable_skeleton, channel_skeleton, is_trapezoid=True)
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
        # gets the indices of where each corner is in the sorted channel skeleton
        sorted_channel_pts = np.array(sorted_channel_pts).tolist()
        channel_skeleton_corners = np.array(channel_skeleton_corners).tolist()
        # we always want to sort the channel pts such that long_corner0 is idx 0 and long_corner1_idx < med_corner1_idx < med_corner0_idx
        long_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner0[0] and x[1] == long_corner0[1]][0]
        long_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner1[0] and x[1] == long_corner1[1]][0]
        med_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner0[0] and x[1] == med_corner0[1]][0]
        med_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner1[0] and x[1] == med_corner1[1]][0]

        if np.abs(long_corner0_idx - long_corner1_idx) > len(sorted_channel_pts)/2:
            sorted_channel_pts = sorted_channel_pts[::-1]
            long_corner0_idx = 0
            long_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner1[0] and x[1] == long_corner1[1]][0]
            med_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner0[0] and x[1] == med_corner0[1]][0]
            med_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner1[0] and x[1] == med_corner1[1]][0]
        pairs = [[long_corner0_idx, long_corner1_idx], [long_corner1_idx, med_corner1_idx], [med_corner1_idx, med_corner0_idx], [med_corner0_idx, long_corner0_idx]]

        if PICK_MODE == 'uni':
            prev_frac = 0
            all_num_pts = [8, 4, 4, 4]
            for pair_idx, pair in enumerate(pairs):
                num_pts = all_num_pts[pair_idx]
                for idx in range(num_pts):
                    rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                    prev_frac = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac, idx, num_pts, channel_cnt_mask)
                    robot.go_home()

                if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                    place_pts = np.linspace(pair[0], pair[1], num_pts).astype(int)
                else:
                    place_pts = np.linspace(pair[0], len(sorted_channel_pts)-1-pair[1], num_pts).astype(int)
                for idx in range(num_pts):
                    channel_idx = place_pts[idx]
                    push_idx(sorted_channel_pts, channel_idx, trap=True)
                
                place_pt1 = sorted_channel_pts[pair[0]]
                place_pt2 = sorted_channel_pts[pair[1]]
                # needs to be swapped as this is how it is expected for the robot
                swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
                side_len = np.abs(pair[0]-pair[1])
                slide_start_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
                slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
                robot.slide_linear(slide_end_pose, slide_start_pose)
                robot.go_home()
        
        if PICK_MODE == 'binary':
            prev_frac1 = 0.5
            prev_frac2 = 0.5

            # long side
            pair = pairs[0]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 1')
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 0, 8, channel_cnt_mask)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 2')
            prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 2, 8, channel_cnt_mask)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 3')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 2, 8, channel_cnt_mask, 
                                                    descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 4')
            prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 3, 8, channel_cnt_mask)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 5')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 3, 8, channel_cnt_mask, 
                                                            descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 6')
            prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 1, 8, channel_cnt_mask, last=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 7')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 1, 8, channel_cnt_mask, 
                                                            descending=True, last=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used)

            if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                corner1 = pair[0]
                corner2 = pair[1]
            else:
                corner1 = pair[0]
                corner2 = len(sorted_channel_pts)-1-pair[1]
            
            nums = [4, 2, 6, 1, 7, 3, 5]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/8)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            # medium sides
            pair1 = pairs[1]
            pair2 = pairs[3]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair1, prev_frac1, 0, 4, channel_cnt_mask)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair2, prev_frac2, 0, 4, channel_cnt_mask, descending=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, channel_idx1_used1, channel_idx2_used1 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair1, prev_frac1, 1, 4, channel_cnt_mask)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair2, prev_frac2, 1, 4, channel_cnt_mask,
                                                                                            descending=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair1, prev_frac1, 1, 4, channel_cnt_mask, last=True,
                                                    channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair2, prev_frac2, 1, 4, channel_cnt_mask, descending=True, last=True,
                                                            channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2)
            
            if np.abs(pair1[0] - pair1[1]) < len(sorted_channel_pts)//2:
                corner1 = pair1[0]
                corner2 = pair1[1]
            else:
                corner1 = pair1[0]
                corner2 = len(sorted_channel_pts)-1-pair1[1]
            nums = [2, 1, 3]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            if np.abs(pair2[0] - pair2[1]) < len(sorted_channel_pts)//2:
                corner1 = pair2[0]
                corner2 = pair2[1]
            else:
                corner1 = pair2[0]
                corner2 = len(sorted_channel_pts)-1-pair2[1]
            nums = [2, 1, 3]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            print('LAST SIDE -- SHORT SIDE')
            # short side
            pair = pairs[2]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, channel_idx1_used1, channel_idx2_used1  = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 0, 4, channel_cnt_mask, last_trap_side=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 0, 4, channel_cnt_mask, descending=True, last_trap_side=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 1, 4, channel_cnt_mask, last_trap_side=True,
                                                            channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 1, 4, channel_cnt_mask, descending=True, 
                                                            last_trap_side=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2)
            
            if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                corner1 = pair[0]
                corner2 = pair[1]
            else:
                corner1 = pair[0]
                corner2 = len(sorted_channel_pts)-1-pair[1]
            nums = [2, 1, 3]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]
            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()
        
        if PICK_MODE == 'hybrid':
            prev_frac1 = 0.5
            prev_frac2 = 0.5
            # long side
            pair = pairs[0]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

            print('STEP 1')
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 0, 4, channel_cnt_mask, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 2')
            prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 1, 4, channel_cnt_mask, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 3')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 1, 4, channel_cnt_mask, 
                                                    descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 4')
            prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 2, 4, channel_cnt_mask, last=True, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('STEP 5')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 2, 4, channel_cnt_mask, 
                                                            descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used, last=True, hybrid=True)

            if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                corner1 = pair[0]
                corner2 = pair[1]
            else:
                corner1 = pair[0]
                corner2 = len(sorted_channel_pts)-1-pair[1]
            
            nums = [2, 1, 3, 0, 4]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_mid_pose, slide_end_pose)
            robot.go_home()

            # medium sides
            pair1 = pairs[1]
            pair2 = pairs[3]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, channel_idx1_used1, channel_idx2_used1 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair1, prev_frac1, 0, 4, channel_cnt_mask, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair2, prev_frac2, 0, 4, channel_cnt_mask, descending=True, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair1, prev_frac1, 2, 4, channel_cnt_mask, last=True,
                                                                                            channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair2, prev_frac2, 2, 4, channel_cnt_mask,
                                                                                            descending=True, last=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2, hybrid=True)
            
            if np.abs(pair1[0] - pair1[1]) < len(sorted_channel_pts)//2:
                corner1 = pair1[0]
                corner2 = pair1[1]
            else:
                corner1 = pair1[0]
                corner2 = len(sorted_channel_pts)-1-pair1[1]
            nums = [2, 4]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[corner1]
            place_pt2 = sorted_channel_pts[corner2]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_start_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_start_pose, slide_end_pose)
            robot.go_home()

            if np.abs(pair2[0] - pair2[1]) < len(sorted_channel_pts)//2:
                corner1 = pair2[0]
                corner2 = pair2[1]
            else:
                corner1 = pair2[0]
                corner2 = len(sorted_channel_pts)-1-pair2[1]
            
            nums = [2, 4]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[corner1]
            place_pt2 = sorted_channel_pts[corner2]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_start_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_start_pose, slide_end_pose)
            robot.go_home()

            print('LAST SIDE -- SHORT SIDE')
            # short side
            pair = pairs[2]
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, channel_idx1_used1, channel_idx2_used1  = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 1, 4, channel_cnt_mask, last_trap_side=True, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 1, 4, channel_cnt_mask, descending=True, last_trap_side=True, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac1, 0, 4, channel_cnt_mask, last_trap_side=True,
                                                            channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1, hybrid=True)
            
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            print('CHECK ISSUE!!')
            prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac2, 0, 4, channel_cnt_mask, descending=True, 
                                                            last_trap_side=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2, hybrid=True)
            
            if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                corner1 = pair[0]
                corner2 = pair[1]
            else:
                corner1 = pair[0]
                corner2 = len(sorted_channel_pts)-1-pair[1]
            nums = [1, 3, 2]
            for num in nums:
                channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
                push_idx(sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
            place_pt2 = sorted_channel_pts[corner1]

            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(corner1-corner2)
            slide_mid_pose = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_end_pose, slide_mid_pose)
            robot.go_home()

            place_pt2 = sorted_channel_pts[corner2]
            slide_end_pose = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, 0.1, camCal, True)
            robot.slide_linear(slide_end_pose, slide_mid_pose)
            robot.go_home()
    
    ### SLIDING/PUSHING ####
    if matched_template != "trapezoid":
        robot.go_home()
        rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        
        if PUSH_MODE == "binary":
            sorted_push_idx = get_binary_search_idx()
        elif PUSH_MODE == "uni" or PUSH_MODE=='hybrid':
            sorted_push_idx = [(1/NUM_PTS_PUSH)*i for i in range(NUM_PTS_PUSH+1)]

        if matched_template == "curved":
            if args.use_slide:
                if PUSH_BEFORE_SLIDE:
                    push_down(sorted_push_idx)
                if PUSH_MODE == 'uni':
                    slide_curved(swapped_sorted_channel_pts, camCal, robot)
                elif PUSH_MODE == 'binary' or PUSH_MODE == "hybrid" or PUSH_MODE == 'golden':
                    midpt_idx = len(swapped_sorted_channel_pts) // 2
                    mid_pose = RigidTransform()
                    mid_pose.rotation = slide_mid_pose.rotation.copy()
                    mid_pose.translation = slide_mid_pose.translation.copy()
                    mid_pose.translation[2] += 0.03

                    robot.move_pose(mid_pose)
                    slide_curved(swapped_sorted_channel_pts[midpt_idx:], camCal, robot)
                    robot.go_home()
                    robot.move_pose(mid_pose)
                    slide_curved(swapped_sorted_channel_pts[:midpt_idx][::-1], camCal, robot)
            else:
                push_down(sorted_push_idx)
        elif matched_template == "straight":
            if args.use_slide:
                if PUSH_BEFORE_SLIDE:
                    push_down(sorted_push_idx)
                if PUSH_MODE == 'uni':
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
                push_down(sorted_push_idx)
    robot.go_home()
    save_eval_imgs()
    print("Done with Experiment!")
