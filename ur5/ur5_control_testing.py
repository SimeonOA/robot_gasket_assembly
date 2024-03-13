from ur5py.ur5 import UR5Robot
import numpy as np
import cv2
# from shape_match import *
from new_shape_match import get_channel #, get_cable
from shape_match import align_channel, get_closest_channel_endpoint,  get_cable
from autolab_core import RigidTransform
import pdb
from real_sense_modules import *
from utils import *
import argparse
from gasketRobot import GasketRobot
from scipy.spatial.transform import Rotation as R
from resources import CROP_REGION, MIDPOINT_THRESHOLD, curved_template_mask, straight_template_mask, trapezoid_template_mask, curved_template_mask_align, straight_template_mask_align
from calibration.image_robot import ImageRobot
from zed_camera import ZedCamera
import math


# CROP_REGION = [64, 600, 189, 922]
# curved_template_mask = cv2.imread('templates_crop_master/master_curved_channel_template.png')
# straight_template_mask = cv2.imread('templates_crop_master/master_straight_channel_template.png')
# trapezoid_template_mask = cv2.imread('templates_crop_master/master_trapezoid_channel_template.png')

START_IDX = 20 # was 8
END_IDX = -START_IDX - 1
# NUM_PTS_PUSH = 8
# NUM_PTS = 4
START_SIDE = 'left'



argparser = argparse.ArgumentParser()
# options for experiment type, either 'no_ends' or 'one_end'
argparser.add_argument('--exp', type=str, default='no_ends_attached')
argparser.add_argument('--num_points', type=int, default=8)
argparser.add_argument('--use_slide', action='store_true', default=False)
# options for where to push on channel: 'uni', 'golden', 'binary', 'slide'
argparser.add_argument('--push_mode', type=str, default='uni')
# options for where to pick the cable at: 'uni', 'golden', 'binary'
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


def match_corners_to_skeleton(corners, skeleton):
    search_pts = np.where(skeleton > 0)
    search_pts = np.vstack((search_pts[0], search_pts[1]))
    # this is now an (N, 2) array
    search_pts = search_pts.T

    # match each corner to the closest pixel on the skeleton
    matched_pts = []
    # this is a (4,2)
    corners = np.array(corners)
    for corner in corners:
        dist_2 = np.sum((search_pts - corner)**2, axis=1)
        matched_pts.append(search_pts[np.argmin(dist_2)])
    return matched_pts


def get_midpt_corners(skeleton, corner0, corner1):
    real_midpt_x = (corner0[0] + corner1[0])//2
    real_midpt_y = (corner0[1] + corner1[1])//2
    real_midpt = np.array([real_midpt_x, real_midpt_y])
    search_pts = np.where(skeleton > 0)
    search_pts = np.vstack((search_pts[0], search_pts[1]))
    # this is now an (N, 2) array
    search_pts = search_pts.T

    # match each corner to the closest pixel on the skeleton
    # this is a (4,2)
    dist_2 = np.sum((search_pts - real_midpt)**2, axis=1)
    matched_midpt = search_pts[np.argmin(dist_2)]
    return matched_midpt


def classify_corners(channel_skeleton_corners):
    dist0 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[1])
    dist1 = np.linalg.norm(channel_skeleton_corners[1]-channel_skeleton_corners[2])
    dist2 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[3])
    max_dist = max([dist0,dist1, dist2])
    # long_cornerX and med_cornerX are the corners on the same side of the trapezoid with one being a corner for the long side and the other being the corner for the medium side
    if max_dist == dist0:
        long_corner0 = channel_skeleton_corners[0]
        long_corner1 = channel_skeleton_corners[1]
        med_corner0 = channel_skeleton_corners[3]
        med_corner1 = channel_skeleton_corners[2]
    elif max_dist == dist1:
        long_corner0 = channel_skeleton_corners[1]
        long_corner1 = channel_skeleton_corners[2]
        med_corner0 = channel_skeleton_corners[0]
        med_corner1 = channel_skeleton_corners[3]
    else:
        long_corner0 = channel_skeleton_corners[0]
        long_corner1 = channel_skeleton_corners[3]
        med_corner0 = channel_skeleton_corners[1]
        med_corner1 = channel_skeleton_corners[2]
    
    return long_corner0, long_corner1, med_corner0, med_corner1

def push_down(sorted_push_idx):
    robot.go_home()
    # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    # rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    # cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
    # sorted_cable_points, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)
    for idx in sorted_push_idx:
        # do i-1 cause i == 1 when we get into here
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
        # y = (0.07*(x-560.13))**2 - 513
        z = -1/1000 #171/1000
        # rx = 0.00478*x - 1.958
        # ry = 0.00192*x - 4.1765
        
        ## 3/2 old implementation of this
        # rx = 0
        # ry = np.pi
        # rz = 0
        # # original_point = [x,y,z,rx,ry,rz]
        # transformed_point = get_rw_pose(pt, swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True).translation

        ## 3/2 new implementation
        transformed_point = get_rw_pose(pt, swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)

        
        #### NOTE: this is currrently hardcoded
        transformed_point.translation[2] = z
        #######

        # rotate by 90 so that gripper edge is perpendicular
        #rot = R.from_euler("xyz",[0,0,np.pi/2]).as_matrix()@R.from_euler("xyz", [rx,ry,rz]).as_matrix()
        robot.rotate_pose90(transformed_point)
        #pose = RigidTransform(rotation=rot, translation=transformed_point)
        # pose = [x,y,z,rx,ry,rz]
        poses.append(transformed_point)

    for pose in poses:
        last_record = time.time()
        robot.move_pose(pose)
        while time.time()-last_record < 0.002:
            pass 

def detect_cable(rgb_img, args):
    # Detecting the cable
    cable_cnt, cable_mask_hollow  = get_cable(img = rgb_img, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize)
    
    # try:
    #     cable_cnt, cable_mask_binary = get_cable(rgb_img)
    # except:
    #     rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    #     cable_cnt, cable_mask_binary = get_cable(rgb_img)

        
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv2.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
    # plt.imshow(rgb_img)
    # plt.imshow(cable_mask_binary, alpha=0.5)
    # plt.show()

    cable_mask_binary = (cable_mask_binary.sum(axis=2)/255).astype('uint8')
     # want to dilate then erode the mask
    # beste_mask = np.best_mask*255
    # plt.imshow(cable_mask_binary)
    # plt.title("before closing")
    # plt.show()
    cable_mask_binary = cv2.morphologyEx(cable_mask_binary,cv2.MORPH_CLOSE,np.ones((5,5), np.uint8))

    # plt.imshow(cable_mask_binary)
    # plt.title("after closing")
    # plt.show()

    cable_skeleton = skeletonize(cable_mask_binary)
    # hough_output = probabilistic_hough_line(cable_skeleton, line_length=10)
    # x = [k[0] for j in hough_output for k in j]
    # y = [k[1] for j in hough_output for k in j]
    # temp = np.zeros_like(cable_skeleton)
    # temp[y,x] = 1
    # cable_skeleton = temp
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    assert len(cable_endpoints) == 2
    return cable_skeleton, cable_length, cable_endpoints, cable_mask_binary

def detect_channel(rgb_img, cable_mask_binary, args):
    # Detecting the channel
    # matched_template, matched_results, channel_cnt, _ = get_channel(img = rgb_img, cable_mask=cable_mask_binary, blur_radius=args.blur_radius, sigma=args.sigma, 
    #                                                  dilate_size=args.dilate_size_channel, canny_threshold=args.canny_threshold_channel, viz=args.visualize)
    matched_template, matched_results, channel_cnt = get_channel(rgb_img)

    if matched_template == 'curved':
        template_mask = curved_template_mask_align
    elif matched_template == 'straight':
        # NOTE: Modified to use large mask for contour fitting
        template_mask = straight_template_mask_align
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask

    # print('================')
    # print('mask stuff')
    # print('================')

    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    # making it 3 channels <--- messes with skeletonization ig?
    # aligned_channel_mask = cv2.merge((aligned_channel_mask, aligned_channel_mask, aligned_channel_mask))
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    # plt.imshow(rgb_img)
    # plt.imshow(aligned_channel_mask, alpha=0.5)
    # plt.show()

    # plt.imshow(cv2.drawContours(rgb_img.copy(), [channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])], -1, 255, 3))
    # plt.title('check channel contour dimensions in detect_channel')
    # plt.show()

    
    # plt.imshow(rgb_img)
    # plt.imshow(aligned_channel_mask, alpha=0.7)
    # plt.show()
    
    # do this so that we have a more accurate reading of where the channel actually is 
    channel_cnt_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    _ = cv2.drawContours(channel_cnt_mask, [channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])],-1, [255,255,255], -1)
    # cv2.drawContours(image=mask,contours=channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]]),contourIdx=-1,color=(0,255,255),thickness=cv2.FILLED)

    # print(aligned_channel_mask) #This was Simeon on Taco Tuesday.

    # skeletonizing the channel
    channel_skeleton = skeletonize(aligned_channel_mask)

    # hough_output = probabilistic_hough_line(channel_skeleton, line_length=10)
    # x = [k[0] for j in hough_output for k in j]
    # y = [k[1] for j in hough_output for k in j]
    # temp = np.zeros_like(channel_skeleton)
    # temp[y,x] = 1
    # channel_skeleton = temp
    # plt.imshow(rgb_img)
    # plt.imshow(channel_skeleton, alpha=0.5, cmap='jet')
    # plt.title("channel skeleton overlaid")
    # plt.show()
    #getting the length and endpoints of the channel
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    if len(channel_endpoints) == 1:
        print("We have a loop!")
    return channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt

def find_closest_point(point_list, point_b):
    # Convert the input lists to NumPy arrays for easier computation
    points = np.array(point_list)
    b = np.array(point_b)
    
    # Calculate the Euclidean distance between point B and each point in the list
    distances = np.linalg.norm(points - b, axis=1)
    
    # Find the index of the point with the smallest distance
    closest_index = np.argmin(distances)
    
    # Retrieve the closest point from the original list
    closest_point = point_list[closest_index]
    
    return closest_point

def get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, is_trapezoid = False, pick_closest_endpoint = False, depth_img = None):
    ### 3/1 changed around a lot of stuff for trapezoid like trying to pick the closest cable endpoint relative to our channel endpoint
    if is_trapezoid:
        # in this case this just the first corner of the trapezoid!
        channel_endpoint_in = channel_endpoints
    else:
        # TODO: try and fix the logic for this honestly
        # just pick an endpoint to be the one that we'll use as our in point
        cable_endpoint_in = cable_endpoints[0]
        channel_endpoint_in, channel_endpoint_out = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
    
    if pick_closest_endpoint:
        # breakpoint()
        # change var name just in case this gets ran a lot
        cable_endpoints1 = np.array(cable_endpoints)
        channel_endpoint_in1 = np.array(channel_endpoint_in)
        # finds the cable endpoint that is closest to the given channel endpoint
        dist_2 = np.sum((cable_endpoints1 - channel_endpoint_in1)**2, axis=1)
        cable_endpoint_in = cable_endpoints1[np.argmin(dist_2)]
        # needs to be a tuple for the sorting code to work
        cable_endpoint_in = tuple(cable_endpoint_in)
    else:
        # just pick an endpoint to be the one that we'll use as our in point
        cable_endpoint_in = cable_endpoints[0]
    channel_endpoint_in = tuple(channel_endpoint_in)
    sorted_channel_pts = sort_skeleton_pts(channel_skeleton, channel_endpoint_in, is_trapezoid)
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

    # trapezoid skeleton is smooth, don't want to delete parts of it. 
    if not is_trapezoid:
        sorted_channel_pts = sorted_channel_pts[START_IDX:END_IDX]
    sorted_cable_pts = sorted_cable_pts[START_IDX:END_IDX]
    
    # filter out points that are invalid/unreasonable in the depth image
    if depth_img is not None:
        pass
        # # WARNING: seems like the depth are expecting the x and y to be flipped!!!!
        # swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
        # swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        # filtered_cable_pts = filter_points(swapped_sorted_channel_pts, depth_img)
        # filtered_channel_pts = filter_points(swapped_sorted_channel_pts, depth_img)
        # # NOTICE THAT THTESE ARE NOW SWAPPED!!!

    # these are aslo now swapped!
    channel_endpoint_in = sorted_channel_pts[0]
    channel_endpoint_out = sorted_channel_pts[-1]

    cable_endpoint_in = sorted_cable_pts[0]
    cable_endpoint_out = sorted_cable_pts[-1]

    # TODO CHECK IF THIS IS WORKING PROPERLY!!!!
    # IDEA: update the channel startpoint to be the closest point on the channel skeleton to the cable endpoint
    # then actually delete the indices before that point from sorted_channel_pts
    
    # channel_endpoint_in = find_closest_point(sorted_channel_pts, cable_endpoint_in)
    # sorted_channel_pts = sorted_channel_pts[sorted_channel_pts.index(channel_endpoint_in):]

    return sorted_cable_pts, sorted_channel_pts

def get_sorted_cable_pts(cable_endpoints, cable_skeleton):
    cable_endpoint_in = cable_endpoints[0]
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

    # trapezoid skeleton is smooth, don't want to delete parts of it. 
    sorted_cable_pts = sorted_cable_pts[START_IDX:END_IDX]

    return sorted_cable_pts

def find_nth_nearest_point(point, sorted_points, given_n):
    # print(endpoint)
    # print(np.array(sorted_points).shape, np.array(endpoint).shape)
    # np.linalg.norm(np.array(sorted_points) - np.array(endpoint))
    # if point == sorted_points[0]:
    #     return sorted_points[n - 1]
    # elif point == sorted_points[-1]:
    #     return sorted_points[-n]
    idx = sorted_points.index(point)

    behind_idx = np.clip(idx - given_n, 0, len(sorted_points)-1)
    infront_idx = np.clip(idx + given_n, 0, len(sorted_points)-1)
    return behind_idx, infront_idx
    # if np.linalg.norm(np.array(point)-np.array(sorted_points[0])) < np.linalg.norm(np.array(point)-np.array(sorted_points[-1])):
    #     behind_idx = np.clip(idx - n, 0, len(sorted_points)-1)
    #     infront_idx = np.clip(idx + n, 0, len(sorted_points)-1)
    #     return behind_idx, infront_idx
    # else:
    #     behind_idx = np.clip(idx - n, 0, len(sorted_points)-1)
    #     infront_idx = np.clip(idx + n, 0, len(sorted_points)-1)
    #     return behind_idx, infront_idx

def get_rotation(point1, point2):
    direction = np.array(point2) - np.array(point1)
    direction = direction / np.linalg.norm(direction) #unit vector

    reference_direction = np.array([1,0])
    cos_theta = np.dot(direction, reference_direction)
    sin_theta = np.sqrt(1 - cos_theta**2)
    dz = np.arctan2(sin_theta, cos_theta)
    dz1 = np.arctan2(direction[1], direction[0])
    euler = np.array([-np.pi, 0, dz1])
    rot_matrix = R.from_euler("xyz", euler).as_matrix()
    return rot_matrix



def get_rw_pose(orig_pt, sorted_pixels, n, ratio, camCal, is_channel_pt, use_depth = False):
    # next_point = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    behind_idx, infront_idx = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    behind_pt = sorted_pixels[behind_idx]
    infront_pt = sorted_pixels[infront_idx]
    # plt.scatter(x=behind_pt[0], y=behind_pt[1], c='m')
    # plt.scatter(x=infront_pt[0], y=infront_pt[1], c='y')
    # plt.imshow(rgb_img)
    # plt.show()

    # offset_point = find_nth_nearest_point(orig_pt, sorted_pixels, int(len(sorted_pixels)*ratio))
    
    # needs to be done since the point was relative to the entire view of the camera but our model is trained on points defined only in the cropped frame of the image
    orig_pt = np.array(orig_pt) #- np.array([CROP_REGION[2], CROP_REGION[0]])
    # next_pt = np.array(next_point) #- np.array([CROP_REGION[2], CROP_REGION[0]])
    orig_rw_xy = camCal.image_pt_to_rw_pt(orig_pt) 
    # next_rw_xy = camCal.image_pt_to_rw_pt(next_pt)
    behind_rw_xy =    camCal.image_pt_to_rw_pt(behind_pt)
    infront_rw_xy = camCal.image_pt_to_rw_pt(infront_pt)
    # offset_rw_xy = camCal.image_pt_to_rw_pt(offset_point)
    # rot = get_rotation(orig_rw_xy, next_rw_xy)
    rot = get_rotation(behind_rw_xy, infront_rw_xy)
    orig_rw_xy = orig_rw_xy / 1000

    # want this z height to have the gripper when closed be just barely above the table
    # will need to tune!
    if not use_depth:
        hardcoded_z = -18  # was -15
        # if we want a point on the channel need to account for the height of the template
        if is_channel_pt:
            hardcoded_z += TEMPLATE_HEIGHT[matched_template] * 1000 #  template z is in cm for some reason
        orig_rw = np.array([orig_rw_xy[0], orig_rw_xy[1],hardcoded_z/1000])
    else:
        print("haven't implemented using depth for pick/place!")
        raise ValueError()
    # converting pose to rigid transform to use with ur5py library
    orig_rt_pose = RigidTransform(rotation=rot, translation=orig_rw)
    return orig_rt_pose

# gets rigid transform given pose

def get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

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
    # hough_output = probabilistic_hough_line(cable_skeleton, line_length=10)
    # x = [k[0] for j in hough_output for k in j]
    # y = [k[1] for j in hough_output for k in j]
    # temp = np.zeros_like(cable_skeleton)
    # temp[y,x] = 1
    # cable_skeleton = temp
    # plt.imshow(cable_skeleton)
    # plt.show()
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    # plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    # plt.imshow(rgb_img)
    # plt.show()

    # plt.imshow(rgb_img)
    # plt.imshow(channel_skeleton, alpha=0.5)
    # plt.imshow(cable_skeleton, alpha=0.5)
    # plt.show()

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

    
    pick_pt = sorted_cable_pts[0] 
    place_pt = sorted_channel_pts[0]
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.title("pick place pts")
    # plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
    # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose_swap, place_pose_swap)
    return swapped_sorted_cable_pts, swapped_sorted_channel_pts, place_pose_swap
    # robot.move_pose(pick_pose)
    
    # place_pose = get_rw_pose(place_pt, sorted_channel_pts, 20, 0.1, camCal)
    # # only do fthis when no ends attached cause we dont care about dragging the rope
    # # z offset for height of channel 
    # place_pose[2] += TEMPLATE_HEIGHT[matched_template]
    # robot.move_pose(place_pose)
    # robot.close_grippers()

    # robot.move_to_channel_overhead(transformed_channel_end, rotation)
    # robot.push(transformed_channel_end, rotation)
    # robot.go_home()
    # robot.open_grippers()

def place_halfway(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio):
    pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio)

def one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

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
    # pick_pt = sorted_cable_pts[-1] 
    # place_pt = sorted_channel_pts[-1]
    # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    # plt.imshow(rgb_img)
    # plt.show()

    # --------- EDIT THIS LATER TO INCLUDE SUPPORT FOR ALL TEMPLATES --------
    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 20, 0.1, camCal, False)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)
    robot.pick_and_place(pick_pose_swap, place_pose_swap)

    # this is the pose of the goal position
    return place_pose_swap

def get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, camCal):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

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
    # --------- EDIT THIS LATER TO INCLUDE SUPPORT FOR ALL TEMPLATES --------
    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    place_pose_swap1 = get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)
    place_pose_swap2 = get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, True)

    # this is the pose of the goal position
    return place_pose_swap1, place_pose_swap2


def attach_intermediary_parts(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, iter, second_endpt_side=None, second=False):
    if second:
        cable_skeleton = skeletonize(cable_mask_binary)
        # hough_output = probabilistic_hough_line(cable_skeleton, line_length=10)
        # x = [k[0] for j in hough_output for k in j]
        # y = [k[1] for j in hough_output for k in j]
        # temp = np.zeros_like(cable_skeleton)
        # temp[y,x] = 1
        # cable_skeleton = temp
        # plt.imshow(cable_skeleton)
        # plt.show()
        cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
        # plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        # plt.imshow(rgb_img)
        # plt.show()

        
        # plt.imshow(rgb_img)
        # plt.imshow(cable_skeleton, alpha=0.7)
        # plt.show()

        # we sort the points on channel and cable to get a relation between the points
        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

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
        
        # pick_pt = sorted_cable_pts[-1] 
        # place_pt = sorted_channel_pts[-1]
        # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        # plt.imshow(rgb_img)
        # plt.show()

        # --------- EDIT THIS LATER TO INCLUDE SUPPORT FOR ALL TEMPLATES --------
        # needs to be swapped as this is how it is expected for the robot
        swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 10, 0.1, camCal, False)
        place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 10, 0.1, camCal, False)
        robot.pick_and_place(pick_pose_swap, place_pose_swap)
        slide_goal_pt = sorted_channel_pts[0]
        slide_goal_pose = get_rw_pose((slide_goal_pt[1], slide_goal_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, False)
    else:
        cable_skeleton = skeletonize(cable_mask_binary)
        # hough_output = probabilistic_hough_line(cable_skeleton, line_length=10)
        # x = [k[0] for j in hough_output for k in j]
        # y = [k[1] for j in hough_output for k in j]
        # temp = np.zeros_like(cable_skeleton)
        # temp[y,x] = 1
        # cable_skeleton = temp
        # plt.imshow(cable_skeleton)
        # plt.show()
        cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
        # plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        # plt.imshow(rgb_img)
        # plt.show()

        # plt.imshow(rgb_img)
        # plt.imshow(channel_skeleton, alpha=0.5)
        # plt.imshow(cable_skeleton, alpha=0.5)
        # plt.show()

        # we sort the points on channel and cable to get a relation between the points
        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

        
        pick_pt = sorted_cable_pts[0] 
        place_pt = sorted_channel_pts[0]
        # plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        # plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        # plt.imshow(rgb_img)
        # plt.show()

        # needs to be swapped as this is how it is expected for the robot
        swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
        # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
        place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

        robot.pick_and_place(pick_pose_swap, place_pose_swap)
    
    # should actually slide across the straight channel (hopefully)
    # robot.push(slide_goal_pose, is_place_pt=True, force_ctrl=True)

def push_idx(sorted_channel_pts, idx):
    if idx >= len(sorted_channel_pts):
        idx = len(sorted_channel_pts) - 1
    push_pt = sorted_channel_pts[idx]
    
    # plt.scatter(x=push_pt[1], y=push_pt[0], c='r')
    # plt.title("Push Points")
    # plt.imshow(rgb_img)
    # plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]

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
    # plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    # plt.imshow(rgb_img)
    # plt.show()

    # plt.imshow(rgb_img)
    # plt.imshow(cable_skeleton, alpha=0.7)
    # plt.show()

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, is_trapezoid, pick_closest_endpoint)
    
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
    # thinking of doing a just sort by which endpoint is closest to the sorted_channel (I feel like this may already be implemented)
    # else:
    #     pass
    cable_idx = math.floor(len(sorted_cable_pts)*ratio)
    channel_idx = math.floor(len(sorted_channel_pts)*ratio)


    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]
    if is_trapezoid:
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

    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=on_channel)
    # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose_swap, place_pose_swap, on_channel)

    return place_pose_swap

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
        # if np.abs((-1-pair[0]) - (-1-pair[1])) < np.abs()
        # if np.abs(len(sorted_channel_pts)-pair[1]) < np.abs(pair[1]-pair[0]):
        place_pts = np.linspace(pair[0], len(sorted_channel_pts)-1-pair[1], num_points).astype(int)
        # else:

    breakpoint()
    # curr_frac = np.abs(place_pts[0]-place_pts[1])/len(sorted_channel_pts)
    # curr_frac = np.abs(place_pts[0]-place_pts[idx])/len(sorted_channel_pts)
    prev_idx = max(0, idx-1)
    curr_frac = np.abs(place_pts[0] - place_pts[idx])/len(sorted_channel_pts)


    # curr_cable_start = int(prev_frac * len(sorted_cable_pts))
    # curr_cable_end = int(prev_frac + curr_frac/num_points * len(sorted_cable_pts))
    # pick_pts = np.linspace(curr_cable_start, curr_cable_end, num_points).astype(int)
    # cable_idx = int(prev_frac + curr_frac) * len(sorted_cable_pts)
    cable_idx = int((prev_frac + curr_frac) * len(sorted_cable_pts))
    channel_idx = place_pts[idx]
    # cable_idx = pick_pts[idx]

    print("my channel idx is", channel_idx)
    print("my cable idx is", cable_idx)
    # breakpoint()


    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]

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

    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=on_channel)
    # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    # robot.pick_and_place(pick_pose_swap, place_pose_swap, on_channel)
    # return prev_frac + curr_frac
    if idx == num_points - 1:
        return prev_frac + curr_frac #np.abs(place_pts[0]-place_pts[1])/len(sorted_channel_pts)#+ curr_frac
    else:
        return prev_frac

# NOTE: this function is busted, idk why
def click_and_move_pts_on_img(rgb_img):

    rope_list = []
    channel_list = []
    # Function to handle mouse click events
    def on_mouse_click(event):
        if event.button == 1:  # Left-click
            rope_list.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot where the click occurred
        elif event.button == 3:  # Right-click
            channel_list.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'bo')  # Plot a blue dot where the click occurred

        plt.draw()

    plt.imshow(rgb_img)
    plt.axis('image')  # Set aspect ratio to be equal
    # Connect the click event to the figure
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Show the plot and wait for user input
    plt.show()

    for pt in rope_list:
        # pretty sure we need it swapped
        rw_pt = camCal.image_pt_to_rw_pt((pt[1], pt[0]))
        rot = R.from_euler("xyz", [0,np.pi,0]).as_matrix()
        trans = [rw_pt[0], rw_pt[1], -15/1000]
        rw_pose = RigidTransform(rotation=rot, translation=trans)
        robot.move_pose(rw_pose)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def search(ratio, min_length):
    """
    given 0 < ratio < 1, output list of numbers between 0 and 1 as pick points
    min_length is when we should stop
    """
    points = []

    def helper_search(left, right):
        length = right - left
        if length <= min_length:
            pass
        else:
            points.append(left + ratio * length)
            left_points = helper_search(left, left + ratio * length)
            right_points = helper_search(left + ratio * length, right)

    helper_search(0, 1)
    return points

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
                trans = [rw_pt[0]/1000, rw_pt[1]/1000, -15/1000]
                rw_pose = RigidTransform(rotation=rot, translation=trans)
                # breakpoint()
                robot.move_pose(rw_pose)



TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
# curved was 2.54cm high, trapezoid was 2.54cm high
TEMPLATE_HEIGHT = {'curved':0.0200, 'straight':0.0127, 'trapezoid':0.0200}

TOTAL_PICK_PLACE = 5


if __name__=='__main__':
    # # loads model for camera to real world
    # camCal = ImageRobot()
    # # # Sets up the realsense and gets us an image
    # pipeline, colorizer, align, depth_scale = setup_rs_camera()
    # time.sleep(1)
    # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    # plt.imsave('test_images/trapezoid_channel_1.png', rgb_img)
    args = argparser.parse_args()
    PUSH_MODE = args.push_mode
    PICK_MODE = args.pick_mode
    EXP_MODE = args.exp
    PUSH_BEFORE_SLIDE = args.push_before_slide
    N = args.exp_num
    robot = GasketRobot()
    robot.set_playload(1)
    robot.end_force_mode()
    # robot.force_mode(robot.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.1)
    robot.go_home()
    
    
    # # Sets up the realsense and gets us an image
    # pipeline, colorizer, align, depth_scale = setup_rs_camera()
    overhead_cam_id = 22008760 # overhead camera
    # side_cam_id = 20120598 # side camera
    # front_eval_cam_id = 20812520 # front eval camera
    side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id)
    # front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth = setup_zed_camera(front_eval_cam_id)
    time.sleep(1)
    # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)

    # Sets up the zed camera and gets us an image
    # camera = ZedCamera(20120598)
    # overhead_cam_id = 22008760 # overhead camera
    # side_cam_id = 20120598 # side camera
    # side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id) #should be overhead camera id
    # rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    
    # plt.imshow(rgb_img)
    # plt.show()
    cropped_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    # plt.imshow(cropped_img)
    # plt.show()
    # plt.imsave('curved_weird_cropped.png', cropped_img)

    # loads model for camera to real world
    camCal = ImageRobot()

    #### Calibrration Testing #########
    # rgb_img = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    # click_and_move_pts_on_img(rgb_img)
    # PointSelector(rgb_img)
    # plt.imshow(rgb_img)
    # plt.show()
    # plt.imsave('curved_weird.png', rgb_img)

    # approximately how many points along the channel we want to move and push to
    NUM_PTS = args.num_points
    NUM_PTS_PUSH = args.num_points

    def get_binary_search_idx():
        sorted_search_idx = []
        for i in range(int(np.log2(NUM_PTS))):
            power = 2**(i+1)
            val = 1/power
            while val <= 1/2:
                if val not in sorted_search_idx:
                    sorted_search_idx.append(val)
                if 1-val not in sorted_search_idx:
                    sorted_search_idx.append(1-val)
                val += 1/power
        
        return sorted_search_idx

    def get_golden_search_idx():
        def get_x1(end, start, golden_ratio):
            return end - (end-start)*golden_ratio
        def get_x2(end, start, golden_ratio):
            return start + (end-start)*golden_ratio  
        golden_ratio = 1 / ((math.sqrt(5) + 1) / 2)
        start, end = 0, 1
        sorted_search_idx = []
        for i in range(NUM_PTS//2):
            x1, x2 = get_x1(end, start, golden_ratio), get_x2(end, start, golden_ratio)
            sorted_search_idx.append(x1)
            sorted_search_idx.append(x2)
            start = min(x1, x2)
            ## Assume f is the identity
            # if f(x1) > f(x2):
            #     end = max(x1, x2)
            # else:
            #     start = min(x1, x2)
        return sorted_search_idx

    if PICK_MODE == "binary" or PICK_MODE == 'hybrid':
        sorted_search_idx = get_binary_search_idx()
    elif PICK_MODE == "golden":
        sorted_search_idx = get_golden_search_idx()
    elif PICK_MODE == "uni":
        sorted_search_idx = [(1/NUM_PTS)*i for i in range(NUM_PTS)]


    # getting all ofNUM_BINARY the ratios for a necessary binary search


    ### PICK AND PLACE ########
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
    
    channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt = detect_channel(rgb_img, cable_mask_binary, args)
    if matched_template != "trapezoid":
        slide_start_pose, slide_end_pose = get_slide_start_end(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
        slide_mid_pose = None
        swapped_sorted_channel_pts = get_sorted_channel_pts(cable_mask_binary, cable_endpoints, channel_endpoints)
        for i in range(len(sorted_search_idx)):
            robot.go_home()
            # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            # do i-1 cause i == 1 when we get into here
            ratio = sorted_search_idx[i]
            if i == 0:
                slide_mid_pose = pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio, channel_cnt_mask)
            else:
                pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, ratio, channel_cnt_mask)
            robot.gripper.open()

            # NOTE: @Tara did you intend to call this PUSH_MODE instead of PICK_MODE? maybe too sleepy to see the reasoning -Karim
            # NOTE: (Will) I am switching this temporarily to PICK_MODE because it seems to align more with documentation
            if PICK_MODE == 'hybrid' and i == 0:
                robot.go_home()
                rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

                # performs the experiment given no ends are attached to the channel
                swapped_sorted_cable_pts, swapped_sorted_channel_pts, slide_start_pose = no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
                    
                existing_pick_pt = swapped_sorted_cable_pts[0]
                second_endpt_side = 'right' if existing_pick_pt[0] < 555 else 'left'
                
                ### NOTE: TAKE NEW PHOTO SINCE CABLE END MAY MOVE AFTER FIRST POINT INSERTION
                robot.go_home()
                ### END NOTE
                
                # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
                rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                slide_end_pose = one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side)
    elif matched_template == "trapezoid":
        corners = get_corners2(rgb_img, channel_cnt)
        channel_skeleton_corners = match_corners_to_skeleton(corners, channel_skeleton)
        long_corner0, long_corner1, med_corner0, med_corner1 = classify_corners(channel_skeleton_corners)
        # plt.title("matched skeleton pts")
        # plt.imshow(channel_skeleton)
        # plt.scatter(channel_skeleton_corners[0][1], channel_skeleton_corners[0][0], c='r')
        # plt.scatter(channel_skeleton_corners[1][1], channel_skeleton_corners[1][0], c='b')
        # plt.scatter(channel_skeleton_corners[2][1], channel_skeleton_corners[2][0], c='g')
        # plt.scatter(channel_skeleton_corners[3][1], channel_skeleton_corners[3][0], c='k')
        # plt.scatter(corners[0][1], corners[0][0], c='r')
        # plt.scatter(corners[1][1], corners[1][0], c='b')
        # plt.scatter(corners[2][1], corners[2][0], c='g')
        # plt.scatter(corners[3][1], corners[3][0], c='k')
        # plt.show()
   
        if PICK_MODE == 'uni':
            

            ## TARA: WRITE IN HERE
            # channel_start_pt = get_midpt_corners(channel_skeleton, long_corner0, long_corner1)
            channel_start_pt = long_corner0

            # plt.title("correct long and med corner pts")
            # plt.imshow(channel_skeleton)
            # plt.scatter(long_corner0[1], long_corner0[0], c='m')
            # plt.scatter(long_corner1[1], long_corner1[0], c='y')
            # plt.scatter(med_corner0[1], med_corner0[0], c='c')
            # plt.scatter(med_corner1[1], med_corner1[0], c='k')
            # plt.scatter(channel_start_pt[1], channel_start_pt[0], c='r')
            # plt.show()
            # sorted_search_idx = [(1/NUM_PTS)*i for i in range(NUM_PTS)]

            

            rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

            # NOTE: @KARIM if stuff is acting weird here try setting `pick_closest_point` to True
            sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_start_pt, cable_skeleton, is_trapezoid=True)
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
            # we do this so we can just do a quick channel_skeleton_corner in sorted_channel_pts
            sorted_channel_pts = np.array(sorted_channel_pts).tolist()
            channel_skeleton_corners = np.array(channel_skeleton_corners).tolist()
            # corner_idxs = np.array([i for i, point in enumerate(sorted_channel_pts) if point in channel_skeleton_corners])
            long_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner0[0] and x[1] == long_corner0[1]][0]
            long_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner1[0] and x[1] == long_corner1[1]][0]
            med_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner0[0] and x[1] == med_corner0[1]][0]
            med_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner1[0] and x[1] == med_corner1[1]][0]

            pairs = [[long_corner0_idx, long_corner1_idx], [long_corner1_idx, med_corner1_idx], [med_corner1_idx, med_corner0_idx], [med_corner0_idx, long_corner0_idx]]
            prev_frac = 0
            for pair_idx, pair in enumerate(pairs):
                # for n in range(args.num_points):
                # pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac, idx, num_points, channel_mask=None)
                for idx in range(NUM_PTS):
                    if pair_idx > 0 and idx == 0:
                        continue
                    rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                    prev_frac = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, camCal, pair, prev_frac, idx, NUM_PTS, channel_cnt_mask)
                    # robot.go_home()

            

        # if pick mode is binary
        # SORTING: 
        # find the midpoint of the largest line segment in the channel
        # sort from that point
        # end of that sorted list will essentially be the same midpoint
        # PICKING:
        # pick the middle of the cable and put it into that midpoint
        # 
        if PICK_MODE == 'binary':
            pass
        if PICK_MODE == 'hybrid':
            pass
        # TODO: pick the corner that is furthest from all of the other ones to be our start one
        channel_start_pt = (channel_skeleton_corners[0][0], channel_skeleton_corners[0][1])


        # 3: sort the channel from that point
        rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_start_pt, cable_skeleton, is_trapezoid=True)

        # 4: find the ratio between the midpoint of the first line segment and the total length of the channel
        # gets the indices of where each corner is in the sorted channel skeleton
        
        # we do this so we can just do a quick channel_skeleton_corner in sorted_channel_pts
        sorted_channel_pts = np.array(sorted_channel_pts).tolist()
        channel_skeleton_corners = np.array(channel_skeleton_corners).tolist()
        corner_idxs = np.array([i for i, point in enumerate(sorted_channel_pts) if point in channel_skeleton_corners])
        
        # want to check that the first pt of sorted_channel_pts is the first corner
        channel_start_pt = list(channel_start_pt)
        assert sorted_channel_pts[0] == channel_start_pt and sorted_channel_pts[corner_idxs[0]] == channel_start_pt

        midpt_idxs = np.array([(corner_idxs[0] + corner_idxs[1])//2,  (corner_idxs[1] + corner_idxs[2])//2,
                      (corner_idxs[2] + corner_idxs[3])//2, (corner_idxs[3] + len(sorted_channel_pts))//2])
        
        midpt_ratios = midpt_idxs/len(sorted_channel_pts)
        corner_ratios = corner_idxs/len(sorted_channel_pts)
        
        def pick_place_trap_side(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, pick_place_ratios, channel_cnt_mask):
            for pp_ratio in pick_place_ratios:
                # need to retake the images
                rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                plt.title('cable skeleton trap side')
                plt.imshow(rgb_img)
                plt.imshow(cable_skeleton, alpha=0.7)
                plt.show()
                pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, pp_ratio, channel_cnt_mask, is_trapezoid=True, pick_closest_endpoint=True)
                robot.go_home()
                # breakpoint()
        
        # order of pick and place: 
        # midpt0, corner0, corner1, pushdown
        side1_ratios = [midpt_ratios[0], corner_ratios[0], corner_ratios[1]]


        print("here are the ratios to be picked up for side 1,", side1_ratios)
        cable_skeleton = skeletonize(cable_mask_binary)

        # breakpoint()
        first_corner_idx = math.floor(len(sorted_channel_pts)*corner_ratios[0])
        second_corner_idx = math.floor(len(sorted_channel_pts)*corner_ratios[1])
        third_corner_idx = math.floor(len(sorted_channel_pts)*corner_ratios[2])
        fourth_corner_idx = math.floor(len(sorted_channel_pts)*corner_ratios[3])
        first_corner_pt = sorted_channel_pts[first_corner_idx] 
        second_corner_pt = sorted_channel_pts[second_corner_idx]
        third_corner_pt = sorted_channel_pts[third_corner_idx]
        fourth_corner_pt = sorted_channel_pts[fourth_corner_idx]
        plt.title("sorted corner pts")
        plt.imshow(rgb_img)
        plt.scatter(first_corner_pt[1], first_corner_pt[0], c='m')
        plt.scatter(second_corner_pt[1], second_corner_pt[0], c='y')
        plt.scatter(third_corner_pt[1], third_corner_pt[0], c='c')
        plt.scatter(fourth_corner_pt[1], fourth_corner_pt[0], c='k')
        plt.show()

        side1_ratios = [midpt_ratios[0], corner_ratios[0], corner_ratios[1]]
        first_idx = math.floor(len(sorted_channel_pts)*side1_ratios[0])
        second_idx = math.floor(len(sorted_channel_pts)*side1_ratios[1])
        third_idx = math.floor(len(sorted_channel_pts)*side1_ratios[2])

        first_pt = sorted_channel_pts[first_idx] 
        second_pt = sorted_channel_pts[second_idx]
        third_pt = sorted_channel_pts[third_idx]
        plt.title("side1 ratio pts")
        plt.scatter(x=first_pt[1], y=first_pt[0], c='r')
        plt.scatter(x=second_pt[1], y=second_pt[0], c='b')
        plt.scatter(x=third_pt[1], y=third_pt[0], c='g')
        plt.scatter(first_corner_pt[1], first_corner_pt[0], c='m')
        plt.scatter(second_corner_pt[1], second_corner_pt[0], c='y')
        plt.scatter(third_corner_pt[1], third_corner_pt[0], c='c')
        plt.scatter(fourth_corner_pt[1], fourth_corner_pt[0], c='k')
        plt.imshow(rgb_img)
        plt.show()

        side2_ratios = [midpt_ratios[1], corner_ratios[1], corner_ratios[2]]
        first_idx = math.floor(len(sorted_channel_pts)*side2_ratios[0])
        second_idx = math.floor(len(sorted_channel_pts)*side2_ratios[1])
        third_idx = math.floor(len(sorted_channel_pts)*side2_ratios[2])

        first_pt = sorted_channel_pts[first_idx] 
        second_pt = sorted_channel_pts[second_idx]
        third_pt = sorted_channel_pts[third_idx]
        plt.title("side2 ratio pts")
        plt.scatter(x=first_pt[1], y=first_pt[0], c='r')
        plt.scatter(x=second_pt[1], y=second_pt[0], c='b')
        plt.scatter(x=third_pt[1], y=third_pt[0], c='g')
        plt.scatter(first_corner_pt[1], first_corner_pt[0], c='m')
        plt.scatter(second_corner_pt[1], second_corner_pt[0], c='y')
        plt.scatter(third_corner_pt[1], third_corner_pt[0], c='c')
        plt.scatter(fourth_corner_pt[1], fourth_corner_pt[0], c='k')
        plt.imshow(rgb_img)
        plt.show()

        side3_ratios = [midpt_ratios[2], corner_ratios[2], corner_ratios[3]]
        first_idx = math.floor(len(sorted_channel_pts)*side3_ratios[0])
        second_idx = math.floor(len(sorted_channel_pts)*side3_ratios[1])
        third_idx = math.floor(len(sorted_channel_pts)*side3_ratios[2])

        first_pt = sorted_channel_pts[first_idx] 
        second_pt = sorted_channel_pts[second_idx]
        third_pt = sorted_channel_pts[third_idx]
        plt.title("side 4ratio pts")
        plt.scatter(x=first_pt[1], y=first_pt[0], c='r')
        plt.scatter(x=second_pt[1], y=second_pt[0], c='b')
        plt.scatter(x=third_pt[1], y=third_pt[0], c='g')
        plt.scatter(first_corner_pt[1], first_corner_pt[0], c='m')
        plt.scatter(second_corner_pt[1], second_corner_pt[0], c='y')
        plt.scatter(third_corner_pt[1], third_corner_pt[0], c='c')
        plt.scatter(fourth_corner_pt[1], fourth_corner_pt[0], c='k')
        plt.imshow(rgb_img)
        plt.show()

        side4_ratios = [midpt_ratios[3], corner_ratios[3], 0.99]
        first_idx = math.floor(len(sorted_channel_pts)*side4_ratios[0])
        second_idx = math.floor(len(sorted_channel_pts)*side4_ratios[1])
        third_idx = math.floor(len(sorted_channel_pts)*side4_ratios[2])

        first_pt = sorted_channel_pts[first_idx] 
        second_pt = sorted_channel_pts[second_idx]
        third_pt = sorted_channel_pts[third_idx]
        plt.title("side1 ratio pts")
        plt.scatter(x=first_pt[1], y=first_pt[0], c='r')
        plt.scatter(x=second_pt[1], y=second_pt[0], c='b')
        plt.scatter(x=third_pt[1], y=third_pt[0], c='g')
        plt.scatter(first_corner_pt[1], first_corner_pt[0], c='m')
        plt.scatter(second_corner_pt[1], second_corner_pt[0], c='y')
        plt.scatter(third_corner_pt[1], third_corner_pt[0], c='c')
        plt.scatter(fourth_corner_pt[1], fourth_corner_pt[0], c='k')
        plt.imshow(rgb_img)
        plt.show()

        # breakpoint()
        
        pick_place_trap_side(cable_mask_binary, cable_endpoints, channel_start_pt, camCal, side1_ratios, channel_cnt_mask)
        if PUSH_MODE == "uni" or PUSH_MODE == "hybrid":
            sorted_push_idx = sample_pts_btwn(corner_idxs[0], corner_idxs[1], 3)
        elif PUSH_MODE == "binary":
            sorted_push_idx = sample_pts_btwn(corner_idxs[0], corner_idxs[1], 3)
            sorted_push_idx = [sorted_push_idx[1], sorted_push_idx[0], sorted_push_idx[2]]
        push_down(sorted_push_idx)
        robot.go_home()

        # midpt1, corner1, corner2, pushdown
        side2_ratios = [midpt_ratios[1], corner_ratios[1], corner_ratios[2]]
        print("here are the ratios to be picked up for side 2,", side2_ratios)
        pick_place_trap_side(cable_mask_binary, cable_endpoints, channel_start_pt, camCal, side2_ratios, channel_cnt_mask)
        if PUSH_MODE == "uni" or PUSH_MODE == "hybrid":
            sorted_push_idx = sample_pts_btwn(corner_idxs[1], corner_idxs[2], 3)
        elif PUSH_MODE == "binary":
            sorted_push_idx = sample_pts_btwn(corner_idxs[1], corner_idxs[2], 3)
            sorted_push_idx = [sorted_push_idx[1], sorted_push_idx[0], sorted_push_idx[2]]
        push_down(sorted_push_idx)
        robot.go_home()


        
        # midpt2, corner2, corner3, pushdown
        side3_ratios = [midpt_ratios[2], corner_ratios[2], corner_ratios[3]]
        print("here are the ratios to be picked up for side 3,", side3_ratios)
        pick_place_trap_side(cable_mask_binary, cable_endpoints, channel_start_pt, camCal, side3_ratios, channel_cnt_mask)
        if PUSH_MODE == "uni" or PUSH_MODE == "hybrid":
            sorted_push_idx = sample_pts_btwn(corner_idxs[2], corner_idxs[3], 3)
        elif PUSH_MODE == "binary":
            sorted_push_idx = sample_pts_btwn(corner_idxs[2], corner_idxs[3], 3)
            sorted_push_idx = [sorted_push_idx[1], sorted_push_idx[0], sorted_push_idx[2]]
        push_down(sorted_push_idx)
        robot.go_home()

        # midpt3, corner3, last_channel_pt, pushdown  
        # just pick 0.99 to get the end but not exact end
        side4_ratios = [midpt_ratios[3], corner_ratios[3], 0.99]
        print("here are the ratios to be picked up for side 4,", side4_ratios)
        pick_place_trap_side(cable_mask_binary, cable_endpoints, channel_start_pt, camCal, side4_ratios, channel_cnt_mask)
        if PUSH_MODE == "uni" or PUSH_MODE == "hybrid":
            sorted_push_idx = sample_pts_btwn(corner_idxs[3], len(sorted_channel_pts)-1, 3)
        elif PUSH_MODE == "binary":
            sorted_push_idx = sample_pts_btwn(corner_idxs[3], len(sorted_channel_pts)-1, 3)
            sorted_push_idx = [sorted_push_idx[1], sorted_push_idx[0], sorted_push_idx[2]]
        push_down(sorted_push_idx)
        robot.go_home()

        # 5: plug this ratio into pick_and_place_ratio this will have us pick and place the cable s.t. it's inserted into the midpoint of the first line segment
        # 6: pick and place the endpoint of the cable into the first corner
        # 7: push down from the endpoint of the cable up until you reach the end of the line segment
        # 8: at this point pick the cable at that point and rotate it such that it aligns with the corner and pushes it down
        # 9: repeat this process by picking the midpoint and insertting it in and then pushing down until you reach a corner and then specifically insert the corner point
        # want to first pick the middle of a side then then 
    
    ### SLIDING/PUSHING ####
    robot.go_home()
    # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)


    '''if args.use_slide:
        sorted_cable_points, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)
        cable_midpt = sorted_cable_points[len(sorted_cable_points) // 2]
        channel_midpt = sorted_channel_pts[len(sorted_channel_pts) // 2]
        dist = np.linalg.norm(np.array(cable_midpt) - np.array(channel_midpt))

        if dist > MIDPOINT_THRESHOLD:
            midpt_ratio = 0.5
            pick_and_place_ratio(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, midpt_ratio, channel_cnt_mask)'''

    if PUSH_MODE == "binary":
        sorted_push_idx = get_binary_search_idx()
    elif PUSH_MODE == "golden":
        sorted_push_idx = get_golden_search_idx()
    elif PUSH_MODE == "uni" or PUSH_MODE=='hybrid':
        sorted_push_idx = [(1/NUM_PTS_PUSH)*i for i in range(NUM_PTS_PUSH+1)]

    # NUM_PTS_PRESS = 12

    if matched_template == "curved":
        if args.use_slide:
            # NOTE: TRY PRESSING DOWN FIRST TO AVOID SLIDING THE CABLE OUT
            # if PUSH_BEFORE_SLIDE:
            #     sorted_push_idx = [(1/NUM_PTS_PRESS)*i for i in range(NUM_PTS_PRESS+1)]
            #     push_down(sorted_push_idx)
            if PUSH_BEFORE_SLIDE:
                # # NOTE: TRY PRESSING DOWN FIRST TO AVOID SLIDING THE CABLE OUT
                # sorted_push_idx = [(1/NUM_PTS_PRESS)*i for i in range(NUM_PTS_PRESS+1)]
                # if PUSH_MODE == "binary":
                #     sorted_push_idx = get_binary_search_idx()
                # elif PUSH_MODE == "uni" or PUSH_MODE =='hybrid':
                #     sorted_push_idx = [(1/NUM_PTS_PUSH)*i for i in range(NUM_PTS_PUSH+1)]
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
        # means that we should just be pushing down normally
        else:
            if not PUSH_BEFORE_SLIDE:
                push_down(sorted_push_idx)
    elif matched_template == "straight":
        if PUSH_BEFORE_SLIDE:
            # # NOTE: TRY PRESSING DOWN FIRST TO AVOID SLIDING THE CABLE OUT
            # sorted_push_idx = [(1/NUM_PTS_PRESS)*i for i in range(NUM_PTS_PRESS+1)]
            # if PUSH_MODE == "binary":
            #     sorted_push_idx = get_binary_search_idx()
            # elif PUSH_MODE == "uni" or PUSH_MODE == 'hybrid':
            #     sorted_push_idx = [(1/NUM_PTS_PUSH)*i for i in range(NUM_PTS_PUSH+1)]
            push_down(sorted_push_idx)

        if args.use_slide:
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
                # want to return to the middle then slide the opposite way
                robot.go_home()
                mid_pose.translation[2] += 0.03
                robot.move_pose(mid_pose)
                mid_pose.translation[2] -= 0.03
                robot.slide_linear(mid_pose, slide_start_pose)
        # means that we should just be pushing down normally
        else:
            if not PUSH_BEFORE_SLIDE:
                push_down(sorted_push_idx)
    elif matched_template == 'trapezoid':
        if args.use_slide:
            # probably want to do something like 4 linear slides or something
            corner1_pose = get_rw_pose(swapped_sorted_channel_pts[corner_idxs[0]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner12_mid_pose = get_rw_pose(swapped_sorted_channel_pts[midpt_idxs[0]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner2_pose = get_rw_pose(swapped_sorted_channel_pts[corner_idxs[1]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner12_mid_pose = get_rw_pose(swapped_sorted_channel_pts[midpt_idxs[1]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True) 
            corner3_pose = get_rw_pose(swapped_sorted_channel_pts[corner_idxs[2]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner12_mid_pose = get_rw_pose(swapped_sorted_channel_pts[midpt_idxs[2]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner4_pose = get_rw_pose(swapped_sorted_channel_pts[corner_idxs[3]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            corner12_mid_pose = get_rw_pose(swapped_sorted_channel_pts[midpt_idxs[3]], swapped_sorted_channel_pts,15,0.1,camCal=camCal, is_channel_pt=True)
            robot.slide_linear(corner1_pose, corner2_pose)
            robot.slide_linear(corner2_pose, corner3_pose)
            robot.slide_linear(corner3_pose, corner4_pose)
            robot.slide_linear(corner4_pose, corner1_pose)
            
        # means that we should just be pushing down normally
        else:
            push_down(sorted_push_idx)

robot.go_home()
# f_name = f'evaluation_images/trapezoid/overhead_{N}_{PICK_MODE}.png'
# overhead_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
# overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
# plt.imsave(f_name, overhead_img)
# f_name = f'evaluation_images/trapezoid/front_{N}_{PICK_MODE}.png'
# front_img = get_zed_img(front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth)
# front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
# plt.imsave(f_name, front_img)