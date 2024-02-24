from ur5py.ur5 import UR5Robot
import numpy as np
from shape_match import *
from autolab_core import RigidTransform
import pdb
from real_sense_modules import *
from utils import *
import argparse
from gasketRobot import GasketRobot
from scipy.spatial.transform import Rotation as R
from resources import CROP_REGION, curved_template_mask, straight_template_mask, trapezoid_template_mask
from calibration.image_robot import ImageRobot
from zed_camera import ZedCamera


# CROP_REGION = [64, 600, 189, 922]
# curved_template_mask = cv2.imread('templates_crop_master/master_curved_channel_template.png')
# straight_template_mask = cv2.imread('templates_crop_master/master_straight_channel_template.png')
# trapezoid_template_mask = cv2.imread('templates_crop_master/master_trapezoid_channel_template.png')

START_IDX = 8
END_IDX = -START_IDX - 1


argparser = argparse.ArgumentParser()
# options for experiment type, either 'no_ends' or 'one_end'
argparser.add_argument('--exp', type=str, default='no_ends_attached')
# options for where to push on channel: 'unidirectional', 'golden', 'binary', 'slide'
argparser.add_argument('--push_mode', type=str, default='unidirectional')
# options for where to pick the cable at: 'unidirectional', 'golden', 'binary', 'slide'
argparser.add_argument('--pick_mode', type=str, default='unidirectional')
argparser.add_argument('--blur_radius', type=int, default=5)
argparser.add_argument('--sigma', type=int, default=0)
argparser.add_argument('--dilate_size_channel', type=int, default=2)
argparser.add_argument('--dilate_size_rope', type=int, default=20)
argparser.add_argument('--canny_threshold_channel', type=tuple, default=(100,255))
argparser.add_argument('--canny_threshold_rope', type=tuple, default=(0,255))
argparser.add_argument('--visualize', default=False, action='store_true')
    
    
def detect_cable(rgb_img, args):
    # Detecting the cable
    cable_cnt, cable_mask_hollow  = get_cable(img = rgb_img, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize)
        
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
    cable_mask_binary = (cable_mask_binary.sum(axis=2)/255).astype('uint8')

    cable_skeleton = skeletonize(cable_mask_binary)
    # breakpoint()
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
    matched_template, matched_results, channel_cnt, _ = get_channel(img = rgb_img, cable_mask=cable_mask_binary, blur_radius=args.blur_radius, sigma=args.sigma, 
                                                     dilate_size=args.dilate_size_channel, canny_threshold=args.canny_threshold_channel, viz=args.visualize)
    
    if matched_template == 'curved':
        template_mask = curved_template_mask
    elif matched_template == 'straight':
        template_mask = straight_template_mask
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask
    print('================')
    print('mask stuff')
    print('================')
    # breakpoint()
    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    # making it 3 channels
    aligned_channel_mask = cv2.merge((aligned_channel_mask, aligned_channel_mask, aligned_channel_mask))
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    # plt.imshow(rgb_img)
    # plt.imshow(aligned_channel_mask, alpha=0.5)
    # plt.show()

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
    # plt.show()
    #getting the length and endpoints of the channel
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    if len(channel_endpoints) == 1:
        print("We have a loop!")
    return channel_skeleton, channel_length, channel_endpoints, matched_template

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

def get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, depth_img = None):
    # just pick an endpoint to be the one that we'll use as our in point
    cable_endpoint_in = cable_endpoints[0]
    channel_endpoint_in, channel_endpoint_out = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
    sorted_channel_pts = sort_skeleton_pts(channel_skeleton, channel_endpoint_in)
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

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

def find_nth_nearest_point(point, sorted_points, given_n):
    # print(endpoint)
    # print(np.array(sorted_points).shape, np.array(endpoint).shape)
    # np.linalg.norm(np.array(sorted_points) - np.array(endpoint))
    # if point == sorted_points[0]:
    #     return sorted_points[n - 1]
    # elif point == sorted_points[-1]:
    #     return sorted_points[-n]
    breakpoint()
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
    euler = np.array([-np.pi, 0, dz])
    rot_matrix = R.from_euler("xyz", euler).as_matrix()
    return rot_matrix



def get_rw_pose(orig_pt, sorted_pixels, n, ratio, camCal, is_channel_pt, use_depth = False):
    # next_point = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    behind_idx, infront_idx = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    behind_pt = sorted_pixels[behind_idx]
    infront_pt = sorted_pixels[infront_idx]
    plt.scatter(x=behind_pt[1], y=behind_pt[0], c='r')
    plt.scatter(x=infront_pt[1], y=infront_pt[0], c='b')
    plt.imshow(rgb_img)
    plt.show()


    breakpoint()
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
        hardcoded_z = -15
        # if we want a point on the channel need to account for the height of the template
        if is_channel_pt:
            hardcoded_z += TEMPLATE_HEIGHT[matched_template]
        orig_rw = np.array([orig_rw_xy[0], orig_rw_xy[1],hardcoded_z/1000])
    else:
        print("haven't implemented using depth for pick/place!")
        raise ValueError()
    # converting pose to rigid transform to use with ur5py library
    orig_rt_pose = RigidTransform(rotation=rot, translation=orig_rw)
    return orig_rt_pose

# gets rigid transform given pose

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
    plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    plt.imshow(rgb_img)
    plt.show()

    # breakpoint()
    plt.imshow(rgb_img)
    plt.imshow(channel_skeleton, alpha=0.5)
    plt.imshow(cable_skeleton, alpha=0.5)
    plt.show()

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

    
    pick_pt = sorted_cable_pts[START_IDX] 
    place_pt = sorted_channel_pts[START_IDX]
    plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    plt.imshow(rgb_img)
    plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
    # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose_swap, place_pose_swap)
    return swapped_sorted_cable_pts, swapped_sorted_channel_pts
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

def one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side):
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
    plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    plt.imshow(rgb_img)
    plt.show()

    breakpoint()
    plt.imshow(rgb_img)
    plt.imshow(cable_skeleton, alpha=0.7)
    plt.show()

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)
    breakpoint()

    # now we want the last endpoints to pick and place
    if second_endpt_side == 'right':
        if sorted_cable_pts[END_IDX][1] >= 555:
            pick_pt = sorted_cable_pts[END_IDX]
        else:
            pick_pt = sorted_cable_pts[START_IDX]
        if sorted_channel_pts[END_IDX][1] >= 555:
            place_pt = sorted_channel_pts[END_IDX]
        else:
            place_pt = sorted_channel_pts[START_IDX]
    else:
        if sorted_cable_pts[END_IDX][1] < 555:
            pick_pt = sorted_cable_pts[END_IDX]
        else:
            pick_pt = sorted_cable_pts[START_IDX]
        if sorted_channel_pts[END_IDX][1] < 555:
            place_pt = sorted_channel_pts[END_IDX]
        else:
            place_pt = sorted_channel_pts[START_IDX]
    # pick_pt = sorted_cable_pts[-1] 
    # place_pt = sorted_channel_pts[-1]
    plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    plt.imshow(rgb_img)
    plt.show()

    # --------- EDIT THIS LATER TO INCLUDE SUPPORT FOR ALL TEMPLATES --------
    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 20, 0.1, camCal, False)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, False)
    robot.pick_and_place(pick_pose_swap, place_pose_swap)
    slide_goal_pt = sorted_channel_pts[0]
    slide_goal_pose = get_rw_pose((slide_goal_pt[1], slide_goal_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, False)
    
    # should actually slide across the straight channel (hopefully)
    # robot.push(slide_goal_pose, is_place_pt=True, force_ctrl=True)


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
        plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        plt.imshow(rgb_img)
        plt.show()

        breakpoint()
        plt.imshow(rgb_img)
        plt.imshow(cable_skeleton, alpha=0.7)
        plt.show()

        # we sort the points on channel and cable to get a relation between the points
        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)
        breakpoint()

        # now we want the last endpoints to pick and place
        if second_endpt_side == 'right':
            if sorted_cable_pts[END_IDX][1] >= 555:
                pick_pt = sorted_cable_pts[END_IDX]
            else:
                pick_pt = sorted_cable_pts[START_IDX]
            if sorted_channel_pts[END_IDX][1] >= 555:
                place_pt = sorted_channel_pts[END_IDX]
            else:
                place_pt = sorted_channel_pts[START_IDX]
        else:
            if sorted_cable_pts[END_IDX][1] < 555:
                pick_pt = sorted_cable_pts[END_IDX]
            else:
                pick_pt = sorted_cable_pts[START_IDX]
            if sorted_channel_pts[END_IDX][1] < 555:
                place_pt = sorted_channel_pts[END_IDX]
            else:
                place_pt = sorted_channel_pts[START_IDX]
        
        # pick_pt = sorted_cable_pts[-1] 
        # place_pt = sorted_channel_pts[-1]
        plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        plt.imshow(rgb_img)
        plt.show()

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
        plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        plt.imshow(rgb_img)
        plt.show()

        # breakpoint()
        plt.imshow(rgb_img)
        plt.imshow(channel_skeleton, alpha=0.5)
        plt.imshow(cable_skeleton, alpha=0.5)
        plt.show()

        # we sort the points on channel and cable to get a relation between the points
        sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

        
        pick_pt = sorted_cable_pts[START_IDX] 
        place_pt = sorted_channel_pts[START_IDX]
        plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
        plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
        plt.imshow(rgb_img)
        plt.show()

        # needs to be swapped as this is how it is expected for the robot
        swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
        # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
        place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

        robot.pick_and_place(pick_pose_swap, place_pose_swap)
    
    # should actually slide across the straight channel (hopefully)
    # robot.push(slide_goal_pose, is_place_pt=True, force_ctrl=True)
        
def pick_and_place_idx(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, idx):
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    plt.imshow(rgb_img)
    plt.show()

    breakpoint()
    plt.imshow(rgb_img)
    plt.imshow(cable_skeleton, alpha=0.7)
    plt.show()

    # we sort the points on channel and cable to get a relation between the points
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

    
    pick_pt = sorted_cable_pts[idx] 
    place_pt = sorted_channel_pts[idx]
    plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    plt.imshow(rgb_img)
    plt.show()

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, 0.1, camCal, is_channel_pt=False)
    # place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, is_channel_pt=True)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, 0.1, camCal, is_channel_pt=True)

    robot.pick_and_place(pick_pose_swap, place_pose_swap)


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
                breakpoint()
                robot.move_pose(rw_pose)



TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
# curved is 2.54cm high, straight is 1.5cm high, trapezoid is 2cm high
TEMPLATE_HEIGHT = {'curved':0.0254, 'straight':0.0127, 'trapezoid':0.0254}

TOTAL_PICK_PLACE = 5


if __name__=='__main__':
    args = argparser.parse_args()
    PUSH_MODE = args.push_mode
    PICK_MODE = args.pick_mode
    EXP_MODE = args.exp
    robot = GasketRobot()
    robot.set_playload(1)
    # robot.force_mode(robot.get_pose(convert=False),[1,1,1,1,1,1],[0,0,0,0,0,0],2,[1,1,1,1,1,1], 0.1)
    robot.go_home()
    
    
    # # Sets up the realsense and gets us an image
    pipeline, colorizer, align, depth_scale = setup_rs_camera()
    time.sleep(1)
    rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)

    # Sets up the zed camera and gets us an image
    # camera = ZedCamera(20120598)
    # overhead_cam_id = 22008760 # overhead camera
    # side_cam_id = 20120598 # side camera
    # side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id) #should be overhead camera id
    # rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
    
    # plt.imshow(rgb_img)
    # plt.show()
    # cropped_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    # plt.imshow(cropped_img)
    # plt.show()

    # loads model for camera to real world
    camCal = ImageRobot()

    #### Calibrration Testing #########
    # rgb_img = get_rs_image(pipeline, align, depth_scale, use_depth=False)
    # click_and_move_pts_on_img(rgb_img)
    # PointSelector(rgb_img)
    plt.imshow(rgb_img)
    plt.show()
    
    NUM_BINARY = 4

    search_ratio = 1/2
    # getting all of the ratios for a necessary binary search
    search_idx = search(search_ratio, search_ratio/NUM_BINARY)
    breakpoint()
    sorted_search_idx = [search_idx[0]]
    stride = round((len(search_idx)-1)/2)
    for i in range(1,round(len(search_idx)/2)):
        sorted_search_idx.append(search_idx[i])
        sorted_search_idx.append(search_idx[i+stride])

    for i in range(1, NUM_BINARY+1):
        if i == 1:
            # gets initial state of cable and channel
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            channel_skeleton, channel_length, channel_endpoints, matched_template = detect_channel(rgb_img, cable_mask_binary, args)

            # performs the experiment given no ends are attached to the channel
            if EXP_MODE == 'no_ends_attached':
                swapped_sorted_cable_pts, swapped_sorted_channel_pts = no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
            
            # even if we are doing no_ends_attached we simply 
            # need to attach one end then we can run the program as if it was always just one end attached        
            # color_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
            # rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            
            # rgb_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
                
            existing_pick_pt = swapped_sorted_cable_pts[0]
            second_endpt_side = 'right' if existing_pick_pt[0] < 555 else 'left'
            # breakpoint()
            
            ### NOTE: TAKE NEW PHOTO SINCE CABLE END MAY MOVE AFTER FIRST POINT INSERTION
            robot.go_home()
            ### END NOTE

            rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, second_endpt_side)
        else:
            breakpoint()
            robot.go_home()
            rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            # do i-1 cause i == 1 when we get into here
            idx = math.floor(sorted_search_idx[i-1]*cable_length)
            pick_and_place_idx(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, idx)


            # swapped_sorted_cable_pts, swapped_sorted_channel_pts = attach_intermediary_parts(cable_mask_binary, cable_endpoints, channel_endpoints, 
            #                                                                                 camCal, iter=i, second_endpt_side=None, second=False)
            # existing_pick_pt = swapped_sorted_cable_pts[0]
            # second_endpt_side = 'right' if existing_pick_pt[0] < 555 else 'left'
            # rgb_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)
            # cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            # attach_intermediary_parts(cable_mask_binary, cable_endpoints, channel_endpoints, camCal, iter=i, second_endpt_side=second_endpt_side, second=True)