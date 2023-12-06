from ur5py import UR5Robot
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


argparser = argparse.ArgumentParser()
# options for experiment type, either 'no_ends' or 'one_end'
argparser.add_argument('--exp', type=str, default='no_ends_attached')
# options for where to push on channel: 'unidirectional', 'golden', 'binary', 'slide'
argparser.add_argument('--push_mode', type=str, default='unidirectional')
# options for where to pick the cable at: 'unidirectional', 'golden', 'binary', 'slide'
argparser.add_argument('--pick_mode', type=str, default='unidirectional')
    
    
def detect_cable(rgb_img):
    # Detecting the cable
    cable_cnt, cable_mask_hollow  = get_cable(img = rgb_img)
        
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)

    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    assert len(cable_endpoints) == 2
    return cable_skeleton, cable_length, cable_endpoints, cable_mask_binary

def detect_channel(rgb_img, cable_mask_binary):
    # Detecting the channel
    matched_template, matched_results, channel_cnt, _ = get_channel(img = rgb_img, cable_mask=cable_mask_binary)
    
    if matched_template == 'curved':
        template_mask = curved_template_mask
    elif matched_template == 'straight':
        template_mask = straight_template_mask
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask
    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    # making it 3 channels
    aligned_channel_mask = cv2.merge((aligned_channel_mask, aligned_channel_mask, aligned_channel_mask))
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    # plt.imshow(rgb_img)
    # plt.imshow(aligned_channel_mask, alpha=0.5)
    # plt.show()

    # skeletonizing the channel
    channel_skeleton = skeletonize(aligned_channel_mask)
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

def find_nth_nearest_point(point, sorted_points, n):
    # print(endpoint)
    # print(np.array(sorted_points).shape, np.array(endpoint).shape)
    # np.linalg.norm(np.array(sorted_points) - np.array(endpoint))
    if point == sorted_points[0]:
        return sorted_points[n - 1]
    elif point == sorted_points[-1]:
        return sorted_points[-n]
    idx = sorted_points.index(point)
    if np.linalg.norm(np.array(point)-np.array(sorted_points[0])) < np.linalg.norm(np.array(point)-np.array(sorted_points[-1])):
        return sorted_points[idx + n - 1]
    else:
        return sorted_points[idx - n]

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
    next_point = find_nth_nearest_point(orig_pt, sorted_pixels, n)
    offset_point = find_nth_nearest_point(orig_pt, sorted_pixels, int(len(sorted_pixels)*ratio))
    
    # needs to be done since the point was relative to the entire view of the camera but our model is trained on points defined only in the cropped frame of the image
    orig_pt = np.array(orig_pt) - np.array([CROP_REGION[2], CROP_REGION[0]])
    next_pt = np.array(next_point) - np.array([CROP_REGION[2], CROP_REGION[0]])
    orig_rw_xy = camCal.image_pt_to_rw_pt(orig_pt) 
    next_rw_xy = camCal.image_pt_to_rw_pt(next_point)   
    offset_rw_xy = camCal.image_pt_to_rw_pt(offset_point)
    rot = get_rotation(orig_rw_xy, next_rw_xy)
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
    # plt.imshow(cable_skeleton)
    # plt.show()
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    plt.imshow(rgb_img)
    plt.show()

    # plt.imshow(rgb_img)
    # plt.imshow(cable_skeleton, alpha=0.5)
    # plt.show()


    
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton)

    breakpoint()
    pick_pt = sorted_cable_pts[0] 
    place_pt = sorted_channel_pts[0]
    plt.scatter(x=pick_pt[1], y=pick_pt[0], c='r')
    plt.scatter(x=place_pt[1], y=place_pt[0], c='b')
    plt.imshow(rgb_img)
    plt.show()

    # needs to be swapped as this is how it is expected
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    pick_pose_swap = get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 20, 0.1, camCal, False)
    place_pose_swap = get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 20, 0.1, camCal, False)
    breakpoint()
    robot.pick_and_place(pick_pose_swap, place_pose_swap)

    # robot.move_pose(pick_pose)
    
    # place_pose = get_rw_pose(place_pt, sorted_channel_pts, 20, 0.1, camCal)
    # # only do this when no ends attached cause we dont care about dragging the rope
    # # z offset for height of channel 
    # place_pose[2] += TEMPLATE_HEIGHT[matched_template]
    # robot.move_pose(place_pose)
    # robot.close_grippers()

    # robot.move_to_channel_overhead(transformed_channel_end, rotation)
    # robot.push(transformed_channel_end, rotation)
    # robot.go_home()
    # robot.open_grippers()

def one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal):
    pass




TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
# curved is 2.54cm high, straight is 1.5cm high, trapezoid is 2cm high
TEMPLATE_HEIGHT = {'curved':0.0254, 'straight':0.02, 'trapezoid':0.0254}

TOTAL_PICK_PLACE = 5


if __name__=='__main__':
    args = argparser.parse_args()
    PUSH_MODE = args.push_mode
    PICK_MODE = args.pick_mode
    EXP_MODE = args.exp
    robot = GasketRobot()
    robot.go_home()
    
    # Sets up the realsense and gets us an image
    pipeline, colorizer, align, depth_scale = setup_rs_camera()
    color_img, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)

    rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    cropped_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    plt.scatter(y=138, x=49)
    plt.imshow(rgb_img)
    plt.show()

    # loads model for camera to real world
    camCal = ImageRobot()

    # gets initial state of cable and channel
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img)
    channel_skeleton, channel_length, channel_endpoints, matched_template = detect_channel(rgb_img, cable_mask_binary)

    # performs the experiment given no ends are attached to the channel
    if EXP_MODE == 'no_ends_attached':
        no_ends_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)
    # even if we are doing no_ends_attached we simply 
    # need to attach one end then we can run the program as if it was always just one end attached        
    one_end_attached(cable_mask_binary, cable_endpoints, channel_endpoints, camCal)