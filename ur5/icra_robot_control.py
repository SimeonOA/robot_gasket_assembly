from r2d2.misc.parameters import varied_camera_1_id, varied_camera_2_id, nuc_ip
from r2d2.misc.server_interface import ServerInterface
import numpy as np
import time
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import r2d2.misc.pointcloud_utils as pc_utils
import matplotlib.pyplot as plt
import cv2
import math
import sys
from primitives.franka import FrankaFranka
from sensing.process_segmasks import get_channel, get_rope
from sensing.utils_binary import skeletonize_img, find_length_and_endpoints, sort_skeleton_pts
from cable_insertion.shape_match import *
import random
from matplotlib.backend_tools import ToolBase, ToolToggleBase
# import pixel_sam

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_path', type=str, default='imgs/curved7.png')
argparser.add_argument('--blur_radius', type=int, default=5)
argparser.add_argument('--sigma', type=int, default=0)
argparser.add_argument('--dilate_size_channel', type=int, default=2) #most recently had it as 15 #DEFAULT WAS ORIGINALLY 2 
argparser.add_argument('--dilate_size_rope', type=int, default=20)
argparser.add_argument('--canny_threshold_channel', type=tuple, default=(0,255)) # DEFAULT WAS ORIGINALLY (100,255)
argparser.add_argument('--canny_threshold_rope', type=tuple, default=(0,255)) # DEFAULT WAS ORIGINALLY (0,255)
argparser.add_argument('--visualize', default=False, action='store_true')
argparser.add_argument('--robot', default=False, action='store_true')
argparser.add_argument('--curved_template_cnt_path', type=str, default='templates/curved_template_full_cnt.npy')
argparser.add_argument('--straight_template_cnt_path', type=str, default='templates/straight_template_full_cnt.npy')
argparser.add_argument('--trapezoid_template_cnt_path', type=str, default='templates/trapezoid_template_full_cnt.npy')

args = argparser.parse_args()



# MODE = 'unidirectional'
MODE = 'binary'
# MODE = 'golden'
TOTAL_PICK_PLACE = 5
calibration_matrices = {
    varied_camera_1_id
    + "_left": np.array([0.30001049, 0.59906832, 0.50958757, -2.0590553, 0.03809194, -2.24397196]),
    varied_camera_1_id
    + "_right": np.array([0.22513224, 0.50150912, 0.50919906, -2.06088171, 0.05364713, -2.23978636]),
    varied_camera_2_id
    + "_left": np.array([ 6.10809570e-01, -1.36855093e-01,  6.22378600e-01,  3.07257795e+00,
       -2.98030630e-03,  1.59287246e+00]),
    varied_camera_2_id + "_right": np.array([ 0.61484969, -0.013918  ,  0.62037367,  3.07275499,  0.00322605,
        1.60402774]),
}

def transform_from_camera_frame(original_point, calibration_matrix, lstsq_x=None, hard_coded_z=None):
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    transformed_point = np.dot(original_point, rotation_matrix.T) + cam_pose
    transformed_point[2] +=0.165 #account for end effector
    if lstsq_x is not None:
      transformed_point[2] = np.dot(np.array([transformed_point[0], transformed_point[1], 1]).reshape((1,3)), lstsq_x)
    elif hard_coded_z is not None: 
      transformed_point[2] = hard_coded_z
    return transformed_point

def transform_from_robot_frame(transformed_point, calibration_matrix):
    transformed_point = transformed_point.copy()
    transformed_point[2] -= 0.165
    cam_pose = calibration_matrix[:3]
    transformed_point = transformed_point - cam_pose
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    original_point = np.dot(transformed_point, np.linalg.inv(rotation_matrix.T))
    return original_point

def point_cloud_to_camera_frame(point_cloud_pt, focal_left_x, focal_left_y, princ_left_x, princ_left_y):
    # Project the 3D point to 2D
    x_2d = (focal_left_x * point_cloud_pt[0] / point_cloud_pt[2]) + princ_left_x
    y_2d = (focal_left_y * point_cloud_pt[1] / point_cloud_pt[2]) + princ_left_y

    print("X-coordinate (2D):", x_2d)
    print("Y-coordinate (2D):", y_2d)
    return x_2d, y_2d

def get_rotation(point1, point2):
    direction = point2 - point1
    direction = direction / np.linalg.norm(direction) #unit vector

    reference_direction = np.array([1,0,0])
    cos_theta = np.dot(direction, reference_direction)
    sin_theta = np.sqrt(1 - cos_theta**2)
    dz = np.arctan2(sin_theta, cos_theta)
    return np.array([-np.pi, 0, dz])

def find_transformed_point_and_rotation_rope(farthest_endpoint, sorted_pixels, n ):
    next_point = find_nth_nearest_point(farthest_endpoint, sorted_pixels, n)
    try:
        err, end_pc_value = point_cloud.get_value(farthest_endpoint[0], farthest_endpoint[1])
        err, next_pc_value = point_cloud.get_value(next_point[0], next_point[1])
        end_pc_value = end_pc_value[:3]
        next_pc_value = next_pc_value[:3]
        transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix, LSTSQ_X,HARD_CODED_Z)
        transformed_next = transform_from_camera_frame(next_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
        rotation = get_rotation(transformed_end, transformed_next)
        if np.isnan(rotation).any() or np.isnan(transformed_end).any() or np.isinf(rotation).any() \
            or np.isinf(transformed_end).any():
            raise Exception("nan or inf in transformed end/rotation")
    except:
        print("error occurred when value")
        idx = sorted_pixels.index(next_point)
        if idx == len(sorted_pixels) - 1:
            next_idx = idx - 1
        else:
            next_idx = idx + 1
        # getting the next thing 
        return find_transformed_point_and_rotation_rope(sorted_pixels[next_idx], sorted_pixels, n+1)
    return transformed_end, rotation

def find_transformed_point_and_rotation_channel(farthest_endpoint, sorted_pixels, n, ratio):
    next_point = find_nth_nearest_point(farthest_endpoint, sorted_pixels, n)
    offset_point = find_nth_nearest_point(farthest_endpoint, sorted_pixels, int(len(sorted_pixels)*ratio))
    try:
        err, end_pc_value = point_cloud.get_value(farthest_endpoint[0], farthest_endpoint[1])
    except:
        print("no point cloud value")
        sys.exit()
    err, next_pc_value = point_cloud.get_value(next_point[0], next_point[1])
    err, offset_pc_value = point_cloud.get_value(offset_point[0], offset_point[1])

    end_pc_value = end_pc_value[:3]
    next_pc_value = next_pc_value[:3]
    offset_pc_value = offset_pc_value[:3]
    transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
    transformed_next = transform_from_camera_frame(next_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)   
    transformed_offset = transform_from_camera_frame(offset_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
    rotation = get_rotation(transformed_end, transformed_next)
    return transformed_end, transformed_next, transformed_offset, rotation

def rope_mask_info(image):
    img = image.get_data()[:,:,:3]
    # img = cv2.imread(img)
    # img = image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(img)[:, :, 0]
    # cable_lower = np.array([22, 22, 180]) # evening (the rope is not bright enough so need to lower the number)

    # cable_lower = np.array([22, 22, 182]) # evening (the rope is not bright enough so need to lower the number)
    # cable_lower = np.array([22, 35, 200]) # day (increase the number to avoid detecting the channel, at the cost of not detecting the rope)
    # cable_upper = np.array([45, 128, 245]) # 

    cable_lower = np.array([0, 24, 181]) # morning (new rope)
    cable_upper = np.array([32, 67, 255]) # morning (new rope)
    # cable_lower = np.array([24, 6, 180]) # evening (new rope)
    # cable_upper = np.array([60, 255, 255]) # evening (new rope)
    rope_mask_img = get_rope(img_rgb, color_bounds=[(cable_lower, cable_upper)], plot=False, dilate=True)
    plt.imshow(rope_mask_img)
    plt.show()
    skeleton_img = skeletonize_img(rope_mask_img)
    length_rope, endpoints_rope = find_length_and_endpoints(skeleton_img)
    sorted_endpoints = sort_skeleton_pts(skeleton_img, endpoints_rope[0])
    for i, pt in enumerate(endpoints_rope):
        endpoints_rope[i] = (pt[1], pt[0])
    for i, pt in enumerate(sorted_endpoints):
        sorted_endpoints[i] = (pt[1], pt[0])
    sorted_endpoints = filter_points(sorted_endpoints, point_cloud)   
    return sorted_endpoints[0],sorted_endpoints[-1], sorted_endpoints, length_rope

def channel_mask_info(image):
    img = image.get_data()[:,:,:3]
    # img = cv2.imread(img)
    # img = image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(img)[:, :, 0]
    # channel_lower = np.array([11, 16, 29]) # morning
    # channel_upper = np.array([179, 55, 110]) # morning
    # channel_lower = np.array([11, 16, 29]) # evening
    # channel_upper = np.array([179, 60, 130]) # evening (decrease the number cuz otherwise the cable will be there as well)
    # channel_upper = np.array([179, 120, 150]) # day (increase the upper bound to still detect the black region)
    # channel_mask_img = get_channel(img_rgb, color_bounds=[(channel_lower, channel_upper)], plot=False)
    channel_lower = np.array([0, 0, 86]) # duct taped trapezoid
    channel_upper = np.array([48, 440, 255]) # duct taped trapezoid
    channel_mask_img = get_rope(img_rgb, color_bounds=[(channel_lower, channel_upper)], plot=False, dilate=False)
    plt.imshow(channel_mask_img)
    plt.show()
    skeleton_img_channel = skeletonize_img(channel_mask_img)
    plt.imshow(skeleton_img_channel)
    plt.show()
    length, endpoints = find_length_and_endpoints(skeleton_img_channel)
    # sorted_endpoints = sort_skeleton_pts(skeleton_img_channel, endpoints[0])
    for i, pt in enumerate(endpoints):
        endpoints[i] = (pt[1], pt[0])
    # for i, pt in enumerate(sorted_endpoints):
    #     sorted_endpoints[i] = (pt[1], pt[0]) 
    # sorted_endpoints = get_channel_sorted_list(skeleton_img_channel, endpoints[0]) 
    return endpoints[0],endpoints[-1],length, skeleton_img_channel

def get_channel_sorted_list(skeleton_img_channel, endpoint):
    endpoint = (endpoint[1], endpoint[0])
    sorted_pixel_list = sort_skeleton_pts(skeleton_img_channel, endpoint)
    for i, pt in enumerate(sorted_pixel_list):
        sorted_pixel_list[i] = (pt[1], pt[0])
    sorted_pixel_list = filter_points(sorted_pixel_list, point_cloud)
    return sorted_pixel_list

def filter_points(sorted_pixels, point_cloud):
    filtered_points = []
    for point in sorted_pixels:
        err, pc_value = point_cloud.get_value(point[0], point[1])
        if np.isnan(pc_value[:3]).any():
            continue
        pc_value = pc_value[:3]
        transformed_point = transform_from_camera_frame(pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
        if transformed_point[2] >-0.1:
            filtered_points.append(point)
    return filtered_points


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_farthest_point(rope_endpoint1,rope_endpoint2,channel_endpoint1,channel_endpoint2):
    # Calculate distances between endpoints
    distance_rc1 = calculate_distance(rope_endpoint1, channel_endpoint1)
    distance_rc2 = calculate_distance(rope_endpoint1, channel_endpoint2)
    distance_cr1 = calculate_distance(rope_endpoint2, channel_endpoint1)
    distance_cr2 = calculate_distance(rope_endpoint2, channel_endpoint2)

    # Determine which endpoints are farthest
    min_distance = min(distance_rc1, distance_rc2,distance_cr1, distance_cr2)
    if min_distance==distance_rc1:
        closest_rope_endpoint, closest_channel_endpoint = rope_endpoint1, channel_endpoint1
    elif min_distance==distance_rc2:
        closest_rope_endpoint, closest_channel_endpoint = rope_endpoint1, channel_endpoint2
    elif min_distance==distance_cr1:
        closest_rope_endpoint, closest_channel_endpoint = rope_endpoint2, channel_endpoint1
    else:
        closest_rope_endpoint, closest_channel_endpoint = rope_endpoint2, channel_endpoint2
    
    # Calculate farthest endpoints
    farthest_rope_endpoint = rope_endpoint2 if np.array_equal(closest_rope_endpoint, rope_endpoint1) else rope_endpoint1
    farthest_channel_endpoint = channel_endpoint2 if np.array_equal(closest_channel_endpoint, channel_endpoint1) else channel_endpoint1
    farthest_rope_endpoint, farthest_channel_endpoint = tuple(farthest_rope_endpoint), tuple(farthest_channel_endpoint)
    print("closest_rope_endpoint, closest_channel_endpoint,farthest_rope_endpoint, farthest_channel_endpoint", 
          tuple(closest_rope_endpoint), closest_channel_endpoint,tuple(farthest_rope_endpoint), farthest_channel_endpoint)
    return tuple(closest_rope_endpoint), tuple(closest_channel_endpoint), farthest_rope_endpoint, farthest_channel_endpoint

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


def check_valid_pointcloud(point, sorted_points, point_cloud):
    err, push_pc_value = point_cloud.get_value(point[0], point[1])
    if np.isnan(push_pc_value[:3]).any():
        idx = sorted_points.index(point)
        return check_valid_pointcloud(sorted_points[idx+1], sorted_points, point_cloud)
    else:
        return push_pc_value

def grab_zed_mat(side_cam,runtime_parameters, image, point_cloud, depth):
    if side_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        side_cam.retrieve_image(image, sl.VIEW.LEFT)
        side_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        side_cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return image, point_cloud, depth

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

def straight_operation(pick_idx, push_idx):
    return

def curved_operation():
    return 

def lstsq_plane_fit(point_cloud, use_corners=False):
    x_lims = (CROP_REGION[2], CROP_REGION[3])
    y_lims = (CROP_REGION[0], CROP_REGION[1])# sample n times for points
    #corners = np.array([[393, 192, 1], [389, 399, 1], [898, 221, 1], [883, 428,1]])
    # all in terms of the robot frame, NOT camera frame
    back_left_corner = [0.38392329, 0.29712275, 0.12101871, 3.1157669,  0.01891393, 0.01662496] 
    front_left_corner = [ 0.66387028,  0.29467574,  0.12385561  ,3.11515728, -0.0171942,   0.01357441] 
    back_right_corner = [ 0.39731589, -0.35454234,  0.12052636  ,3.12372894,  0.06641094,  0.04344724] 
    front_right_corner = [ 0.66744441, -0.33716521,  0.12332594  ,3.11005075, -0.0128768,   0.03549414] 
    center = [ 0.53591853 ,-0.01974466 , 0.12187542,  3.10052929,  0.04072593,  0.02758502]
    corners = [back_left_corner, front_left_corner, back_right_corner, front_right_corner]
    if use_corners:
        n = 4
        data = np.ones((n, 3))
        height = np.zeros((n, 1))
        for i in range(n):
            x_choice = corners[i][0]
            y_choice = corners[i][1]
            data[i] = np.array([x_choice, y_choice, 1])
            height[i] = corners[i][2]
    else:
        n = 100
        data = np.ones((n, 3))
        height = np.zeros((n, 1))
        for i in range(n):
            x_choice = random.randint(*x_lims)
            y_choice = random.randint(*y_lims)
            data[i] = np.array([x_choice, y_choice, 1])
            height[i] = point_cloud.get_value(x_choice, y_choice)[1][2]
    print(data)
    x = np.linalg.lstsq(data, height)
    real_x = x[0]
    test_pt = corners[0][:3]
    test_pt[2] = 1
    print(np.dot(np.array(test_pt), real_x))
    return real_x

global rope_list
global channel_list

def click_and_move_pts_on_img(rgb_img, matched_template, lstsq_x=None, hard_coded_z=None):
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
    plt.title('Click on four points')

    # Connect the click event to the figure
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Show the plot and wait for user input
    plt.show()

    point_dict = {rope_point: 1 for rope_point in rope_list}
    point_dict.update({channel_point: 0 for channel_point in channel_list})

    print("points clicked",point_dict)
    # Save the points array if needed
    np.savetxt('points.txt', np.array(rope_list + channel_list))

    
    robot.close_grippers()



    while True:
        use_click_pts = True
        use_lstsq = False
        if use_lstsq:
            for point in point_dict:
                indicator = point_dict[point]
                transformed_end = camera_to_robot(transformation_matrix, point)
                # means the point is on the rope therefore we want to reach to the table
                if indicator == 1:
                    # we shouldn't need to do anything here cause depth is already set for the table
                    pass 
                # point is on the channel so we want to be right above it
                elif indicator == 0:
                    transformed_end[2] += TEMPLATE_HEIGHT[matched_template]
                # increasded z height for the grippers being closed
                transformed_end[2] += 0.019
                print('FINAL transformed_end', transformed_end)
                breakpoint()
                robot.go_to_ee_pose(transformed_end)
                robot.go_home()
        elif use_click_pts:
            for point in point_dict:
                indicator = point_dict[point]
                err, end_pc_value = point_cloud.get_value(point[0], point[1])
                end_pc_value = end_pc_value[:3]
                transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
                print("transformed_end", transformed_end)
                # means the point is on the rope therefore we want to reach to the table
                if indicator == 1:
                    # we shouldn't need to do anything here
                    pass 
                # point is on the channel so we want to be right above it
                elif indicator == 0:
                    transformed_end[2] += TEMPLATE_HEIGHT[matched_template]
                # increasded z height for the grippers being closed
                transformed_end[2] += 0.019
                transformed_end[1] += 0.038 # this is a temporary thing just to see what's happeninig
                print('FINAL transformed_end', transformed_end)
                robot.go_to_ee_pose(transformed_end)
                # plt.scatter(x=point[0],y=point[1] , c='r')
                # plt.imshow(rgb_img)
                # plt.show()
                robot.go_home()
        else:
            def interpolate_3d_points(point1, point2, num_points):
                # Ensure that num_points is at least 2 to include the two endpoints
                num_points = max(num_points, 2)

                # Create a list to store the interpolated points
                interpolated_points = []

                # Calculate the step size for interpolation
                step_size = 1.0 / (num_points - 1)

                for i in range(num_points):
                    # Calculate the interpolation factor
                    t = i * step_size

                    # Interpolate each component (x, y, z) separately
                    x = point1[0] * (1 - t) + point2[0] * t
                    y = point1[1] * (1 - t) + point2[1] * t
                    z = point1[2] * (1 - t) + point2[2] * t

                    interpolated_points.append([x, y, z])

                return interpolated_points
            should_interp_pts = False
            if should_interp_pts:
                point1 = [0.35921905, 0.37465663, 0.13982772]
                # front right corner from view of laptop to table
                # point2 = [ 0.42230256, -0.37695241,  0.13982772  ]
                # point on the center of the rectangle  on the table (kinda, it's not accurate)
                point2 = [0.55237145, 0.00257713, 0.141455  ]
                interpolated_points = interpolate_3d_points(point1, point2, 5)
                for point in interpolated_points:
                    robot.go_to_ee_pose(point)
                robot.go_home()
            #  if we aren't interp pts just have the franka go ot the same point repeatedly to check accuracy
            else:
                final_transformed_end = [0.38927874, 0.01418093, 0.139799]
                print("goal", final_transformed_end)
                robot.go_to_ee_pose(final_transformed_end)
                final_state = robot.robot.get_ee_pose()
                print("state after tring to go there", final_state)
                print("error between goal and final state", np.array(final_state[:3]) - np.array(final_transformed_end))
                robot.go_home()

        

    


def pushdown(sorted_channel_pts, push_idx, MODE, img = None):
    if MODE == 'unidirectional':
        pass
    if MODE == 'binary':
        search_ratio = 1/2
        push_idx = search(search_ratio, 1)
    if MODE == 'golden':
        search_ratio = 0.81
        push_idx = search(search_ratio, 1)
    while push_idx:
            idx = push_idx.pop(0)
            transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = \
                find_transformed_point_and_rotation_channel(sorted_channel_pts[idx], sorted_channel_pts, 30, 1/8)
            # robot.move_to_channel(transformed_channel_end, transformed_channel_offset, rotation)
            if not img is None:
                plt.scatter(x=sorted_channel_pts[idx][0],y=sorted_channel_pts[idx][1] , c='r')
                plt.imshow(rgb_img)
                plt.show()
            robot.push(transformed_channel_end, rotation)
def robot_to_camera(robot_point):
    detected_pc_pt = transform_from_robot_frame(robot_point, left_cam_calibration_matrix)
    detected_cam_pixel = point_cloud_to_camera_frame(detected_pc_pt, focal_left_x, focal_left_y, princ_left_x, princ_left_y)
    return detected_cam_pixel

def get_endpoints_in_get_sorted_pts(cable_endpoints, channel_endpoints):
    # just pick an endpoint to be the one that we'll use as our in point
    cable_endpoint_in = cable_endpoints[0]
    channel_endpoint_in, channel_endpoint_out = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
    sorted_channel_pts = sort_skeleton_pts(channel_skeleton, channel_endpoint_in)
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

    # WARNING: seems like tthe pointcloud and depth are expecting the x and y to be flipped!!!!
    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]

    # filter out points that are invalid in the point cloud

    # NOTICE THAT THTESE ARE NOW SWAPPED!!!
    sorted_channel_pts = filter_points(swapped_sorted_channel_pts, point_cloud) 
    # these are aslo now swapped!
    channel_endpoint_in = sorted_channel_pts[0]
    channel_endpoint_out = sorted_channel_pts[-1]
    sorted_cable_pts = filter_points(swapped_sorted_cable_pts, point_cloud)
    # these are aslo now swapped!
    cable_endpoint_in = sorted_cable_pts[0]
    cable_endpoint_out = sorted_cable_pts[-1]


    # IDEA: update the channel startpoint to be the closest point on the channel skeleton to the cable endpoint
    # then actually delete the indices before that point from sorted_channel_pts
    channel_endpoint_in = find_closest_point(sorted_channel_pts, cable_endpoint_in)
    sorted_channel_pts = sorted_channel_pts[sorted_channel_pts.index(channel_endpoint_in):]

    return cable_endpoint_in, cable_endpoint_out, sorted_cable_pts, channel_endpoint_in, channel_endpoint_out, sorted_channel_pts



def pick(pick_pt):
    transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(pick_pt, sorted_cable_pts, 20)
        
    detected_cam_pixel = robot_to_camera(transformed_rope_end)
    pick_pt = sorted_cable_pts[0]
    place_pt = sorted_channel_pts[0]
    plt.scatter(x=pick_pt[0], y=pick_pt[1], c='r')
    plt.scatter(x=place_pt[0], y=place_pt[1], c='b')
    plt.scatter(x=detected_cam_pixel[0], y=detected_cam_pixel[1], c='g')
    plt.imshow(rgb_img)
    plt.show()
    breakpoint()
    transformed_rope_end[2] += 0.017 #accountting for the height of closed gripper
    robot.grab(transformed_rope_end, rotation)

def place(place_pt):
    transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(place_pt, sorted_channel_pts, 30, 1/8)
        
    detected_cam_pixel = robot_to_camera(transformed_channel_end)
    pick_pt = sorted_cable_pts[0]
    place_pt = sorted_channel_pts[0]
    plt.scatter(x=pick_pt[0], y=pick_pt[1], c='r')
    plt.scatter(x=place_pt[0], y=place_pt[1], c='b')
    plt.scatter(x=detected_cam_pixel[0], y=detected_cam_pixel[1], c='g')
    plt.imshow(rgb_img)
    plt.show()

    # only do this when no ends attached cause we dont care about dragging the rope
    transformed_channel_end[2] += TEMPLATE_HEIGHT[matched_template]
    robot.move_to_channel_overhead(transformed_channel_end, rotation)
    robot.push(transformed_channel_end, rotation)
    robot.go_home()
    robot.open_grippers()


def lstsq_fit_camera_to_robot(camera_point):
    # n = 4
    # data = np.ones((n, 3))
    # height = np.zeros((n, 1))
    # for i in range(n):
    #     x_choice = camera_corners[i][0]
    #     y_choice = camera_corners[i][1]
    #     data[i] = np.array([x_choice, y_choice, 1])
    #     height[i] = corners[i][2]
    # print(data)
    # x = np.linalg.lstsq(data, height)
    # real_x = x[0]
    # test_pt = corners[0][:3]
    # test_pt[2] = 1
    # print(np.dot(np.array(test_pt), real_x))
    # return real_x

    from scipy.linalg import lstsq

    # Define the camera frame and robot frame points
    four_corners_camera_pixels = np.array([
        [883.4830384559145, 418.4703438833429],
        [894.0108934054686, 210.7844780603225],
        [388.67385582687524, 193.5570790519614],
        [389.6309335495619, 398.3717117069216]
    ])

    four_corners_robot_points_xyz = np.array([
        [0.35614908, 0.36473045, 0.13559806],
        [0.66066617, 0.37906951, 0.1388433],
        [0.71358085, -0.32604247, 0.13867502],
        [0.4265801, -0.36815172, 0.13547967]
    ])

    # Add a column of ones to the camera frame points for the translation component
    ones_column = np.ones((4, 1))
    four_corners_camera_pixels_homogeneous = np.hstack((four_corners_camera_pixels, ones_column))

    # Solve for the transformation matrix using the least squares method
    transformation_matrix, _, _, _ = lstsq(four_corners_camera_pixels_homogeneous, four_corners_robot_points_xyz)

    # Extract the transformation matrix components
    #a, b, c, d, e, f, g, h, i, j, k, l = transformation_matrix

    # Apply the transformation to camera frame points
    camera_point_homog = np.array([camera_point[0], camera_point[1], 1])

    robot_point = np.dot(camera_point_homog, transformation_matrix)

    # Extract x, y, z coordinates
    print("Robot Point:", robot_point)

    print("Transformation Matrix:")
    print(transformation_matrix)

    return transformation_matrix
    # print("\nTransformed Points in Robot Frame (x, y, z):")
    # print(transformed_points_xyz)

def camera_to_robot(transformation_matrix, camera_point):
    camera_point_homog = np.array([camera_point[0], camera_point[1], 1])

    robot_point = np.dot(camera_point_homog, transformation_matrix)
    return robot_point


def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
        '''
        pixel_coord: [x, y] in pixel coordinates
        cam_intrinsics: 3x3 camera intrinsics matrix
        '''
        DIST_TO_TABLE = 0.775
        print('cam instrics', cam_intrinsics)
        pixel_coord = np.array(pixel_coord)
        point_3d_cam = DIST_TO_TABLE * np.linalg.inv(cam_intrinsics).dot(np.r_[pixel_coord, 1.0])
        point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
        # print(T_CAM_BASE.matrix)
        point_3d_world = point_3d_world[:3]/point_3d_world[3]
        print('non-homogenous = ', point_3d_world)
        return point_3d_world

def get_transform_matrix(calibration_matrix):
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    # rot = rotation_matrix(np.pi, [0,1,0], cam_pose)[:3,:3]
    return RigidTransform(rotation_matrix, cam_pose, from_frame="camera", to_frame="world")


four_corners_camera_pixels = [(883.4830384559145, 418.4703438833429),
                              (894.0108934054686, 210.7844780603225),
                              (388.67385582687524, 193.5570790519614),
                              (389.6309335495619, 398.3717117069216)]
# in order of front_left, back_left, back_right, front_right
four_corners_robot_points = [[0.35614908, 0.36473045, 0.13559806, 3.14159125, 0.01023882, 0.07635253],
                             [ 0.66066617,  0.37906951,  0.1388433 , -3.12389301, -0.05048417, 0.06182553],
                             [ 0.71358085, -0.32604247,  0.13867502,  3.06851884, -0.03097824, 0.06269901],
                             [ 0.4265801 , -0.36815172,  0.13547967,  3.11990928,  0.06704974, 0.11874178]
                             ]
four_corners_robot_points_xyz = [corner[:3] for corner in four_corners_robot_points]

if __name__ == "__main__":

    
    transformation_matrix = lstsq_fit_camera_to_robot((883.4830384559145, 418.4703438833429))
    # # left_cam_calibration_matrix = calibration_matrices[varied_camera_2_id + "_left"]
    
    # left_cam_calibration_matrix = np.array([ 5.87372635e-01,  3.48340382e-02,  7.42931818e-01,  3.08162185e+00,
    #    -1.33631679e-03,  1.69080620e+00])
    # #np.array([ 5.11421595e-01, -7.65744179e-04,  7.37325573e-01,  3.09109159e+00,
    #    #-9.21380307e-04,  1.84634534e+00])
    

    tvecs = np.array([5.87372635e-01, -0.0575, 0.78])
    rvecs = np.array([np.pi, 0, np.pi/2])
    left_cam_calibration_matrix = np.concatenate((tvecs, rvecs))

    T_CAM_BASE = get_transform_matrix(left_cam_calibration_matrix)

    bottom_left_robot = np.array([0.37928192, 0.41199588]) # z value = 0.1450749
    top_left_robot = np.array([0.68508192, 0.41699588]) # z value = 0.1450749
    bottom_right_robot = np.array([0.40838191999999995, -0.34030412]) # z value = 0.1450749
    top_right_robot = np.array([0.70818192, -0.31990412])

    bottom_right_camera = np.array([382.93138949075484, 202.17077855614195])
    bottom_left_camera = np.array([890.1825825147218, 216.5269443964429])
    top_left_camera = np.array([881.5688830105412, 423.25573249677655])
    top_right_camera = np.array([386.7597003815018, 406.98541121110213])

    center_camera = np.array([625.0720533304975, 317.97718300123626])
    centter_robot = np.array([0.55708192, 0.030995879999999972, 0.1450749])

    # Construct the transformation matrix from the robot frame to the camera frame
    robot_points = np.array([bottom_left_robot, top_left_robot, bottom_right_robot, top_right_robot])
    camera_points = np.array([bottom_left_camera, top_left_camera, bottom_right_camera, top_right_camera])

    # # Solve for the transformation matrix using the known correspondences
    # A = np.vstack((robot_points.T, np.ones(robot_points.shape[0])))
    # B = camera_points
    # breakpoint()
    # transformation_matrix = np.linalg.lstsq(A.T, B, rcond=None)[0]

    # # Invert the transformation matrix to get the transformation from camera frame to robot frame
    # inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # # Now, you can use the inverse transformation matrix to map points from camera frame to robot frame
    # def map_camera_to_robot(camera_point):
    #     camera_point_homogeneous = np.append(camera_point, 1)
    #     robot_point_homogeneous = np.dot(inverse_transformation_matrix, camera_point_homogeneous)
    #     return robot_point_homogeneous[:2]


    # Define functions to perform separate interpolations for x and y
    def interpolate_x(camera_x):
        # x = point1[0] * (1 - t) + point2[0] * t
        return np.interp(camera_x, [bottom_right_camera[0], top_right_camera[0], top_left_camera[0], bottom_left_camera[0]], [bottom_right_robot[0], top_right_robot[0], top_left_robot[0], bottom_left_camera[0]])

    def interpolate_y(camera_y):
        return np.interp(camera_y, [bottom_right_camera[1], bottom_left_camera[1],  top_right_camera[1], top_left_camera[1]], [bottom_right_robot[1], bottom_left_robot[1], top_right_robot[1], top_left_robot[1]])

    robot_x = interpolate_x(center_camera[0])
    robot_y = interpolate_y(center_camera[1])
    robot_point = np.array([robot_x, robot_y])
    print("the interpolated robot point is", robot_point)
    # robot = FrankaFranka()
    # robot.go_home()
    # robot.close_grippers()
    # print(robot.robot.get_ee_pose())

    # Create a Camera object
    # id = int(varied_camera_2_id)
    side_cam = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.set_from_serial_number(20120598) #overhead camera
    status = side_cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Camera Failed To Open")
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_FILL
    calibration_params = side_cam.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    focal_left_y = calibration_params.left_cam.fy
    princ_left_x = calibration_params.left_cam.cx
    princ_left_y = calibration_params.left_cam.cy
    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis
    # tz = calibration_params.T.z
    # Horizontal field of view of the left eye in degrees
    # h_fov = calibration_params.left_cam.h_fov

    image = sl.Mat()
    point_cloud = sl.Mat()
    depth = sl.Mat()



    curved_template_mask = cv2.imread('/home/r2d2/robot_cable_insertion/franka/templates_crop_master/master_curved_channel_template.png')
#   curved_template_mask = cv2.imread('/home/r2d2/robot_cable_insertion/franka/templates_crop_master/master_curved_fill_template.png')
    straight_template_mask = cv2.imread('/home/r2d2/robot_cable_insertion/franka/templates_crop_master/master_straight_channel_template.png')
    trapezoid_template_mask = cv2.imread('/home/r2d2/robot_cable_insertion/franka/templates_crop_master/master_trapezoid_channel_template.png')

    TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
    # curved is 2.54cm high, straight is 1.5cm high, trapezoid is 2cm high
    TEMPLATE_HEIGHT = {'curved':0.0254, 'straight':0.015, 'trapezoid':0.02}
    # first elem is currved width/height, second elem is straight width/height, third elem is trapezoid width/height
    TEMPLATE_RECTS = [(587.4852905273438, 168.0382080078125),(2.75, 26.5), (12, 5.75)]
    TEMPLATE_RATIOS = [max(t)/min(t) for t in TEMPLATE_RECTS]
    CROP_REGION = [120, 478, 374, 1000] # [minY, maxY, minX, maxX]




# ACTUAL CONTROL CODE HERE!!!!!!
    TOTAL_PICK_PLACE = 5
    count = 0

    image, point_cloud, depth = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)
    rgb_img = image.get_data()[:,:,:3]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(depth.get_data()[CROP_REGION[0]+50:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]])
    plt.show()
    breakpoint()
    
    # get the plane and try to interpolate depth so it goes smoothly across
    
    LSTSQ_X = lstsq_plane_fit(point_cloud, use_corners=True)
    HARD_CODED_Z = np.mean([0.12101871,0.12385561,0.12052636,0.12332594])

        # Global variables to store the clicked points

    # click_and_move_pts_on_img(rgb_img, matched_template='straight')

    TEST_DEPTH = False
    if TEST_DEPTH:
        robot.close_grippers()
        use_lstsq = False
        use_hardcoded = True
        while True:
            four_corners_and_center = [(392.5021667176222, 196.42831222002155), (391.5450889949354, 398.3717117069216), (881.5688830105412, 434.7406651690173), (899.7533597415891, 222.2694107325633), (626.9862087758709, 308.406405774369)] #[(392.5021667176222, 195.47123449733488), (391.5450889949354, 398.3717117069216), (882.525960733228, 427.0840433875235), (898.7962820189023, 225.14064390062345), (637.5140637254251, 314.1488721104893)]
            points_array = np.array(four_corners_and_center)
            breakpoint()
            for point in points_array:
                err, end_pc_value = point_cloud.get_value(point[0], point[1])
                end_pc_value = end_pc_value[:3]
                transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix, LSTSQ_X, HARD_CODED_Z)
                print("transformed_end", transformed_end)
                if use_lstsq:
                    transformed_end[2] = np.dot(np.array([transformed_end[0], transformed_end[1], 1]).reshape((1,3)), LSTSQ_X)
                elif use_hardcoded: 
                    transformed_end[2] = HARD_CODED_Z
                print("transformed_end after changing", transformed_end)
                # seems to be consistently 8.5cm off table
                # transformed_end[2] -= 0.075
                # offset for the closed gripper since it is longer than  the gripper when open
                transformed_end[2] += 0.017
                breakpoint()
                robot.go_to_ee_pose(transformed_end)
                breakpoint()
                robot.go_home()






    cable_cnt, cable_mask_hollow  = get_cable(img_path=None, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize, img=rgb_img)
        
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)


    matched_template, matched_results, channel_cnt, _ = get_channel(img_path=None, blur_radius=args.blur_radius, 
                                        sigma=args.sigma, dilate_size=args.dilate_size_channel, 
                                        canny_threshold=args.canny_threshold_channel, viz=args.visualize, img = rgb_img, cable_mask=cable_mask_binary)
    
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
    plt.imshow(rgb_img)
    plt.imshow(aligned_channel_mask, alpha=0.5)
    plt.show()
    channel_skeleton = skeletonize(aligned_channel_mask)
    plt.imshow(rgb_img)
    plt.imshow(channel_skeleton, alpha=0.5, cmap='jet')
    plt.show()
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    

    # if neither end is attached we need to place one endpoint in and then begin the entire process
    NO_ENDS_ATTACHED = True
    if NO_ENDS_ATTACHED:
        cable_skeleton = skeletonize(cable_mask_binary)
        plt.imshow(cable_skeleton)
        plt.show()
        cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
        plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        plt.imshow(rgb_img)
        plt.show()

        plt.imshow(rgb_img)
        plt.imshow(cable_skeleton, alpha=0.5)
        plt.show()


        
        cable_endpoint_in, cable_endpoint_out, sorted_cable_pts, channel_endpoint_in, channel_endpoint_out,\
        sorted_channel_pts = get_endpoints_in_get_sorted_pts(cable_endpoints, channel_endpoints)


        pick_pt = sorted_cable_pts[0]
        place_pt = sorted_channel_pts[0]
        plt.scatter(x=pick_pt[0], y=pick_pt[1], c='r')
        plt.scatter(x=place_pt[0], y=place_pt[1], c='b')
        plt.imshow(rgb_img)
        plt.show()
        breakpoint()

        transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(pick_pt, sorted_cable_pts, 20)
        
        detected_pc_pt = transform_from_robot_frame(transformed_rope_end, left_cam_calibration_matrix)
        detected_cam_pixel = point_cloud_to_camera_frame(detected_pc_pt, focal_left_x, focal_left_y, princ_left_x, princ_left_y)
        pick_pt = sorted_cable_pts[0]
        place_pt = sorted_channel_pts[0]
        plt.scatter(x=pick_pt[0], y=pick_pt[1], c='r')
        plt.scatter(x=place_pt[0], y=place_pt[1], c='b')
        plt.scatter(x=detected_cam_pixel[0], y=detected_cam_pixel[1], c='g')
        plt.imshow(rgb_img)
        plt.show()
        breakpoint()
        transformed_rope_end[2] += 0.017 #accountting for the height of closed gripper
        robot.grab(transformed_rope_end, rotation)
        transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(place_pt, sorted_channel_pts, 30, 1/8)
        
        detected_pc_pt = transform_from_robot_frame(transformed_channel_end, left_cam_calibration_matrix)
        detected_cam_pixel = point_cloud_to_camera_frame(detected_pc_pt, focal_left_x, focal_left_y, princ_left_x, princ_left_y)
        pick_pt = sorted_cable_pts[0]
        place_pt = sorted_channel_pts[0]
        plt.scatter(x=pick_pt[0], y=pick_pt[1], c='r')
        plt.scatter(x=place_pt[0], y=place_pt[1], c='b')
        plt.scatter(x=detected_cam_pixel[0], y=detected_cam_pixel[1], c='g')
        plt.imshow(rgb_img)
        plt.show()

        # only do this when no ends attached cause we dont care about dragging the rope
        transformed_channel_end[2] += TEMPLATE_HEIGHT[matched_template]
        robot.move_to_channel_overhead(transformed_channel_end, rotation)
        robot.push(transformed_channel_end, rotation)
        robot.go_home()
        robot.open_grippers()
        
    pick_idx = []
    push_idx = []

    while True:
        #START here!!!
        image, point_cloud, depth = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)

        rgb_img = image.get_data()[:,:,:3]

        cable_cnt, cable_mask_hollow  = get_cable(img_path=None, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize, img=rgb_img)
        
        cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
        cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
        cv.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
        cable_skeleton = skeletonize(cable_mask_binary)
        plt.imshow(cable_skeleton)
        plt.show()
        cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
        plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
        plt.imshow(rgb_img)
        plt.show()

        # need to figure which cable endpoint is in the channel and the closeseted channel endpoint and sort from there
        # check which cable endpoint is in the channel then pick the channel endpoint that is closest to that cable endpoint
        # then sort from there
        cable_endpoint_in, cable_endpoint_out = get_cable_endpoint_in_channel(cable_endpoints, aligned_channel_mask)
        # cable_endpoint_in = cable_endpoints[0]
        channel_endpoint_in, channel_endpoint_out = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
        sorted_channel_pts = sort_skeleton_pts(channel_skeleton, channel_endpoint_in)
        channel_endpoint_in = find_closest_point(sorted_channel_pts, cable_endpoint_in)
        sorted_channel_pts = sorted_channel_pts[sorted_channel_pts.index(channel_endpoint_in):]
        sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)



        # WARNING: seems like tthe pointcloud and depth are expecting the x and y to be flipped!!!!
        swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]

        # filter out points that are invalid in the point cloud

        # NOTICE THAT THTESE ARE NOW SWAPPED!!!
        sorted_channel_pts = filter_points(swapped_sorted_channel_pts, point_cloud) 
        # these are aslo now swapped!
        channel_endpoint_in = sorted_channel_pts[0]
        channel_endpoint_out = sorted_channel_pts[-1]
        sorted_cable_pts = filter_points(swapped_sorted_cable_pts, point_cloud)
        # these are aslo now swapped!
        cable_endpoint_in = sorted_cable_pts[0]
        cable_endpoint_out = sorted_cable_pts[-1]


        # IDEA: update the channel startpoint to be the closest point on the channel skeleton to the cable endpoint
        # then actually delete the indices before that point from sorted_channel_pts
        channel_endpoint_in = find_closest_point(sorted_channel_pts, cable_endpoint_in)
        sorted_channel_pts = sorted_channel_pts[sorted_channel_pts.index(channel_endpoint_in):]


        # noticie the cahnge in plotting cause we've swapped the x and y!!!
        plt.scatter(x=channel_endpoint_in[0], y=channel_endpoint_in[1], c='r')
        plt.scatter(x=cable_endpoint_in[0], y=cable_endpoint_in[1], c='b')
        plt.imshow(rgb_img)
        plt.show()

        if not push_idx and not pick_idx:
            push_idx = [(i * (len(sorted_channel_pts)-1)//(TOTAL_PICK_PLACE)) for i in range(0,TOTAL_PICK_PLACE+1)]
            pick_place_pixels_per_waypoint = len(sorted_channel_pts)//(TOTAL_PICK_PLACE)

            # use the min length to avoid pulling the rope too far
            min_length = min(len(sorted_channel_pts), len(sorted_cable_pts))
            # dont want the first idx cause that's 0, i.e. our starting point
            place_idx = [i for i in range(0,min_length, pick_place_pixels_per_waypoint)][1:]
            # place_idx = [(i * (len(sorted_cable_pts)//(TOTAL_PICK_PLACE)))-1 for i in range(0,TOTAL_PICK_PLACE+1)][1:]
            pick_idx = place_idx.copy()
        plt.scatter(x=[sorted_channel_pts[i][0] for i in push_idx], y=[sorted_channel_pts[i][1] for i in push_idx], c='m')
        plt.scatter(x=[sorted_cable_pts[i][0] for i in pick_idx], y=[sorted_cable_pts[i][1] for i in pick_idx], c='g')
        plt.scatter(x=[sorted_channel_pts[i][0] for i in place_idx], y=[sorted_channel_pts[i][1] for i in place_idx], c='y')
        plt.title('pick and push places')
        plt.imshow(rgb_img)
        plt.show()

        # just put the endpoint to the end no matter what


        # WARNING: seems like tthe pointcloud and depth are expecting (x,y) aka (col, row)!!!!!!!!!!!
        if matched_template == 'curved' or matched_template == 'straight':
        # picks the endpoint and places it at the end and pushes down
            pick_pt = sorted_cable_pts[pick_idx.pop()]
            transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(pick_pt, sorted_cable_pts, 20)
            # breakpoint()
            robot.grab(transformed_rope_end, rotation)
        
            place_pt = sorted_channel_pts[place_idx.pop()]
            transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(place_pt, sorted_channel_pts, 20, 1/8)
            # accounts for height of template
            transformed_channel_end[2] += TEMPLATE_HEIGHT[matched_template]
            transformed_channel_offset[2] += TEMPLATE_HEIGHT[matched_template]
            robot.move_to_channel(transformed_channel_end, transformed_channel_offset, rotation)
            robot.open_grippers()
            robot.push(transformed_channel_end, rotation)
        
        # FOR STRAIGHT GO TO END AND THEN PUSH ALL THE WAY ACROSS        
        if matched_template == 'straight':
            pushdown(sorted_channel_pts, push_idx, MODE, rgb_img)
            exit()
        # FOR CURVED PICK TO A POINT AND IMMEDIATELY PUSH DOWN
        elif matched_template == 'curved':
            if count == 1:
                # picks in the mdidle and then pushes it down
                cable_midway = sorted_cable_pts[len(sorted_cable_pts)//2]
                transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(cable_midway, sorted_cable_pts, 20)

                robot.grab(transformed_rope_end, rotation)

                channel_midway = sorted_cable_pts[len(sorted_channel_pts)//2]
                transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(channel_midway, sorted_channel_pts, 30, 1/8)
                transformed_channel_end[2] += TEMPLATE_HEIGHT[matched_template]
                transformed_channel_offset[2] += TEMPLATE_HEIGHT[matched_template]
                robot.move_to_channel(transformed_channel_end, transformed_channel_offset, rotation)

                robot.push(transformed_channel_end, rotation)
            elif count == 2:
                pushdown(sorted_channel_pts, push_idx, MODE)
                exit()
            count += 1
            continue
        
        # do unidirectional
        elif matched_template == 'trapezoid':
            if pick_idx == []:
                pushdown(sorted_channel_pts, push_idx, MODE)
                exit()
























        continue




        # image = cv2.imread("/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_channel_error355.png")
        # image = img = cv2.imread("/home/r2d2/Robot-Cable-Insertion_TRI-Demo/colors/color6.png")
        try:
            # breakpoint()
            rope_end1,rope_end2, sorted_rope_pixels, total_rope_pixels = rope_mask_info(image)
        except Exception as e:
            print("no rope found")
            img = image.get_data()[:,:,:3]
            number = random.randint(0,1000)
            cv2.imwrite(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_rope_error{number}.png", img)

            # sys.exit()
        # img = image.get_data()[:,:,:3]
        # breakpoint()

        # cv2.imwrite("/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask.png", img)
        #find channel
        try:
            # channel_end1,channel_end2,total_channel_pixels, skeleton_channel_img = channel_mask_info(image)
            breakpoint()
            channel_mask = pixel_sam.get_mask()
            skeleton_channel_img = skeletonize_img(channel_mask)
            total_channel_pixels, channel_endpoints = find_length_and_endpoints(skeleton_channel_img) 
            channel_end1, channel_end2 = channel_endpoints[0], channel_endpoints[-1]
            sorted_channel_pixels = get_channel_sorted_list(skeleton_channel_img, channel_end1)
            channel_end1, channel_end2 = sorted_channel_pixels[0], sorted_channel_pixels[-1]
        except Exception as e:
            print("no channel found")
            print(e)
            img = image.get_data()[:,:,:3]
            number = random.randint(0,1000)
            cv2.imwrite(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_channel_error{number}.png", img)

            sys.exit()
        #calculate which rope end is the one you grabbed!
        # closest_rope_endpoint, closest_channel_endpoint,farthest_rope_endpoint, farthest_channel_endpoint = calculate_farthest_point(np.array(rope_end1), np.array(rope_end2), np.array(channel_end1), np.array(channel_end2))
        farthest_channel_endpoint = channel_end2 if channel_end2[0] < channel_end1[0] else channel_end1
        closest_channel_endpoint = channel_end1 if np.array_equal(farthest_channel_endpoint, channel_end2) else channel_end2
        closest_rope_endpoint = rope_end2 if np.linalg.norm(np.asarray(rope_end2) - np.asarray(closest_channel_endpoint)) < np.linalg.norm(np.asarray(rope_end1) - np.asarray(closest_channel_endpoint)) else rope_end1
        farthest_rope_endpoint = rope_end1 if np.array_equal(closest_rope_endpoint, rope_end2) else rope_end2
        print("rope_end1, rope_end2, channel_end1, channel_end2", rope_end1, rope_end2, channel_end1, channel_end2)
        print("closest_rope_endpoint, closest_channel_endpoint,farthest_rope_endpoint, farthest_channel_endpoint", closest_rope_endpoint, closest_channel_endpoint,farthest_rope_endpoint, farthest_channel_endpoint)
        sorted_channel_pixels = get_channel_sorted_list(skeleton_channel_img, closest_channel_endpoint)

        transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(farthest_rope_endpoint, sorted_rope_pixels, 20)

        robot.grab(transformed_rope_end, rotation)
        # lst = [farthest_channel_endpoint, closest_channel_endpoint, sorted_channel_pixels[0], sorted_channel_pixels[-1]]
        # plt.scatter(x=[i[0] for i in lst], y=[i[1] for i in lst], c=['r','b','g','y'])
        # plt.imshow(skeleton_channel_img)
        # plt.show()
        transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(farthest_channel_endpoint, sorted_channel_pixels, 30, 1/8)
        try:
            robot.move_to_channel(transformed_channel_end, transformed_channel_offset, rotation)
        except:
            breakpoint()
            sys.exit()
        time.sleep(1)
        robot.open_grippers()
        robot.go_home()


        #push rope
        n = 9
        parts = [i * (len(sorted_channel_pixels)//(n)) for i in range(0,n+1)]
        print("total_channel_pixels and len(sorted_channel_pixels)", total_channel_pixels,  len(sorted_channel_pixels))
        print(len(parts),parts)
        plt.scatter(x=[sorted_channel_pixels[part][0] for part in parts], y=[sorted_channel_pixels[part][1] for part in parts])
        plt.imshow(skeleton_channel_img)
        plt.show()

        # hardcoded push-slide procedure
        err, closest_channel_endpoint_pc = point_cloud.get_value(closest_channel_endpoint[0], closest_channel_endpoint[1])
        closest_channel_endpoint_pc=closest_channel_endpoint_pc[:3]
        closest_channel_endpoint_trasnform = transform_from_camera_frame(closest_channel_endpoint_pc, left_cam_calibration_matrix)
        err, farthest_channel_endpoint_pc = point_cloud.get_value(farthest_channel_endpoint[0], farthest_channel_endpoint[1])
        farthest_channel_endpoint_pc = farthest_channel_endpoint_pc[:3]
        farthest_channel_endpoint_trasnform = transform_from_camera_frame(farthest_channel_endpoint_pc, left_cam_calibration_matrix)

        starting_x = closest_channel_endpoint_trasnform[0]-0.005; starting_y = closest_channel_endpoint_trasnform[1]
        ending_x = farthest_channel_endpoint_trasnform[0]+0.01; ending_y = farthest_channel_endpoint_trasnform[1]
        fraction_start = [1/8, 3/8, 5/8]
        fraction_end = [3/8, 5/8, 1]
        fraction_start_x = np.interp(fraction_start, [0,1], [starting_x, ending_x])
        fraction_start_y = np.interp(fraction_start, [0,1], [starting_y, ending_y])
        fraction_end_x = np.interp(fraction_end, [0,1], [starting_x, ending_x])
        fraction_end_y = np.interp(fraction_end, [0,1], [starting_y, ending_y])
        fraction_z = [(closest_channel_endpoint_trasnform[2] + farthest_channel_endpoint_trasnform[2])/2] * len(fraction_start)
        fraction_start_points = np.vstack((fraction_start_x, fraction_start_y, fraction_z)).T # (n,3)
        fraction_end_points = np.vstack((fraction_end_x, fraction_end_y, fraction_z)).T # (n,3)


        
        rotation[2] -= 1/2*np.pi
        robot.close_grippers()
        time.sleep(1)

        #First push_slide
        robot.push_slide(fraction_start_points[0], fraction_end_points[0], rotation, start_offset=0.0, end_offset=-0.02,duration=2)
        time.sleep(1)

        #Second push_slide

        robot.push_slide(fraction_start_points[1], fraction_end_points[1], rotation, start_offset=-0.025, end_offset=-0.025,duration=1)
        time.sleep(1)

        #Third push_slide
        robot.push_slide(fraction_start_points[2], fraction_end_points[2], rotation, start_offset=-0.025, end_offset=-0.015,duration=1)
        time.sleep(1)



        # for i in range(len(fraction_start_points)):
        #     try:
        #         robot.push_slide(fraction_start_points[i], fraction_end_points[i], rotation)
        #     except:
        #         breakpoint()
        #         sys.exit()
        #     time.sleep(1)
        robot.open_grippers()
        robot.go_home()

        # start_end_parts = [(1,3),(3,5),(5,7)]
        # rotation[2] -= 1/2*np.pi
        # for start, end in start_end_parts:
        #     err, push_pc_value_start = point_cloud.get_value(sorted_channel_pixels[parts[start]][0], sorted_channel_pixels[parts[start]][1])
        #     # push_pc_value_start = check_valid_pointcloud(sorted_channel_pixels[parts[start]], sorted_channel_pixels, point_cloud)
        #     err, push_pc_value_end = point_cloud.get_value(sorted_channel_pixels[parts[end]][0], sorted_channel_pixels[parts[end]][1])
        #     # push_pc_value_end = check_valid_pointcloud(sorted_channel_pixels[parts[end]], sorted_channel_pixels, point_cloud)
        #     push_pc_value_start = push_pc_value_start[:3]
        #     push_pc_value_end = push_pc_value_end[:3]
        #     push_transform_pt_start = transform_from_camera_frame(push_pc_value_start, left_cam_calibration_matrix)
        #     push_transform_pt_end = transform_from_camera_frame(push_pc_value_end, left_cam_calibration_matrix)
        #     try:
        #         robot.push_slide(push_transform_pt_start, push_transform_pt_end ,rotation)
        #     except:
        #         breakpoint()
        #         sys.exit()
        #     time.sleep(1)
        # sorted_channel_pixels



        # image, point_cloud = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud)
        # new_rope_end1,new_rope_end2, new_sorted_rope_pixels, new_total_rope_pixels = rope_mask_info(image)
        # new_channel_end1,new_channel_end2, new_sorted_channel_pixels, new_total_channel_pixels = channel_mask_info(image)
        # transformed_rope_end, rotation = find_transformed_point_and_rotation_rope(new_rope_end1, new_sorted_rope_pixels, 20)
        # rotation[2] += 1/2*np.pi
        # #find 6 points along the rope to push
        # n=6
        # parts = [i * (new_total_rope_pixels // (n - 1)) for i in range(n)]
        # new_sorted_rope_pixels.reverse()
        # for i in range(n):
        #     err, end_pc_value = point_cloud.get_value(new_sorted_rope_pixels[parts[i]][0], new_sorted_rope_pixels[parts[i]][1])
        #     end_pc_value = end_pc_value[:3]
        #     trans_point = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix)
        #     # err, final_pc_value = point_cloud.get_value(trans_point[0], trans_point[1])
        #     try:
        #         robot.push(trans_point, rotation)
        #     except:
        #         breakpoint()
        #     time.sleep(1)




        #close the camera
    robot.go_home()
    side_cam.close()