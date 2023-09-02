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
import random
import pixel_sam
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


def transform_from_camera_frame(original_point, calibration_matrix):
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    transformed_point = np.dot(original_point, rotation_matrix.T) + cam_pose
    transformed_point[2] +=0.165 #account for end effector
    return transformed_point

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
    err, end_pc_value = point_cloud.get_value(farthest_endpoint[0], farthest_endpoint[1])
    try:
        err, next_pc_value = point_cloud.get_value(next_point[0], next_point[1])
    except:
        print("no point cloud value")
        breakpoint()
        sys.exit()
    end_pc_value = end_pc_value[:3]
    next_pc_value = next_pc_value[:3]
    transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix)
    transformed_next = transform_from_camera_frame(next_pc_value, left_cam_calibration_matrix)
    rotation = get_rotation(transformed_end, transformed_next)
    return transformed_end, rotation

def find_transformed_point_and_rotation_channel(farthest_endpoint, sorted_pixels, n, ratio):
    next_point = find_nth_nearest_point(farthest_endpoint, sorted_pixels, n)
    offset_point = find_nth_nearest_point(farthest_endpoint, sorted_pixels, int(len(sorted_pixels)*ratio))
    try:
        err, end_pc_value = point_cloud.get_value(farthest_endpoint[0], farthest_endpoint[1])
    except:
        print("no point cloud value")
        breakpoint()
        sys.exit()
    err, next_pc_value = point_cloud.get_value(next_point[0], next_point[1])
    err, offset_pc_value = point_cloud.get_value(offset_point[0], offset_point[1])

    end_pc_value = end_pc_value[:3]
    next_pc_value = next_pc_value[:3]
    offset_pc_value = offset_pc_value[:3]
    transformed_end = transform_from_camera_frame(end_pc_value, left_cam_calibration_matrix)
    transformed_next = transform_from_camera_frame(next_pc_value, left_cam_calibration_matrix)
    transformed_offset = transform_from_camera_frame(offset_pc_value, left_cam_calibration_matrix)
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
        transformed_point = transform_from_camera_frame(pc_value, left_cam_calibration_matrix)
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

def find_nth_nearest_point(endpoint, sorted_points, n):
    # if endpoint == sorted_points[0]:
    #     return sorted_points[n - 1]
    # elif endpoint == sorted_points[-1]:
    #     return sorted_points[-n]
    if np.linalg.norm(np.array(endpoint)-np.array(sorted_points[0])) < np.linalg.norm(np.array(endpoint)-np.array(sorted_points[-1])):
        return sorted_points[n - 1]
    else:
        return sorted_points[-n]


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



if __name__ == "__main__":


    left_cam_calibration_matrix = calibration_matrices[varied_camera_2_id + "_left"]


    r = FrankaFranka()
    r.go_home()


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
    image = sl.Mat()
    point_cloud = sl.Mat()
    depth = sl.Mat()

    #START here!!!
    image, point_cloud, depth = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)

    img = image.get_data()[:,:,:3]
    number = random.randint(0,1000)
    cv2.imwrite(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_rope_error{number}.png", img)

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

    r.grab(transformed_rope_end, rotation)
    # lst = [farthest_channel_endpoint, closest_channel_endpoint, sorted_channel_pixels[0], sorted_channel_pixels[-1]]
    # plt.scatter(x=[i[0] for i in lst], y=[i[1] for i in lst], c=['r','b','g','y'])
    # plt.imshow(skeleton_channel_img)
    # plt.show()
    transformed_channel_end, transformed_channel_next, transformed_channel_offset, rotation = find_transformed_point_and_rotation_channel(farthest_channel_endpoint, sorted_channel_pixels, 30, 1/8)
    try:
        r.move_to_channel(transformed_channel_end, transformed_channel_offset, rotation)
    except:
        breakpoint()
        sys.exit()
    time.sleep(1)
    r.open_grippers()
    r.go_home()


    #push rope
    # n = 9
    # parts = [i * (len(sorted_channel_pixels)//(n)) for i in range(0,n+1)]
    # print("total_channel_pixels and len(sorted_channel_pixels)", total_channel_pixels,  len(sorted_channel_pixels))
    # print(len(parts),parts)
    # plt.scatter(x=[sorted_channel_pixels[part][0] for part in parts], y=[sorted_channel_pixels[part][1] for part in parts])
    # plt.imshow(skeleton_channel_img)
    # plt.show()

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
    r.close_grippers()
    time.sleep(1)

    #First push_slide
    r.push_slide(fraction_start_points[0], fraction_end_points[0], rotation, start_offset=0.0, end_offset=-0.02,duration=2)
    time.sleep(1)

    #Second push_slide

    r.push_slide(fraction_start_points[1], fraction_end_points[1], rotation, start_offset=-0.025, end_offset=-0.025,duration=1)
    time.sleep(1)

    #Third push_slide
    r.push_slide(fraction_start_points[2], fraction_end_points[2], rotation, start_offset=-0.025, end_offset=-0.015,duration=1)
    time.sleep(1)



    # for i in range(len(fraction_start_points)):
    #     try:
    #         r.push_slide(fraction_start_points[i], fraction_end_points[i], rotation)
    #     except:
    #         breakpoint()
    #         sys.exit()
    #     time.sleep(1)
    r.open_grippers()
    r.go_home()

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
    #         r.push_slide(push_transform_pt_start, push_transform_pt_end ,rotation)
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
    #         r.push(trans_point, rotation)
    #     except:
    #         breakpoint()
    #     time.sleep(1)




    #close the camera
    side_cam.close()