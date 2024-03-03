
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics, Point, PointCloud
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import time
import os
import sys
import traceback
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import pdb
from queue import Empty
from multiprocessing import Queue, Process
from random import shuffle
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from plantcv import plantcv as pcv
import pyzed.sl as sl
from resources import CROP_REGION

def setup_zed_camera(cam_id):
    side_cam = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    # init_params.camera_fps = 30
    init_params.set_from_serial_number(cam_id) #overhead camera
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
    return side_cam, runtime_parameters, image, point_cloud, depth

def grab_zed_mat(side_cam,runtime_parameters, image, point_cloud, depth):
    if side_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        side_cam.retrieve_image(image, sl.VIEW.LEFT)
        side_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        side_cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return image, point_cloud, depth

def get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth):

    image, _, _ = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)
    color_img = image.get_data()[:,:,:3]
    rgb_img = color_img
    # rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    return rgb_img 

def get_corners2(rgb_img, channel_cnt):
    channel_cnt = channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    p = cv2.arcLength(channel_cnt, True) # cnt is the rect Contours
    appr = cv2.approxPolyDP(channel_cnt, 0.02*p, True) # appr contains the 4 points

    appr = sorted(appr, key=lambda c: c[0][0])

    #pa = top lef point
    #pb = bottom left point
    #pc = top right point
    #pd = bottom right point

    pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
    pc, pd = sorted(appr[2:4], key=lambda c: c[0][1])
    # plt.imshow(cv2.drawContours(rgb_img.copy(), [channel_cnt], -1, 255, 3))
    # plt.scatter(pa[0,0], pa[0,1], c='r')
    # plt.scatter(pb[0,0], pb[0,1], c='b')
    # plt.scatter(pc[0,0], pc[0,1], c='g')
    # plt.scatter(pd[0,0], pd[0,1], c='k')
    # plt.show()

    return [[pa[0,1], pa[0,0]], [pb[0,1], pb[0,0]], [pc[0,1], pc[0,0]], [pd[0,1], pd[0,0]]] 

def get_corners(skeleton_img):
    # Convert to grayscale

    # 3/1/24 added in type conversion and stacking for it to work with the cv2 functions 
    skeleton_img = skeleton_img.astype(np.uint8)
    skeleton_img = skeleton_img*255
    # pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skeleton_img, size=40)
    
    skeleton_img = cv2.merge((skeleton_img, skeleton_img, skeleton_img))
    gray = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold if your mask isn't binary
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find corners
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=100)

    # Convert the corners to integers
    corners = np.int0(corners)

    # image = skeleton_img.copy()
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(image, (x, y), 1, 255, -1)
    # plt.imshow(image)
    # plt.show()

    corners = corners.reshape((4,2))
    corners = [[corners[i][1], corners[i][0]] for i in range(len(corners))]
    return corners

def sort_skeleton_pts(skeleton_img, endpoint):
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    start_pt = endpoint
    q = [start_pt]
    sorted_pts = []
    # dfs from start_pt which is one of our waypoints 
    while len(q) > 0:
        next_loc = q.pop(0)
        visited.add(next_loc)
        sorted_pts.append(next_loc)
        counter = 0
        for n in NEIGHS:
            test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
            if (test_loc in visited):
                continue
            if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                    or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                continue
            if skeleton_img[test_loc[0]][test_loc[1]].sum() > 0:
                q.append(test_loc)
    # plt.scatter(x = [j[1] for j in sorted_pts], y=[i[0] for i in sorted_pts],c='w')
    # plt.scatter(x=[sorted_pts[-1][1], sorted_pts[0][1]], y=[sorted_pts[-1][0], sorted_pts[0][0]], c='g')
    # plt.scatter(x=[endpoint[1]], y=[endpoint[0]], c='r')
    # plt.imshow(skeleton_img)
    # plt.show()
    return sorted_pts

def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)


def determine_length(object_points, endpoint_list, waypoints_list, g):
    visited = set()
    distances_list = dict()
    waypoint_found = dict()
    waypoints_sorted = [] 
    start_point = [int(endpoint_list[0][0]), int(endpoint_list[0][1])]
    end_point = [int(endpoint_list[1][0]), int(endpoint_list[1][1])]
    waypoints_sorted.append(start_point)
    # import pdb
    # pdb.set_trace()
    start_point_adj = g.ij_to_point(start_point).data
    end_point_adj = g.ij_to_point(end_point).data
    q = [start_point_adj]
    test_loc_temp= [start_point_adj[0], start_point_adj[1], start_point_adj[2]] 
    NEIGHS = [-1,  1]
    counter = 0
    copy_waypoints = np.array([g.ij_to_point(waypoint).data for waypoint in waypoints_list])
    copy_waypoints_dict = dict([((g.ij_to_point(waypoint).data[0], g.ij_to_point(waypoint).data[1],\
                                  g.ij_to_point(waypoint).data[2]), waypoint) for waypoint in waypoints_list])

    while len(q) > 0:  
        next_loc = q.pop()
        # if next_loc == start_point_adj:
        if (next_loc == start_point_adj).all():
            visited.add((start_point_adj[0], start_point_adj[1], start_point_adj[2]))
        else:
            visited.add(tuple(start_point_adj))
        #return index of next_point in object_points
        index = check_index(object_points.data, next_loc, ind = True)
        require = False
        min_dist = float('inf')
        for n in NEIGHS:
            test_loc = object_points.data.T[index+n]
            # print("Test_loc:", test_loc)
            # print("End Point Adj:", end_point_adj)
            if (test_loc == end_point_adj).all():
                break
            else: 
                if np.any(np.all(test_loc == copy_waypoints, axis=1)):
                    if (tuple(test_loc) in visited):
                        continue
                    require = True
                    diff = test_loc_temp-test_loc
                    dist = diff.dot(diff)
                    waypoint_found[dist] = test_loc
                    if dist < min_dist:
                        min_dist = dist
                else:
                    q.append(test_loc)
        counter += 1
        if require == True:  
            distances_list[waypoint_found[min_dist]]= min_dist
            test_loc_temp = waypoint_found[min_dist]
            q.append(test_loc_temp)
            waypoints_sorted.append(copy_waypoints_dict[test_loc_temp])
        # if counter % 1000 == 0:
        #     print("counter:", counter)
    return sum(distances_list.values()), waypoints_sorted

def determine_length_pixels(endpoint_list,waypoints_sorted):
    pixel_distance = waypoints_sorted[0] - endpoint_list[0] 
    total_dist = pixel_distance.dot(pixel_distance)
    for i in range(1, len(waypoints_sorted)):
        pixel_distance = waypoints_sorted[i]- waypoints_sorted[i-1]
        total_dist += pixel_distance.dot(pixel_distance)
    pixel_distance = endpoint_list[1]- waypoints_sorted[len(waypoints_sorted)]
    total_dist += pixel_distance.dot(pixel_distance)
    return total_dist
    
    
def dfs_resample(distance,sample_no, segmented_cloud, iface, transformed_rope_cloud):
    sampling_index = distance/sample_no
    new_transf = iface.T_PHOXI_BASE.inverse()
    transformed_segmented_cloud = new_transf.apply(segmented_cloud)
    cloud_image = iface.cam.intrinsics.project_to_image(
        transformed_rope_cloud, round_px=False)
    kernel = np.ones((6, 6), np.uint8)
    image = cv2.erode(image, kernel)

def evenly_sample_points_dist(points, num_points, distance):
    selected_points = []
    # getting the first endpoints in
    selected_points.append(points[0])
    selected_points.append(points[-1])
    while len(selected_points) < num_points:
        dists = np.array([np.linalg.norm(p-sp) for sp in selected_points for p in points])
        dists = dists.reshape(len(selected_points), len(points))
        num_close_pts = np.sum(dists <= distance, axis = 0)
        next_point_idx = np.argmax(num_close_pts)
        next_point = points[next_point_idx]
        selected_points.append(next_point)
        points = np.delete(points, next_point_idx, axis = 0)
    return selected_points


def evenly_sample_points(points,num_points):
    selected_points = []
    # getting the first endpoints in
    selected_points.append(points[0])
    selected_points.append(points[-1])
    while len(selected_points) < num_points:
        dists = np.array([min([np.linalg.norm(p-sp)]) for sp in selected_points] for p in points)
        next_point_idx = np.argmax(dists)
        next_point = points[next_point_idx]
        selected_points.append(next_point)
    return selected_points

def make_bounding_boxes(img):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
    
    # save resulting image
    # cv2.imwrite('two_blobs_result.jpg',result)      

    # show thresh and result  
    # plt.title("bounding box for detection")  
    # plt.imshow(result, interpolation="nearest")
    # plt.show()

    return result


def skeletonize_img(img):
    img[img[:,:,2] < 250] = 0
    img[img[:,:,2] > 250] = 1
    img = img[:,:,0]
    gray = img*255
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    # blurred_image = gaussian_filter(image, sigma=1)

    # dilate the image to just eliminate risk of holes causes severances in the skeleton
    dilated_img = image.copy()
    kernel = np.ones((5,5), np.uint8)
    cv2.dilate(image, kernel, dst=dilated_img, iterations=1)
    # perform skeletonization
    skeleton = skeletonize(dilated_img)

    #find candidates who 

    # display results
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
    #                         sharex=True, sharey=True)

    # ax = axes.ravel()

    # ax[0].imshow(image, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('original', fontsize=20)

    # ax[1].imshow(skeleton, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('skeleton', fontsize=20)

    # fig.tight_layout()
    # plt.show()

    return skeleton

'''
def skeletonize_img(img):

    gray = img*255
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    # blurred_image = gaussian_filter(image, sigma=1)

    # dilate the image to just eliminate risk of holes causes severances in the skeleton
    dilated_img = image.copy()
    kernel = np.ones((3,3), np.uint8)
    cv2.dilate(image, kernel, dst=dilated_img, iterations=1)
    # perform skeletonization
    skeleton = skeletonize(dilated_img)

    #find candidates who 

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    return skeleton
'''

def prune_branches(branch_endpoints, skeleton_img):
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    # all of the pixels that we passed by
    past_pixels_lst = []
    save_pixels = []
    for branch_endpoint in branch_endpoints:
        if branch_endpoint in save_pixels:
            continue
        q = [branch_endpoint]
        while len(q) > 0:
            next_loc = q.pop()
            visited.add(next_loc)
            count = 0
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    past_pixels_lst.append(test_loc)
                    count += 1
                    if test_loc not in visited:
                        q.append(test_loc)
            # means we've reached the main branch
            if count >= 3:
                # breakpoint()
                # need to remove these cause they're probably actually a part of the main branch
                # need to also figure out a way to remove branch_endpoints from the q if they

# THIS IS IMPORTANT READ THIS LINE!!!!! #################################################################


            # need to also figure out a way to remove branch_endpoints from the q if they are a part of the 3/4 neighbors that our count found
            # thinking of having those points be in a remove list and then doing .remove from the q instead of trying to eliminate the last x cause the same point will appear twice
            # could also have 

            
# READ THE PARAGRAPH IN BETWEEN!!!!! ########################################################################################################
                # save_pixels.extend()
                q = q[:-count]
                # saved_pixels = past_pixels_lst[count:]
                # plt.scatter(x=[i[1] for i in past_pixels_lst], y=[i[0] for i in past_pixels_lst], c='g')
                past_pixels_lst = past_pixels_lst[:-count]
                for past_pixel in past_pixels_lst:
                    skeleton_img[past_pixel[0]][past_pixel[1]] = False
                # # plt.scatter(x=[i[1] for i in saved_pixels], y=[i[0] for i in saved_pixels], c='g')
                # plt.scatter(x=[i[1] for i in past_pixels_lst], y=[i[0] for i in past_pixels_lst], c='b')
                # plt.scatter(x=[i[1] for i in q], y=[i[0] for i in q], c='r')
                # plt.scatter(x=[i[1] for i in branch_endpoints], y=[i[0] for i in branch_endpoints], c='y')
                # plt.scatter(x=branch_endpoint[1], y=branch_endpoint[0], c='w')
                # plt.scatter(x=next_loc[1], y=next_loc[0], c='m')
                # plt.imshow(skeleton_img)
                # plt.show()
                break
        # sets all of those passed pixels to False
    # breakpoint()
    # plt.title("pruned skeleton")
    # plt.imshow(skeleton_img)
    # plt.show()
    return skeleton_img

def find_length_and_endpoints(skeleton_img):
    # breakpoint()
    # instead of doing DFS just do BFS and the last 2 points to have which end up having no non-visited neighbor 

    #### IDEA: do DFS but have a left and right DFS with distances for one being negative and the other being positive 
    # done because skeleton from sam mask has 3 channels for some reason
    # only valid channel is the green one
    if len(skeleton_img.shape) == 3:
        skeleton_img = skeleton_img[:,:,1]
        skeleton_img = skeleton_img.astype(bool)
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))

    if nonzero_pts is None:
        nonzero_pts = [[[0,0]]]
    total_length = len(nonzero_pts)
    # pdb.set_trace()
    start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])
    # run dfs from this start_pt, when we encounter a point with no more non-visited neighbors that is an endpoint
    endpoints = []
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    q = [start_pt]
    dist_q = [0]
    # tells us if the first thing we look at is actually an endpoint
    initial_endpoint = False
    # carry out floodfill
    q = [start_pt]
    paths = [[]]
    # carry out floodfill
    IS_LOOP = False
    IS_LOOP_VISIT = False
    # ENTIRE_VISITED = [False] * total_length
    def dfs(q, dist_q, visited, paths, start_pixel, increment_amt):
        '''
        q: queue with next point on skeleton for one direction
        dist_q: queue with distance from start point to next point for one direction
        visited: queue with visited points for only one direction
        increment_amt: counter that indicates direction +/- 1
        '''

        # is_loop = ENTIRE_VISITED[start_pixel + increment_amt*len(visited)]
        # if is_loop:
        #     return is_loop


        while len(q) > 0:
            next_loc = q.pop()
            distance = dist_q.pop()
            # paths[-1].append(next_loc)
            # if next_loc in visited:
            #     return True # we have a loop
            # print("distance: ",distance)
            visited.add(next_loc)
            counter = 0
            count_visited = []
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                # print("count_visited ", count_visited)
                if (test_loc in visited):
                    count_visited.append(test_loc)
                    if len(count_visited) >= 3:
                        pass
                        # plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
                        # plt.scatter(x = [j[1] for j in count_visited], y=[i[0] for i in count_visited],c='m')
                        # plt.scatter(x = [j[1] for j in visited], y=[i[0] for i in visited],c='r')
                        # plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
                        # plt.scatter(x=next_loc[1], y=next_loc[0], c='b')
                        # plt.imshow(skeleton_img)
                        # plt.show()
                        # return True
                    continue
                if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                    continue
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    counter += 1
                    #length_checker += 1
                    q.append(test_loc)
                    dist_q.append(distance+increment_amt)
            # this means we haven't added anyone else to the q so we "should" be at an endpoint
            if counter == 0:
                endpoints.append([next_loc, distance])
            # if next_loc == start_pt and counter == 1:
            #     endpoints.append([next_loc, distance])
            #     initial_endpoint = True
        return False # we don't have a loop
    counter = 0
    length_checker = 0
    increment_amt = 1
    visited = set([start_pt])
    for n in NEIGHS:
        test_loc = (start_pt[0]+n[0], start_pt[1]+n[1])
        # one of the neighbors is valued at one so we can dfs across it
        if skeleton_img[test_loc[0]][test_loc[1]] == True:
            counter += 1
            q = [test_loc]
            dist_q = [0]
            start_pixel = test_loc[0]*len(skeleton_img[0]) + test_loc[1]
            # breakpoint()
            IS_LOOP = dfs(q, dist_q, visited, paths, start_pixel, increment_amt)
            # if IS_LOOP:
            #     break
            # the first time our distance will be incrementing but the second time
            # , i.e. when dfs'ing the opposite direction our distance will be negative to differentiate both paths
            increment_amt = -1
    # we only have one neighbor therefore we must be an endpoint
    # only works for the start point so we need to check
    if counter == 1:
        distance = 0
        endpoints.insert(0, [start_pt, distance])
        initial_endpoint = True

    # pdb.set_trace()
    final_endpoints = []
    # largest = second_largest = None
    # for pt, distance in endpoints:
    #     if largest is None or distance > endpoints[largest][1]:
    #         second_largest = largest
    #         largest = endpoints.index([pt, distance])
    #     elif second_largest is None or distance > endpoints[second_largest][1]:
    #         second_largest = endpoints.index([pt, distance])
    # if initial_endpoint:
    #     final_endpoints = [endpoints[0][0], endpoints[largest][0]]
    # else:
    #     final_endpoints = [endpoints[second_largest][0], endpoints[largest][0]]
    
    largest_pos = largest_neg = None
    print("IS LOOP:", IS_LOOP)
    print("IS LOOP VISIT: ", IS_LOOP_VISIT)
    if IS_LOOP:
        final_endpoints = [endpoints[0][0]] # MAYBE RAND INDEX LATER
    else:
        for pt, distance in endpoints:
            if largest_pos is None or distance > endpoints[largest_pos][1]:
                largest_pos = endpoints.index([pt, distance])
            elif largest_neg is None or distance < endpoints[largest_neg][1]:
                largest_neg = endpoints.index([pt, distance])
        if initial_endpoint:
            final_endpoints = [endpoints[0][0], endpoints[largest_pos][0]]
        else:
            final_endpoints = [endpoints[largest_neg][0], endpoints[largest_pos][0]]
    # breakpoint()
    print("num endpoints = ", len(endpoints))
    print("endpoints are: ", endpoints)
    branch_endpoints = endpoints.copy()
    branch_endpoints = [x[0] for x in branch_endpoints]
    for final_endpoint in final_endpoints:
        branch_endpoints.remove(final_endpoint)
    pruned_skeleton = prune_branches(branch_endpoints, skeleton_img.copy())

    # plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    # plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    # plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    # plt.title("final endpoints")
    # plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    # plt.imshow(skeleton_img, interpolation="nearest")
    # plt.show() 

    print("the final endpoints are: ", final_endpoints)
    # breakpoint()
    # pdb.set_trace()
    # display results

    print("the total length is ", total_length)
    return total_length, final_endpoints

# Function to check the index of a  
def check_index(p_array, part, ind = False):
    newp_array = p_array.T
    # print("newp_array:", newp_array)
    p_shape = newp_array.shape
    check = np.where(newp_array == part)
    # print ("check[0]:", check[0])
    # print ("len(check[0]):", len(check[0]))
    # print ("part:", part)
    # print ("len(part):", len(part))
    if len(check[0]) != len(part):
        return False
    else:
        if ind == True:
            return check[0][0]
        return(np.all(np.isclose(check[0], check[0][0])))
    
def find_all_solns(compressed_map, max_edges):
    all_solns = []
    tightness = 0
    while (True):
        all_solns = []
        for r in range(len(compressed_map)):
            for c in range(len(compressed_map[r])):
                if (compressed_map[r][c] != 0):
                    curr_edges = 0
                    for add in range(1, 2):
                        if (compressed_map[min(len(compressed_map)-add, r+add)][c] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][c] == 0):
                            curr_edges += 1
                        if (compressed_map[r][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[r][max(0, c-add)] == 0):
                            curr_edges += 1
                        if (compressed_map[min(len(compressed_map)-add, r+add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[min(len(compressed_map)-add, r+add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][max(0, c-add)] == 0):
                            curr_edges += 1
                    if (max_edges-tightness <= curr_edges <= max_edges+tightness):
                        all_solns += [(c, r)]
        print("ALL SOLUTIONS TIGHTNESS "+str(tightness) + ": "+str(all_solns))
        if (len(all_solns) >= 2):
            min_x = 100000
            max_x = 0
            for soln in all_solns:
                if soln[0] < min_x:
                    min_x = soln[0]
                if soln[0] > max_x:
                    max_x = soln[0]
            if (max_x-min_x) > 2:
                break
        tightness += 1
    return all_solns

def find_endpoints_compressed_map(compressed_map, max_edges, compress_factor, place, place_2, new_di_data, DISPLAY=False):
    all_solns = find_all_solns(compressed_map, max_edges)
    min_dist = 100000000
    max_dist = 0
    min_all_solns = (0, 0)
    max_all_solns = (0, 0)
    for soln in all_solns:
        soln = soln *compress_factor
        print("soln[1]", soln[1])
        #soln[1] = soln[1] - int(compress_factor/2)
        #soln[0] = soln[0] - int(compress_factor/2)
        dist1 = np.linalg.norm(
            np.array([place[1]-soln[1], place[0]-soln[0]]))
        dist2 = np.linalg.norm(
            np.array([place_2[1]-soln[1], place_2[0]-soln[0]]))
        if dist1 > max_dist or dist2 > max_dist:
            max_dist = max(dist1, dist2)
            max_all_solns = soln
        if dist1 < min_dist or dist2 < min_dist:
            min_dist = min(dist1, dist2)
            min_all_solns = soln

    scaled_test_loc = [max_all_solns[0]*compress_factor,
                    max_all_solns[1]*compress_factor]
    scaled_test_loc_2 = []
    scaled_test_loc_2 = [min_all_solns[0]*compress_factor,
                        min_all_solns[1]*compress_factor]
    if (scaled_test_loc[0] != 0):
        scaled_test_loc[0] = scaled_test_loc[0] - int(compress_factor/2)
        print("scaled_test_loc[0]", scaled_test_loc[0])
    if (scaled_test_loc[1] != 0):
        scaled_test_loc[1] = scaled_test_loc[1] - int(compress_factor/2)
    if (scaled_test_loc_2[0] != 0):
        scaled_test_loc_2[0] = scaled_test_loc_2[0] - \
            int(compress_factor/2)
    if (scaled_test_loc_2[1] != 0):
        scaled_test_loc_2[1] = scaled_test_loc_2[1] - \
            int(compress_factor/2)
    if DISPLAY:
        plt.title("compressed_map")
        plt.imshow(compressed_map, interpolation="nearest")
        plt.show()
    min_dist = 10000
    min_dist_2 = 1000
    candidate_rope_loc = (0, 0)
    candidate_rope_loc_2 = (0, 0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (di_data[r][c][0] != 0):
                dist = np.linalg.norm(
                    np.array([r-scaled_test_loc[1], c-scaled_test_loc[0]]))
                if (dist < min_dist):
                    candidate_rope_loc = (c, r)
                    min_dist = dist
                dist_2 = np.linalg.norm(
                    np.array([r-scaled_test_loc_2[1], c-scaled_test_loc_2[0]]))
                if (dist_2 < min_dist_2):
                    candidate_rope_loc_2 = (c, r)
                    min_dist_2 = dist_2
    min_loc = candidate_rope_loc
    min_loc_2 = (0, 0)
    print("FITTED POINT: " + str(min_loc))
    min_loc_2 = candidate_rope_loc_2
    print("FITTED POINT OF OTHER END: " + str(min_loc_2))
    if DISPLAY:
        plt.title("new_di_data")
        plt.scatter(x=[min_loc[0], min_loc_2[0]], y = [min_loc[1], min_loc_2[1]], c='w')
        plt.scatter(x=[j[0]*compress_factor - int(compress_factor/2) for j in all_solns], y = [j[1]*compress_factor - int(compress_factor/2) for j in all_solns], c='b')
        plt.imshow(new_di_data, interpolation="nearest")
        plt.show()

    pick = min_loc
    print("This is Pick", pick)
    pick_2 = (0, 0)
    pick_2 = min_loc_2

    return pick, pick_2


def coord_to_point(coord, iface, img):
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    xind, yind = coord
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    new_point_data = np.array(
        [point.y, point.x, point.z])
    new_point = Point(
        new_point_data, frame=point.frame)
    return new_point

def inpaint_depth(three_mat_depth):
    three_mat_depth = three_mat_depth.copy()
    zero_pixels = np.where(three_mat_depth == 0)

    for i in range(len(zero_pixels[0])):
        x = zero_pixels[0][i]
        y = zero_pixels[1][i]

        patch_size = 5
        patch = three_mat_depth[x-patch_size:x+patch_size+1, y-patch_size:y+patch_size+1]
        patch_nonzero = patch[np.nonzero(patch)]
        avg_value = np.mean(patch)
        if 0 < x < len(three_mat_depth[0]) and 0 < y < len(three_mat_depth):
            three_mat_depth[x,y] = three_mat_depth[x-1,y]

    mask = (three_mat_depth ==0).astype(np.uint8)
    return cv2.inpaint(three_mat_depth, mask, 7, cv2.INPAINT_NS)

def crop_img(vert_end, horiz_end, img):
    one_mask_vert = np.ones((vert_end, 1032))
    zero_mask_vert = np.zeros((772 - vert_end, 1032))
    full_mask_vert = np.vstack((one_mask_vert, zero_mask_vert))
    one_mask_horiz = np.ones((772, 1032 - horiz_end))
    zero_mask_horiz = np.zeros((772, horiz_end))
    full_mask_horiz = np.hstack((zero_mask_horiz, one_mask_horiz))
    full_mask = np.logical_and(full_mask_vert, full_mask_horiz)
    cropped_img = full_mask * img
    # plt.title('cropped img')
    # plt.imshow(cropped_img)
    # plt.show()
    return cropped_img



