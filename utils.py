from interface_rws import Interface
from yumirws.yumi import YuMiArm, YuMi
from push import push_action_endpoints
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from grasp import Grasp, GraspSelector
from tcps import *
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics, Point, PointCloud
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from rotate import rotate_from_pointcloud, rotate
import time
import os
import sys
import traceback
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import pdb
from utils import *


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
    plt.title("bounding box for detection")  
    plt.imshow(result, interpolation="nearest")
    plt.show()

    return result

def skeletonize_img(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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

def find_length_and_endpoints(skeleton_img):
    # instead of doing DFS just do BFS and the last 2 points to have which end up having no non-visited neighbor 

    #### IDEA: do DFS but have a left and right DFS with distances for one being negative and the other being positive 
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
    # carry out floodfill
    def dfs(q, dist_q, visited, increment_amt):
        while len(q) > 0:
            next_loc = q.pop()
            distance = dist_q.pop()
            visited.add(next_loc)
            counter = 0
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if (test_loc in visited):
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
            dfs(q, dist_q, visited, increment_amt)
            # the first time our distance will be incrementing but the second time
            # , i.e. when dfs'ing the opposite direction our distance will be negative to differentiate both paths
            increment_amt = -1
    # we only have one neighbor therefore we must be an endpoint
    if counter == 1:
        distance = 0
        endpoints.append([start_pt, distance])
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
    for pt, distance in endpoints:
        if largest_pos is None or distance > endpoints[largest_pos][1]:
            largest_pos = endpoints.index([pt, distance])
        elif largest_neg is None or distance < endpoints[largest_neg][1]:
            largest_neg = endpoints.index([pt, distance])
    if initial_endpoint:
        final_endpoints = [endpoints[0][0], endpoints[largest_pos][0]]
    else:
        final_endpoints = [endpoints[largest_neg][0], endpoints[largest_pos][0]]
    
    plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.title("final endpoints")
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 
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
    
def find_endpoints_compressed_map(compressed_map, max_edges):
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