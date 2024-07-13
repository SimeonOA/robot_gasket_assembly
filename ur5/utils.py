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
import pyzed.sl as sl
from scipy.spatial.distance import cdist
from resources import CROP_REGION

def setup_zed_camera(cam_id):
    side_cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    # init_params.camera_fps = 30
    init_params.set_from_serial_number(cam_id)
    status = side_cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Camera Failed To Open")
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_FILL
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
    if len(appr) == 5:
        temp = np.array(appr).reshape(-1,2)
        pairwise_dist = cdist(temp, temp) + np.eye(temp.shape[0])*1e6
        nearest_neigh_loc = np.array(np.where(pairwise_dist == np.min(pairwise_dist)))
        assert np.array_equal(nearest_neigh_loc, nearest_neigh_loc.T)
        temp = np.array([temp[nearest_neigh_loc[0][0]], temp[nearest_neigh_loc[0][1]]])
        cnt = channel_cnt.reshape(-1,2)
        keep = [x for x in cnt if x[0] >= min(temp[0][0], temp[1][0]) and x[0] <= max(temp[0][0],temp[1][0]) and x[1] >= min(temp[0][1],temp[1][1]) and x[1] <= max(temp[0][1],temp[1][1])]
        interp_idx = np.argmax(cdist(temp[:2],keep).sum(axis=0))
        new_pt = keep[interp_idx].reshape(-1,2)
        appr = [x for i, x in enumerate(appr) if i not in nearest_neigh_loc[0]]
        appr.insert(nearest_neigh_loc[0][0], new_pt)
    #pa = top lef point, pb = bottom left point
    pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
    #pc = top right point, pd = bottom right point
    pc, pd = sorted(appr[2:4], key=lambda c: c[0][1])
    # Visualization:
    # plt.imshow(cv2.drawContours(rgb_img.copy(), [channel_cnt], -1, 255, 3))
    # plt.scatter(pa[0,0], pa[0,1], c='r')
    # plt.scatter(pb[0,0], pb[0,1], c='b')
    # plt.scatter(pc[0,0], pc[0,1], c='g')
    # plt.scatter(pd[0,0], pd[0,1], c='k')
    # plt.show()
    return [[pa[0,1], pa[0,0]], [pb[0,1], pb[0,0]], [pd[0,1], pd[0,0]], [pc[0,1], pc[0,0]]] 

def get_corners(skeleton_img):
    skeleton_img = skeleton_img.astype(np.uint8)*255
    skeleton_img = cv2.merge((skeleton_img, skeleton_img, skeleton_img))
    gray = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=100)
    corners = np.int0(corners)
    # Visualization:
    # image = skeleton_img.copy()
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(image, (x, y), 1, 255, -1)
    # plt.imshow(image)
    # plt.show()
    corners = corners.reshape((4,2))
    corners = [[corners[i][1], corners[i][0]] for i in range(len(corners))]
    return corners

def sort_skeleton_pts(skeleton_img, endpoint, is_trapezoid=False):
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    start_pt = endpoint
    q = [start_pt]
    sorted_pts = []
    # dfs from start_pt which is one of our waypoints 
    if not is_trapezoid:
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
    else:
        while len(q) > 0:
            next_loc = q.pop()
            visited.add(next_loc)
            sorted_pts.append(next_loc)
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if (test_loc in visited):
                    continue
                if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                    continue
                if skeleton_img[test_loc[0]][test_loc[1]].sum() > 0:
                    q.append(test_loc)
    # Visualization:
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

def get_midpt(pt1, pt2):
    return ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)

def sample_pts_btwn(pt1, pt2, n):
    x1, y1 = pt1
    x2, y2 = pt2
    x_vals = np.linspace(x1,x2,n, endpoint=True)
    y_vals = np.linspace(y1,y2,n, endpoint=True)
    pts = np.column_stack((x_vals, y_vals))
    # print("this is pt1", pt1)
    # print("this is pt2", pt2)
    # print("and this is all of the pts", pts)
    return pts

def determine_length(object_points, endpoint_list, waypoints_list, g):
    visited = set()
    distances_list = dict()
    waypoint_found = dict()
    waypoints_sorted = [] 
    start_point = [int(endpoint_list[0][0]), int(endpoint_list[0][1])]
    end_point = [int(endpoint_list[1][0]), int(endpoint_list[1][1])]
    waypoints_sorted.append(start_point)
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

def evenly_sample_points_dist(points, num_points, distance):
    selected_points = []
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
    selected_points.append(points[0])
    selected_points.append(points[-1])
    while len(selected_points) < num_points:
        dists = np.array([min([np.linalg.norm(p-sp)]) for sp in selected_points] for p in points)
        next_point_idx = np.argmax(dists)
        next_point = points[next_point_idx]
        selected_points.append(next_point)
    return selected_points

def make_bounding_boxes(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY)[1]
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
    # save resulting image
    # cv2.imwrite('two_blobs_result.jpg',result)      
    # Visualization:
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
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # blurred_image = gaussian_filter(image, sigma=1)
    # dilate the image to just eliminate risk of holes causes severances in the skeleton
    dilated_img = image.copy()
    kernel = np.ones((5,5), np.uint8)
    cv2.dilate(image, kernel, dst=dilated_img, iterations=1)
    skeleton = skeletonize(dilated_img)
    # Visualization:
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
                q = q[:-count]
                past_pixels_lst = past_pixels_lst[:-count]
                for past_pixel in past_pixels_lst:
                    skeleton_img[past_pixel[0]][past_pixel[1]] = False
                # Visualization:
                # plt.scatter(x=[i[1] for i in saved_pixels], y=[i[0] for i in saved_pixels], c='g')
                # plt.scatter(x=[i[1] for i in past_pixels_lst], y=[i[0] for i in past_pixels_lst], c='b')
                # plt.scatter(x=[i[1] for i in q], y=[i[0] for i in q], c='r')
                # plt.scatter(x=[i[1] for i in branch_endpoints], y=[i[0] for i in branch_endpoints], c='y')
                # plt.scatter(x=branch_endpoint[1], y=branch_endpoint[0], c='w')
                # plt.scatter(x=next_loc[1], y=next_loc[0], c='m')
                # plt.imshow(skeleton_img)
                # plt.show()
                break
        # sets all of those passed pixels to False
    # Visualization:
    # plt.title("pruned skeleton")
    # plt.imshow(skeleton_img)
    # plt.show()
    return skeleton_img

def find_length_and_endpoints(skeleton_img):
    # instead of doing DFS just do BFS and the last 2 points to have which end up having no non-visited neighbor 
    if len(skeleton_img.shape) == 3:
        skeleton_img = skeleton_img[:,:,1]
        skeleton_img = skeleton_img.astype(bool)
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))
    if nonzero_pts is None:
        nonzero_pts = [[[0,0]]]
    total_length = len(nonzero_pts)
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
    # ENTIRE_VISITED = [False] * total_length
    def dfs(q, dist_q, visited, paths, start_pixel, increment_amt):
        '''
        q: queue with next point on skeleton for one direction
        dist_q: queue with distance from start point to next point for one direction
        visited: queue with visited points for only one direction
        increment_amt: counter that indicates direction +/- 1
        '''
        while len(q) > 0:
            next_loc = q.pop()
            distance = dist_q.pop()
            visited.add(next_loc)
            counter = 0
            count_visited = []
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                # print("count_visited ", count_visited)
                if (test_loc in visited):
                    count_visited.append(test_loc)
                    continue
                if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                    continue
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    counter += 1
                    q.append(test_loc)
                    dist_q.append(distance+increment_amt)
            # this means we haven't added anyone else to the q so we "should" be at an endpoint
            if counter == 0:
                endpoints.append([next_loc, distance])
        return False # we don't have a loop
    counter = 0
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
            IS_LOOP = dfs(q, dist_q, visited, paths, start_pixel, increment_amt)
            increment_amt = -1
    # we only have one neighbor therefore we must be an endpoint
    # only works for the start point so we need to check
    if counter == 1:
        distance = 0
        endpoints.insert(0, [start_pt, distance])
        initial_endpoint = True
    final_endpoints = []
    largest_pos = largest_neg = None

    if IS_LOOP:
        final_endpoints = [endpoints[0][0]]
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
    branch_endpoints = endpoints.copy()
    branch_endpoints = [x[0] for x in branch_endpoints]
    for final_endpoint in final_endpoints:
        branch_endpoints.remove(final_endpoint)
    pruned_skeleton = prune_branches(branch_endpoints, skeleton_img.copy())
    # Visualization:
    # plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    # plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    # plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    # plt.title("final endpoints")
    # plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    # plt.imshow(skeleton_img, interpolation="nearest")
    # plt.show() 
    return total_length, final_endpoints

# Function to check the index of a  
def check_index(p_array, part, ind = False):
    newp_array = p_array.T
    check = np.where(newp_array == part)
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
    # Visualization:
    # plt.title('cropped img')
    # plt.imshow(cropped_img)
    # plt.show()
    return cropped_img



