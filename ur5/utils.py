import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from resources import *
from autolab_core import RigidTransform

def get_sorted_cable_pts(cable_endpoints, cable_skeleton, is_trap=False):
    cable_endpoint_in = cable_endpoints[0]
    sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)
    # trapezoid skeleton is smooth, don't want to delete parts of it. 
    if not is_trap:
        sorted_cable_pts = sorted_cable_pts[START_IDX:END_IDX]
    return sorted_cable_pts
    
def press_idx(robot, sorted_channel_pts, idx, trap=False, viz=False, rgb_img=None):
    if idx >= len(sorted_channel_pts):
        idx = len(sorted_channel_pts) - 1
    press_pt = sorted_channel_pts[idx]
    if viz:
        plt.scatter(x=press_pt[1], y=press_pt[0], c='r')
        plt.title("Press Point")
        plt.imshow(rgb_img)

    # needs to be swapped as this is how it is expected for the robot
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    if trap:
        press_pose_swap = robot.get_rw_pose((press_pt[1], press_pt[0]), swapped_sorted_channel_pts[::-1], 15, is_channel_pt=True)
    else:
        press_pose_swap = robot.get_rw_pose((press_pt[1], press_pt[0]), swapped_sorted_channel_pts, 15, is_channel_pt=True)
    press_above_translation = press_pose_swap.translation
    press_above_translation[2] += 0.02
    press_above_pose = RigidTransform(press_pose_swap.rotation, press_above_translation)
    robot.rotate_pose90(press_above_pose)
    robot.press(press_pose_swap)
    robot.move_pose(press_above_pose, interp="tcp")

def find_nth_nearest_point(point, sorted_points, given_n):
    idx = sorted_points.index(point)
    behind_idx = np.clip(idx - given_n, 0, len(sorted_points)-1)
    infront_idx = np.clip(idx + given_n, 0, len(sorted_points)-1)
    return behind_idx, infront_idx

def get_rotation(point1, point2):
    direction = np.array(point2) - np.array(point1)
    direction = direction / np.linalg.norm(direction)
    dz = np.arctan2(direction[1], direction[0])
    euler = np.array([-np.pi, 0, dz])
    rot_matrix = R.from_euler("xyz", euler).as_matrix()
    return rot_matrix

def get_closest_channel_endpoint(cable_endpoint, channel_endpoints):
    min_dist = np.inf
    closest_channel_endpoint = None
    for channel_endpoint in channel_endpoints:
        dist = np.linalg.norm(np.array(channel_endpoint) - np.array(cable_endpoint))
        if dist < min_dist:
            min_dist = dist
            closest_channel_endpoint = channel_endpoint
    if closest_channel_endpoint == None:
        print('neither endpoint was found in the channel')
    farthest_channel_endpoint = channel_endpoints[0] if channel_endpoints[1] == closest_channel_endpoint else channel_endpoints[1]
    return closest_channel_endpoint, farthest_channel_endpoint

def get_sorted_pts(cable_endpoints, channel_endpoints, cable_skeleton, channel_skeleton, is_trapezoid = False, pick_closest_endpoint = False):
    if is_trapezoid:
        # in this case this just the first corner of the trapezoid
        channel_endpoint_in = channel_endpoints
    else:
        cable_endpoint_in = cable_endpoints[0]
        channel_endpoint_in, channel_endpoint_out = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
    
    if pick_closest_endpoint:
        cable_endpoints_np = np.array(cable_endpoints)
        channel_endpoint_in_np = np.array(channel_endpoint_in)
        dist_2 = np.sum((cable_endpoints_np - channel_endpoint_in_np)**2, axis=1)
        cable_endpoint_in = cable_endpoints_np[np.argmin(dist_2)]
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
    else:
        sorted_cable_pts = sorted_cable_pts[3:-3]
    
    channel_endpoint_in = sorted_channel_pts[0]
    channel_endpoint_out = sorted_channel_pts[-1]
    cable_endpoint_in = sorted_cable_pts[0]
    cable_endpoint_out = sorted_cable_pts[-1]
    return sorted_cable_pts, sorted_channel_pts

def classify_corners(channel_skeleton_corners):
    dist0 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[1])
    dist1 = np.linalg.norm(channel_skeleton_corners[1]-channel_skeleton_corners[2])
    dist2 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[3])
    dist3 = np.linalg.norm(channel_skeleton_corners[2]-channel_skeleton_corners[3])
    max_dist = max([dist0,dist1, dist2, dist3])
    # long_cornerX and med_cornerX are the corners on the same side of the trapezoid with one being a corner for the long side and the other being the corner for the medium side
    if max_dist == dist0:
        long_corner0, long_corner1 = channel_skeleton_corners[0], channel_skeleton_corners[1]
        med_corner0, med_corner1 = channel_skeleton_corners[3], channel_skeleton_corners[2]
    elif max_dist == dist1:
        long_corner0, long_corner1 = channel_skeleton_corners[1], channel_skeleton_corners[2]
        med_corner0, med_corner1 = channel_skeleton_corners[0], channel_skeleton_corners[3]
    elif max_dist == dist2:
        long_corner0, long_corner1 = channel_skeleton_corners[0], channel_skeleton_corners[3]
        med_corner0, med_corner1 = channel_skeleton_corners[1], channel_skeleton_corners[2]
    else:
        long_corner0, long_corner1 = channel_skeleton_corners[2], channel_skeleton_corners[3]
        med_corner0, med_corner1 = channel_skeleton_corners[1], channel_skeleton_corners[0]
    return long_corner0, long_corner1, med_corner0, med_corner1

def get_corners(channel_cnt):
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
    # pa = top left point, pb = bottom left point
    pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
    # pc = top right point, pd = bottom right point
    pc, pd = sorted(appr[2:4], key=lambda c: c[0][1])
    # Visualization:
    # plt.imshow(cv2.drawContours(rgb_img.copy(), [channel_cnt], -1, 255, 3))
    # plt.scatter(pa[0,0], pa[0,1], c='r')
    # plt.scatter(pb[0,0], pb[0,1], c='b')
    # plt.scatter(pc[0,0], pc[0,1], c='g')
    # plt.scatter(pd[0,0], pd[0,1], c='k')
    # plt.show()
    return [[pa[0,1], pa[0,0]], [pb[0,1], pb[0,0]], [pd[0,1], pd[0,0]], [pc[0,1], pc[0,0]]] 

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

def get_binary_search_idx(num_pts):
        sorted_search_idx = []
        for i in range(int(np.log2(num_pts))):
            power = 2**(i+1)
            val = 1/power
            while val <= 1/2:
                if val not in sorted_search_idx:
                    sorted_search_idx.append(val)
                if 1-val not in sorted_search_idx:
                    sorted_search_idx.append(1-val)
                val += 1/power
        return sorted_search_idx

def find_closest_point(point_list, point_b):
    points = np.array(point_list)
    b = np.array(point_b)
    distances = np.linalg.norm(points - b, axis=1)
    closest_index = np.argmin(distances)
    closest_point = point_list[closest_index]
    return closest_point

def match_corners_to_skeleton(corners, skeleton):
    search_pts = np.where(skeleton > 0)
    search_pts = np.vstack((search_pts[0], search_pts[1]))
    search_pts = search_pts.T
    # match each corner to the closest pixel on the skeleton
    matched_pts = []
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
    search_pts = search_pts.T
    # match each corner to the closest pixel on the skeleton
    dist_2 = np.sum((search_pts - real_midpt)**2, axis=1)
    matched_midpt = search_pts[np.argmin(dist_2)]
    return matched_midpt