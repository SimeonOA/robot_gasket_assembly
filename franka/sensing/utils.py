# from autolab_core import RigidTransform, PointCloud, RgbdImage, ColorImage, DepthImage
# from autolab_core.image import BINARY_IM_DEFAULT_THRESH
# from numpy.lib.histograms import histogram_bin_edges
# from collision import CollisionInterface
from queue import Empty
# from yumiplanning.yumi_kinematics import YuMiKinematics
# from yumiplanning.yumi_planner import Planner
import numpy as np
from multiprocessing import Queue, Process
from random import shuffle
import math
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor

def get_N_lowest_pts(img, N):
    flattened_img = img.flatten()
    n_lowest_pts_idx = np.argpartition(flattened_img, -N)[-N:]
    # since the indices are flattened we need to convert back to 2D coords
    n_lowest_pts_coords = np.unravel_index(n_lowest_pts_idx, img.shape)
    # breakpoint()
    n_lowest_pts = list(zip(n_lowest_pts_coords[0], n_lowest_pts_coords[1]))
    # pretty sure first list is our y-coords and second list is our x-coords
    plt.title("n lowest points")
    plt.scatter(x=n_lowest_pts_coords[1], y=n_lowest_pts_coords[0])
    plt.imshow(img)
    plt.show()
    return n_lowest_pts

def get_fit_plane_ransac(points, depth_vals):
    # Assuming you have a 3D point cloud stored in a numpy array called 'points'
    X = points #points[:, :2]  # Extracting the x and y coordinates
    Z = depth_vals #points[:, 2]   # Extracting the z coordinate

    # Fitting a plane using RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X, Z)

    # Accessing the fitted plane parameters
    A = ransac.estimator_.coef_[0]
    B = ransac.estimator_.coef_[1]
    C = -1
    D = ransac.estimator_.intercept_

    # solving the equation: {A}x + {B}y + {C}z + {D} = 0 for z
    # return ransac
    return lambda x,y: A*x + B*y + D

def waypoints_dfs(skeleton, ):
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
def generate_waypoints_relative(skeleton, start_pt, pixels_per_waypoint):
    return

def generate_waypoints_naive(skeleton, skeleton_len, start_pt, num_waypoints):
    # POSSIBLE BUG: skeleton_len may also inclue the strands of the skeleton, need to check!!!
    pixels_per_waypoint = skeleton_len//num_waypoints

    


def skeletonize_img(img):
    # Invert the horse image
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # since we are currently using a binary mask we need to mult by 255 to get it in grayscale
    gray = img*255

    # threshold
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    blurred_image = gaussian_filter(image, sigma=1)

    # perform skeletonization
    skeleton = skeletonize(blurred_image)

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

# version used for TRI demo, does not have is_loop
def find_length_and_endpoints_tri(skeleton_img):
    #### IDEA: do DFS but have a left and right DFS with distances for one being negative and the other being positive 
    nonzero_pts = None
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))
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
            next_loc = q.pop(0)
            distance = dist_q.pop(0)
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

    # import pdb; pdb.set_trace()
    # print("endpoints are", endpoints)
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
        if largest_pos == 0:
                idx = largest_neg
        else:
                idx = largest_pos
        final_endpoints = [endpoints[0][0], endpoints[idx][0]]
    else:
        final_endpoints = [endpoints[largest_neg][0], endpoints[largest_pos][0]]
    plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 
    # pdb.set_trace()
    # display results

    print("the total length is ", total_length)
    return total_length, final_endpoints

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
    IS_LOOP = False
    ENTIRE_VISITED = [False] * int(np.nonzero(skeleton_img).sum())
    def dfs(q, dist_q, visited, start_pixel, increment_amt):
        '''
        q: queue with next point on skeleton for one direction
        dist_q: queue with distance from start point to next point for one direction
        visited: queue with visited points for only one direction
        increment_amt: counter that indicates direction +/- 1
        '''

        is_loop = ENTIRE_VISITED[start_pixel + increment_amt*len(visited)]
        if is_loop:
            return is_loop


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
        return is_loop
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
            IS_LOOP = dfs(q, dist_q, visited, increment_amt)
            if IS_LOOP:
                break
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
            if skeleton_img[test_loc[0]][test_loc[1]] == True:
                q.append(test_loc)
    plt.scatter(x = [j[1] for j in sorted_pts], y=[i[0] for i in sorted_pts],c='w')
    plt.scatter(x=[sorted_pts[-1][1], sorted_pts[0][1]], y=[sorted_pts[-1][0], sorted_pts[0][0]], c='g')
    plt.scatter(x=[endpoint[1]], y=[endpoint[0]], c='r')
    plt.imshow(skeleton_img)
    plt.show()
    return sorted_pts



class GraspSegmenter:

    #  zed camera gives us rgb and depth separately so we just save them differently
    def __init__(self, depth_image, rgb_image):
        self.color = rgb_image
        self.depth = depth_image
        print("self.color is", self.color.shape)
        print("self.depth is", self.depth.shape)

    def segment_cable(self, loc, orient_mode=False):
        '''
        returns a PointCloud corresponding to cable points along the provided location
        inside the depth image
        '''
        q = [loc]
        pts = []
        closepts = []
        visited = set()
        print("Loc", loc)
        #  DOUBLE CHECK WHETHER OR NOT IT IS (loc[1], loc[0]) or NOT!!!!
        start_point = self.depth[loc[1]][loc[0]]
        print("start_point", start_point)

        start_color = self.color[loc[1]][loc[0]]
        RADIUS2 = 1  # distance from original point before termination
        CLOSE2 = .002**2
        DELTA = 0.01#.00080#.00075
        NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        COLOR_THRESHOLD = 45
        counter = 0
        waypoints = []
        weird_pts = []
        # carry out floodfill
        while len(q) > 0:
            # if we're in orient_mode we only want the local segment of the cable to find the principle axis, not the entire cable
            if (orient_mode):
                if counter > 600:
                    break
            next_loc = q.pop(0)
            next_point = self.depth[next_loc[1]][next_loc[0]]
            # NOT SURE IF THIS IS NECESARY
            next_color = np.asarray(self.color[next_loc[1]][next_loc[0]])
            next_color = next_color.astype(np.int16)
            visited.add(next_loc)
            diff = start_point-next_point
            # print("diff is", diff)
            dist = diff #diff.dot(diff)
            if (dist > RADIUS2):
                continue
            pts.append(next_loc)
            # has us updating the list of points to our waypoint
            if counter % 1000 == 0:
                waypoints.append((next_loc[1],next_loc[0]))
            counter += 1
            if (dist < CLOSE2):
                closepts.append(next_point)
            # add neighbors if they're within delta of current height
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                # NOT SURE IF len(self.depth[0]) is EQUIVALENT TO self.depth.width!!!!!
                if test_loc[0] >= len(self.depth[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(self.depth) or test_loc[1] < 0:
                    continue
                if (test_loc in visited):
                    continue
                # want to check if the points were adding are of similar color cause the cable is a uniform color
                # the channel currently is not the same color so this is another method to differentiate between them
                test_pt = self.depth[test_loc[1]][test_loc[0]]
                # this is cause the data is grayscaled so we can just look at the Red channel 
                test_color = np.asarray(self.color[test_loc[1]][test_loc[0]])
                # pdb.set_trace()
                test_color = test_color.astype(np.int16)
                color_diff = np.linalg.norm(next_color-test_color)
                if (20 < color_diff < 45):
                    weird_pts.append([test_loc[1], test_loc[0]])
                # print("the delta is", abs(test_pt
                if (abs(test_pt-next_point) < DELTA): #and color_diff < COLOR_THRESHOLD):
                    q.append(test_loc)

        return np.array(pts), np.array(closepts), waypoints, weird_pts

    def segment_channel(self, loc, use_pixel = False):
        '''
        returns a PointCloud corresponding to cable points along the provided location
        inside the depth image
        '''
        q = [loc]
        pts = []
        pts_pixels = []
        closepts = []
        visited = set()
        start_point = self.depth[loc[1]][loc[0]]
        RADIUS2 = 1  # distance from original point before termination
        CLOSE2 = .002**2
        DELTA = 0.01 #0.0002*3
        NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        counter = 0
        waypoints = []
        endpoints = []
        # carry out floodfill
        while len(q) > 0:
            next_loc = q.pop(0)
            next_point = self.depth[next_loc[1]][next_loc[0]]
            # Next Point Example: [0.33499247 0.00217638 0.05645943]
            visited.add(next_loc)
            diff = start_point-next_point
            dist = diff #diff.dot(diff)
            if (dist > RADIUS2):
                continue
            pts_pixels.append(next_loc)
            pts.append(next_point)
            # has us updating the list of points to our waypoint
            if counter % 800 == 0:
                waypoints.append((next_loc[1],next_loc[0]))
            counter += 1
            if (dist < CLOSE2):
                closepts.append(next_point)
            # add neighbors if they're within delta of current height
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if (test_loc in visited):
                    continue
                # 
                if test_loc[0] >= len(self.depth[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(self.depth) or test_loc[1] < 0:
                    continue
                test_pt = self.depth[test_loc[1]][test_loc[0]]
                if (abs(test_pt-next_point) < DELTA):
                    q.append(test_loc)
        if (use_pixel == False):
            # NOT SURE WHETHER OT TRANSPOSE OR NOT
            return np.array(pts).T, np.array(closepts).T, waypoints, endpoints
        else:
            return pts_pixels, np.array(pts).T, np.array(closepts).T, waypoints, endpoints
  