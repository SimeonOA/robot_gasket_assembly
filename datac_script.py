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

cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0, cable)
behavior_cloning_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../multi-fidelity-behavior-cloning"
sys.path.insert(0, behavior_cloning_path)

DISPLAY = True
TWO_ENDS = False
PUSH_DETECT = True

# Directory to save the images
save_dir = "data/"

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# SPEED=(.5,6*np.pi)
SPEED = (.025, 0.3*np.pi)
# iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
#                   ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                  METAL_GRIPPER.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)



def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)

def coord_to_point(coord):
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    xind, yind = coord
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    new_point_data = np.array(
        [point.y, point.x, point.z])
    new_point = Point(
        new_point_data, frame=point.frame)
    return new_point


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
    
def dfs_resample(distance,sample_no, segmented_cloud):
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
    plt.imshow(result, interpolation="nearest")
    plt.show()

    return result

def skeletonize_img(img):
    # Invert the horse image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    blurred_image = gaussian_filter(image, sigma=1)

    # perform skeletonization
    skeleton = skeletonize(blurred_image)

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
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='b')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 
    # pdb.set_trace()
    # display results

    print("the total length is ", total_length)
    return total_length, final_endpoints





original_channel_waypoints = []
original_depth_image_scan = None
last_depth_image_scan = None
channel_endpoints = None
prev_channel_pt = []
CABLE_PIXELS_TO_DIST = None
CHANNEL_PIXELS_TO_DIST = None
try:
    while True:
        tt = time.time()
        iface.home()
        iface.open_grippers()
        iface.sync()
        # set up a simple interface for clicking on a point in the image
        img = iface.take_image()

        g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
        # NEW --------------------------------------------------------------------------------

        # ----------------------Find brightest pixel for segment_cable
        if DISPLAY:
            plt.imshow(img.color.data, interpolation="nearest")
            plt.axis('off')
            color_save_path = os.path.join(save_dir, f"{tt}_color.jpg")
            plt.savefig(color_save_path, bbox_inches='tight', pad_inches=0)
            plt.show()
        three_mat_color = img.color.data
        three_mat_depth = img.depth.data
        if original_depth_image_scan is None:
            original_depth_image_scan = three_mat_depth
        last_depth_image_scan = three_mat_depth

        edges_pre = np.uint8(three_mat_depth*10)
        edges = cv2.Canny(edges_pre,10,20)
        # plt.imshow(edges, cmap = 'gray')
        # plt.show()
        
        
        ### BEGIN FINDING THE CHANNEL AND CABLE POINTS!!!

        # ----------------------FIND END OF CHANNEL
        lower = 254
        upper = 256
        channel_start = (0, 0)
        max_edges = 0
        candidate_channel_pts = []
        
        # guess for what 0.5in is in terms of depth
        depth_diff_goal = 0.016
        # threshold to allow for error
        depth_threshold = 0.002

        # plt.imshow(edges)
        # plt.show()
        


        for r in range(len(edges)):
            for c in range(len(edges[r])):
                if (lower < edges[r][c]< upper):
                    diff1 = 0
                    diff2 = 0
                    diff3 = 0
                    diff4 = 0
                    for add in range(1, 4):
                        if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
                            break
                        # top - bottom
                        diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c]) 
                        # left - right
                        diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
                        # top left - bottom right
                        diff3 = abs(three_mat_depth[r-add][c-add] - three_mat_depth[r+add][r+add])
                        # top right - bottom left
                        diff4 = abs(three_mat_depth[r-add][c+add] - three_mat_depth[r+add][r-add])

                        if diff1 > 0.03 or diff2 > 0.03 or diff3 > 0.03 or diff4 > 0.03:
                            continue
                        if 0.01 <= np.mean(np.array([diff1, diff2, diff3, diff4])) <= 0.014:
                            candidate_channel_pts += [(r,c)]
                            
                    if diff1 > 0.02 or diff2 > 0.02 or diff3 > 0.02 or diff4 > 0.02:
                        continue
                    if 0.01 <= np.mean(np.array([diff1, diff2, diff3, diff4])) <= 0.014:
                        candidate_channel_pts += [(r,c)]
                        #print("the detected avg was: ", np.mean(np.array([diff1, diff2, diff3, diff4])))
        print("Candidate Edge pts: ", candidate_channel_pts)
        # need to figure out which edge point is in fact the best one for our channel
        # i.e. highest up, and pick a point that is actually in the channel
        max_depth = 100000
        min_depth = 0
        channel_edge_pt = (0,0)
        channel_start = (0,0)
        sorted_candidate_channel_pts = sorted(candidate_channel_pts, key=lambda x: three_mat_depth[x[0]][x[1]])



        print("The sorted list is: ", sorted_candidate_channel_pts)
        #channel_edge_pt = sorted_candidate_channel_pts[0]
        possible_cable_edge_pt = sorted_candidate_channel_pts[-1]
        #print("the edge with lightest depth is: ", three_mat_depth[channel_edge_pt[0]][channel_edge_pt[1]])
        print("the edge with deepest depth is: ", three_mat_depth[possible_cable_edge_pt[0]][possible_cable_edge_pt[1]])

        for candidate_pt in candidate_channel_pts:
            r = candidate_pt[0]
            c = candidate_pt[1]
            print("r", r, "c", c, "my depth is: ", three_mat_depth[r][c])
            if 0 < three_mat_depth[r][c] < max_depth:
                print("max depth:", max_depth)
                channel_edge_pt = (r,c)
                max_depth = three_mat_depth[r][c]
            if three_mat_depth[r][c] > min_depth:
                possible_cable_edge_pt = (r,c)
                min_depth = three_mat_depth[r][c]
        print("The edge of the channel is: ", channel_edge_pt)
        r,c = channel_edge_pt
        possible_channel_pts = []


        ##### NEED TO REMOVE THE EDGES OF VALUE 0 FROM THE SAMPLE BASE!!!!!
        index = 0
        while index < len(sorted_candidate_channel_pts) and channel_start == (0,0):
            channel_edge_pt = sorted_candidate_channel_pts[index]
            r,c = channel_edge_pt
            if three_mat_depth[r][c] == 0.0:
                index += 1
                continue
            for add in range(1, 4):
                if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
                    break
                # left - right
                diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c])
                diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
                if 0.01 <= diff1 < 0.014: # prev upper was 0.016
                    if three_mat_depth[r-add][c] > three_mat_depth[r+add][c]:
                        channel_start = (r-add, c)
                        possible_channel_pts += [(r-add, c)]
                    else:
                        channel_start = (r+add, c)
                        possible_channel_pts += [(r+add, c)]
                if 0.01 <= diff2 < 0.014: #prev upper was 0.016
                    if three_mat_depth[r][c-add] > three_mat_depth[r][c+add]:
                        channel_start = (r, c-add)
                        possible_channel_pts += [(r, c-add)]
                    else:
                        channel_start = (r, c+add)
                        possible_channel_pts += [(r, c+add)]
            # the point in the channel was not found, so we need to look at the next best one
            if channel_start == (0,0):
                index += 1
        # channel_start = (channel_edge_pt[1], channel_edge_pt[0])
        print("possible channel pts: ", possible_channel_pts)
        print("The chosen channel_pt is: ", channel_start)

      
        print("CHANNEL_START: "+str(channel_start))
        
        # FINDING THE POINT ON THE CABLE!!!
        r = possible_cable_edge_pt[0]
        c = possible_cable_edge_pt[1]
        index = 0
        cable_pt = (0,0)
        while index < len(sorted_candidate_channel_pts) and cable_pt == (0,0):
            possible_cable_pt = sorted_candidate_channel_pts[-index]
            r,c = possible_cable_pt
            if three_mat_depth[r][c] == 0.0:
                index += 1
                continue
            for add in range(1, 8):
                # once we've found a suitable cable point we want to just exit
                if cable_pt != (0,0):
                    break
                if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
                    break
                # left - right
                diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c])
                diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
                if 0.01 <= diff1 < 0.020: # prev upper was 0.016
                    # the depth that is lower (i.e. point is closer to the camera is the point that is actually on the cable)
                    if three_mat_depth[r-add][c] > three_mat_depth[r+add][c]:
                        cable_pt = (r+add, c)
                    else:
                        cable_pt = (r-add,c)
                if 0.01 <= diff2 < 0.020: #prev upper was 0.016
                    if three_mat_depth[r][c-add] > three_mat_depth[r][c+add]:
                        cable_pt = (r,c+add)
                    else:
                        cable_pt = (r,c-add)
        # the point in the cable was not found, so we need to look at the next best one
            if cable_pt == (0,0):
                index += 1
        
        loc = (cable_pt[1], cable_pt[0])
        max_scoring_loc = loc
        print("the cable point is ", cable_pt)


        plt.imshow(edges, cmap='gray')
        # plt.scatter(x = [j[1] for j in candidate_channel_pts], y=[i[0] for i in candidate_channel_pts],c='r')
        # plt.scatter(x=channel_edge_pt[1], y=channel_edge_pt[0], c='b')
        # plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        # plt.scatter(x=cable_pt[1], y=cable_pt[0], c='w')
        plt.imshow(three_mat_depth, interpolation="nearest")
        plt.axis('off')
        depth_save_path = os.path.join(save_dir, f"{tt}_depth.jpg")
        plt.savefig(depth_save_path,bbox_inches='tight', pad_inches=0)
        plt.show()
        ### END OF THIS WORK!!!

        print("Starting segment_cable pt: "+str(max_scoring_loc))
        # ----------------------Segment
        rope_cloud, _, cable_waypoints = g.segment_cable(loc)
        # ----------------------Remove block

        new_transf = iface.T_PHOXI_BASE.inverse()
        transformed_rope_cloud = new_transf.apply(rope_cloud)
        di = iface.cam.intrinsics.project_to_image(
            transformed_rope_cloud, round_px=False)
        if DISPLAY:
            plt.imshow(di._image_data(), interpolation="nearest")
            plt.axis('off')
            cable_save_path = os.path.join(save_dir, f"{tt}_cable.jpg")
            plt.savefig(cable_save_path,bbox_inches='tight', pad_inches=0)
            plt.show()
        
        # make_bounding_boxes(di._image_data())
        # cable_skeleton = skeletonize_img(di._image_data())
        
        # cable_len, cable_endpoints_1 = find_length_and_endpoints(cable_skeleton)
        
        ### MODIFIED BY KARIM AFTER COMMENTING OUT CORY"S CODE
        delete_later = []
        ### MODIFIED BY KARIM AFTER COMMENTING OUT CORY"S CODE

        di_data = di._image_data()
        for delete in delete_later:
            di_data[delete[1]][delete[0]] = [float(0), float(0), float(0)]

        mask = np.zeros((len(di_data), len(di_data[0])))
        loc_list = [loc]

        # modified segment_cable code to build a mask for the cable

        # pick the brightest rgb point in the depth image
        # increment in each direction for it's neighbors looking to see if it meets the thresholded rgb value
        # if not, continue
        # if yes set it's x,y position to the mask matrix with the value 1
        # add that value to the visited list so that we don't go back to it again

        new_di_data = np.zeros((len(di_data), len(di_data[0])))
        xdata = []
        ydata = []

        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                new_di_data[r][c] = di_data[r][c][0]
                if (new_di_data[r][c] > 0):
                    xdata += [c]
                    ydata += [r]

        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if (new_di_data[r][c] != 0):
                    curr_edges = 0
                    for add in range(1, 8):
                        if (new_di_data[min(len(new_di_data)-add, r+add)][c] != 0):
                            curr_edges += 1
                        if (new_di_data[max(0, r-add)][c] != 0):
                            curr_edges += 1
                        if (new_di_data[r][min(len(new_di_data[0])-add, c+add)] != 0):
                            curr_edges += 1
                        if (new_di_data[r][max(0, c-add)] != 0):
                            curr_edges += 1
                        if (new_di_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)] != 0):
                            curr_edges += 1
                        if (new_di_data[min(len(new_di_data)-add, r+add)][max(0, c-add)] != 0):
                            curr_edges += 1
                        if (new_di_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)] != 0):
                            curr_edges += 1
                        if (new_di_data[max(0, r-add)][max(0, c-add)] != 0):
                            curr_edges += 1
                    if (curr_edges < 11):
                        new_di_data[r][c] = 0.0

        new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
        # if DISPLAY:
        #     plt.imshow(new_di._image_data(), interpolation="nearest")
        #     plt.show()

        new_di_data = gaussian_filter(new_di_data, sigma=1)

        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if (new_di_data[r][c] != 0):
                    new_di_data[r][c] = 255
        new_di_data = gaussian_filter(new_di_data, sigma=1)

        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if (new_di_data[r][c] != 0):
                    new_di_data[r][c] = 255
        new_di_data = gaussian_filter(new_di_data, sigma=1)

        save_loc = (0, 0)
        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if (new_di_data[r][c] != 0):
                    new_di_data[r][c] = 255
                    save_loc = (c, r)
        new_di_data = gaussian_filter(new_di_data, sigma=1)

        compress_factor = 30
        rows_comp = int(math.floor(len(di_data)/compress_factor))
        cols_comp = int(math.floor(len(di_data[0])/compress_factor))
        compressed_map = np.zeros((rows_comp, cols_comp))

        for r in range(rows_comp):
            if r != 0:
                r = float(r) - 0.5
            for c in range(cols_comp):
                if c != 0:
                    c = float(c) - 0.5
                for add in range(1, 5):
                    if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(c*compress_factor)] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(max(0, r*compress_factor-add))][int(c*compress_factor)] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(r*compress_factor)][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(r*compress_factor)][int(max(0, c*compress_factor-add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(max(0, c*compress_factor-add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(max(0, r*compress_factor-add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
                    if (new_di_data[int(max(0, r*compress_factor-add))][int(max(0, c*compress_factor-add))] != 0):
                        compressed_map[int(r)][int(c)] = 255
                        break
        max_edges = 0
        test_locs = (0, 0)
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
                    if (curr_edges > max_edges):
                        test_loc = (c, r)
                        max_edges = curr_edges
        if 'test_loc' in globals():
            print(test_loc)
            print("scaled: " +
                str((test_loc[0]*compress_factor, test_loc[1]*compress_factor)))
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

        channel_start_d = (channel_start[1], channel_start[0])
        # channel_cloud, _, channel_waypoints, possible_channel_end_pts = g.segment_channel(channel_start_d)
        channel_cloud_pixels, channel_cloud, _, channel_waypoints, possible_channel_end_pts = \
            g.segment_channel(channel_start_d, use_pixel=True)
        waypoint_first= g.ij_to_point(channel_waypoints[0]).data
        # print('channel cloud shape', channel_cloud.shape)
        # print('channel waypoints one case:', channel_waypoints[0])
        # print('channel waypoints one case adjusted', waypoint_first)
        # print('channel cloud one case', channel_cloud[-1])
        # print('channel cloud one case', channel_cloud.data[-1])
        # print('channel cloud', channel_cloud)
        # print('location test', np.where(channel_cloud.data == channel_cloud.data[-1]))
        # print('channel waypoints', channel_waypoints)
        # plt.scatter(x = [j[1] for j in channel_waypoints], y=[i[0] for i in channel_waypoints],c='c')
        # plt.scatter(x = [j[1] for j in cable_waypoints], y=[i[0] for i in cable_waypoints],c='0.75')
        # plt.scatter(x = [j[1] for j in possible_channel_end_pts], y=[i[0] for i in possible_channel_end_pts],c='0.45')
        # plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        # plt.scatter(x=cable_pt[1], y=cable_pt[0], c='w')
        # plt.imshow(three_mat_depth, interpolation="nearest")
        # plt.show()
        
        transformed_channel_cloud = new_transf.apply(channel_cloud)
        image_channel = iface.cam.intrinsics.project_to_image(
            transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
        image_channel_data = image_channel._image_data()
        
        # make_bounding_boxes(image_channel_data)
        
        # image_channel_data = gaussian_filter(image_channel_data, sigma=1)
        # channel_skeleton = skeletonize_img(image_channel_data)

        # channel_len, channel_endpoints = find_length_and_endpoints(channel_skeleton)


        copy_channel_data = copy.deepcopy(image_channel_data)
        lower = 80
        upper = 255



        for r in range(len(image_channel_data)):
            for c in range(len(image_channel_data[r])):
                if (new_di_data[r][c] != 0):
                    image_channel_data[r][c][0] = 0.0
                    image_channel_data[r][c][1] = 0.0
                    image_channel_data[r][c][2] = 0.0

        # Finish Thresholding, now find corner to place
        if DISPLAY:
            print("channel tracking!") # So we know if the channel tracking works appropriately
            plt.imshow(copy_channel_data, interpolation="nearest")
            plt.axis('off')
            channel_save_path = os.path.join(save_dir, f"{tt}_channel.jpg")
            plt.savefig(channel_save_path, bbox_inches='tight', pad_inches=0)
            plt.show()
        for r in range(len(copy_channel_data)):
            for c in range(len(copy_channel_data[r])):
                if copy_channel_data[r][c][0] != 0:
                    copy_channel_data[r][c][0] = 255.0
                    copy_channel_data[r][c][1] = 255.0
                    copy_channel_data[r][c][2] = 255.0
        img_skeleton = cv2.cvtColor(copy_channel_data, cv2.COLOR_RGB2GRAY)

        features = cv2.goodFeaturesToTrack(img_skeleton, 10, 0.01, 200)
        print("OPEN CV2 FOUND FEATURES: ", features)
        endpoints = [x[0] for x in features]

        closest_to_origin = (0, 0)
        furthest_from_origin = (0, 0)
        min_dist = 10000000
        max_dist = 0
        for endpoint in endpoints:
            dist = np.linalg.norm(np.array([endpoint[0], endpoint[1]-400]))
            if dist < min_dist:
                min_dist = dist
                closest_to_origin = endpoint
        for endpoint in endpoints:
            dist = np.linalg.norm(
                np.array([closest_to_origin[0]-endpoint[0], closest_to_origin[1]-endpoint[1]]))
            if dist > max_dist:
                max_dist = dist
                furthest_from_origin = endpoint
        endpoints = [closest_to_origin, furthest_from_origin]
        print("ENDPOINTS SELECTED: " + str(endpoints))
        # if DISPLAY:
        #     print("img skeleton")
        #     plt.scatter(x=[j[0][0] for j in features], y = [j[0][1] for j in features], c = '0.2')
        #     plt.scatter(x=[j[0] for j in endpoints], y = [j[1] for j in endpoints], c = 'm')
        #     plt.imshow(img_skeleton, interpolation="nearest")
        #     plt.show()
        # ----------------------FIND END OF CHANNEL
        # Use estimation
        place = (0, 0)
        place_2 = (0, 0)
        # Use left side
        if (endpoints[0][0] < endpoints[1][0]):
            place = endpoints[0]
            place_2 = endpoints[1]
        else:
            place = endpoints[1]
            place_2 = endpoints[0]
        print("ACTUAL PLACE: "+str(place))
        print("ACTUAL PLACE 2: "+str(place_2))
        
        
    #     plt.scatter(x = [j[1] for j in channel_waypoints], y=[i[0] for i in channel_waypoints],c='c')
    #     plt.scatter(x = [j[1] for j in cable_waypoints], y=[i[0] for i in cable_waypoints],c='0.75')
    #     plt.scatter(x=[pick[0], place[0]], y = [pick[1], place[1]], c='b')
    #     plt.scatter(x=[waypoint_pick[1], waypoint_place[1]], y = [waypoint_pick[0], waypoint_place[0]], c='r')
        # plt.imshow(three_mat_depth, interpolation="nearest")
        # plt.show()

        # need to do this so that the points are in the proper place!
        # waypoint_pick1 = waypoint_pick[1], waypoint_pick[0]
        # waypoint_place1 = waypoint_place[1], waypoint_place[0]
        
        # start_place = place
        # old_place = waypoint_place


        # continuously rescan and update the cable point cloud to gather new endpoints and waypoints, dont need to rescan channel hopefully
        # we know when to stop when the next place location is some distance close enough to either the second end point of the channel (straight/curved channel) or the start point of the channel (trapezoid)
        # while 


    
    # this is for if anything breaks then we just move onto the pushing task 
except Exception:
    traceback.print_exc()
    ACCEPTABLE_DEPTH = 0.02
    img = iface.take_image()
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    # NEW --------------------------------------------------------------------------------

    # ----------------------Find brightest pixel for segment_cable
    # if DISPLAY:
    #     plt.imshow(img.color.data, interpolation="nearest")
    #     plt.show()
    three_mat_color = img.color.data
    three_mat_depth = img.depth.data

    last_depth_image_scan = three_mat_depth

    print("BEGINNING PUSHING")

print("Done with script, can end")
