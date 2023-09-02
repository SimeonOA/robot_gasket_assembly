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
from push import *

cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0, cable)
behavior_cloning_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../multi-fidelity-behavior-cloning"
sys.path.insert(0, behavior_cloning_path)

DISPLAY = True
TWO_ENDS = False
PUSH_DETECT = True
# used to get the relative depths for flattening the depth image
CALIBRATE = False


# SPEED=(.5,6*np.pi)
SPEED = (.025, 0.3*np.pi)
# iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
#                   ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                  METAL_GRIPPER.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)

######################### MAIN ####################################################################################################### 

original_channel_waypoints = []
original_depth_image_scan = None
last_depth_image_scan = None
channel_endpoints = None
prev_channel_pt = []
CABLE_PIXELS_TO_DIST = None
CHANNEL_PIXELS_TO_DIST = None


try:
    while True:
        iface.home()
        iface.open_grippers()
        iface.sync()
        # set up a simple interface for clicking on a point in the image
        img = iface.take_image()

        g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)

        if DISPLAY:
            plt.title("color image")
            plt.imshow(img.color.data, interpolation="nearest")
            plt.show()

        three_mat_color = img.color.data
        three_mat_depth = img.depth.data
        
        # need to eliminate bottom segment of the workspace cause the dropoff creates weird problems
        # starting from the top of the image, the amount of pixels we want in the vertical direction
        vert_pixels = 600
        horiz_pixels = 200  
        three_mat_depth = crop_img(vert_pixels, horiz_pixels, three_mat_depth)

        if original_depth_image_scan is None:
            original_depth_image_scan = three_mat_depth
        last_depth_image_scan = three_mat_depth

        # levels out all of the pixels so that the workspace is treated as flat
        # basically tries to remove the slant that the workspace has
        if CALIBRATE:
            highest_depth = np.max(three_mat_depth)
            mask = (np.ones(three_mat_depth.shape)*highest_depth) - three_mat_depth
            # ensures that those 0 null values in the original scan are still set to 0
            mask[mask > 0.5] = 0
            with open('flat_table_depth_mask.npy', "wb") as f:
                np.save(f, mask, allow_pickle=False)
        else:
            with open('flat_table_depth_mask.npy', "rb") as f:
                mask = np.load(f)

        print(mask)
        three_mat_depth_flat = mask + three_mat_depth
        plt.title("image depth flat")
        plt.imshow(three_mat_depth_flat, interpolation='nearest')
        plt.show()

        plt.title("image depth not flat")
        plt.imshow(three_mat_depth, interpolation='nearest')
        plt.show()

        # just need it to be same dimensions as three_mat_depth
        dilated_depth_img = three_mat_depth.copy()
        kernel = np.ones((3,3), np.uint8)
        cv2.dilate(three_mat_depth, kernel, dst=dilated_depth_img, iterations=1)
        plt.title("dilated depth img")
        plt.imshow(dilated_depth_img, interpolation='nearest')
        plt.show()

        # dilates the image to try to resolve 0 depth values
        # three_mat_depth = dilated_depth_img
        

        if np.array_equal(three_mat_depth, dilated_depth_img):
            is_dilated = True
        else:
            is_dilated = False   


        # WITHOUT DILATION THRESHOLD1 = 10, THRESHOLD2 = 20
        if is_dilated:
            edges_pre = np.uint8(three_mat_depth*100.0)
            plt.imshow(edges_pre, interpolation='nearest')
            plt.title('edges_pre with dilation')
            plt.show()
            edges = cv2.Canny(edges_pre,1,3, L2gradient = False)
        else:
            edges_pre = np.uint8(three_mat_depth*10)
            plt.imshow(edges_pre, interpolation='nearest')
            plt.title('edges_pre without dilation')
            plt.show()
            edges = cv2.Canny(edges_pre,10,20, L2gradient = False)
        
        
        ### BEGIN FINDING THE CHANNEL AND CABLE POINTS!!!

        # ----------------------FIND END OF CHANNEL
        channel_start = (0, 0)
        max_edges = 0
        candidate_channel_pts = []
        
        # guess for what 0.5in is in terms of depth
        depth_diff_goal = 0.016
        # threshold to allow for error
        depth_threshold = 0.002

        plt.title("canny edges")
        plt.imshow(edges)
        plt.show()

        
        # testing out using contours instead to try and figure out cable and channels
        # edges_copy = edges.copy()
        # test = cv2.cvtColor(three_mat_color, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('test', test)
        # cv2.waitKey(0)
        # color_edges = cv2.Canny(test,10,20)
        # cv2.imshow("colored edges", color_edges)
        # cv2.waitKey(0)
        # if not is_dilated:
        #     blurred_depth_image = gaussian_filter(three_mat_depth, sigma=1)
        #     plt.imshow(blurred_depth_image, interpolation='nearest')
        #     plt.title("blurred depth image")
        #     plt.show()
        # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("num contours: " + str(len(contours)))
        # cv2.drawContours(edges, contours, -1, (255,0,0), 3)
        # cv2.imshow('contours', edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #  ------------------------------------------
        

        #depth_image = cv2.imread(three_mat_depth, cv2.IMREAD_UNCHANGED)
        


        # performing image in painting, to remove all of the 0 values in the depth image with an average
        # three_mat_depth = inpaint_depth(three_mat_depth)
        

        print('is_dilated', is_dilated)
        if is_dilated:
            lower_thresh = 0.009
            upper_thresh = 0.011
        else:
            lower_thresh = 0.009
            upper_thresh = 0.01

        # searches the depth values surrounding a given pixel to see if
        # the average change is depth is around 0.5in
        for r in range(len(edges)):
            for c in range(len(edges[r])):
                if (edges[r][c] == 255):
                    diff1 = 0
                    diff2 = 0
                    diff3 = 0
                    diff4 = 0
                    count_diff = 0
                    total_diff = 0
                    for add in range(1, 3):
                        # checks if out of bounds
                        if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
                            break
                        # top - bottom
                        diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c]) 
                        if diff1 < 0.03:
                            total_diff += diff1
                            count_diff += 1
                        # left - right
                        diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
                        if diff2 < 0.03:
                            total_diff += diff2
                            count_diff += 1
                        # top left - bottom right
                        diff3 = abs(three_mat_depth[r-add][c-add] - three_mat_depth[r+add][r+add])
                        if diff3 < 0.03:
                            total_diff += diff3
                            count_diff += 1
                        # top right - bottom left
                        diff4 = abs(three_mat_depth[r-add][c+add] - three_mat_depth[r+add][r-add])
                        if diff4 < 0.03:
                            total_diff += diff4
                            count_diff += 1
                    
                    if count_diff != 0 and lower_thresh <= (total_diff/count_diff) <= upper_thresh:
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
        print("CHANNEL_START: "+str(channel_start))
        
        # FINDING A POINT ON THE CABLE
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
        plt.title("cable and channel detected points")
        plt.scatter(x = [j[1] for j in candidate_channel_pts], y=[i[0] for i in candidate_channel_pts],c='r')
        plt.scatter(x=channel_edge_pt[1], y=channel_edge_pt[0], c='b')
        plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        plt.scatter(x=cable_pt[1], y=cable_pt[0], c='w')
        plt.imshow(three_mat_depth, interpolation="nearest")
        plt.show()
        ### END OF THIS WORK!!!

        print("Starting segment_cable pt: "+str(max_scoring_loc))
        # ----------------------Segment
        rope_cloud, _, cable_waypoints, weird_pts = g.segment_cable(loc)
        # ----------------------Remove block

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

        ax = axes.ravel()
        plt.title("weird points for using similar color to remove certain pixels")
        plt.scatter(x = [j[1] for j in weird_pts], y=[i[0] for i in weird_pts],c='w')
        ax[0].imshow(img.color.data, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('color', fontsize=20)

        ax[1].imshow(edges, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('edge', fontsize=20)

        fig.tight_layout()
        plt.show()


        new_transf = iface.T_PHOXI_BASE.inverse()
        transformed_rope_cloud = new_transf.apply(rope_cloud)
        di = iface.cam.intrinsics.project_to_image(
            transformed_rope_cloud, round_px=False)
        if DISPLAY:
            plt.title("depth image, image data")
            plt.imshow(di._image_data(), interpolation="nearest")
            plt.show()
        
        make_bounding_boxes(di._image_data())
        cable_skeleton = skeletonize_img(di._image_data())
        
        cable_len, cable_endpoints_1 = find_length_and_endpoints(cable_skeleton)

        
        # code for a compressed map to find endpoints
        # not at all well-converted for current code do not attempt without looking at 106a_main.py and editing utils
        # pick, pick_2 = find_endpoints_compressed_map(compressed_map, max_edges, place, place_2, new_di_data, DISPLAY=False)

       

        # channel_cloud, _, channel_waypoints, possible_channel_end_pts = g.segment_channel(channel_start_d)
        # ----------------------Segment
        channel_start_d = (channel_start[1], channel_start[0])
        channel_cloud_pixels, channel_cloud, _, channel_waypoints, possible_channel_end_pts = \
            g.segment_channel(channel_start_d, use_pixel=True)
        # ----------------------

        # waypoint_first= g.ij_to_point(channel_waypoints[0]).data
        print('channel cloud shape', channel_cloud.shape)
        print('channel waypoints one case:', channel_waypoints[0])
        # print('channel waypoints one case adjusted', waypoint_first)
        print('channel cloud one case', channel_cloud[-1])
        print('channel cloud one case', channel_cloud.data[-1])
        print('channel cloud', channel_cloud)
        print('location test', np.where(channel_cloud.data == channel_cloud.data[-1]))
        print('channel waypoints', channel_waypoints)
        plt.scatter(x = [j[1] for j in channel_waypoints], y=[i[0] for i in channel_waypoints],c='c')
        plt.scatter(x = [j[1] for j in cable_waypoints], y=[i[0] for i in cable_waypoints],c='0.75')
        plt.scatter(x = [j[1] for j in possible_channel_end_pts], y=[i[0] for i in possible_channel_end_pts],c='0.45')
        plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        plt.scatter(x=cable_pt[1], y=cable_pt[0], c='w')
        plt.title("cable and channel start points and waypoints")
        plt.imshow(three_mat_depth, interpolation="nearest")
        plt.show()
        
        transformed_channel_cloud = new_transf.apply(channel_cloud)
        image_channel = iface.cam.intrinsics.project_to_image(
            transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
        image_channel_data = image_channel._image_data()
        
        make_bounding_boxes(image_channel_data)
        
        image_channel_data = gaussian_filter(image_channel_data, sigma=1)
        channel_skeleton = skeletonize_img(image_channel_data)

        channel_len, channel_endpoints = find_length_and_endpoints(channel_skeleton)


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
            plt.title("copy_channel_data")
            plt.imshow(copy_channel_data, interpolation="nearest")
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
        if DISPLAY:
            print("img skeleton")
            plt.title("img_skeleton data")
            plt.scatter(x=[j[0][0] for j in features], y = [j[0][1] for j in features], c = '0.2')
            plt.scatter(x=[j[0] for j in endpoints], y = [j[1] for j in endpoints], c = 'm')
            plt.imshow(img_skeleton, interpolation="nearest")
            plt.show()
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
        

        len_3d, channel_waypoints_sorted = determine_length(channel_cloud, endpoints, channel_waypoints)
        print("Len 3D", len_3d)
        print("Channel Waypoints Sorted", channel_waypoints_sorted)
        
        len_pixel = determine_length_pixels(endpoints, channel_waypoints_sorted) 
        print("Len Pixel:", len_pixel)




    #### FINDING ENDPOINTS OF THE CABLE ##################
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
        # assert pick is not None and place is not None
        # take_action(pick, place, 0)
        # START PICK AND PLACE ___________________________________

        ##### THE ORDER IN WHICH YOU PICK AND PLACE POINTS SHOULD BE 
        linalg_norm = lambda x,y: np.sqrt((x[0]-y[1])**2 + (x[1]-y[0])**2)
        # INSERT BFS CODE HERE!!!  
         

            
        # our new "place" i.e. channel endpoint relative to where we pick waypoints is the prev_channel_pt
        # we want to update this for the next interation as well
        sorted_channel_waypoints = sorted(channel_waypoints, key = lambda x: linalg_norm(place , x))
        curr_channel_pt = place
        if prev_channel_pt != []:
            place = prev_channel_pt
        prev_channel_pt = curr_channel_pt



        # if the endpoint of the rope is too close to the end point of the channel we just say our cable endpoint is our channel endpoint
        # OR if the rope is already attached to the channel by virtue of not being TWO_ENDS
        CABLE_CHANNEL_ENDPOINT_ACCEPTABLE_DIFF = 60
        if not TWO_ENDS or np.linalg.norm(np.array([pick[1]-place[1],pick[0]-place[0]])) < CABLE_CHANNEL_ENDPOINT_ACCEPTABLE_DIFF:
            pick = place    
        
        sorted_cable_waypoints = sorted(cable_waypoints, key = lambda x: linalg_norm(pick,x))
        print("checking that cable waypoints works")
        for i in sorted_cable_waypoints:
            print(linalg_norm(pick,i))
        # basically saying we want our waypoints to be about a fifth of the distance between the channel end points
        acceptable_dist = linalg_norm(place ,place_2) / 4
        for waypoint in sorted_cable_waypoints:
            if linalg_norm(pick , waypoint) > acceptable_dist:
                dist2waypoint = linalg_norm(pick , waypoint)
                print("this is pick", pick)
                print("this is waypoint", waypoint)
                print("this is the calculated norm between them", dist2waypoint)
                waypoint_pick = waypoint
                break




        closest_dst = 10000
        sorted_channel_waypoints = sorted(channel_waypoints, key = lambda x: linalg_norm(place , x))
        for channel_waypoint in sorted_channel_waypoints:
            diff_in_dst = abs(linalg_norm(place , channel_waypoint ) - dist2waypoint)
            print("this is the calculated nrom between them", linalg_norm(place , channel_waypoint ))
            if diff_in_dst < closest_dst:
                waypoint_place = channel_waypoint
                dist2waypoint1 = linalg_norm(place, channel_waypoint)
                closest_dst = diff_in_dst
        print("this is place", place)
        print("this is waypoint_place", waypoint_place)
        print("this is the calculated nrom between them", dist2waypoint1)
        

        plt.title("pick and place locations")
        plt.scatter(x = [j[1] for j in channel_waypoints], y=[i[0] for i in channel_waypoints],c='c')
        plt.scatter(x = [j[1] for j in cable_waypoints], y=[i[0] for i in cable_waypoints],c='0.75')
        plt.scatter(x=[pick[0], place[0]], y = [pick[1], place[1]], c='b')
        plt.scatter(x=[waypoint_pick[1], waypoint_place[1]], y = [waypoint_pick[0], waypoint_place[0]], c='r')
        plt.imshow(three_mat_depth, interpolation="nearest")
        plt.show()

        # need to do this so that the points are in the proper place!
        waypoint_pick1 = waypoint_pick[1], waypoint_pick[0]
        waypoint_place1 = waypoint_place[1], waypoint_place[0]
        
        # if the endpoint of the rope is too close to the end point of the channel
        if np.linalg.norm(np.array([pick[1]-place[1],pick[0]-place[0]])) < CABLE_CHANNEL_ENDPOINT_ACCEPTABLE_DIFF:
            take_action(waypoint_pick1, waypoint_place1, 0)
            print("skip")
        else:
            take_action_2(pick, waypoint_pick1, place, waypoint_place1)

        start_place = place
        old_place = waypoint_place


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
    if DISPLAY:
        plt.title("color image data")
        plt.imshow(img.color.data, interpolation="nearest")
        plt.show()
    three_mat_color = img.color.data
    three_mat_depth = img.depth.data

    last_depth_image_scan = three_mat_depth

    print("BEGINNING PUSHING")

    total_pushes = 0


    # PACKING __________________________________________________
    # if not PUSH_DETECT:
    #     push_action_endpoints(
    #         new_place_point, [new_endpoint_1_point, new_endpoint_2_point], iface)
    # else:
    #     # Move across the entire length of cable in interval and see if any point is not well fit. If ANY point is 
    #     # found to not be well fit, run push action again 
    #     while (True):
    #         push_action_endpoints(
    #             new_place_point, [new_endpoint_1_point, new_endpoint_2_point], iface, False)
    #         img = iface.take_image()
    #         depth = img.depth.data
    #         print(depth)
    #         start = np.array([endpoints[0][0], endpoints[0][1]])
    #         end = np.array([endpoints[1][0], endpoints[1][1]])
    #         move_vector = (end-start)/np.linalg.norm(end-start)
    #         current = copy.deepcopy(start)
    #         loop_again = False
    #         tolerance = 0.0042
    #         interval_scaling = 4
    #         print("START: ", start)
    #         print("END: ", end)
    #         print("MOVE VECTOR: ", move_vector)
    #         for count in range(210):
    #             curr_depth = depth[int(math.floor(current[1]))][int(
    #                 math.floor(current[0]))]
    #             depth_lower = depth[int(math.floor(
    #                 current[1]+9))][int(math.floor(current[0]))]
    #             print("CURRENT POINT: ", [int(math.floor(current[0])), int(math.floor(
    #                 current[1]))], " CURRENT DEPTH: ", curr_depth, " DEPTH_LOWER: ", depth_lower)
    #             if (curr_depth != 0 and depth_lower != 0 and (abs(depth_lower - curr_depth) > tolerance)):
    #                 loop_again = True
    #                 print("EXCEEDED DEPTH TOLERANCE!")
    #             current[0] += move_vector[0]*interval_scaling
    #             current[1] += move_vector[1]*interval_scaling
    #         if DISPLAY:
    #             plt.imshow(img.depth.data, interpolation="nearest")
    #             plt.show()
    #         if not loop_again:
    #             break

print("Done with script, can end")
