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


def take_action(pick, place, angle):

    #handle the actual grabbing motion
    l_grasp=None
    r_grasp=None
    #single grasp

#----------------------------------------------
    #### NEED TO ADD CODE THAT FIGURES OUT WHICH SPOT IS EASIER FOR THE ROBOT TO PICK AND PLACE FROM!!!
#--------------------------------------------

    print("This is pick_action:", pick)
    grasp = g.single_grasp(pick,.002,iface.L_TCP)
    g.col_interface.visualize_grasps([grasp.pose],iface.L_TCP)
    #wrist = grasp.pose*iface.R_TCP.inverse()
    print("grabbing with left arm")
    l_grasp=grasp
    l_grasp.pose.from_frame=YK.l_tcp_frame
    iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.set_speed((.1, 1))
    yumi_left = iface.y.left
    iface.close_grippers()
    
    print("This is place action:", place)

#--------- ---------------
    # NEED TO MODIFY THE PLACE LOCATION SO THAT THE ROBOT ACTUALLY DROPS IT ABOVE THE CHANNEL AND DOESN"T COLLIDE WITH IT!!!!
#-------------------
    count = 0
    grasp = None
    while count < 15:
        try:
            grasp = g.single_grasp(place,.000,iface.L_TCP, place_mode=True)
            g.col_interface.visualize_grasps([grasp.pose],iface.L_TCP)
            #wrist = grasp.pose*iface.R_TCP.inverse()
            print("the grasp pose is", grasp.pose)
            print("grabbing with left arm")
            l_grasp=grasp
            l_grasp.pose.from_frame=YK.l_tcp_frame
            iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp, dive = False)
            iface.set_speed((.1, 1))
            yumi_left = iface.y.left
            break
        except:
            print("failed to find a grasp")
            count += 1
    # the grasp being none means that our standard place function failed so now we will try Cory's original work
    if grasp == None:
        iface.go_delta(l_trans=[0, 0, 0.1])  # lift
        time.sleep(1)
        iface.set_speed((.1, 1))
        place_point = coord_to_point(place)
        pick_point = coord_to_point(pick)
        delta_xy = [place_point[i] - pick_point[i] for i in range(2)]
        #delta_z = [place_point[2]-pick_point[2]] #[three_mat_depth[place[1]][place[0]] - three_mat_depth[pick[1]][pick[0]]]
        delta_z = [0]
        delta = delta_xy + delta_z
        change_height = 0
        delta[2] = delta[2] + change_height
        iface.go_delta(l_trans=[0, 0, 0.1])
        iface.go_delta(l_trans=delta)
        time.sleep(1)
        iface.go_delta(l_trans=[0, 0, -0.06])
        time.sleep(3)
        iface.open_grippers()
    # iface.set_speed(SPEED)
    iface.set_speed((.1, 1))
    time.sleep(2)

    iface.go_delta(l_trans=[0, 0, 0.1])  # lift
    iface.home()
    iface.sync()




    print(yumi_left.get_pose())

    # iface.go_delta(l_trans=[0, 0, 0.2])  # lift
    # # if angle != 0:
    # #    pick[2] = pick[2] + .05
    # iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
    #                                              from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # # if angle != 0:
    # #    rotate(angle, iface)
    # #    iface.go_delta(l_trans=[0, 0, -.05])
    # # iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]),
    # #                                            from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))

    # iface.close_grippers()
    # time.sleep(3)
    # iface.go_delta(l_trans=[0, 0, 0.1])  # lift
    # time.sleep(1)
    # iface.set_speed((.1, 1))
    # delta = [place[i] - pick[i] for i in range(3)]
    # change_height = 0
    # delta[2] = delta[2] + change_height
    # iface.go_delta(l_trans=delta)
    # time.sleep(1)
    # iface.go_delta(l_trans=[0, 0, -0.06])
    # time.sleep(3)
    # iface.open_grippers()
    # iface.home()
    # iface.sync()
    # # iface.set_speed(SPEED)
    # iface.set_speed((.1, 1))
    # time.sleep(2)
    # iface.open_grippers()


# Use this for two ends of the cable free
def take_action_2(pick, pick_2, place, place_2):

    # print("grabbing with left arm")
    # # GRIP LEFT
    # iface.set_speed((.1, 1))
    # iface.go_delta(l_trans=[0, 0, 0.2])  # lift
    # iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
    #                                              from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # # GRIP RIGHT
    # iface.go_delta(r_trans=[0, 0, 0.23])  # lift
    # time.sleep(3)
    # iface.go_cartesian(r_targets=[RigidTransform(translation=pick_2, rotation=Interface.GRIP_DOWN_R,
    #                                              from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # iface.close_grippers()
    # time.sleep(3)
    # # LIFT AND MOVE LEFT
    # iface.go_delta(l_trans=[0, 0, 0.18])  # lift
    # time.sleep(1)
    # delta = [place[i] - pick[i] for i in range(3)]
    # change_height = 0
    # delta[2] = delta[2] + change_height
    # iface.go_delta(l_trans=delta)
    # # LIFT AND MOVE RIGHT
    # iface.go_delta(r_trans=[0, 0, 0.09])  # lift
    # time.sleep(1)
    # delta = [place_2[i] - pick_2[i] for i in range(3)]
    # change_height = 0
    # delta[2] = delta[2] + change_height
    # # Re-write go-delta because previous was error!
    # l_delta, r_delta = None, None
    # r_trans = delta
    # if r_trans is not None:
    #     r_cur = iface.y.right.get_pose()
    #     r_delta = RigidTransform(
    #         translation=r_trans, from_frame=r_cur.to_frame, to_frame=r_cur.to_frame)
    #     r_new = r_delta*r_cur
    # if r_delta is not None:
    #     iface.y.right.goto_pose(r_new, speed=iface.speed)
    # # DROP BOTH
    # time.sleep(2)
    # iface.go_delta(l_trans=[0, 0, -0.12])
    # #iface.go_delta(r_trans=[0, 0, -0.015])
    # time.sleep(3)
    # iface.open_grippers()
    # time.sleep(2)
    # iface.go_delta(l_trans=[0, 0, 0.1])
    # iface.go_delta(r_trans=[0, 0, 0.1])

    print("This is pick_action:", pick, pick_2)
    l_grasp, r_grasp = g.double_grasp(pick, pick_2,.007, .007, iface.L_TCP, iface.R_TCP)
    print("grabbing with left arm")
    l_grasp.pose.from_frame=YK.l_tcp_frame
    r_grasp.pose.from_frame=YK.r_tcp_frame
    iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.set_speed((.1, 1))

    print("This is the place action", place, place_2)
    l_grasp, r_grasp = g.double_grasp(place, place_2,.000, .000, iface.L_TCP, iface.R_TCP)
    print("grabbing with left arm")
    l_grasp.pose.from_frame=YK.l_tcp_frame
    r_grasp.pose.from_frame=YK.r_tcp_frame
    iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.set_speed((.1, 1))

    iface.home()
    iface.sync()
    time.sleep(2)

def push_down(point, depth=0.012):
    iface.set_speed((.1, 1))


    xind, yind = point
    lin_ind = int(original_depth_image_scan.depth.ij_to_linear(np.array(xind), np.array(yind)))
    push_point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    new_push_point_data = np.array(
        [push_point.y, push_point.x, push_point.z + depth])
    new_push_point = Point(
        new_push_point_data, frame=push_point.frame)
    
    
    iface.go_cartesian(l_targets=[RigidTransform(translation=new_push_point, rotation=Interface.GRIP_DOWN_R,
                                                     from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    time.sleep(2)
    iface.go_delta(l_trans=[0, 0, 0.1])  # lift

    return None

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
    
def determine_length(object_points, endpoint_list, waypoints_list):
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

    pdb.set_trace()
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
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='b')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 
    pdb.set_trace()
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

    ### Cory's work!!!
        # pixel_r = 0
        # pixel_c = 0
        # points_3d = iface.cam.intrinsics.deproject(img.depth)
        # lower = 0
        # upper = 190
        # delete_later = []
        # max_score = 0
        # max_scoring_loc = (0, 0)
        # highlight_upper = 256
        # highlight_lower = 254
        # for r in range(len(three_mat_color)):
        #     for c in range(len(three_mat_color[r])):
        #         if (highlight_lower < three_mat_color[r][c][0] <= highlight_upper and 210 < three_mat_color[r][c][1] <= highlight_upper and 210 < three_mat_color[r][c][2] <= highlight_upper):
        #             curr_score = 0
        #             for add in range(1, 10):
        #                 if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[max(0, r-add)][c][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[r][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[max(0, r-add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #             if (curr_score > max_score):
        #                 max_scoring_loc = (c, r)
        #                 max_score = curr_score

        #         if (lower < three_mat_color[r][c][1] < upper):
        #             delete_later += [(c, r)]
        # copy_color = copy.deepcopy(three_mat_color)
        # remove_glare_tolerance = 170
        # remove_glare_tolerance_2 = 120
        # for r in range(len(copy_color)):
        #     for c in range(len(copy_color[r])):
        #         dist = dist = np.linalg.norm(
        #             np.array([r-max_scoring_loc[1], c-max_scoring_loc[0]]))
        #         if dist < remove_glare_tolerance:
        #             copy_color[r][c][0] = 0.0
        #             copy_color[r][c][1] = 0.0
        #             copy_color[r][c][2] = 0.0
        #         if dist < remove_glare_tolerance_2:
        #             delete_later += [(c, r)]
        # highlight_lower = 200
        # highlight_upper = 255
        # max_score = 0
        # max_scoring_loc = (0, 0)
        # for r in range(len(copy_color)):
        #     for c in range(len(copy_color[r])):
        #         if (highlight_lower < copy_color[r][c][0] <= highlight_upper and 210 < copy_color[r][c][1] <= highlight_upper and 210 < copy_color[r][c][2] <= highlight_upper):
        #             curr_score = 0
        #             for add in range(1, 13):
        #                 if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][c][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[max(0, r-add)][c][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[r][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[r][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[max(0, r-add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
        #                     curr_score += 1
        #                 if (highlight_lower < copy_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
        #                     curr_score += 1
        #             if (curr_score > max_score):
        #                 max_scoring_loc = (c, r)
        #                 max_score = curr_score
        # if DISPLAY:
        #     plt.imshow(copy_color, interpolation="nearest")
        #     plt.show()
        # loc = max_scoring_loc
    ### Cory's Work!!!
        
        
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

        plt.imshow(edges)
        plt.show()
        

        #depth_image = cv2.imread(three_mat_depth, cv2.IMREAD_UNCHANGED)
        
        # performing image in painting, to remove all of the 0 values in the depth image with an average
        # zero_pixels = np.where(three_mat_depth == 0)

        # for i in range(len(zero_pixels[0])):
        #     x = zero_pixels[0][i]
        #     y = zero_pixels[1][i]

        #     patch_size = 5
        #     patch = three_mat_depth[x-patch_size:x+patch_size+1, y-patch_size:y+patch_size+1]
        #     patch_nonzero = patch[np.nonzero(patch)]
        #     avg_value = np.mean(patch)
        #     if 0 < x < len(three_mat_depth[0]) and 0 < y < len(three_mat_depth):
        #         three_mat_depth[x,y] = three_mat_depth[x-1,y]

        # mask = (three_mat_depth ==0).astype(np.uint8)
        # three_mat_depth = cv2.inpaint(three_mat_depth, mask, 7, cv2.INPAINT_NS)


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

                        # if 0.01 < diff1 < 0.014 or 0.01 < diff2 < 0.014 or 0.01 < diff3 < 0.014 or 0.01 < diff4 < 0.014:
                        #     candidate_channel_pts += [(r,c)]     
                    # throw away values that we know differ by too much, this is cause if you take the avg of diffs 
                    # if diff1 > 0.02:
                    #     diff1 = 0
                    # if diff2 > 0.02:
                    #     diff2 = 0
                    # if diff3 > 0.02:
                    #     diff3 = 0
                    # if diff4 > 0.02:
                    #     diff4 = 0 
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

        # for r in range(len(three_mat_color)):
        #     for c in range(len(three_mat_color[r])):
        #         if (lower < three_mat_color[r][c][0] < upper):
        #             curr_edges = 0
        #             for add in range(1, 11):
        #                 if (lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[max(0, r-add)][c][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[r][max(0, c-add)][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
        #                     curr_edges += 1
        #                 if (lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] < upper):
        #                     curr_edges += 1
        #             if (curr_edges > max_edges):
        #                 max_edges = curr_edges
        #                 channel_start = (c, r)
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
        plt.scatter(x = [j[1] for j in candidate_channel_pts], y=[i[0] for i in candidate_channel_pts],c='r')
        plt.scatter(x=channel_edge_pt[1], y=channel_edge_pt[0], c='b')
        plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        plt.scatter(x=cable_pt[1], y=cable_pt[0], c='w')
        plt.imshow(three_mat_depth, interpolation="nearest")
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
            plt.show()
        
        make_bounding_boxes(di._image_data())
        cable_skeleton = skeletonize_img(di._image_data())
        
        cable_len, cable_endpoints_1 = find_length_and_endpoints(cable_skeleton)
        
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
        if DISPLAY:
            plt.imshow(new_di._image_data(), interpolation="nearest")
            plt.show()

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

        # origin_x = 0
        # origin_y = 0
        # min_dist = 100000000
        # max_dist = 0
        # min_all_solns = (0, 0)
        # max_all_solns = (0, 0)
        # for soln in all_solns:
        #     dist = np.linalg.norm(
        #         np.array([origin_x-soln[1], origin_y-soln[0]]))
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_all_solns = soln
        #     if TWO_ENDS:
        #         if dist > max_dist:
        #             max_dist = dist
        #             max_all_solns = soln

        # scaled_test_loc = [min_all_solns[0]*compress_factor,
        #                    min_all_solns[1]*compress_factor]
        # scaled_test_loc_2 = []
        # if TWO_ENDS:
        #     scaled_test_loc_2 = [max_all_solns[0]*compress_factor,
        #                          max_all_solns[1]*compress_factor]
        # if (scaled_test_loc[0] != 0):
        #     scaled_test_loc[0] = scaled_test_loc[0] - int(compress_factor/2)
        # if (scaled_test_loc[1] != 0):
        #     scaled_test_loc[1] = scaled_test_loc[1] - int(compress_factor/2)
        # if TWO_ENDS:
        #     if (scaled_test_loc_2[0] != 0):
        #         scaled_test_loc_2[0] = scaled_test_loc_2[0] - \
        #             int(compress_factor/2)
        #     if (scaled_test_loc_2[1] != 0):
        #         scaled_test_loc_2[1] = scaled_test_loc_2[1] - \
        #             int(compress_factor/2)
        # if DISPLAY:
        #     plt.imshow(compressed_map, interpolation="nearest")
        #     plt.show()
        # min_dist = 10000
        # min_dist_2 = 1000
        # candidate_rope_loc = (0, 0)
        # candidate_rope_loc_2 = (0, 0)
        # for r in range(len(new_di_data)):
        #     for c in range(len(new_di_data[r])):
        #         if (di_data[r][c][0] != 0):
        #             dist = np.linalg.norm(
        #                 np.array([r-scaled_test_loc[1], c-scaled_test_loc[0]]))
        #             if (dist < min_dist):
        #                 candidate_rope_loc = (c, r)
        #                 min_dist = dist
        #             if TWO_ENDS:
        #                 dist_2 = np.linalg.norm(
        #                     np.array([r-scaled_test_loc_2[1], c-scaled_test_loc_2[0]]))
        #                 if (dist_2 < min_dist_2):
        #                     candidate_rope_loc_2 = (c, r)
        #                     min_dist_2 = dist_2
        # min_loc = candidate_rope_loc
        # min_loc_2 = (0, 0)
        # if TWO_ENDS:
        #     min_loc_2 = candidate_rope_loc_2
        #     print("FITTED POINT: " + str(min_loc))
        #     print("FITTED POINT OF OTHER END: " + str(min_loc_2))
        # if DISPLAY:
        #     plt.scatter(x=[min_loc[0], min_loc_2[0]], y = [min_loc[1], min_loc_2[1]], c='w')
        #     plt.scatter(x=[j[0]*compress_factor - int(compress_factor/2) for j in all_solns], y = [j[1]*compress_factor - int(compress_factor/2) for j in all_solns], c='b')
        #     plt.imshow(new_di_data, interpolation="nearest")
        #     plt.show()




        # # ----------------------FIND END OF CHANNEL
        # lower = 254
        # upper = 256
        # channel_start = (0, 0)
        # max_edges = 0
        # candidate_channel_pts = []
        
        # # guess for what 0.5in is in terms of depth
        # depth_diff_goal = 0.016
        # # threshold to allow for error
        # depth_threshold = 0.002

        # plt.imshow(edges)
        # plt.show()
        

        # #depth_image = cv2.imread(three_mat_depth, cv2.IMREAD_UNCHANGED)
        
        # # performing image in painting, to remove all of the 0 values in the depth image with an average
        # # zero_pixels = np.where(three_mat_depth == 0)

        # # for i in range(len(zero_pixels[0])):
        # #     x = zero_pixels[0][i]
        # #     y = zero_pixels[1][i]

        # #     patch_size = 5
        # #     patch = three_mat_depth[x-patch_size:x+patch_size+1, y-patch_size:y+patch_size+1]
        # #     patch_nonzero = patch[np.nonzero(patch)]
        # #     avg_value = np.mean(patch)
        # #     if 0 < x < len(three_mat_depth[0]) and 0 < y < len(three_mat_depth):
        # #         three_mat_depth[x,y] = three_mat_depth[x-1,y]

        # # mask = (three_mat_depth ==0).astype(np.uint8)
        # # three_mat_depth = cv2.inpaint(three_mat_depth, mask, 7, cv2.INPAINT_NS)


        # for r in range(len(edges)):
        #     for c in range(len(edges[r])):
        #         if (lower < edges[r][c]< upper):
        #             diff1 = 0
        #             diff2 = 0
        #             diff3 = 0
        #             diff4 = 0
        #             for add in range(1, 4):
        #                 if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
        #                     break
        #                 # top - bottom
        #                 diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c]) 
        #                 # left - right
        #                 diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
        #                 # top left - bottom right
        #                 diff3 = abs(three_mat_depth[r-add][c-add] - three_mat_depth[r+add][r+add])
        #                 # top right - bottom left
        #                 diff4 = abs(three_mat_depth[r-add][c+add] - three_mat_depth[r+add][r-add])

        #                 # if 0.01 < diff1 < 0.014 or 0.01 < diff2 < 0.014 or 0.01 < diff3 < 0.014 or 0.01 < diff4 < 0.014:
        #                 #     candidate_channel_pts += [(r,c)]     
        #             # throw away values that we know differ by too much, this is cause if you take the avg of diffs 
        #             # if diff1 > 0.02:
        #             #     diff1 = 0
        #             # if diff2 > 0.02:
        #             #     diff2 = 0
        #             # if diff3 > 0.02:
        #             #     diff3 = 0
        #             # if diff4 > 0.02:
        #             #     diff4 = 0 
        #             if diff1 > 0.02 or diff2 > 0.02 or diff3 > 0.02 or diff4 > 0.02:
        #                 continue
        #             if 0.01 <= np.mean(np.array([diff1, diff2, diff3, diff4])) <= 0.014:
        #                 candidate_channel_pts += [(r,c)]
        #                 #print("the detected avg was: ", np.mean(np.array([diff1, diff2, diff3, diff4])))
        # print("Candidate Edge pts: ", candidate_channel_pts)
        # # need to figure out which edge point is in fact the best one for our channel
        # # i.e. highest up, and pick a point that is actually in the channel
        # max_depth = 100000
        # min_depth = 0
        # channel_edge_pt = (0,0)
        # channel_start = (0,0)
        # sorted_candidate_channel_pts = sorted(candidate_channel_pts, key=lambda x: three_mat_depth[x[0]][x[1]])



        # print("The sorted list is: ", sorted_candidate_channel_pts)
        # #channel_edge_pt = sorted_candidate_channel_pts[0]
        # possible_cable_edge_pt = sorted_candidate_channel_pts[-1]
        # #print("the edge with lightest depth is: ", three_mat_depth[channel_edge_pt[0]][channel_edge_pt[1]])
        # print("the edge with deepest depth is: ", three_mat_depth[possible_cable_edge_pt[0]][possible_cable_edge_pt[1]])

        # for candidate_pt in candidate_channel_pts:
        #     r = candidate_pt[0]
        #     c = candidate_pt[1]
        #     print("r", r, "c", c, "my depth is: ", three_mat_depth[r][c])
        #     if 0 < three_mat_depth[r][c] < max_depth:
        #         print("max depth:", max_depth)
        #         channel_edge_pt = (r,c)
        #         max_depth = three_mat_depth[r][c]
        #     if three_mat_depth[r][c] > min_depth:
        #         possible_cable_edge_pt = (r,c)
        #         min_depth = three_mat_depth[r][c]
        # print("The edge of the channel is: ", channel_edge_pt)
        # r,c = channel_edge_pt
        # possible_channel_pts = []


        # ##### NEED TO REMOVE THE EDGES OF VALUE 0 FROM THE SAMPLE BASE!!!!!
        # index = 0
        # while index < len(sorted_candidate_channel_pts) and channel_start == (0,0):
        #     channel_edge_pt = sorted_candidate_channel_pts[index]
        #     r,c = channel_edge_pt
        #     if three_mat_depth[r][c] == 0.0:
        #         index += 1
        #         continue
        #     for add in range(1, 4):
        #         if (r-add < 0 or c-add < 0) or (r+add >= len(three_mat_depth) or c+add >= len(three_mat_depth[r])):
        #             break
        #         # left - right
        #         diff1 = abs(three_mat_depth[r-add][c] - three_mat_depth[r+add][c])
        #         diff2 = abs(three_mat_depth[r][c-add] - three_mat_depth[r][c+add])
        #         if 0.01 <= diff1 < 0.014: # prev upper was 0.016
        #             if three_mat_depth[r-add][c] > three_mat_depth[r+add][c]:
        #                 channel_start = (r-add, c)
        #                 possible_channel_pts += [(r-add, c)]
        #             else:
        #                 channel_start = (r+add, c)
        #                 possible_channel_pts += [(r+add, c)]
        #         if 0.01 <= diff2 < 0.014: #prev upper was 0.016
        #             if three_mat_depth[r][c-add] > three_mat_depth[r][c+add]:
        #                 channel_start = (r, c-add)
        #                 possible_channel_pts += [(r, c-add)]
        #             else:
        #                 channel_start = (r, c+add)
        #                 possible_channel_pts += [(r, c+add)]
        #     # the point in the channel was not found, so we need to look at the next best one
        #     if channel_start == (0,0):
        #         index += 1
        # # channel_start = (channel_edge_pt[1], channel_edge_pt[0])
        # print("possible channel pts: ", possible_channel_pts)
        # print("The chosen channel_pt is: ", channel_start)


        # plt.imshow(edges, cmap='gray')
        # plt.scatter(x = [j[1] for j in candidate_channel_pts], y=[i[0] for i in candidate_channel_pts],c='r')
        # plt.scatter(x=channel_edge_pt[1], y=channel_edge_pt[0], c='b')
        # plt.scatter(x=channel_start[1], y=channel_start[0], c='m')
        # plt.scatter(x=possible_cable_edge_pt[1], y=possible_cable_edge_pt[0], c='w')
        # plt.imshow(three_mat_depth, interpolation="nearest")
        # plt.show()
        # # for r in range(len(three_mat_color)):
        # #     for c in range(len(three_mat_color[r])):
        # #         if (lower < three_mat_color[r][c][0] < upper):
        # #             curr_edges = 0
        # #             for add in range(1, 11):
        # #                 if (lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[max(0, r-add)][c][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[r][max(0, c-add)][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
        # #                     curr_edges += 1
        # #                 if (lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] < upper):
        # #                     curr_edges += 1
        # #             if (curr_edges > max_edges):
        # #                 max_edges = curr_edges
        # #                 channel_start = (c, r)
        # print("CHANNEL_START: "+str(channel_start))
        # channel_start_d = (channel_start[1], channel_start[0])
        # channel_cloud, _ = g.segment_channel(channel_start_d)
        # transformed_channel_cloud = new_transf.apply(channel_cloud)
        # image_channel = iface.cam.intrinsics.project_to_image(
        #     transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
        # image_channel_data = image_channel._image_data()
        # copy_channel_data = copy.deepcopy(image_channel_data)
        # lower = 80
        # upper = 255

        channel_start_d = (channel_start[1], channel_start[0])
        # channel_cloud, _, channel_waypoints, possible_channel_end_pts = g.segment_channel(channel_start_d)
        channel_cloud_pixels, channel_cloud, _, channel_waypoints, possible_channel_end_pts = \
            g.segment_channel(channel_start_d, use_pixel=True)
        waypoint_first= g.ij_to_point(channel_waypoints[0]).data
        print('channel cloud shape', channel_cloud.shape)
        print('channel waypoints one case:', channel_waypoints[0])
        print('channel waypoints one case adjusted', waypoint_first)
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





        # if not TWO_ENDS:
        #     take_action(pick, place, 0)
        #     # print("skip")
        # else:
        #     take_action_2(pick, pick_2, place, place_2)





        # # Convert Pick, Place, and Pick 2 to world coordinates:
        # xind, yind = pick
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
        # xind2, yind2 = place
        # lin_ind2 = int(img.depth.ij_to_linear(np.array(xind2), np.array(yind2)))

        # xind3, yind3 = pick_2
        # lin_ind3 = 0
        # if TWO_ENDS:
        #     lin_ind3 = int(img.depth.ij_to_linear(
        #         np.array(xind3), np.array(yind3)))

        # xind4, yind4 = place_2
        # lin_ind4 = 0
        # if TWO_ENDS:
        #     lin_ind4 = int(img.depth.ij_to_linear(
        #         np.array(xind4), np.array(yind4)))

        # points_3d = iface.cam.intrinsics.deproject(img.depth)
        # point = iface.T_PHOXI_BASE*points_3d[lin_ind]
        # point = [p for p in point]
        # point[2] -= 0.0028  # manually adjust height a tiny bit
        # place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]
        # point_2 = None
        # if TWO_ENDS:
        #     point_2 = iface.T_PHOXI_BASE*points_3d[lin_ind3]
        #     point_2 = [p for p in point_2]
        #     point_2[2] -= 0.003  # manually adjust height a tiny bit
        # # Convert Channel End and Channel start to world coordinates
        # xind, yind = place
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
        # place_point = iface.T_PHOXI_BASE*points_3d[lin_ind]
        # place_point_2 = (0, 0)
        # if TWO_ENDS:
        #     place_point_2 = iface.T_PHOXI_BASE*points_3d[lin_ind4]

        # xind, yind = endpoints[0]
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
        # endpoint_1_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

        # xind, yind = endpoints[1]
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
        # endpoint_2_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

        # new_place_point_data = np.array(
        #     [place_point.y, place_point.x, place_point.z])
        # new_place_point = Point(new_place_point_data, frame=place_point.frame)

        # new_endpoint_1_point_data = np.array(
        #     [endpoint_1_point.y, endpoint_1_point.x, endpoint_1_point.z])
        # new_endpoint_1_point = Point(
        #     new_endpoint_1_point_data, frame=endpoint_1_point.frame)

        # new_endpoint_2_point_data = np.array(
        #     [endpoint_2_point.y, endpoint_2_point.x, endpoint_2_point.z])
        # new_endpoint_2_point = Point(
        #     new_endpoint_2_point_data, frame=endpoint_2_point.frame)

        # # FIND THE PRINCIPLE AXIS FOR THE GRIPPER AND ROTATE, THEN TAKE ACTION

        # rope_cloud_data_x = rope_cloud.x_coords
        # rope_cloud_data_y = rope_cloud.y_coords
        # rope_cloud_data_z = rope_cloud.z_coords
        # tolerance = 0.03
        # new_rope_cloud_data_x = []
        # new_rope_cloud_data_y = []
        # new_rope_cloud_data_z = []
        # for count in range(len(rope_cloud_data_x)):
        #     dist = np.linalg.norm(np.array(
        #         [rope_cloud_data_x[count]-point[0], rope_cloud_data_y[count]-point[1]]))
        #     if dist < tolerance:
        #         new_rope_cloud_data_x += [rope_cloud_data_x[count]]
        #         new_rope_cloud_data_y += [rope_cloud_data_y[count]]
        #         new_rope_cloud_data_z += [rope_cloud_data_z[count]]
        # new_rope_cloud_data_x = np.array(new_rope_cloud_data_x)
        # new_rope_cloud_data_y = np.array(new_rope_cloud_data_y)
        # new_rope_cloud_data_z = np.array(new_rope_cloud_data_z)

        # new_rope_cloud = PointCloud(np.array(
        #     [new_rope_cloud_data_x, new_rope_cloud_data_y, new_rope_cloud_data_z]), frame=rope_cloud.frame)

        # transformed_new_rope_cloud = new_transf.apply(new_rope_cloud)
        # di_2 = iface.cam.intrinsics.project_to_image(
        #     transformed_new_rope_cloud, round_px=False)
        # if DISPLAY:
        #     plt.imshow(di_2._image_data(), interpolation="nearest")
        #     plt.show()
        # # START PICK AND PLACE ___________________________________
        # if not TWO_ENDS:
        #     take_action(point, place_point, 0)
        #     # print("skip")
        # else:
        #     take_action_2(point, point_2, place_point, place_point_2)
    
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
        plt.imshow(img.color.data, interpolation="nearest")
        plt.show()
    three_mat_color = img.color.data
    three_mat_depth = img.depth.data

    last_depth_image_scan = three_mat_depth

    print("BEGINNING PUSHING")

    total_pushes = 0
    # binary push method, look at the midpoint waypoint in the channel, see if it's pushed down or not
    # need the chnanel waypoints to be sorted in terms of distance to a given endpoint, whichever endpoint does not matter
    def binary_push(sorted_channel_waypoints):
        iface.close_grippers()
        total_pushes = 0
        high = len(sorted_channel_waypoints) - 1
        low = 0
        # gets the indices in binary order to be evaluated
        def binary_order(low, high):
            if low > high:
                return []
            if low == high:
                return [low]
            mid = (high+low)//2
            print(low, high, mid)
            return [mid]+binary_order(low, mid-1)+binary_order(mid+1, high)
        
        binary_indices = binary_order(low, high)
        pushes_this_run = 0
        while True:
            img = iface.take_image()
            last_depth_image_scan = img.depth.data
            for i in binary_indices: 
                channel_waypoint = sorted_channel_waypoints[i]
                r,c = channel_waypoint
                if  last_depth_image_scan[r][c] - original_depth_image_scan[r][c] > ACCEPTABLE_DEPTH:
                    pushes_this_run += 1
                    push_down(channel_waypoint)
            if pushes_this_run == 0:
                return total_pushes
            total_pushes += pushes_this_run
    
    
    
    
    
    
    # linear push method, just travel across the waypoints until everything is pressed down 
    def linear_push():
        iface.close_grippers()
        total_pushes = 0
        pushes_this_run = 0
        while True:
            img = iface.take_image()
            last_depth_image_scan = img.depth.data
            for channel_waypoint in sorted_channel_waypoints:
                r,c = channel_waypoint
                if  last_depth_image_scan[r][c] - original_depth_image_scan[r][c] > ACCEPTABLE_DEPTH:
                    pushes_this_run += 1
                    push_down(channel_waypoint)
            if pushes_this_run == 0:
                return total_pushes
            total_pushes += pushes_this_run


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
