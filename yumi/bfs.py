from interface_rws import Interface
from yumirws.yumi import YuMiArm, YuMi
from push import push_action_endpoints
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from grasp import Grasp, GraspSelector
from rotate import rotate_from_pointcloud, rotate
from tcps import *
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics, Point, PointCloud
import collections
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import time
import os
import sys

SPEED = (.025, 0.3*np.pi)
iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                  METAL_GRIPPER.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
new_transf = iface.T_PHOXI_BASE.inverse()

img = iface.take_image()
g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)



def check_index(p_array, part, ind = False):
    newp_array = p_array
    p_shape = newp_array.shape
    check = np.where(newp_array == part)
    if len(check[0]) != len(part):
        return False
    else:
        if ind == True:
            return check[0][0]
        return(np.all(np.isclose(check[0], check[0][0])))
    

def determine_length(waypoints_list):
    visited = set()
    distances_list = dict()
    waypoint_found = dict()
    print(len(waypoints_list))
    #start_point = endpoint_list[0]
    # start_point_adj = g.ij_to_point(next_loc).data
    # q = [start_point_adj]
    # NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # counter = 0
    copy_waypoints = np.array([g.ij_to_point(waypoint).data for waypoint in waypoints_list])
    #interest_pt = object_points[start_point]

    test_waypoint = g.ij_to_point((484, 510)).data
    print("test_waypoint", test_waypoint)
    print("copy_waypoint", copy_waypoints)
    print("copy_waypoint_1", copy_waypoints[1])
    # if (test_waypoint in copy_waypoints):
    if np.any(np.all(test_waypoint == copy_waypoints, axis=1)):
        print("Yes")
        index = check_index(copy_waypoints, test_waypoint, ind = True)
        print("Index", index)



    # while len(q) > 0:  
    #     next_loc = q.pop()
    #     visited.add(next_loc)
    #     #return index of next_point in object_points
    #     index = check_index(object_points, next_loc, ind = True)
    #     require = False
    #     min_dist = float('inf')
    #     for n in NEIGHS:
    #         test_loc = object_points[index[0]+n[0], index[1]+n[1]]
    #         if (test_loc in visited):
    #             continue
    #         else: 
    #             #test_pt = g.ij_to_point(test_loc).data
    #             if (test_loc in copy_waypoints):
    #                 require = True
    #                 diff = start_point-test_loc
    #                 dist = diff.dot(diff)
    #                 waypoint_found[dist] = test_loc
    #                 if dist < min_dist:
    #                     min_dist = dist
    #             else:
    #                 q.append(test_loc)
    #     if require == True:  
    #         #index = check_index(object_points, waypoint_found[min_dist], ind = True)
    #         distances_list[waypoint_found[min_dist]]= min_dist
    #         q.append(waypoint_found[min_dist])


channel_waypoints = [(484, 510), (447, 540), (426, 578), (388, 607), (377, 636), (330, 667), (313, 689), (290, 724), (264, 745),
                      (244, 784), (223, 816), (258, 773), (289, 724), (335, 671), (370, 621), (436, 561), (484, 508), (501, 472), (532, 435), 
                      (578, 399), (599, 361), (632, 323), (663, 285), (698, 247), (734, 209), (707, 239), (668, 280), (625, 321), (596, 361), (556, 400), (541, 439), (499, 478)]

channel_waypoints = np.array([list(elem) for elem in channel_waypoints])
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

# selected_pts_1 = evenly_sample_points_dist(channel_waypoints, 20, 13)
# selected_pts_2 = evenly_sample_points(channel_waypoints, 20)

# xs = [x[0] for x in channel_waypoints]
# ys = [x[1] for x in channel_waypoints]
# plt.scatter(x=xs, y=ys, c='r')
# xs = [x[0] for x in selected_pts_1]
# ys = [x[1] for x in selected_pts_1]
# plt.scatter(x=xs, y=ys, c='g')
# xs = [x[0] for x in selected_pts_2]
# ys = [x[1] for x in selected_pts_2]
# plt.scatter(x=xs, y=ys, c='b')
# plt.show()

def make_bounding_boxes():
    img = iface.take_image()
    # convert to grayscale
    img2 = np.array(img.color.data, dtype=np.uint8)
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("x,y,w,h:",x,y,w,h)
    
    # save resulting image
    cv2.imwrite('two_blobs_result.jpg',result)      

    # show thresh and result    
    cv2.imshow("bounding_box", result)

make_bounding_boxes()


#determine_length(channel_waypoints)


# def determine_length(object_points, endpoint_list, waypoints_list):
#     # transformed_channel_cloud = new_transf.apply(object_points)
#     # image_obj_points = iface.cam.intrinsics.project_to_image(
#     #     transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
#     # image_obj_points_data = image_obj_points._image_data()
#     #g = GraspSelector(image_obj_points, iface.cam.intrinsics, iface.T_PHOXI_BASE)
#     visited = set()
#     distances_list = dict()
#     waypoint_found = dict()
#     start_point = endpoint_list[0]
#     q = [start_point]
#     # RADIUS2 = 1  # distance from original point before termination
#     # CLOSE2 = .002**2
#     # DELTA = 0.0002*3
#     NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
#     counter = 0
#     copy_waypoints = copy.deepcopy(waypoints_list)
#     # waypoints = []
#     # endpoints = []
#     #index into object_point 
#     interest_pt = object_points[start_point]

#     while len(q) > 0:  
#         next_loc = q.pop()
#         next_point = g.ij_to_point(next_loc).data
#         visited.add(next_loc)
#         #return index of next_point in object_points
#         index = check_index(object_points, next_point, ind = True)
#         require = False
#         min_dist = float('inf')
#         for n in NEIGHS:
#             test_loc = object_points[index[0]+n[0], index[1]+n[1]]
#             if (test_loc in visited):
#                 continue
#             # if test_loc[0] >= self.depth.width or test_loc[0] < 0 \
#             #         or test_loc[1] >= self.depth.height or test_loc[1] < 0:
#             #     continue
#             else: 
#                 test_pt = g.ij_to_point(test_loc).data
#                 if (test_loc in copy_waypoints):
#                     require = True
#                     diff = start_point-test_pt
#                     dist = diff.dot(diff)
#                     waypoint_found[dist] = test_loc
#                     if dist < min_dist:
#                         min_dist = dist
#                 else:
#                     q.append(test_loc)
#         if require == True:  
#             index = check_index(object_points, waypoint_found[min_dist], ind = True)
#             distances_list[waypoint_found[min_dist]]= min_dist
#             q.append(waypoint_found[min_dist])
#     return PointCloud(np.array(pts).T, "base_link"), PointCloud(np.array(closepts).T, "base_link").mean(), waypoints, endpoints

#     bfs_traversal = []
#     vis = [False]*V
#     for i in range(V):
 
#         # To check if already visited
#         if (vis[i] == False):
#             q = []
#             vis[i] = True
#             q.append(i)
 
#             # BFS starting from ith node
#             while (len(q) > 0):
#                 g_node = q.pop(0)
 
#                 bfs_traversal.append(g_node)
#                 for it in adj[g_node]:
#                     if (vis[it] == False):
#                         vis[it] = True
#                         q.append(it)
#     return length_list

# def determine_length(object_points, endpoint_list, waypoints_list):
#     visited = set()
#     distances_list = dict()
#     waypoint_found = dict()
#     start_point = endpoint_list[0]
#     q = [start_point]
#     NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
#     counter = 0
#     copy_waypoints = copy.deepcopy(waypoints_list)
#     interest_pt = object_points[start_point]

#     while len(q) > 0:  
#         next_loc = q.pop()
#         next_point = g.ij_to_point(next_loc).data
#         visited.add(next_loc)
#         #return index of next_point in object_points
#         index = check_index(object_points, next_point, ind = True)
#         require = False
#         min_dist = float('inf')
#         for n in NEIGHS:
#             test_loc = object_points[index[0]+n[0], index[1]+n[1]]
#             if (test_loc in visited):
#                 continue
#             else: 
#                 test_pt = g.ij_to_point(test_loc).data
#                 if (test_loc in copy_waypoints):
#                     require = True
#                     diff = start_point-test_pt
#                     dist = diff.dot(diff)
#                     waypoint_found[dist] = test_loc
#                     if dist < min_dist:
#                         min_dist = dist
#                 else:
#                     q.append(test_loc)
#         if require == True:  
#             #index = check_index(object_points, waypoint_found[min_dist], ind = True)
#             distances_list[waypoint_found[min_dist]]= min_dist
#             q.append(waypoint_found[min_dist])
