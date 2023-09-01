from interface_rws import Interface
from scipy.ndimage.filters import gaussian_filter
from grasp import Grasp, GraspSelector
from tcps import *
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics, Point, PointCloud
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
import sys
cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0, cable)

"""
#SPEED=(.5,6*np.pi)
SPEED=(.025,0.3*np.pi)
iface=Interface("1703005",METAL_GRIPPER.as_frames(YK.l_tcp_frame,YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame),speed=SPEED)
"""


def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)


def push_action(pick, iface):
    print("pushing down")
    iface.set_speed((.25, 5))
    iface.close_grippers()
    num = 5

    pick_temp = copy.deepcopy(pick)
    pick_temp2 = copy.deepcopy(pick)

    while num > 0:
        print("Push Number", num, "Push coord: ", pick_temp)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp, rotation=Interface.GRIP_DOWN_R,
                                                     from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
        time.sleep(2)
        iface.go_delta(l_trans=[0, 0, 0.1])  # lift
        pick_temp[1] -= 0.02
        num -= 1

    num2 = 5
    while num2 > 0:
        print("Push Number", num2)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp2, rotation=Interface.GRIP_DOWN_R,
                                                     from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
        time.sleep(2)
        iface.go_delta(l_trans=[0, 0, 0.1])  # lift
        pick_temp2[1] -= 0.023
        num2 -= 1


def push_action_endpoints(pick, endpoints, iface, double=True):
    print("pushing down")
    iface.set_speed((.25, 5))
    iface.close_grippers()

    dist_1 = np.linalg.norm(
        np.array([pick[0]-endpoints[0][0], pick[1]-endpoints[0][1]]))
    dist_2 = np.linalg.norm(
        np.array([pick[0]-endpoints[1][0], pick[1]-endpoints[1][1]]))
    if (dist_1 > dist_2):
        start = np.array([float(endpoints[1][1]), float(endpoints[1][0])])
        end = np.array([float(endpoints[0][1]), float(endpoints[0][0])])
    else:
        start = np.array([float(endpoints[0][1]), float(endpoints[0][0])])
        end = np.array([float(endpoints[1][1]), float(endpoints[1][0])])

    move_vector = (end-start)/np.linalg.norm(end-start)
    print("START: "+str(start))
    print("END: " + str(end))
    # print(move_vector)
    DEPTH = .097 #.1243
    start = np.array([start[0], start[1], DEPTH])
    pick_temp = copy.deepcopy(start)
    pick_temp2 = copy.deepcopy(start)
    interval_scaling = 0.02
    num = 31
    left_homed = False
    while num > 0:
        print("Push Number", num, " Coord: ", pick_temp)
        if pick_temp[1] >= 0.01090098:
            iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp, rotation=Interface.GRIP_DOWN_R,
                                                         from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
            time.sleep(0.25)
            iface.go_delta(l_trans=[0, 0, 0.07])  # lift
        else:
            if not left_homed:
                pick_temp[0] += 0.01
                iface.home_left()
                left_homed = True
                iface.go_cartesian(r_targets=[RigidTransform(translation=pick_temp + np.array([0, 0, 0.05]), rotation=Interface.GRIP_DOWN_R,
                                                             from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
            iface.go_cartesian(r_targets=[RigidTransform(translation=pick_temp, rotation=Interface.GRIP_DOWN_R,
                                                         from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
            time.sleep(0.25)
            iface.go_delta(r_trans=[0, 0, 0.07])  # lift

        pick_temp[0] += move_vector[0]*interval_scaling
        pick_temp[1] += move_vector[1]*interval_scaling
        num -= 1
        print(pick_temp)
        print(num)

    iface.home_right()
    if double:
        num2 = 31
        left_homed = False
        while num2 > 0:
            print("Push Number", num2)
            if pick_temp2[1] >= 0.01090098:
                iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp2, rotation=Interface.GRIP_DOWN_R,
                                                             from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
                time.sleep(1)
                iface.go_delta(l_trans=[0, 0, 0.1])  # lift
            else:
                if not left_homed:
                    pick_temp2[0] += 0.01
                    iface.home_left()
                    left_homed = True
                    iface.go_cartesian(r_targets=[RigidTransform(translation=pick_temp2 + np.array([0, 0, 0.05]), rotation=Interface.GRIP_DOWN_R,
                                                                 from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
                iface.go_cartesian(r_targets=[RigidTransform(translation=pick_temp2, rotation=Interface.GRIP_DOWN_R,
                                                             from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
                time.sleep(1)
                iface.go_delta(r_trans=[0, 0, 0.1])  # lift

            pick_temp2[0] += move_vector[0]*interval_scaling
            pick_temp2[1] += move_vector[1]*interval_scaling
            num2 -= 1
            print(pick_temp2)
    iface.home()


def take_action(pick, place, angle, iface, g):

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
def take_action_2(pick, pick_2, place, place_2, iface, g):

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

def push_down(point, iface, original_depth_image_scan, points_3d, depth=0.012):
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

# linear push method, just travel across the waypoints until everything is pressed down 
def linear_push(sorted_channel_waypoints, iface, original_depth_image_scan, ACCEPTABLE_DEPTH):
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

# binary push method, look at the midpoint waypoint in the channel, see if it's pushed down or not
    # need the chnanel waypoints to be sorted in terms of distance to a given endpoint, whichever endpoint does not matter
def binary_push(sorted_channel_waypoints, iface, original_depth_image_scan, ACCEPTABLE_DEPTH):
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
    