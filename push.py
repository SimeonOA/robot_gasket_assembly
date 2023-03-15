from interface_rws import Interface
from scipy.ndimage.filters import gaussian_filter
from grasp import Grasp, GraspSelector
from tcps import *
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics

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


def push_action(pick):
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
