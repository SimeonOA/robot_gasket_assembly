from autolab_core import RigidTransform,RgbdImage,DepthImage,ColorImage, CameraIntrinsics

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
import sys
cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0,cable)
from interface_rws import Interface 
from tcps import *
from grasp import Grasp, GraspSelector
from scipy.ndimage.filters import gaussian_filter

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
    iface.set_speed((.25,5))
    iface.close_grippers()
    num = 5

    pick_temp = copy.deepcopy(pick)
    pick_temp2 = copy.deepcopy(pick)

    while num > 0:
        print("Push Number",num,"Push coord: ", pick_temp)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp, rotation=Interface.GRIP_DOWN_R,
            from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
        time.sleep(2)
        iface.go_delta(l_trans=[0,0,0.1]) # lift
        pick_temp[1] -= 0.02
        num -= 1 

    num2 = 5
    while num2 > 0:
        print("Push Number", num2)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp2, rotation=Interface.GRIP_DOWN_R,
            from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
        time.sleep(2)
        iface.go_delta(l_trans=[0,0,0.1]) # lift
        pick_temp2[1] -= 0.023
        num2 -= 1 

def push_action_endpoints(pick,endpoints,iface):
    #SPEED=(.5,6*np.pi)
    #SPEED=(.025,0.3*np.pi)
    #iface=Interface("1703005",METAL_GRIPPER.as_frames(YK.l_tcp_frame,YK.l_tip_frame),
    #    ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame),speed=SPEED)
    print("pushing down")
    iface.set_speed((.25,5))
    iface.close_grippers()
    

    dist_1 = np.linalg.norm(np.array([pick[0]-endpoints[0][0],pick[1]-endpoints[0][1]]))
    dist_2 = np.linalg.norm(np.array([pick[0]-endpoints[1][0],pick[1]-endpoints[1][1]]))
    if(dist_1 > dist_2):
        start = np.array([float(endpoints[1][1]), float(endpoints[1][0])])
        end = np.array([float(endpoints[0][1]), float(endpoints[0][0])])
    else:
        start = np.array([float(endpoints[0][1]), float(endpoints[0][0])])
        end = np.array([float(endpoints[1][1]), float(endpoints[1][0])])
    
    move_vector = (end-start)/np.linalg.norm(end-start)
    #print(move_vector)
    DEPTH = .123
    start = np.array([start[0],start[1],DEPTH])
    pick_temp = copy.deepcopy(start)
    pick_temp2 = copy.deepcopy(start)
    interval_scaling = 0.02
    num = 25
    while num > 0:
        print("Push Number", num, " Coord: ", pick_temp)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp, rotation=Interface.GRIP_DOWN_R,
            from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
        time.sleep(1)
        iface.go_delta(l_trans=[0,0,0.1]) # lift
        pick_temp[0] += move_vector[0]*interval_scaling
        pick_temp[1] += move_vector[1]*interval_scaling
        num -= 1 
        print(pick_temp)
        print(num)

    num2 = 25
    while num2 > 0:
        print("Push Number", num2)
        iface.go_cartesian(l_targets=[RigidTransform(translation=pick_temp2, rotation=Interface.GRIP_DOWN_R,
            from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
        time.sleep(1)
        iface.go_delta(l_trans=[0,0,0.1]) # lift
        pick_temp2[0] += move_vector[0]*interval_scaling
        pick_temp2[1] += move_vector[1]*interval_scaling
        num2 -= 1 
        print(pick_temp2)

    #handle the actual grabbing motion
    #l_grasp=None
    #r_grasp=None
    #single grasp
    #grasp = g.single_grasp(pick,.007,iface.R_TCP)
    # g.col_interface.visualize_grasps([grasp.pose],iface.R_TCP)
    #wrist = grasp.pose*iface.R_TCP.inverse()
    
    #r_grasp=grasp
    #r_grasp.pose.from_frame=YK.r_tcp_frame
    #iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
   
    # iface.go_cartesian(r_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
    #     from_frame = YK.r_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
    
    
    
    # time.sleep(1)
    # delta = [place[i] - pick[i] for i in range(3)]
    # iface.go_delta(l_trans=delta)
    # iface.open_grippers()
    # iface.home()
    # iface.sync()
    # iface.set_speed(SPEED)
    # time.sleep(2)
    # iface.open_grippers()

#policy = CornerPullingBCPolicy()
"""
while True:
    q = input("Enter to home arms, anything else to quit\n")
    if not q=='':break
    iface.home()
    iface.open_grippers()
    iface.sync()
    #set up a simple interface for clicking on a point in the image
    #img=iface.take_image()

    #g = GraspSelector(img,iface.cam.intrinsics,iface.T_PHOXI_BASE)

    #pick,place=click_points(img) #left is pick point, right is place point
    # pick = min_loc
    # place = best_location
    #place = FILL IN HERE
    # VAINAVI: will need to observe and crop image most likely
    #action = policy.get_action(img.color._data)
    #pick, place = act_to_kps(action)
    # breakpoint()

    # assert pick is not None and place is not None

    # Convert to world coordinates:
    loc = np.array([0.5, 0.4, 0.127])
    #loc = [l for l in loc]
    #push_action(loc)
    # xind, yind = loc
    # lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
    # # xind2, yind2 = place
    # # lin_ind2 = int(img.depth.ij_to_linear(np.array(xind2),np.array(yind2)))

    

    #points_3d = iface.cam.intrinsics.deproject_pixel(self, depth, loc)
    #point=iface.T_PHOXI_BASE*points_3d[lin_ind]
    #point = [p for p in point]
    #point[2] += 0.005 # manually adjust height a tiny bit
    #place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]
    #point = [0.55538863, 0.21792354, 0.10870991]
    #point_2 = [0.55538863, 0.21792354, 0.10870991]
    #point_3 = [0.39306104, 0.2665695 , 0.05135202]
    point = [0.21792354, 0.55538863, 0.10870991]
    point_2 = [0.21792354, 0.55538863, 0.10870991]
    point_3 = [0.2665695, 0.39306104 , 0.05135202]
    
    push_action_endpoints(point,[point_2,point_3], iface)
    # break
    #g.close()
print("Done with script, can end")
"""