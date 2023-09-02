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


def rotate_from_pointcloud(pointcloud):
    # returns angle in radians that the gripper should be rotated to be aligned with the pointcloud
    # pointcloud is a 3xN array of points
    x, y, z = pointcloud.x_coords, pointcloud.y_coords, pointcloud.z_coords
    # Note that `z` is unused as we don't care about the height/depth of the pointcloud
    
    # find the two points which are furthest apart
    p1, p2 = np.array([x[0],y[0]]), np.array([x[1],y[1]])
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            p1_new, p2_new = np.array([x[i],y[i]]), np.array([x[j],y[j]])
            if np.linalg.norm(p1_new-p2_new) > np.linalg.norm(p1-p2):
                p1, p2 = p1_new, p2_new

    print(f"{p1=}")
    print(f"{p2=}")
    slope = (p1[1]-p2[1])/(p1[0]-p2[0] + 1e-6)
    slope = abs(slope)
    # if slope is greater than 1, then the line is more vertical than horizontal
    # this means that the gripper should be rotated 45 degrees
    print(f"{slope=}")
    if slope < 1:
        deg = 45
    else:
        deg = 0
    print(f"DEGREES: {deg}")
    angle = deg*np.pi/180
    return angle

def rotate(angle, iface, DEBUG=False):
    # Rotates the left gripper by angle radians
    time.sleep(5)
    # rotate the gripper by angle radians
    for i in range(1):
        if DEBUG:
            print("loop", i)
        print("before get pose")
        l_cur = iface.y.left.get_pose()
        print("after get pose")
        l_delta = RigidTransform(
            translation=np.array([0.0,0.0,0.0]),
            # default:
            # rotation=Interface.GRIP_DOWN_R,
            rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]),
            from_frame=YK.l_tcp_frame,
            to_frame=YK.l_tcp_frame
        )
        l_new = l_cur*l_delta
        iface.y.left.goto_pose(l_new,speed=iface.speed)
    time.sleep(5)

if __name__ == "__main__":    
    SPEED=(.5,6*np.pi)
    SPEED=(.025,0.3*np.pi)
    iface=Interface("1703005",METAL_GRIPPER.as_frames(YK.l_tcp_frame,YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame),speed=SPEED)
    rotate(np.pi/6)
    print("done")
