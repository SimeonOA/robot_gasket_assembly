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

#SPEED=(.5,6*np.pi)
SPEED=(.025,0.3*np.pi)
iface=Interface("1703005",METAL_GRIPPER.as_frames(YK.l_tcp_frame,YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame),speed=SPEED)

def rotate(angle, DEBUG=False):
    # Rotates the left gripper by angle radians
    for i in range(10):
        if DEBUG:
            print("loop", i)
        l_cur = iface.y.left.get_pose()
        l_delta = RigidTransform(
            translation=np.array([0.0,0.0,0.0]),
            # rotation=Interface.GRIP_DOWN_R,
            rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]),
            from_frame=YK.l_tcp_frame,
            to_frame=YK.l_tcp_frame
        )
        l_new = l_cur*l_delta
        iface.y.left.goto_pose(l_new,speed=iface.speed)
    
rotate(np.pi/6)
print("done")
