from ur5py.ur5 import UR5Robot
import numpy as np
# from shape_match import *
from new_shape_match import get_channel
from shape_match import get_cable, align_channel, get_closest_channel_endpoint
from autolab_core import RigidTransform
import pdb
from real_sense_modules import *
from utils import *
import argparse
from gasketRobot import GasketRobot
from scipy.spatial.transform import Rotation as R
from resources import CROP_REGION, curved_template_mask, straight_template_mask, trapezoid_template_mask
from calibration.image_robot import ImageRobot
from zed_camera import ZedCamera

robot = GasketRobot()
pose = robot.get_pose(convert=False)
print(pose)
pose1 = [-0.4667621349341, -0.5508576307893888, -0.0034437216212429983, 2.4227376296251646, -1.9904911168788861, -0.00038438761708320217]