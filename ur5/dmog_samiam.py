import socket
import time
import timeit
import subprocess
import sys
import copy
import random
import rtde_control
import rtde_receive
import re
import torchvision.transforms as transforms

import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from itertools import repeat
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pyrealsense2 as rs

from sklearn.decomposition import PCA
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from pickle import dump, load
from copy import deepcopy

from real_sense_modules import setup_rs_camera, get_rs_image, viz_rs_images
from resources import x_crop, y_crop

from shapely.geometry import LineString, Polygon
from scipy.spatial import ConvexHull
from scipy.ndimage import label
from scipy.ndimage import convolve
import math as m
import IPython
import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
#from fastsam import FastSAM, FastSAMPrompt

from scipy.spatial import distance
from itertools import chain, product

import extcolors
from PIL import Image

import serial, datetime
from serial.tools import list_ports
from cal_cam.image_robot import ImageRobot
from ur5py import UR5Robot, UR2RT, RT2UR
import pdb

from laundrynet import LaundryNet
from autolab_core import RigidTransform

import warnings
warnings.filterwarnings("ignore")

import torchvision.transforms.functional as TF
import os
# import test_nn_pytorch_bce_feature as tnpbf



class Identity(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x

class SiameseNetwork(nn.Module):
	def __init__(self, num_ftrs_resnet):
		super(SiameseNetwork, self).__init__()
		self.fc = nn.Linear(num_ftrs_resnet*2, 1)

	def forward(self, input1, input2):
		# In this function we pass in both images and obtain both vectors which are returned
		input1_tor = torch.tensor(input1)
		input2_tor = torch.tensor(input2)
		output12 = torch.cat((input1_tor, input2_tor),0)
		output = self.fc(output12)
		return output


class process_data(object):   
	def __init__(self, target_size = [224, 224], inspect = False):
		self.target_size = target_size
		self.inspect = inspect       

	# loads image and masks for the computation
	def load_img_masks(self, image, masks):
		self.image = image
		self.masks = masks

	# loads candidate grasp points and orientations to try
	def load_cand_pts_orns(self, cand_pts, orn_granularity):
		self.cand_pts = cand_pts
		self.cand_orns = m.pi*np.array(range(orn_granularity))/orn_granularity

	def process_(self, resnet, pt_index, orn_index, device):
		
		self.w = np.shape(self.image)[0]
		self.h = np.shape(self.image)[1]
		_image = Image.new("RGB", (self.w, self.h), color=(255, 255, 255))

		# load picking position and orientation
		pos = self.cand_pts[pt_index]
		orn = self.cand_orns[orn_index]

		# get masked pile image
		num_masks = len(self.masks)

		if num_masks == 0:
			print('No item in the masks.')
			return False

		pile_mask = self.masks[0]#['segmentation'] 
		for i in range(num_masks-1):
			pile_mask = pile_mask | self.masks[i+1]#['segmentation'] 
		
		masked_pile_images = self.apply_mask(self.image, pile_mask)
		masked_pile_images_ = self.extend_translate_rotate_cut(ori_img=masked_pile_images, tran=pos, rot=orn, size_=self.target_size)

		masked_pile_feat_ = self.get_feature(masked_pile_images_, resnet, device)

		masked_image_arrs_ = []
		masked_image_feats_ = []
		# get masked individual image
		# plt.ion()
		for i in range(num_masks):
			mask = self.masks[i]#['segmentation'] 
			masked_image = self.apply_mask(_image, mask)
			masked_image_ = self.extend_translate_rotate_cut(ori_img=masked_image, tran=pos, rot=orn, size_=self.target_size)

			masked_image_arrs_.append(np.asarray(masked_image_))
			masked_image_feats_.append(self.get_feature(masked_image_, resnet, device))


		masked_image_arrs_ = np.asarray(masked_image_arrs_)
		masked_image_feats_ = np.asarray(masked_image_feats_)

		return masked_pile_images_, masked_pile_feat_, masked_image_arrs_, masked_image_feats_
	
	def get_feature(self, img, resnet, device):
		transformation = transforms.Compose([transforms.Resize((224,224)),
									 transforms.ToTensor()])
		
		img= transformation(img)
		img = img.to(device)
		img_feature = resnet(img.unsqueeze(0))
		feature_array = img_feature.cpu().numpy().reshape(-1)
		# feature_array = img_feature.cuda().numpy().reshape(-1)
		
		return feature_array

	def apply_mask(self, image, mask):
		# Step 1: Convert the image and mask to PyTorch tensors
		image_tensor = TF.to_tensor(image)

		#image_tensor = image_tensor.transpose(0, 2)
		#image_tensor = image_tensor.transpose(0, 1)

		#print('Image tensor shape')
		#print(np.shape(image_tensor))


		
		mask_tensor = torch.tensor(mask, dtype=torch.bool)
		#mask_tensor = TF.to_tensor(mask)
		mask_tensor_to_stack = [mask_tensor, mask_tensor, mask_tensor]
		mask_tensor = np.dstack(mask_tensor_to_stack)
		mask_tensor = torch.tensor(mask_tensor)

		#print('Mask tensor shape before')
		#print(np.shape(mask_tensor))

		mask_tensor = mask_tensor.transpose(0,2)
		if not np.shape(mask_tensor) == np.shape(image_tensor):
			mask_tensor = mask_tensor.transpose(1,2)

		#print('Mask tensor shape after')
		#print(np.shape(mask_tensor))

		masked_image_tensor = image_tensor * mask_tensor
		# Step 3: Convert the selected pixels tensor back to a PIL image
		masked_image = TF.to_pil_image(masked_image_tensor)

		return masked_image
	
	def extend_translate_rotate_cut(self, ori_img, tran, rot, size_):
		# extend
		new_w = self.w*3
		new_h = self.h*3
		expanded_image = Image.new("RGB", (new_w, new_h), color=(0, 0, 0))
		x_offset = int((new_w - self.w) / 2)
		y_offset = int((new_h - self.h) / 2)
		expanded_image.paste(ori_img, (x_offset, y_offset))

		# translate
		translate_x = self.w//2-tran[1]
		translate_y = self.h//2-tran[0]

		image_tensor = TF.to_tensor(expanded_image)
		translated_image_tensor = TF.affine(image_tensor, angle=0, translate=(translate_x, translate_y), scale=1, fill=[0,], shear=0)

		# rotate
		degrees_to_rotate = rot/np.pi*180
		if degrees_to_rotate > 180:
			degrees_to_rotate -= 360
		rotation_center = (new_w//2, new_h//2)
		rotated_image_tensor = TF.affine(translated_image_tensor, angle=degrees_to_rotate, translate=(0, 0), scale=1, shear=0, fill=[0,], center = rotation_center)

		# cut
		crop_x1 = rotation_center[0] - size_[0] // 2
		crop_y1 = rotation_center[1] - size_[1] // 2
		crop_x2 = crop_x1 + size_[0]
		crop_y2 = crop_y1 + size_[1]

		cropped_image_tensor = TF.crop(rotated_image_tensor, crop_y1, crop_x1, size_[1], size_[0])

		cropped_image = TF.to_pil_image(cropped_image_tensor)

		return cropped_image

	def process_data_sequence_(self, resnet):

		num_pts = len(self.cand_pts)
		num_orns = len(self.cand_orns)

		masked_pile_images_arr = [[None] * num_pts] * num_orns 
		masked_pile_feat_arr = [[None] * num_pts] * num_orns 
		masked_image_arrs_arr = [[None] * num_pts] * num_orns 
		masked_image_feats_arr = [[None] * num_pts] * num_orns

		for pt_index in range(num_pts):
			for orn_index in range(num_orns):
				m_p_i_, m_p_f_, m_i_a_, m_i_f_ = self.process_(resnet, pt_index, orn_index, self.device)

				masked_pile_images_arr[pt_index][orn_index] = m_p_i_
				masked_pile_feat_arr[pt_index][orn_index] = m_p_f_
				masked_image_arrs_arr[pt_index][orn_index] = m_i_a_
				masked_image_feats_arr[pt_index][orn_index] = m_i_f_
		
		return masked_pile_images_arr, masked_pile_feat_arr, masked_image_arrs_arr, masked_image_feats_arr



class DMog():

	def __init__(self):

		self.LN = LaundryNet()

		self.real_robot = True

		self.load_params()

		self.setup_robot_and_camera() 

	def load_params(self):
		self.load_laundry_net_params()
		self.load_workspace_params()
		self.load_scene_reset_params()
		self.load_robot_motion_params()
		self.load_segmentation_params()
		self.load_imagerobot()
		self.load_viz_debug_params()
		self.load_calibration_params()
		# self.load_process_data()

	def setup_robot_and_camera(self):

		#  Realsense setup
		self.depth_flag = True
		if self.real_robot:
			self.pipeline, self.colorizer, self.align, self.depth_scale = setup_rs_camera(use_depth=self.depth_flag)

		self.depth_lookup_table = np.load("depth_lookup_table.npy")

		# Robot connection
		self.HOST = "172.22.22.3"
		self.ROBOT_PORT = 30003
		self.GRIPPER_PORT = 63352
		if self.real_robot:
			self.connect_robot()

		# Image to Real coordinates
		model_path = './cal_cam/'
		model_name = model_path+'cam_robot_regr.pkl'
		scaler_name = model_path+'cam_robot_scaler.pkl'
		self.cam_model, self.cam_scaler = self.load_model(model_name, scaler_name)

		# Image processing
		self.IM_THRESH_LOW = 130
		self.IM_THRESH_HIGH = 255

		self.KERNEL_SIZE = 5

		self.IM_CIRCLE_P_1 = 5
		self.IM_CIRCLE_P_2 = 2
		self.IM_X_MAX = 444
		self.IM_Y_MAX = 575
		self.CAM_TO_TABLE_SURFACE = 0.82

	def load_laundry_net_params(self):
		self.PROXIMITY_THRESHOLD = 10 #for the narrow strip for fold
		self.PICK_NOISE = 20
		self.NUM_IMAGE_SAMPLES_FOR_CLUSTER = 100 # Sample the image
		self.GRIPPER_WIDTH_PIXELS = 30
		self.WEIGHT_POLICY_THRESH = 40
		self.ROBOT_MAX_HEIGHT_PIXELS = 300

		self.MAX_HEAP_DEFORMATION = 85/1000. # Measured with 10 data points

		self.MAX_NUM_STACKS = 10
		self.FOLD_WIDTH = 150

	def load_scene_reset_params(self):

		# Speed of basket place or pick
		self.place_basket_speed = 1
		self.place_basket_acc = 1

		self.x_basket_grasp = 515
		basket_edge_pt_y = -600 #self.y_home-basket_edge_to_center --- WAS -650
		basket_base_z = -10 # Where to place the gripper (on basket) --- WAS 90
		
		
		basket_edge_to_center = 139  
		drop_offset = 230
		

		self.basket_start_pt = np.array([self.x_basket_grasp, basket_edge_pt_y, basket_base_z])
		self.basket_start_pt_drop = np.array([self.x_home, basket_edge_pt_y, basket_base_z+drop_offset])
		self.basket_mid_air_pt = np.array([-20, -500, 320])
		self.basket_end_pt = np.array([-350, -450, 270])
		#self.basket_end_joints = np.array([67.0, -105.0, 105.0, -170.0, 60.0, 130.0])*m.pi/180

		self.basket_start_orn = m.pi/2
		self.basket_mid_air_orn = m.pi/2
		self.basket_end_rpy = [0.76, 0.312, 1.04]

		self.num_shuffle_actions = 10

		return None

	def load_robot_motion_params(self):
		# Shaking
		self.SHAKE_EPS = 20/1000.
		self.shake_acc = 10.0
		self.shake_speed = 3.1

		# General motion
		# Original values: 0.5, 0.75, 2.0, 2.0
		self.close_gripper_time = 0.5
		self.open_gripper_time = 0.5

		self.fast=True

		if self.fast == True:
			self.joint_speed = 2
			self.joint_acc = 2
			self.reset_speed = 2
			self.reset_acc = 2
		else:
			self.joint_speed = 0.3
			self.joint_acc = 0.3
			self.reset_speed = 0.3
			self.reset_acc = 0.3
		self.gripper_close_speed = 255  # Between 0 and 255

		# If using moveJ
		self.grasp_above_time = 2.0
		self.reach_grasp_z_time = 2.0

	def load_workspace_params(self):
		# Fixed positions
		self.z_above = 350
		#self.z_grasp = -20
		self.z_grasp = -85
		self.z_shuffle = 30
		self.z_stack = 30

		# Basket location
		
		self.x_home = 415
		self.y_home = -430
		self.z_home = 350

		#  Z at which robot translates between pick and place
		self.translation_z = self.z_above
		self.rearrangement_translation_z = 150
		self.z_place = 350
		# self.place_point = [self.x_home, self.y_home, self.z_place]

		# Workspace boundaries
		self.Y_BOUND = -277
		self.X_BOUND = -500

		# Point where we check for trailing cloth
		self.CHECKPOINT = [330, -580, 421]

	def load_segmentation_params(self):
		self.model_size = 'vit_b'
		self.sam_path = './sam_weights/'

		if self.model_size == 'vit_h':
			self.sam_model_name = 'sam_vit_h_4b8939.pth'
		elif self.model_size == 'vit_l':
			self.sam_model_name = 'sam_vit_l_0b3195.pth'
		elif self.model_size == 'vit_b':
			self.sam_model_name = 'sam_vit_b_01ec64.pth'
		else:
			self.sam_model_name = ''

		# Standard values for the cleanup process
		self.metr = 'area'
		self.stab_score_default = 1
		self.pred_iou_default = 1

		self.conv_size = 5
		self.max_first = False
		self.size_thresh = 600
		self.im_thresh = 130
		self.maxval = 1
		self.kernel_size = 5
		self.hole_remove_conv = 10
		self.cloth_thresh = 0.25

		self.transparency = 0.6
		
		self.distance_thresh=15
		#self.distance_thresh = 35
		self.nbr_width=20

		self.diff_thresh = 14

		self.dist_baseline_thresh = 15
		self.rat = 2
		self.orn_granularity = 6
		self.sample_num = 1000

		self.baseline_prob_scaling_factor = 100

		self.height_thresh = 1.37

	def create_param_dict(self):
		param_dict = dict()

		param_dict['model_size'] = self.model_size
		param_dict['metr'] = self.metr
		param_dict['stab_score_default'] = self.stab_score_default
		param_dict['pred_iou_default'] = self.pred_iou_default

		param_dict['conv_size'] = self.conv_size
		param_dict['max_first'] = self.max_first
		param_dict['size_thresh'] = self.size_thresh
		param_dict['im_thresh'] = self.im_thresh
		param_dict['maxval'] = self.maxval
		param_dict['kernel_size'] = self.kernel_size
		param_dict['hole_remove_conv'] = self.hole_remove_conv
		param_dict['cloth_thresh'] = self.cloth_thresh

		param_dict['transparency'] = self.transparency
		
		param_dict['distance_thresh'] = self.distance_thresh
		param_dict['nbr_width'] = self.nbr_width

		param_dict['diff_thresh'] = self.diff_thresh

		param_dict['dist_baseline_thresh'] = self.dist_baseline_thresh
		param_dict['rat'] = self.rat
		param_dict['orn_granularity'] = self.orn_granularity
		param_dict['sample_num'] = self.sample_num

		param_dict['joint_speed']= self.joint_speed
		param_dict['joint_acc'] = self.joint_acc

		param_dict['open_gripper_time'] = self.open_gripper_time
		param_dict['close_gripper_time'] = self.close_gripper_time

		param_dict['x_crop'] = x_crop
		param_dict['y_crop'] = y_crop

		param_dict['rw_lower_bds'] = self.image_pt_to_rw_pt([y_crop[0], x_crop[0]])[0]
		param_dict['rw_upper_bds'] = self.image_pt_to_rw_pt([y_crop[1], x_crop[1]])[0]

		param_dict['x_home'] = self.x_home
		param_dict['y_home'] = self.y_home
		param_dict['z_home'] = self.z_home

		param_dict['z_above'] = self.z_above
		param_dict['z_grasp'] = self.z_grasp

		param_dict['num_shuffle_actions'] = self.num_shuffle_actions
		param_dict['baseline_prob_scaling_factor'] = self.baseline_prob_scaling_factor

		param_dict['height_thres'] = self.height_thresh

		return param_dict

	def load_imagerobot(self):
		self.calibration_path = './cal_cam/cam_cal_09_11_23.csv'
		self.model_path = './'
		self.ir = ImageRobot(self.model_path)
		self.ir.train_model(self.calibration_path)

	def load_viz_debug_params(self):
		self.show_steps = False
		self.debug_motion_plan = False


	def load_nn_params(self):

		self.nn_model_path = './seg_net_models/my_net_2023_08_30_11_12_39.pt'

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		
		# load the data processor (generates NN inputs)
		self.proc_data = process_data()

		# load the fixed-weight resnet
		self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT) 
		self.num_ftrs_resnet = self.resnet.fc.in_features
		self.resnet.fc = nn.Flatten()
		for param in self.resnet.parameters():
			param.requires_grad = False
		self.resnet = self.resnet.to(self.device)
		self.resnet.eval()

		# load the siamese network
		self.sia_net = SiameseNetwork(self.num_ftrs_resnet)
		self.sia_net.load_state_dict(torch.load(self.nn_model_path))
		self.sia_net = self.sia_net.to(self.device)
		self.sia_net.eval()


	def load_calibration_params(self):

		self.use_pick_pt_depth = True
		self.use_hardcoded_cal = False

		self.x_intercept = 150.1111247
		self.x_coef_on_x = -2.07787301 #* (360-140)/(360-120)
		self.x_coef_on_y = 0.02772887548

		self.y_intercept = -759.9587912
		self.y_coef_on_y = 2.069261384
		self.y_coef_on_x = 0.02158838398


		# UPPER PLANE VALUES

		self.upper_height_value = 1.582
		self.upper_z_value = 150

		self.upper_x_intercept = 72.43845257
		self.upper_x_coef_on_x = -1.665
		self.upper_x_coef_on_y = 0.01

		self.upper_y_intercept = -709.8271378
		self.upper_y_coef_on_y = 1.656541759
		self.upper_y_coef_on_x = 0.03923585725





		self.surface_height_value = 1.255
		self.surface_z_value = -16

		#self.x_intercept = 150.1111247
		#self.x_coef_on_x = -2.07787301 #* (360-140)/(360-120)
		#self.x_coef_on_y = 0.02772887548

		#self.y_intercept = -759.9587912
		#self.y_coef_on_y = 2.069261384
		#self.y_coef_on_x = 0.02158838398

		f_x = -0.4818473759
		c_x = 73.77723968

		f_y = 0.4821812388
		c_y = 365.0698399

		self.K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

		# depth * np.linalg.inv(self.K) * np.r_[pixel, 1.0]




	def go_home(self):
		xyz_home = [[self.x_home, self.y_home, self.z_home]]
		self.move_ur5py(xyz_home, [0], [self.joint_speed], [self.joint_acc])
		self.wrist_unwinder()
		return None



	### RUNS EXPERIMENTS WITH SCENE RESETS

	def experiment(self, policy='random'):
		# start_time = timeit.default_timer()
		time.sleep(3)
		weight = self.recordWeight(0)
		NUM_SAMPLE_RUNS = 3
		print ('Method:', policy)
		start_pt = 1
		print(start_pt)
		end_pt = start_pt + NUM_SAMPLE_RUNS
		print(end_pt)
		for sample_num in range(start_pt, end_pt):
			print ('Sample number:', sample_num)

			self.data_path = './dmog_data/method_{}/sample_{}/'.format(policy, sample_num)
			subprocess.call(['rm', '-rf', self.data_path])
			subprocess.call(['mkdir', '-p', self.data_path])

			print ('N00000')
			initial_weight = self.recordWeight(0)
			self.reset_scene()
			self.move_above_place_point()
			self.shuffle_placed_clothes_v2()
			self.move_above_place_point()

			sample_start_time = timeit.default_timer()
			self.clear_table(policy=policy, sample_num=sample_num)
			sample_total_time = timeit.default_timer() - sample_start_time

			time.sleep(3)
			final_weight = self.recordWeight(0)

			np.save(self.data_path+'initial_weight', initial_weight)
			np.save(self.data_path+'final_weight', final_weight)
			np.save(self.data_path+'sample_total_time', sample_total_time)

			print ('Initial load weight', initial_weight)
			print ('Final load weight', final_weight)
			print ('Sample total time', sample_total_time)

	def reset_scene(self):

		print('Emptying basket...')
		self.empty_basket(self.basket_start_pt, self.basket_start_orn,
			self.basket_mid_air_pt, self.basket_mid_air_orn,
			self.basket_end_pt, self.basket_end_rpy)

		if self.debug_motion_plan:
			ac_q = self.robot.get_joints()
			print(ac_q)
			pdb.set_trace()

		self.place_empty_basket(self.basket_start_pt, self.basket_start_orn,
			self.basket_mid_air_pt, self.basket_mid_air_orn )

		# self.shuffle_basket_clothes()

	def empty_basket(self, start_pt, start_orn, mid_air_pt, mid_air_orn, end_pt, end_rpy):

		self.wrist_unwinder()

		start_pt_above = deepcopy(start_pt)
		start_pt_above[-1] = mid_air_pt[-1]

		# Open gripper
		self.s_gripper.sendall(b'SET POS 100\n')
		time.sleep(self.open_gripper_time)

		xyz_list = [start_pt_above, start_pt]
		orn_list = [start_orn, start_orn]
		speed_list = [self.joint_speed, self.place_basket_speed]
		acc_list = [self.joint_acc, self.place_basket_acc]

		print('Going to location')
		print('xyz_list')
		print(xyz_list)
		self.move_ur5py(xyz_list, orn_list, speed_list, acc_list, unwind=False)

		# # Move above grasp point
		# self.move_ur5py(start_pt_above, start_orn, self.joint_speed, self.joint_acc)
		# # Move to basket grasp point
		# self.move_ur5py(start_pt, start_orn, self.place_basket_speed, self.place_basket_acc)

		# Close gripper
		self.s_gripper.sendall(b'SET POS 255\n')
		time.sleep(self.close_gripper_time)

		xyz_list = [start_pt_above, mid_air_pt]
		orn_list = [start_orn, mid_air_orn]
		speed_list = [self.joint_speed, self.joint_speed]
		acc_list = [self.joint_acc, self.joint_acc]
		self.move_ur5py(xyz_list, orn_list, speed_list, acc_list, unwind=False)

		#pdb.set_trace()

		xyz_list = [end_pt]
		rpy_list = [end_rpy]
		speed_list = [self.joint_speed]
		acc_list = [self.joint_acc]
		self.move_ur5py(xyz_list, rpy_list, speed_list, acc_list, use_rpy=True, unwind=False)
		#time.sleep(1)

		# # Move above the basket grasp point
		# self.move_ur5py(start_pt_above, start_orn, self.joint_speed, self.joint_acc)
		# # Move to the mid-air point
		# self.move_ur5py(mid_air_pt, mid_air_orn, self.joint_speed, self.joint_acc)
		# # Move to the basket end point with the right orn
		# self.move_ur5py(end_pt, end_orn, self.joint_speed, self.joint_acc)

		return None

	def place_empty_basket(self, start_pt, start_orn, mid_air_pt, mid_air_orn):

		start_pt_above = deepcopy(start_pt)
		start_pt_above[-1] = mid_air_pt[-1]

		xyz_list = [mid_air_pt, start_pt_above, start_pt]
		orn_list = [mid_air_orn, start_orn, start_orn]
		speed_list = [self.joint_speed, self.joint_speed, self.place_basket_speed]
		acc_list = [self.joint_acc, self.joint_acc, self.place_basket_acc]

		self.move_ur5py(xyz_list, orn_list, speed_list, acc_list, unwind=False)
		# self.move_ur5py(mid_air_pt, mid_air_orn, self.joint_speed, self.joint_acc)
		# self.move_ur5py(start_pt_above, start_orn, self.joint_speed, self.joint_acc)
		# self.move_ur5py(start_pt, start_orn, self.place_basket_speed, self.place_basket_acc)

		# Open gripper
		self.s_gripper.sendall(b'SET POS 0\n')
		time.sleep(self.open_gripper_time)

		self.move_ur5py([start_pt_above], [start_orn], [self.joint_speed], [self.joint_acc], unwind=False)

		return None



	def shuffle_placed_clothes_v2(self):

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)

		translation_z_reset = 270
		bd_pt_1 = [-350, -670, translation_z_reset]
		bd_pt_2 = [-20, -670, translation_z_reset]
		bd_pt_3 = [-20, -380, translation_z_reset]
		bd_pt_4 = [-350, -380, translation_z_reset]

		samples_per_edge = 4
		place_pts_bd_1 = np.random.uniform(low=bd_pt_1, high=bd_pt_2, size=(samples_per_edge, 1, 3))
		place_pts_bd_2 = np.random.uniform(low=bd_pt_2, high=bd_pt_3,  size=(samples_per_edge, 1, 3))
		place_pts_bd_3 = np.random.uniform(low=bd_pt_3, high=bd_pt_4, size=(samples_per_edge, 1, 3))
		place_pts_bd_4 = np.random.uniform(low=bd_pt_4, high=bd_pt_1, size=(samples_per_edge, 1, 3))

		all_place_pts_bd = [place_pts_bd_1, place_pts_bd_2, place_pts_bd_3, place_pts_bd_4]

		bd_pt_1_pick = [-350, -670, self.z_grasp]
		bd_pt_2_pick = [-20, -670, self.z_grasp]
		bd_pt_3_pick = [-20, -380, self.z_grasp]
		bd_pt_4_pick = [-350, -380, self.z_grasp]

		pick_pts_bd_1 = np.random.uniform(low=bd_pt_1_pick, high=bd_pt_3_pick,  size=(2*samples_per_edge, 1, 3))
		pick_pts_bd_2 = np.random.uniform(low=bd_pt_2_pick, high=bd_pt_4_pick,  size=(2*samples_per_edge, 1, 3))
		all_pick_pts_bd = [pick_pts_bd_1, pick_pts_bd_2]

		place_pts = []
		for place_pts_bd in all_place_pts_bd:
			for i in range(place_pts_bd.shape[0]):
				place_pts.append(place_pts_bd[i][0])

		pick_pts = []
		for pick_pts_bd in all_pick_pts_bd:
			for i in range(pick_pts_bd.shape[0]):
				pick_pts.append(pick_pts_bd[i][0])

		shuffled_picks = np.random.permutation(pick_pts)
		shuffled_place = np.random.permutation(place_pts)

		
		for shuffle_action in range(self.num_shuffle_actions):
			# pick_pt = shuffled_picks[shuffle_action]
			pick_pt, _ = self.gen_random_pick_point(cloth_points, depth_image, x_bds = [30,300], y_bds = [30,210])
			place_pt = shuffled_place[shuffle_action]

			pick_orn = np.random.uniform(low=0., high=m.pi/4.)

			self.pick_and_place(pick_pt, place_pt, pick_orn, translation_z_reset, use_shake_action=False, use_reset_speed=True)

			# # Move above the current place point
			# orn = pick_orn
			# above_place_pt = deepcopy(place_pt)
			# above_place_pt[-1] = translation_z_reset
			# self.move_ur5py(above_place_pt, orn, self.reset_speed, self.reset_acc)

		# Move home at the end
		home_place_point = self.gen_place_point()
		#orn = pick_orn
		self.move_ur5py([home_place_point], [pick_orn], [self.reset_speed], [self.reset_acc])

		# Generate 10 pick points at random
		# Pick at one pick point and place at any of the 10 place points
		# At random
		return None

	def test_ur5_motion(self):

		print ('First check')
		# Move to checkpoint
		# Move the robot to the check point (to take another image)
		check_point_orn = 0
		self.move_ur5py([self.CHECKPOINT], [check_point_orn], [self.reset_speed], [self.reset_acc])

		# Move home at the end
		home_place_point = self.gen_place_point()
		self.move_ur5py([home_place_point], [check_point_orn], [self.reset_speed], [self.reset_acc])

		print ('Made it it')
		IPython.embed()
		xyz_list = [self.CHECKPOINT, home_place_point]
		orn_list = [check_point_orn, check_point_orn]
		joint_speed_list = [self.reset_speed, self.reset_speed]
		joint_acc_list = [self.reset_acc, self.reset_acc]

		self.move_ur5py(xyz_list, orn_list, joint_speed_list, joint_acc_list)


		return None


	def gen_random_pick_point(self, cloth_points, depth_image, x_bds, y_bds):
		num_cloth_points = len(cloth_points)
		rand_pt_index = np.random.randint(num_cloth_points)
		im_pt = cloth_points[rand_pt_index]
		im_pt[0] = min(max(im_pt[0], x_bds[0]), x_bds[1])
		im_pt[1] = min(max(im_pt[1], y_bds[0]), y_bds[1])

		rand_real_pt = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]

		grasp_orn = self.get_pca_grasp_orn(cloth_points, im_pt)

		depth_at_pick_pt = depth_image[im_pt[0], im_pt[1]]
		pick_z = self.get_pick_depth(depth_at_pick_pt)

		pick_pt =  [rand_real_pt[0], rand_real_pt[1], pick_z]

		return pick_pt, grasp_orn

	def gen_place_point(self):

		change_mag = 75.0
		# change_mag = 0
		delta_x = np.random.uniform(-change_mag, change_mag)
		delta_y = 0 #np.random.uniform(-change_mag, change_mag)

		place_point = [self.x_home+delta_x, self.y_home+delta_y, self.z_home]
		print("place:", place_point)
		return place_point

	
	def get_pca_grasp_orn(self, cloth_points, pick_point_img):

		# IPython.embed()
		#
		# import numpy as np
		# import matplotlib.pyplot as plt
		# from sklearn.decomposition import PCA
		#

		pick_xy = deepcopy(pick_point_img[0:2])


		pick_xy = [pick_xy[0], pick_xy[1]]
		PCA_RADIUS = 10000 #pixels
		pca_points = []
		for pt in cloth_points:
			pca_points.append(pt)

		pca = PCA(n_components=2)
		pca.fit(np.array(pca_points))

		comp = pca.components_
		minor_axis = comp[1]
		grasp_orn = m.atan2(minor_axis[1], minor_axis[0])

		if grasp_orn < 0:
			grasp_orn += m.pi

		return grasp_orn

	def connect_robot(self):
		print('Connecting robot ...')

		self.s_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		time.sleep(0.1)
		print("gripper was socketed")
		print(self.HOST)
		print(self.GRIPPER_PORT)
		self.s_gripper.connect((self.HOST, self.GRIPPER_PORT))
		time.sleep(0.1)
		print("gripper was connected")

		# Set gripper operating speed
		self.s_gripper.sendall(b"SET SPE %d\n" % self.gripper_close_speed)
		time.sleep(0.1)
		print("gripper operating speed set")

		self.rtde_c = None
		self.robot = UR5Robot()
		# self.rtde_c = rtde_control.RTDEControlInterface("172.22.22.3")
		# print("ff")

		time.sleep(0.1)

		print('Done connecting robot ...')



	def move_ur5py(self, xyz_list, orn_list, joint_speed_list, joint_acc_list, use_rpy=False, unwind=True):
		#if unwind_first:
		#	self.wrist_unwinder()

		if self.debug_motion_plan:
				pdb.set_trace()
		for i in range(len(xyz_list)):
			lst = [xyz_list[i][0]/1000, xyz_list[i][1]/1000, xyz_list[i][2]/1000]

			if xyz_list[i][1] > -320:
				lst[1] = -.320
			if xyz_list[i][1] < -750:
				lst[1] = -.750

			if xyz_list[i][0] > 550:
				lst[0] = .550
			if xyz_list[i][0] < -490:
				lst[0] = -.490

			if use_rpy:
				rpy = orn_list[i]
				lst.extend(rpy)
			else:
				gorn = orn_list[i]
				axisang = self.rotateZAxis(gorn)
				lst.extend(axisang)
			#lst.append(0)
			#print(lst)
			
			actual_q = self.robot.get_joints()
			desired_q = RT2UR(UR2RT(lst))
			
			if self.debug_motion_plan:
				#import pdb
				pdb.set_trace()
			
			
			#print("This is gorn itself...")
			#print(gorn)
			#print("This is what we're sending ur5py...")
			#print(UR2RT(lst))


			self.robot.move_pose(UR2RT(lst), vel=joint_speed_list[i], acc=joint_acc_list[i])
			time.sleep(0.25)
			if unwind:
				self.wrist_unwinder()

			

	def move_above_place_point(self):
		place_point = self.gen_place_point()
		orn = 0
		self.move_ur5py([place_point], [orn], [self.joint_speed], [self.joint_acc])
		return None

	def drop_at_point(self, xyz_above_place, orn_above_place):
		self.move_ur5py(xyz_above_place, orn_above_place, self.joint_speed, self.joint_acc)
		# Open gripper
		self.s_gripper.sendall(b'SET POS 0\n')
		time.sleep(self.open_gripper_time)
		return None

	
	def pick_and_place(self, pick_point, place_point, grasp_orn, translation_z, use_shake_action=False, save=False, depth_at_pick_pt=1.1, use_reset_speed=False):

		if use_reset_speed:
			speed_val = self.reset_speed
			acc_val = self.reset_acc
		else:
			speed_val = self.joint_speed
			acc_val = self.joint_acc
		# Open gripper
		#print('Opening gripper...')
		self.s_gripper.sendall(b'SET POS 0\n')
		time.sleep(self.open_gripper_time)

		x_pick, y_pick, z_pick = pick_point
		x_place, y_place, z_place = place_point

		# Above pick point
		xyz_above_pick = [x_pick, y_pick, translation_z]
		orn_above_pick = grasp_orn

		# Pick point
		xyz_pick = [x_pick, y_pick, z_pick]
		orn_pick = grasp_orn

		# Above place point
		above_place_z = np.max([translation_z, z_place])
		xyz_above_place = [x_place, y_place, above_place_z]
		orn_above_place = grasp_orn

		# Place point
		xyz_place = [x_place, y_place, z_place]
		orn_place = grasp_orn

		xyz_list = [xyz_above_pick, xyz_pick]
		orn_list = [orn_above_pick, orn_pick]
		speed_list = [speed_val, speed_val]
		acc_list = [acc_val, acc_val]
		#print('Moving to pick...')
		self.move_ur5py(xyz_list, orn_list, speed_list, acc_list)

		# self.move_ur5py(xyz_above_pick, orn_above_pick, speed_val, acc_val)
		# self.move_ur5py(xyz_pick, orn_pick, speed_val, acc_val)

		# Close gripper
		#print('Closing gripper...')
		self.s_gripper.sendall(b'SET POS 255\n')
		time.sleep(self.close_gripper_time)

		xyz_list = [xyz_above_pick, xyz_above_place]
		orn_list = [orn_above_pick, orn_above_place]
		speed_list = [speed_val, speed_val]
		acc_list = [acc_val, acc_val]
		#print('Moving to place...')
		self.move_ur5py(xyz_list, orn_list, speed_list, acc_list)

		# Open gripper
		self.s_gripper.sendall(b'SET POS 0\n')
		time.sleep(self.open_gripper_time)

		return None


	def pick_and_place_sequence(self, point_seq, grasp_orn_seq, translation_z, use_shake_action=False, save=False, depth_at_pick_pt=1.1, use_reset_speed=False):

		#print(point_seq)
		#print(point_seq[3][0])

		for i in range(len(point_seq)-1):
			curr_pt = [point_seq[i][0][0], point_seq[i][0][1], self.z_grasp]
			next_pt = [point_seq[i+1][0][0], point_seq[i+1][0][1], self.z_grasp]
			#print(curr_pt)
			#print(grasp_orn_seq[0][i])
			self.wrist_unwinder()
			self.pick_and_place(curr_pt, next_pt, grasp_orn_seq[0][i], translation_z, use_shake_action=use_shake_action, save=save, depth_at_pick_pt=depth_at_pick_pt, use_reset_speed=use_reset_speed)

		#return None


	def gather_at_point(self, pick_point_seq, grasp_orn_seq, gathering_pt, translation_z, use_shake_action=False, save=False, depth_at_pick_pt=1.1, use_reset_speed=False):

		for i in range(len(pick_point_seq)):
			self.pick_and_place(pick_point_seq[i], gathering_pt, grasp_orn_seq[i], translation_z, use_shake_action=use_shake_action, save=save, depth_at_pick_pt=depth_at_pick_pt, use_reset_speed=use_reset_speed)

		#return None



	def shake_action(self, shake_point, shake_orn, num_shake_actions=3):

		x_shake, y_shake, z_shake = shake_point

		z_shake_low = z_shake #- self.SHAKE_EPS
		z_shake_high = z_shake #+ self.SHAKE_EPS

		x_shake_low = x_shake #- self.SHAKE_EPS
		x_shake_high = x_shake #+ self.SHAKE_EPS

		shake_orn_low = shake_orn
		shake_orn_high = shake_orn + m.pi/6.

		xyz_shake_low = [x_shake_low, y_shake, z_shake_low]
		xyz_shake_high = [x_shake_high, y_shake, z_shake_high]

		# xyz_shake_low = deepcopy(shake_point)
		# xyz_shake_high = deepcopy(shake_point)

		#orn_shake_low = shake_orn_low
		#orn_shake_high = shake_orn_high

		for _ in range(num_shake_actions):
			self.move_ur5py(xyz_shake_low, shake_orn_low, self.shake_speed, self.shake_acc)
			self.move_ur5py(xyz_shake_high, shake_orn_high, self.shake_speed, self.shake_acc)
		return None

	def wrist_unwinder(self):
		ac_q = self.robot.get_joints()
		if abs(ac_q[5]) > m.pi:
			new_q = ac_q
			if ac_q[5] > m.pi:
				new_q[5] = ac_q[5]-m.pi
			if ac_q[5] < -m.pi/2:
				new_q[5] = ac_q[5]+m.pi
		
			self.robot.move_joint(new_q, vel=3, acc=3)
			#time.sleep(0.1)

		return None
			

	def get_state(self, use_depth=False, viz_image=False, is_weight = False):

		rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline,
									self.align, self.depth_scale, use_depth=use_depth)

		# Use x_crop and y_crop to get workspace image
		rs_color_image = cv2.resize(rs_color_image, (640, 480))
		color_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		# 4
		rs_scaled_depth_image = cv2.resize(rs_scaled_depth_image, (640, 480))
		#color_image = cv2.resize(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), (575, 444))

		


		if viz_image:
			MAX_TIME = 10000
			start_time = timeit.default_timer()
			total_time = 0

			while(True or total_time <= MAX_TIME):

				rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=use_depth)
				color_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
				r, g, b = cv2.split(color_image)
				color_image = cv2.merge((b, g, r))

				cv2.imshow('Color image', color_image)

				if use_depth:
					depth_image = rs_scaled_depth_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
					cv2.imshow('Depth image', depth_image)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				total_time = timeit.default_timer() - start_time

			cv2.destroyAllWindows()
			cv2.waitKey()

		if use_depth:
			depth_image = rs_scaled_depth_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
			#depth_image = cv2.resize(depth_image, (575, 444))
		else:
			depth_image = None

		#plt.imshow(color_image)
		#plt.show()
		#import pdb; pdb.set_trace()

		cloth_points, cloth_depths, bw_image = self.get_cloth_points(color_image, depth_image, use_depth=use_depth)

		return cloth_points, cloth_depths, color_image, depth_image, bw_image


	def random_policy(self, cloth_points, rgb_image, viz_policy=False, save=False):

		# Sample point at random from cloth points
		num_cloth_points = len(cloth_points)
		rand_pt_index = np.random.randint(num_cloth_points)
		im_pt = cloth_points[rand_pt_index]

		rand_real_pt = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]

		if viz_policy:
			c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			cv2.imshow('Random Policy', c_im)
			cv2.waitKey()

		if save:
			c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			cv2.imwrite(self.trial_path+'random_policy_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		# grasp_orn = self.get_pca_grasp_orn(cloth_points, im_pt)
		grasp_orn = np.random.uniform(low=0, high=m.pi/2.)
		return [rand_real_pt[0], rand_real_pt[1], self.z_grasp], grasp_orn, im_pt

	

	def get_clusters(self, cloth_points, cloth_depths, rgb_image, bw_image,
								depth_image, pick_method='centroid', save=False):

		clusters = []

		init_clusters = self.get_init_clusters(cloth_points, cloth_depths,
									rgb_image, bw_image, depth_image, save=save)

		for cluster_ind in range(len(init_clusters)):

			cluster = init_clusters[cluster_ind]

			pick_point, pick_orn, pick_im_pt = self.get_cluster_pick_values(cloth_points,
					cluster['centroid'],
					cluster['max_height_point'],
					pick_method=pick_method)

			# Add extra files to the dict
			cluster['pick_point'] = pick_point
			cluster['pick_orn'] = pick_orn
			cluster['pick_im_pt'] = pick_im_pt

			clusters.append(cluster)

		sorted_clusters = sorted(clusters, key=lambda x: x["pred_weight"], reverse=True)

		return sorted_clusters


	def get_cluster_pick_values(self, cloth_points, centroid, max_height_pt, pick_method):

		if pick_method == 'centroid':
			pick_im_pt = deepcopy(centroid)
		elif pick_method == 'max_height':
			pick_im_pt = deepcopy(max_height_pt)
		else:
			print ('Invalid pick method')
			pick_im_pt = []
		pick_im_pt[0] = pick_im_pt[0] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		pick_im_pt[1] = pick_im_pt[1] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		pick_im_pt[0], pick_im_pt[1] = self.im_pt_exceeded(pick_im_pt[0], pick_im_pt[1])
		rw_pick = self.ir.image_pt_to_rw_pt([pick_im_pt[1], pick_im_pt[0]])

		rw_x = rw_pick[0][0]
		rw_y = rw_pick[0][1]

		pick_point = [rw_x, rw_y, self.z_grasp]
		pick_orn = self.get_pca_grasp_orn(cloth_points, pick_im_pt)

		return pick_point, pick_orn, pick_im_pt

	def im_pt_exceeded(self, p0, p1):

		if p0 >= self.IM_X_MAX:
			p0 = self.IM_X_MAX-1
		if p1 >= self.IM_Y_MAX:
			p1 = self.IM_Y_MAX-1

		return p0, p1

	def convert_pts(self, reshaped_pca_pts, centers, max_height_pts):

		new_max_height_pts = []

		for max_height_pt in max_height_pts:
			new_max_height_pts.append([max_height_pt[1], max_height_pt[0]])


		conv_centers = []
		for center in centers:
			new_center = [center[1], center[0]]
			conv_centers.append(new_center)

		new_reshaped_pca_pts = []
		for pca_pts in reshaped_pca_pts:
			new_pca_pts = []
			for pt in pca_pts:
				new_pt = [pt[1], pt[0]]
				new_pca_pts.append(new_pt)

			new_reshaped_pca_pts.append(new_pca_pts)

		return np.array(new_reshaped_pca_pts), conv_centers, new_max_height_pts

	def get_max_height_props(self, all_cluster_pts, depth_image, bw_image):

		all_max_height_pts = []
		all_max_heights = []
		all_avg_heights = []
		all_max_height_pick_orns = []

		for cluster_pts in all_cluster_pts:
			cloth_depths = []
			cloth_points_to_use = []
			for pt in cluster_pts:
				im_x, im_y = pt
				if bw_image[im_x, im_y] == 0:
					# Cloth points
					depth_val = depth_image[im_x, im_y]
					# print (depth_val)
					if depth_val != 0:
						cloth_depths.append(1/depth_val)
						cloth_points_to_use.append(pt)

			if len(cloth_depths) > 0:
				max_height_index = np.argmax(cloth_depths)
				max_height_pt = cloth_points_to_use[max_height_index]
				max_height = cloth_depths[max_height_index]
				avg_height = np.mean(cloth_depths)
				max_height_pick_orn = self.get_pca_grasp_orn(cloth_points_to_use, max_height_pt)

			else:
				max_height_pt = [0,0]
				max_height = 1.2
				avg_height = 1.2
				max_height_pick_orn = 0.0

			all_avg_heights.append(avg_height)
			all_max_heights.append(max_height)
			all_max_height_pts.append(max_height_pt)
			all_max_height_pick_orns.append(max_height_pick_orn)

		# cv2.imshow('Title', bw_image)
		# cv2.waitKey()
		# for cluster_pts in all_cluster_pts:
		# 	plt.scatter(np.array(cluster_pts)[:,0], np.array(cluster_pts)[:,1])
		#
		# plt.show()
		# plt.cla()

		return all_max_heights, all_max_height_pts, all_avg_heights, all_max_height_pick_orns

	def get_downsampled_points(self, init_points):

		num_samples = int(len(init_points)/20)

		indices = np.random.permutation(len(init_points))

		result = np.array(init_points)[indices[0:num_samples]]

		return result

	def get_max_volume_props(self, cluster_cloth_pts, depth_image, bw_image):
		disk_radius = 50 # Pixels

		max_volumes = []
		max_volume_points = []
		max_volume_pick_orns = []

		for cluster_pts in cluster_cloth_pts:
			all_pt_volumes = []
			all_pts = []
			downsampled_pts = self.get_downsampled_points(cluster_pts)
			for pt in downsampled_pts:
				# Compute all points in cluster pts that are a disk_radius away
				relevant_pts = self.compute_disk_pts(pt, downsampled_pts, disk_radius)
				# print (len(relevant_pts))
				relevant_pt_heights = []
				for relevant_pt in relevant_pts:
					pt_depth = depth_image[relevant_pt[0], relevant_pt[1]]
					if pt_depth > 0:
						pt_height = 1/pt_depth
						relevant_pt_heights.append(pt_height)

				pt_volume = np.sum(relevant_pt_heights)
				all_pt_volumes.append(pt_volume)
				all_pts.append(pt)

			if len(all_pt_volumes) > 0:
				max_volume_pt_ind = np.argmax(all_pt_volumes)
				max_volume = all_pt_volumes[max_volume_pt_ind]
				max_volume_point = all_pts[max_volume_pt_ind]
				max_volume_pick_orn = self.get_pca_grasp_orn(cluster_pts, max_volume_point)
			else:
				max_volume = 1.0
				max_volume_point = [0.0, 0.0]
				max_volume_pick_orn = 0

			max_volumes.append(max_volume)
			max_volume_points.append(max_volume_point)
			max_volume_pick_orns.append(max_volume_pick_orn)

		return max_volumes, max_volume_points, max_volume_pick_orns

	def compute_disk_pts(self, pt, cluster_pts, disk_radius):

		# IPython.embed()
		relevant_pts = []
		# print (len(cluster_pts))
		interval = 3
		count_interval = 0
		for c_pt in cluster_pts:
			count_samples = 0
			# if count_interval == interval*count_samples:
			dist = np.linalg.norm(np.array(c_pt) - np.array(pt))
			if dist <= disk_radius:
				relevant_pts.append(c_pt)
				count_samples += 1

			count_interval += 1

		# print ('DOne with disks')
		# print ('relevant pts', len(relevant_pts))
		return relevant_pts

	def get_init_clusters(self, cloth_points, cloth_depths, rgb_image, bw_image, depth_image, useLN = False, save=False):

		if not useLN:
			pred_weight = 0
			ln_im_pick_pt = 0
			ln_pick_point = 0
			ln_pick_orn = 0
		
		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

		pca_points, contours, cnt_centers, cnt_radius, _ = self.pca_cluster(grayscale_image)

		cluster_cloth_pts = self.get_cluster_points(contours, bw_image)

		# clusters = self.get_clusters_cc(bw_image)

		# clusters = self.get_clusters_cc(grayscale_image)

		max_heights, max_height_pts, avg_heights, max_height_pick_orns = self.get_max_height_props(cluster_cloth_pts, depth_image, bw_image)

		# max_volumes, max_volume_points, max_volume_pick_orns = self.get_max_volume_props(cluster_cloth_pts, depth_image, bw_image)

		max_volumes = deepcopy(max_heights)
		max_volume_points = deepcopy(max_height_pts)
		max_volume_pick_orns = deepcopy(max_height_pick_orns)


		cnt_centers = []

		if useLN:
			cnt_ln_pick_points = []
			cnt_ln_pick_orns = []
			cnt_ln_im_pick_pts = []
			cnt_ln_pred_weights = []

		# IPython.embed()
		for cnt in cluster_cloth_pts:
			try:
				cnt_center	= np.mean(cnt, axis=0)
			except:
				print ('Contour has no points')
				return []

			cnt_centers.append([int(cnt_center[0]), int(cnt_center[1])])

			if useLN:
				ln_pick_point, ln_pick_orn, ln_im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cnt, cloth_depths, depth_image, save=False, focus_cluster=True)
				cnt_ln_pick_points.append(ln_pick_point)
				cnt_ln_pick_orns.append(ln_pick_orn)
				cnt_ln_im_pick_pts.append([ln_im_pick_pt[0]*240/444, ln_im_pick_pt[1]*330/575])
				cnt_ln_pred_weights.append(pred_weight)

		num_cnts = len(contours)
		reshaped_pca_pts = np.reshape(pca_points, (num_cnts, 4, 2))

		reshaped_pca_pts, _, _ = self.convert_pts(reshaped_pca_pts, cnt_centers, max_height_pts)

		# components = self.get_clusters_cc(bw_image)
		#
		# for ind in range(len(components)):
		# 	for im_pt in components[ind]:
		# 		# im_pt = deepcopy(max_height_pts[0])
		# 		c_clus = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), 1)
		# 	cv2.imshow('Show cluster', c_clus)
		# 	cv2.waitKey()

		#
		# im_pt = deepcopy(max_height_pts[0])
		# c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
		# cv2.imshow('SHow Max height', c_im)
		# cv2.waitKey()

		# im_pt = cnt_centers[0]
		# c_im2 = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (0, 0,255), self.IM_CIRCLE_P_2)
		# cv2.imshow('Show cnt centeres', c_im2)
		# cv2.waitKey()


		all_cnt_areas = []

		if save:
			drawn_cnts_image = rgb_image.copy()
			all_cnt_colors = [(0, 255, 0), (255, 255, 0), (0, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 0)]

		cnt_ind = 0
		for cnt in contours:
			cnt_area = cv2.contourArea(np.array(cnt, dtype=int))
			if save:
				drawn_cnts_image = cv2.drawContours(drawn_cnts_image, np.array(cnt, dtype=int), -1, all_cnt_colors[cnt_ind], 3)
			all_cnt_areas.append(cnt_area)
			cnt_ind += 1

		if save:
			for pca_pts in reshaped_pca_pts:
				for pt in pca_pts:
					# Add the pca points per cluster
					drawn_cnts_image = cv2.circle(drawn_cnts_image, [int(pt[1]), int(pt[0])], 3, (0, 255, 0), 1)

		init_clusters = []

		for cluster_ind in range(num_cnts):
			area = all_cnt_areas[cluster_ind]
			pca_points = reshaped_pca_pts[cluster_ind]
			centroid = cnt_centers[cluster_ind]

			avg_height = avg_heights[cluster_ind]
			max_height = max_heights[cluster_ind]
			max_height_point = max_height_pts[cluster_ind]
			max_height_pick_orn = max_height_pick_orns[cluster_ind]

			# Volume
			max_volume = max_volumes[cluster_ind]
			max_volume_point = max_volume_points[cluster_ind]
			max_volume_pick_orn = max_volume_pick_orns[cluster_ind]

			# Centroid
			centroid_pick_orn = self.get_pca_grasp_orn(cluster_cloth_pts[cluster_ind], centroid)

			if useLN:
				ln_im_pick_pt = cnt_ln_im_pick_pts[cluster_ind]
				ln_pick_orn = cnt_ln_pick_orns[cluster_ind]
				ln_pick_point = cnt_ln_pick_points[cluster_ind]
				pred_weight = cnt_ln_pred_weights[cluster_ind]

			cluster = {'area': area,
					'pca_points': pca_points,
					'centroid': centroid,
					'max_height': max_height,
					'max_height_point': max_height_point,
					'max_height_pick_orn': max_height_pick_orn,
					'avg_height': avg_height,
					'pred_weight': pred_weight,
					'ln_im_pick_pt': ln_im_pick_pt,
					'ln_pick_point': ln_pick_point,
					'ln_pick_orn': ln_pick_orn,
					'max_volume': max_volume,
					'max_volume_point': max_volume_point,
					'max_volume_pick_orn': max_volume_pick_orn,
					'centroid_pick_orn': centroid_pick_orn,
					'cluster_pts': cluster_cloth_pts[cluster_ind]}

			init_clusters.append(cluster)

		if save:
			cv2.imwrite(data_path + 'rgb_im_cnts.jpg', rgb_im_cnts, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		return init_clusters

	def get_cluster_points(self, all_contours, bw_image):

		# IPython.embed()


		all_cnt_pts = [[] for _ in range(len(all_contours))]

		x_full = np.linspace(start=0, stop=bw_image.shape[0], num=bw_image.shape[0], dtype=int)
		y_full = np.linspace(start=0, stop=bw_image.shape[1], num=bw_image.shape[1], dtype=int)

		x_subs = random.sample(x_full.tolist(), self.NUM_IMAGE_SAMPLES_FOR_CLUSTER)
		y_subs = random.sample(y_full.tolist(), self.NUM_IMAGE_SAMPLES_FOR_CLUSTER)

		# x_subs = x_full.tolist()
		# y_subs = y_full.tolist()

		# At least have boundaries in the contour points
		# IPython.embed()

		num_cnts = len(all_contours)
		for cnt_ind in range(num_cnts):
			cnt = all_contours[cnt_ind]
			len_cnt = len(cnt)
			cnt_reshaped = np.reshape(cnt, (len_cnt, 2))
			for c_reshaped in cnt_reshaped:
				im_y = c_reshaped[0]
				im_x = c_reshaped[1]
				all_cnt_pts[cnt_ind].append([int(im_x), int(im_y)])

		# cc = all_cnt_pts[0]
		# plt.scatter(np.array(cc)[:,0], np.array(cc)[:,1], color='blue')

		for im_x in x_subs:
			for im_y in y_subs:
				for cnt_ind in range(len(all_contours)):
					cnt = all_contours[cnt_ind]
					dist_cnt = cv2.pointPolygonTest(np.array(cnt, dtype=int),(im_y, im_x),True)
					if dist_cnt > 0:
						# Interior
						# CHeck if it is also a cloth point
						if bw_image[im_x, im_y] == 0:
							all_cnt_pts[cnt_ind].append([im_x, im_y])


		# dd = all_cnt_pts[0]
		# plt.scatter(np.array(cc)[:,0], np.array(cc)[:,1], color='red')
		# plt.show()

		return all_cnt_pts

	def max_volume_constraint_policy(self, rgb_image, depth_image, bw_image, save=False, viz_policy='False'):

		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

		pca_points, contours, cnt_centers, cnt_radius, _ = self.pca_cluster(grayscale_image)

		cluster_cloth_pts = self.get_cluster_points(contours, bw_image)
		max_volumes, max_volume_points, max_volume_pick_orns = self.get_max_volume_props(cluster_cloth_pts, depth_image, bw_image)

		max_index = np.argmax(max_volumes)

		pick_im_pt = max_volume_points[max_index]

		pick_im_pt[0] = pick_im_pt[0] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		pick_im_pt[1] = pick_im_pt[1] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		pick_im_pt[0], pick_im_pt[1] = self.im_pt_exceeded(pick_im_pt[0], pick_im_pt[1])

		pick_orn = max_volume_pick_orns[max_index]

		rw_pick = self.ir.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])

		rw_x = rw_pick[0][0]
		rw_y = rw_pick[0][1]
		depth_at_pick_pt = depth_image[pick_im_pt[0], pick_im_pt[1]]
		pick_z = self.get_pick_depth(depth_at_pick_pt)
		pick_point = [rw_x, rw_y, pick_z]

		if save:
			c_im = cv2.circle(rgb_image, [pick_im_pt[1], pick_im_pt[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			cv2.imwrite(self.trial_path+'max_volume_contraint_policy_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		return pick_point, pick_orn, pick_im_pt

	def rank_clusters(self, clusters):
		ranked_clusters = sorted(clusters, key=lambda cluster: cluster['score'], reverse=True)
		return ranked_clusters

	def compute_cluster_grasp(self, cluster_props):
		# Just cluster centroid for now and the minor axis
		# TODO: Combine max height + cloth variation + region density

		cluster_centroid = cluster_props['centroid']
		cluster_pcs = cluster_props['pcs']

		cl_x, cl_y = cluster_centroid
		cl_rw = self.ir.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
		cluster_grasp = [cl_rw[0], cl_rw[1], self.z_grasp]

		# Grasp should be along the minor axis!
		cl_grasp_vec = deepcopy(cluster_pcs[0])
		grasp_orn = m.atan2(cl_grasp_vec[0], cl_grasp_vec[1])

		return cluster_grasp_point, cluster_grasp_orn

	def make_cluster_dense(self, cluster_props):
		cluster_pcs = cluster_props['pcs']

		# Major axis push
		cluster_pcs[0]

		# Pick and place?
		# Push using PCA?
		return None

	def get_largest_cluster(self, clusters):

		all_cluster_areas = []
		for cluster_props in clusters:
			cluster_area = cluster_props['area']
			all_cluster_areas.append(cluster_area)

		max_ind = np.argmax(all_cluster_areas)

		largest_cluster_props = clusters[max_ind]

		return largest_cluster_props

	def get_closest_cluster(self, des_cluster, clusters):

		des_cluster_id = des_cluster['id']
		des_cluster_centroid = des_cluster['centroid']

		all_dists = []
		for cluster_prop in clusters:
			cluster_centroid = cluster_prop['centroid']
			cluster_id = cluster_prop['id']
			if cluster_id != des_cluster_id:
				dist_to_des_cluster = np.linalg.norm(des_cluster_centroid-cluster_centroid)
				all_dists.append(dist_to_des_cluster)


		if len(all_dists) > 0:
			min_dist_ind = np.argmin(all_dists)
			closest_cluster_props = clusters[min_dist_ind]
		else:
			closest_cluster_props = []

		return closest_cluster_props

	def get_cloth_points(self, rgb_image, depth_image, use_depth=False):


		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		(thresh, bw_image) = cv2.threshold(grayscale_image, self.IM_THRESH_LOW, self.IM_THRESH_HIGH, cv2.THRESH_BINARY)

		bw_image = cv2.dilate(bw_image, kernel=np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE)))


		# IPython.embed()
		# cv2.imshow('gs', grayscale_image)
		# cv2.imshow('thresh', thresh)
		# cv2.imshow('bw im', bw_image)
		# cv2.imshow('Depth', depth_image)
		# cv2.waitKey()

		cloth_points = []
		cloth_depths = []

		# print ('Xxx', bw_image.shape[0])
		# print ('Yyyy', bw_image.shape[1])
		for im_x in range(self.GRIPPER_WIDTH_PIXELS, bw_image.shape[0]):
			for im_y in range(bw_image.shape[1]):
				if bw_image[im_x, im_y] == 0:
					# Cloth points
					depth_val = depth_image[im_x, im_y]
					if depth_val != 0:
						cloth_depths.append(1/depth_val)
						cloth_points.append([im_x, im_y])

		# print ('cloth depths', cloth_depths)
		return cloth_points, cloth_depths, bw_image

	def get_cloth_points2(self, rgb_image, use_depth=False):


		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		(thresh, bw_image) = cv2.threshold(grayscale_image, self.IM_THRESH_LOW, self.IM_THRESH_HIGH, cv2.THRESH_BINARY)

		bw_image = cv2.dilate(bw_image, kernel=np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE)))


		# IPython.embed()
		# cv2.imshow('gs', grayscale_image)
		# cv2.imshow('thresh', thresh)
		# cv2.imshow('bw im', bw_image)
		# cv2.imshow('Depth', depth_image)
		# cv2.waitKey()

		cloth_points = []
		cloth_depths = []

		# print ('Xxx', bw_image.shape[0])
		# print ('Yyyy', bw_image.shape[1])
		for im_x in range(self.GRIPPER_WIDTH_PIXELS, bw_image.shape[0]):
			for im_y in range(bw_image.shape[1]):
				if bw_image[im_x, im_y] == 0:
					# Cloth points
					depth_val = depth_image[im_x, im_y]
					if depth_val != 0:
						cloth_points.append([im_x, im_y])

		# print ('cloth depths', cloth_depths)
		return cloth_points, cloth_depths, bw_image

	def min_depth_policy(self, cloth_points, cloth_depths):

		min_depth_index = np.argmin(cloth_depths)
		im_pt = cloth_points[min_depth_index]

		real_pt = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]
		grasp_cand = [rand_real_pt, np.random.uniform(0, m.pi/2.)]

		return grasp_cand

	def image_pt_to_rw_pt(self, image_pt, depth=None):
		#reversed_image_pt = [image_pt[1], image_pt[0]]

		#self.x_intercept = 150.1111247
		#self.x_coef_on_x = -2.07787301
		#self.x_coef_on_y = 0.02772887548

		#self.y_intercept = -759.9587912
		#self.y_coef_on_y = 2.069261384
		#self.y_coef_on_x = 0.02158838398
		
		if depth:

			height = 800 - depth

			height_fraction = height/(self.upper_z_value - self.surface_z_value)

			print("height fraction:" + str(height_fraction))
		
			image_pt_tr = self.cam_scaler.transform([image_pt])
			rw_pt_surface = self.cam_model.predict(image_pt_tr)

			rw_pt_upper = [0,0]
			rw_pt_upper[0] = self.upper_x_intercept + self.upper_x_coef_on_x*image_pt[0] + self.upper_x_coef_on_y*image_pt[1]
			rw_pt_upper[1] = self.upper_y_intercept + self.upper_y_coef_on_x*image_pt[0] + self.upper_y_coef_on_y*image_pt[1]

			print(rw_pt_upper)
			print(rw_pt_surface)

			rw_pt_surface = np.array(rw_pt_surface)
			rw_pt_upper = np.array(rw_pt_upper)
			
			if height_fraction > 0.15:
				rw_pt = height_fraction * rw_pt_upper + (1 - height_fraction) * rw_pt_surface
			else:
				rw_pt = rw_pt_surface

		else:
			if self.use_hardcoded_cal:
				rw_pt = [0,0]
				rw_pt[0] = self.x_intercept + self.x_coef_on_x*image_pt[1] + self.x_coef_on_y*image_pt[0]
				rw_pt[1] = self.y_intercept + self.y_coef_on_x*image_pt[1] + self.y_coef_on_y*image_pt[0]

			else:
				image_pt_tr = self.cam_scaler.transform([image_pt])
				rw_pt = self.cam_model.predict(image_pt_tr)
		
		
		return rw_pt

	def gen_lookup_depth_table(self, depth_image):

		image_x = depth_image.shape[0]
		image_y = depth_image.shape[1]

		rw_coords_table = np.zeros((image_x, image_y, 2))

		for x_coord in range(image_x):
			for y_coord in range(image_y):
				im_pt = [x_coord, y_coord]
				rw_pt = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]
				rw_coords_table[x_coord, y_coord] = rw_pt.copy()

		# Save depth lookup table and load
		np.save("depth_lookup_table", rw_coords_table)

		self.depth_lookup_table = np.load("depth_lookup_table.npy")

		return rw_coords_table

	def load_model(self, model_name, scaler_name):
		model = load(open(model_name, 'rb'))
		scaler = load(open(scaler_name, 'rb'))
		return model, scaler

	def cluster_detector(self, grayscale_image, window=False):
		if window == False:
			ret, thresh1 = cv2.threshold(grayscale_image, self.IM_THRESH_LOW,
										self.IM_THRESH_HIGH, cv2.THRESH_BINARY_INV)
			smallest_sock_area = 1500 # changed
			centers = []
			num_of_clusters = 0
			clusters = []
			MIN_ASPECT_RATIO = 0.1
		else:
			thresh1 = grayscale_image
			smallest_sock_area = 1 # changed
			centers = []
			num_of_clusters = 0
			clusters = []
			MIN_ASPECT_RATIO = 0.01 #CHANGED

		contours, heirachy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
		# cv2.drawContours(grayscale_image, sorted_contours, -1, (0,255,0), 1)
		# cv2.imshow('window', grayscale_image)
		# cv2.waitKey()

		cnt_radius = []
		for index, cnt in enumerate(sorted_contours):
			(x, y), radius = cv2.minEnclosingCircle(cnt)
			area = cv2.contourArea(cnt)
			(x_br, y_br, w_br, h_br) = cv2.boundingRect(cnt)
			aspect_ratio = float(w_br)/float(h_br)
			# print (aspect_ratio)
			# print('')

			if area >= smallest_sock_area:
				if aspect_ratio >= MIN_ASPECT_RATIO:
					clusters.append(cnt)
					centers.append([int(x), int(y)])
					cnt_radius.append(radius)
					num_of_clusters += 1

		cluster_pts = []
		for index, cnt in enumerate(clusters):
			extra_points = cnt.copy()
			prev_pt = cnt[0][0]
			end_pt = cnt[len(cnt) - 1][len(cnt[len(cnt) - 1]) - 1]
			ends_dist = distance.euclidean(prev_pt, end_pt)
			if ends_dist > 50:
				fm = np.linspace([prev_pt[0], prev_pt[1]], [end_pt[0], end_pt[1]])
				fill_mid = [[i] for i in fm]
				s = np.concatenate((extra_points, fill_mid), axis=0)
				extra_points = s
			for i in range(0, len(cnt)):
				for j in range(0, len(cnt[i])):
					curr_pt = cnt[i][j]
					dist = distance.euclidean(curr_pt, prev_pt)
					if dist > 40:
						fm = np.linspace([prev_pt[0], prev_pt[1]], [curr_pt[0], curr_pt[1]])
						fill_mid = [[i] for i in fm]
						s = np.concatenate((extra_points, fill_mid), axis=0)
						extra_points = s
					prev_pt = curr_pt
			cluster_pts.append(extra_points)
		# print(centers)
		# x = []
		# y = []
		# for i in range(len(cluster_pts)):
		# 	for j in range(len(cluster_pts[i])):
		# 		for k in cluster_pts[i][j]:
		# 			x.append(k[0])
		# 			y.append(k[1])
		# 			cv2.circle(grayscale_image, (int(k[0]), int(k[1])), 5, (255, 0, 0), 5)
		# for i in centers:
		# 	cv2.circle(grayscale_image, (i[0], i[1]), self.IM_CIRCLE_P_1, (255, 0, 0), 2)
		# cv2.imshow("image", grayscale_image)
		# cv2.waitKey()
		# print ('Num clusters', len(centers))
		# IPython.embed()
		bounding_rectangles = [cv2.boundingRect(np.array(cluster_pts[i], dtype=int)) for i in range(len(cluster_pts))]
		return cluster_pts, centers, cnt_radius, bounding_rectangles

	def capture_data_samples(self, sample_num=0, data_path='./dmog_data/'):

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		cv2.imwrite(data_path+'rgb_sample_{}.jpg'.format(sample_num), rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
		np.save(data_path+'depth_arr_{}'.format(sample_num), depth_image)
		np.save(data_path+'cloth_points_{}'.format(sample_num), cloth_points)
		np.save(data_path+'cloth_depths_{}'.format(sample_num), cloth_depths)
		np.save(data_path+'bw_image_{}'.format(sample_num), bw_image)

		return None

	def load_sample_data(self, sample_num=0, data_path='./dmog_data/'):

		rgb_image = cv2.imread(data_path+'sample_{}.jpg'.format(sample_num))
		cloth_points = np.load(data_path+'cloth_points_{}.npy'.format(sample_num), allow_pickle=False)
		cloth_depths = np.load(data_path+'cloth_depths_{}.npy'.format(sample_num))
		bw_image = np.load(data_path+'bw_image_{}.npy'.format(sample_num))
		depth_image = np.load(data_path+'depth_arr_{}.npy'.format(sample_num))

		return cloth_points, cloth_depths, rgb_image, depth_image, bw_image


	def rearrangement_sample(self, sample):
		trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='weight')
		trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='volume')
		trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='height')

		# Max height

		# Garment-Net

		# Max volume

		return None




	def full_experiment_pipeline(self, policy='random', rearrangement='no_rearrangement', predictor='baseline', num_samples=25, save=True):
		self.wrist_unwinder()
		self.go_home()
		
		
		if policy in ['segmentation_and_height', 'segmentation_and_volume', 'height', 'volume', 'weight']:
			self.use_pick_pt_depth = True
		else:
			self.use_pick_pt_depth = False
		num_garments = int(input("Enter total number of garments: "))
		if policy == 'segmentation':
			self.mask_generator = self.loadSAM()
			if predictor == 'NN':
				self.load_nn_params()
		if policy == "segmentation_and_height":
			print("Loading SAM...")
			self.mask_generator = self.loadSAM()
			if predictor == 'NN':
				self.load_nn_params()
		for sample_num in range(num_samples):
			current_ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
			if save:
				self.data_path = './experiment_files/data_{}/method_{}/predictor_{}/sample_{}/'.format(rearrangement, policy, predictor, current_ts)
				if not os.path.isdir(self.data_path):
					subprocess.call(['mkdir', '-p', self.data_path])

				param_dict = self.create_param_dict()
				np.save(self.data_path+'param_dict', param_dict)
			start_weight = self.recordWeight(0)
			print('start weight: ' + str(start_weight))

			if False: # RESET
				print('Resetting scene...')

				self.reset_scene()

				print('Scene reset!')
				self.move_above_place_point()

				time.sleep(3)
				scene_weight = self.recordWeight(0)
				if scene_weight > 14.0:
					print('Clothing dump failed, please put garments onto work surface')
					ruh_roh = cv2.imread('ruh_roh.jpeg')
					ruh_roh = cv2.cvtColor(ruh_roh, cv2.COLOR_BGR2RGB)
					plt.imshow(ruh_roh)
					plt.show()

				self.shuffle_placed_clothes_v2()

			


			start_time = timeit.default_timer()

			

			print("Starting sample number: "+str(sample_num))
			
			info_dictionary = self.one_sample_no_rearrangement(policy, save=save, predictor=predictor, rearrangement=rearrangement) # 

			
			final_time = timeit.default_timer()

			time.sleep(3)
			final_weight = self.recordWeight(0)

			print('final weight: ' + str(final_weight))

			num_dropped = 0

			if final_weight < start_weight - 14.0:
				print('Clothing not in basket!')
				ruh_roh = cv2.imread('ruh_roh.jpeg')
				ruh_roh = cv2.cvtColor(ruh_roh, cv2.COLOR_BGR2RGB)
				plt.imshow(ruh_roh)
				plt.show()
				num_dropped = int(input("Enter number of dropped garments: "))

				

			if save:
				weight_seq = info_dictionary['weight_seq']
				weight_seq.append(final_weight)
				time_seq = info_dictionary['time_seq']
				time_seq.append(final_time-start_time)
				rgb_image_seq = info_dictionary['rgb_image_seq']
				bw_image_seq = info_dictionary['bw_image_seq']
				depth_image_seq = info_dictionary['depth_image_seq']
				pick_seq = info_dictionary['pick_seq']

				garment_counts = [num_garments, num_dropped]

				np.save(self.data_path + 'weight_seq', weight_seq)
				np.save(self.data_path + 'time_seq', time_seq)
				#pdb.set_trace()
				np.save(self.data_path + 'pick_seq', np.asanyarray(pick_seq, dtype=object))
				np.save(self.data_path + 'garment_counts', garment_counts)
				np.save(self.data_path + 'num_attempts', self.num_attempts)
				if rearrangement == 'rearrangement':
					np.save(self.data_path + 'num_rearrangements', self.num_rearrangements)
				elif rearrangement == 'no_rearrangement':
					np.save(self.data_path + 'num_rearrangements', 0)

				if not rearrangement == 'no_rearrangement':
					np.save(self.data_path + 'num_rearrange_seq', info_dictionary['num_rearrange_seq'])

				if policy=='segmentation':
					masks_seq = info_dictionary['masks_seq']
					masks_raw_seq = info_dictionary['masks_raw_seq']
					cand_pts_seq = info_dictionary['cand_pts_seq']
					cand_orns_seq = info_dictionary['cand_orns_seq']
					new_compute_seq = info_dictionary['new_compute_seq']
					compute_time_seq = info_dictionary['compute_time_seq']
					
					np.save(self.data_path + 'masks_seq', masks_seq)
					np.save(self.data_path + 'masks_raw_seq', masks_raw_seq)
					np.save(self.data_path + 'cand_pts_seq', cand_pts_seq)
					np.save(self.data_path + 'cand_orns_seq', cand_orns_seq)
					np.save(self.data_path + 'new_compute_seq', new_compute_seq)
					np.save(self.data_path + 'compute_time_seq', compute_time_seq)

				num_trials = len(rgb_image_seq)

				for trial_num in range(num_trials):
					self.trial_path = self.data_path + 'trial_{}/'.format(trial_num)
					subprocess.call(['mkdir', '-p', self.trial_path])

					rgb_image = cv2.cvtColor(rgb_image_seq[trial_num], cv2.COLOR_BGR2RGB)
					cv2.imwrite(self.trial_path+'rgb_image.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
					np.save(self.trial_path + 'bw_image', bw_image_seq[trial_num])
					np.save(self.trial_path + 'im_pick_pt', pick_seq[trial_num][0])
					np.save(self.trial_path + 'pick_orn', pick_seq[trial_num][1])
					np.save(self.trial_path + 'depth_image', depth_image_seq[trial_num])

					if policy=='segmentation':
						np.save(self.trial_path + 'masks', masks_seq[trial_num])
						np.save(self.trial_path + 'masks_raw', masks_raw_seq[trial_num])

		
	def one_sample_no_rearrangement(self, policy='random', save=True, predictor='baseline', rearrangement='no_rearrangement'):
		if policy == 'segmentation' and rearrangement == 'sequence_full':
			print ('======= segmentation in full sequences policy =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq = self.laundry_seg_algo_sequences_long(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
		elif policy == 'segmentation_and_height' and rearrangement == 'sequence_selected':
			print ('======= segmentation and height policy with selected sequences =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq, num_rearrange_seq = self.laundry_hybrid_seg_height_sequences(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq, 'num_rearrange_seq':num_rearrange_seq}
		elif policy == 'segmentation_and_consolidation':
			print ('======= segmentation and consolidation policy =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq = self.laundry_seg_algo_with_consolidation(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
		elif policy == 'segmentation_and_height' and rearrangement == 'no_rearrangement':
			print ('======= segmentation and height policy =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq = self.laundry_seg_and_height_algo(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
		elif policy == 'segmentation_and_volume' and rearrangement == 'no_rearrangement':
			print ('======= segmentation and volume policy =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq = self.laundry_seg_and_vol_algo(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
		elif policy == 'segmentation' and rearrangement == 'no_rearrangement':
			print ('======= segmentation policy =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq = self.laundry_seg_algo(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
		elif policy == 'segmentation' and rearrangement == 'sequence_selected':
			print ('======= segmentation policy with selected sequences =======')
			weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq, num_rearrange_seq = self.laundry_seg_seq_selected(save=save, predictor=predictor)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, \
				'pick_seq':pick_seq, 'masks_seq':masks_seq, 'masks_raw_seq':masks_raw_seq, 'cand_pts_seq':cand_pts_seq, 'cand_orns_seq':cand_orns_seq, \
				'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq, 'num_rearrange_seq':num_rearrange_seq}
		elif policy in ['random', 'weight', 'volume', 'height']:
			to_print = '======= {} policy ======='.format(policy)
			print(to_print)
			if rearrangement == 'rearrangement':
				weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq = self.laundry_algo(pick_method=policy, save=save, use_fold=True, use_stack =True)
			else:
				weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq = self.laundry_new_algo(pick_method=policy, save=save, use_fold=False, use_stack =False)
			to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'depth_image_seq':depth_image_seq, 'pick_seq':pick_seq}
		#elif policy == 'random':
		#	print ('======= Random Policy =======')
		#	weight_seq, time_seq, rgb_image_seq, bw_image_seq, pick_seq = self.laundry_new_algo(pick_method = 'random', save=save)
		#	to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'bw_image_seq':bw_image_seq, 'pick_seq':pick_seq}
		else:
			print ('Pick a different policy, sorry.')

		return to_return



	def no_rearrangement_sample(self, policy, current_ts, save=False, viz_policy=False):

		MAX_TRIALS = 100
		num_trials = 0
		MIN_CLOTH_POINTS = 200

		self.data_path = './no_rearrangement_data/method_{}/sample_{}/'.format(policy, current_ts)
		initial_weight = self.recordWeight(0)
		weight_sequence = [initial_weight]

		sample_initial_time = timeit.default_timer()
		time_sequence = [0]
		

		if policy == 'segmentation':
				print ('======= Segmentation Policy ====== ')
				weight_seq, time_seq, rgb_image_seq, pick_seq, masks_seq, new_compute_seq, compute_time_seq = self.laundry_seg_algo(save=save)
				to_return = {'weight_seq':weight_seq, 'time_seq':time_seq, 'rgb_image_seq':rgb_image_seq, 'pick_seq':pick_seq, 'masks_seq':masks_seq, 'new_compute_seq':new_compute_seq, 'compute_time_seq':compute_time_seq}
				return to_return

		while num_trials < MAX_TRIALS :
			self.trial_path = self.data_path + 'trial_{}/'.format(num_trials)
			#subprocess.call(['mkdir', '-p', self.trial_path])

			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)

			# Record the initial weight
			init_action_weight = self.recordWeight(0)

			num_cloth_points = len(cloth_points)

			print("Number of cloth points", num_cloth_points)

			if num_cloth_points > MIN_CLOTH_POINTS:
				subprocess.call(['mkdir', '-p', self.trial_path])

				if policy == 'random':
					print ('======= Random Policy ====== ')
					pick_point, pick_orn, im_pick_pt = self.random_policy(cloth_points, rgb_image, viz_policy=viz_policy, save=save)
				elif policy == 'weight':
					print ('======= Max-Weight Policy ====== ')
					pick_point, pick_orn, im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, save=save)
				elif policy == 'height':
					print ('======= Max-Height Policy ====== ')
					pick_point, pick_orn, im_pick_pt = self.max_height_policy(cloth_points, cloth_depths, rgb_image, viz_policy=viz_policy, save=save)
				elif policy == 'volume':
					print ('======= Max-Volume Policy ====== ')
					pick_point, pick_orn, im_pick_pt = self.max_volume_constraint_policy( rgb_image, depth_image, bw_image, save=save, viz_policy=viz_policy)
				else:
					print ("Invalid policy")

				place_point = self.gen_place_point()
				self.pick_and_place(pick_point, place_point, pick_orn, self.translation_z, use_shake_action=False)

				curr_time = timeit.default_timer()
				time_sequence.append(curr_time-sample_initial_time)

				rw_pick_pt = deepcopy(pick_point)
				time.sleep(3)
				weight = self.recordWeight(0)
				weight_sequence.append(weight)
				#print('Weight is:' + str(weight) + 'g')

				cv2.imwrite(self.trial_path+'rgb_image.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
				np.save(self.trial_path+'bw_image', bw_image)
				np.save(self.trial_path + "depth_image", depth_image)
				#np.save(self.trial_path+'cloth_points', cloth_points)
				np.save(self.trial_path+'cloth_depths', cloth_depths)
				np.save(self.trial_path+'im_pick_pt', im_pick_pt)
				np.save(self.trial_path+'pick_orn', pick_orn)
				np.save(self.trial_path+'rw_pick_pt', rw_pick_pt)
				np.save(self.trial_path+'weight', weight)
				np.save(self.trial_path+'init_action_weight', init_action_weight)
				np.save(self.trial_path+'num_cloth_points', num_cloth_points)


				if policy == 'weight':
					print ('Predicted weight:', pred_weight)
					# actual_weight = 1e3
					#
					# try:
					# 	actual_weight =  float(str(weight)[:-2]) -  float(str(init_action_weight)[:-2])
					# except:
					# 	print ('Weight not properly read')
					# print ('Actual weight:', actual_weight)
					np.save(self.trial_path+'pred_weight', pred_weight)
					# np.save(self.trial_path+'actual_weight', actual_weight)
			else:
				break
			num_trials += 1

		print ('Number of trips', num_trials)

		return time_sequence, weight_sequence, num_trials

	def clear_table(self, policy='random', sample_num=0, viz_policy=False, save=False):
		MAX_TRIALS = 100
		num_trials = 0
		MIN_CLOTH_POINTS = 1000

		self.data_path = './pick_data/method_{}/sample_{}/'.format(policy, sample_num)

		while num_trials < MAX_TRIALS:
			self.trial_path = self.data_path + 'trial_{}/'.format(num_trials)
			subprocess.call(['mkdir', '-p', self.trial_path])

			if self.real_robot:
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			else:
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.load_sample_data(sample_num=sample_num)

			# cluster_coverage, div_pts, div_vals, cluster_cloth_pts = self.get_cluster_props(rgb_image)
			# IPython.embed()
			# Record the initial weight
			init_action_weight = self.recordWeight(0)

			num_cloth_points = len(cloth_points)

			# self.capture_data_samples(sample_num=0)

			print("Number of cloth points", num_cloth_points)

			if num_cloth_points > MIN_CLOTH_POINTS:
				if policy == 'random':
					print ('======= Random Policy ====== ')
					pick_point, pick_orn, im_pick_pt = self.random_policy(cloth_points, rgb_image, viz_policy=viz_policy, save=save)
				elif policy == 'laundry_net':
					print ('======= Max Weight Policy ====== ')
					pick_point, pick_orn, im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, save=save)
				elif policy == 'max_height':
					print ('======= Max-Height Policy ====== ')
					pick_point, pick_orn, im_pick_pt = self.max_height_policy(cloth_points, cloth_depths, rgb_image, viz_policy=viz_policy, save=save)
				elif policy == 'max_volume':
					pick_point, pick_orn, im_pick_pt = self.max_volume_policy(cloth_points, cloth_depths, rgb_image, bw_image, depth_image, viz_policy=viz_policy, save=save)
				elif policy == 'laundry_net_stack':
					self.laundry_net_stack_policy(viz_policy=viz_policy, save=save)
				elif policy == 'laundry_net_stack_fold':
					self.laundry_net_stack_fold_policy(viz_policy=viz_policy, save=save)
				elif policy == 'laundry_net_fold':
					self.laundry_net_fold_policy(viz_policy=viz_policy, save=save)
				else:
					print ("Invalid policy")

				place_point = self.gen_place_point()

				if policy == 'laundry_net_stack' or policy == 'laundry_net_stack_fold' or policy == 'laundry_net_fold':

					# self.pick_and_place(pick_point, place_point, pick_orn, self.translation_z, use_shake_action=False)
					# rw_pick_pt = deepcopy(pick_point)
					time.sleep(3)
					weight = self.recordWeight(0)
					print('Weight is:', weight)

					cv2.imwrite(self.trial_path+'rgb_image.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
					cv2.imwrite(self.trial_path+'bw_image.jpg', bw_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
					np.save(self.trial_path + "depth_image", depth_image)
					np.save(self.trial_path+'cloth_points', cloth_points)
					np.save(self.trial_path+'cloth_depths', cloth_depths)
					# np.save(self.trial_path+'im_pick_pt', im_pick_pt)
					# np.save(self.trial_path+'pick_orn', pick_orn)
					# np.save(self.trial_path+'rw_pick_pt', rw_pick_pt)
					np.save(self.trial_path+'weight', weight)
					np.save(self.trial_path+'init_action_weight', init_action_weight)
					np.save(self.trial_path+'num_cloth_points', num_cloth_points)

				if self.real_robot and policy != 'laundry_net_stack' and policy != 'laundry_net_stack_fold' and policy != 'laundry_net_fold' :
					self.pick_and_place(pick_point, place_point, pick_orn, self.translation_z, use_shake_action=False)
					rw_pick_pt = deepcopy(pick_point)
					# start_time = timeit.default_timer()
					time.sleep(3)
					weight = self.recordWeight(0)
					print('Weight is:', weight)

					cv2.imwrite(self.trial_path+'rgb_image.jpg', rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
					cv2.imwrite(self.trial_path+'bw_image.jpg', bw_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
					np.save(self.trial_path + "depth_image", depth_image)
					np.save(self.trial_path+'cloth_points', cloth_points)
					np.save(self.trial_path+'cloth_depths', cloth_depths)
					np.save(self.trial_path+'im_pick_pt', im_pick_pt)
					np.save(self.trial_path+'pick_orn', pick_orn)
					np.save(self.trial_path+'rw_pick_pt', rw_pick_pt)
					np.save(self.trial_path+'weight', weight)
					np.save(self.trial_path+'init_action_weight', init_action_weight)
					np.save(self.trial_path+'num_cloth_points', num_cloth_points)
					if policy == 'laundry_net':
						print ('Predicted weight:', pred_weight)
						actual_weight = 1e3

						try:
							actual_weight =  float(str(weight)[:-2]) -  float(str(init_action_weight)[:-2])
						except:
							print ('Weight not properly read')
						print ('Actual weight:', actual_weight)
						np.save(self.trial_path+'pred_weight', pred_weight)
						np.save(self.trial_path+'actual_weight', actual_weight)

			else:
				break

			num_trials += 1

		print ('Number of trips', num_trials)

		return None

	def test_laundry(self):

		# trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='weight')

		# IPython.embed()
		self.pick_experiments(policy='random')
		self.get_state(use_depth=True)

		# trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='weight')
		# trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='volume')
		# trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='height')

		# IPython.embed()

		# num_samples = 3
		# all_results = []
		# for _ in range(num_samples):
		# 	trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='weight')
		# 	trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='volume')
		# 	trips, fold_actions, stack_actions = self.laundry_algo(use_fold=True, use_stack=True, pick_method='height')
		#
		# 	# all_results.append([trips, fold_actions, stack_actions])
		# 	IPython.embed()

		return None


	def max_height_policy(self, cloth_points, cloth_depths, rgb_image, viz_policy=False, save=False):

		max_height_index = np.argmax(cloth_depths)
		im_pt = cloth_points[max_height_index]

		im_pt[0] = im_pt[0] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		im_pt[1] = im_pt[1] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
		im_pt[0], im_pt[1] = self.im_pt_exceeded(im_pt[0], im_pt[1])

		rw_x, rw_y = self.image_pt_to_rw_pt([im_pt[0], im_pt[1]])
		pick_point = [rw_x, rw_y, self.z_grasp]
		pick_orn = self.get_pca_grasp_orn(cloth_points, im_pt)

		if viz_policy:
			c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), 3)
			cv2.imshow('Max height', c_im)
			cv2.waitKey()

		if save:
			c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), 3)
			cv2.imwrite(self.trial_path+'max_height_policy_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		return pick_point, pick_orn, im_pt

	def laundry_net_policy(self, rgb_image, cloth_points, cloth_depths, depth_im,  viz_policy=False, mask_point=[0,0], use_mask=False, save=False, focus_cluster=False):
		# print(rgb_image.shape)
		# print(cloth_points.shape)
		# print(cloth_depths.shape)
		# print(depth_im.shape)
		# pdb.set_trace()
		if focus_cluster:
			cloth_points_cropped = deepcopy(cloth_points)
		else:
			cloth_points_cropped = self.LN.get_cloth_points_eroded(rgb_image, depth_im)

		if len(cloth_points_cropped) > 0:
			opt_grasp_cand, pred_weight = self.LN.get_optimal_grasp_cand(rgb_image, cloth_points_cropped, cloth_points, cloth_depths, depth_im, mask_point=mask_point, use_mask=use_mask)
			print("opt_grasp_cand: ", opt_grasp_cand)
			print("pred_weight: ", pred_weight)
			if len(opt_grasp_cand) > 0:
				im_pt = opt_grasp_cand[0:2]

				im_pt[0] = im_pt[0] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
				im_pt[1] = im_pt[1] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
				im_pt[0], im_pt[1] = self.im_pt_exceeded(im_pt[0], im_pt[1])

				pick_orn = opt_grasp_cand[2]


				try:
					rw_x, rw_y = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]
				except:
					print ('Didnt work')
					IPython.embed()
				depth_at_pick_pt = depth_im[im_pt[0], im_pt[1]]
				pick_z = self.get_pick_depth(depth_at_pick_pt)
				pick_point = [rw_x, rw_y, pick_z]

				if viz_policy:
					c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
					cv2.imshow('Max Weight', c_im)
					cv2.waitKey()

				if save:
					c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
					cv2.imwrite(self.trial_path+'max_weight_policy_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

			else:
				pick_point = []
				pick_orn = 0
				im_pt = []
				pred_weight = 1e3
		else:
			pick_point = []
			pick_orn = 0
			im_pt = []
			pred_weight = 1e3

		return pick_point, pick_orn, im_pt, pred_weight

	def heap_policy(self, cloth_points, cloth_depths, rgb_image, viz_policy=False):

		max_height_index = np.argmax(cloth_depths)
		im_pt_max = cloth_points[max_height_index]

		min_height_index = np.argmin(cloth_depths)
		im_pt_min_init = cloth_points[min_height_index]

		# Update im_pt_min such that it's not at the edge
		# Use a vector from the min point to the max point
		min_to_max_vec = np.array(im_pt_max) - np.array(im_pt_min_init)
		min_max_dist = np.linalg.norm(min_to_max_vec)
		unit_min_to_max_vec = min_to_max_vec/min_max_dist

		eps = 0.1*min_max_dist # 10 percent of dist from min to max

		im_pt_min = np.array(np.array(im_pt_min_init) + eps*unit_min_to_max_vec, dtype=int)

		rw_x_min, rw_y_min = self.image_pt_to_rw_pt([im_pt_min[1], im_pt_min[0]])[0]
		pick_point_min = [rw_x_min, rw_y_min, self.z_grasp]
		pick_orn_min = self.get_pca_grasp_orn(cloth_points, im_pt_min)

		rw_x_max, rw_y_max = self.image_pt_to_rw_pt([im_pt_max[0], im_pt_max[1]])[0]
		pick_point_max = [rw_x_max, rw_y_max, self.z_grasp]
		pick_orn_max = self.get_pca_grasp_orn(cloth_points, im_pt_max)

		place_point_min = deepcopy(pick_point_max)
		place_point_min[-1] = self.translation_z

		if self.real_robot:
			self.pick_and_place(pick_point_min, place_point_min, pick_orn_min, self.translation_z, use_shake_action=False)

		if viz_policy:
			c_im = cv2.circle(rgb_image, [im_pt_min[1], im_pt_min[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			c_im = cv2.circle(c_im, [im_pt_max[1], im_pt_max[0]], self.IM_CIRCLE_P_1, (255,0,0), self.IM_CIRCLE_P_2)
			cv2.imshow('Min-to-Max Policy', c_im)
			cv2.waitKey()

		return pick_point_max, pick_orn_min

	def min_height_policy(self, cloth_points, cloth_depths, rgb_image, viz_policy=False):

		min_height_index = np.argmin(cloth_depths)
		im_pt = cloth_points[min_height_index]

		[rw_x, rw_y] = self.image_pt_to_rw_pt([im_pt[1], im_pt[0]])[0]
		pick_point = [rw_x, rw_y, self.z_grasp]
		pick_orn = self.get_pca_grasp_orn(cloth_points, im_pt)

		if viz_policy:
			c_im = cv2.circle(rgb_image, [im_pt[1], im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
			cv2.imshow('Min height', c_im)
			cv2.waitKey()

		return pick_point, pick_orn, im_pt

	def convert_depth_image_to_z_val(self, depth_image):

		# self.gen_lookup_depth_table(depth_image)

		im_center_pt = [int(depth_image.shape[0]/2.), int(depth_image.shape[1]/2.)]
		center_pt_rw = self.image_pt_to_rw_pt([im_center_pt[1], im_center_pt[0]])[0]

		# Initialize the new top_down z values
		top_down_depth_values = depth_image.copy()

		for x_t in range(depth_image.shape[0]):
			for y_t in range(depth_image.shape[1]):
				target_pt_rw = self.image_pt_to_rw_pt([y_t, x_t])[0]
				planar_dist = np.linalg.norm(center_pt_rw - target_pt_rw)
				radial_val = depth_image[x_t, y_t]
				if radial_val > 0:
					top_down_z_value = np.sqrt(radial_val**2 - planar_dist**2)
				else:
					print ("Invalid depth reading")
					top_down_z_value = 0

				top_down_depth_values[x_t, y_t] = top_down_z_value

		# IPython.embed()
		# Further post processing
		# For all 0 readings use the max value encountered in the z_values
		max_z_val = np.max(top_down_depth_values)
		# normalized_depth_values = (top_down_depth_values/max_z_val)*255

		# Replace zero values
		x_locations, y_locations = np.where(top_down_depth_values == 0 )

		for x_loc in x_locations:
			for y_loc in y_locations:
				top_down_depth_values[x_loc, y_loc] = max_z_val

		depth_min = np.min(top_down_depth_values)
		depth_max = np.max(top_down_depth_values)
		# Normalize the depth values
		b = 1*depth_min/(depth_min - depth_max)
		a = (1 - b)/depth_max

		normalized_top_down_depth_values = top_down_depth_values.copy()
		for x_t in range(depth_image.shape[0]):
			for y_t in range(depth_image.shape[1]):
				prev_val = top_down_depth_values[x_t, y_t]
				normalized_value = a*prev_val + b
				normalized_top_down_depth_values[x_t, y_t] = 255*normalized_value

		# IPython.embed()

		# cv2.imshow("Normals", normalized_top_down_depth_values)
		# cv2.waitKey()

		return top_down_depth_values, normalized_top_down_depth_values

	def max_height_stack(self):

		pick_point = []
		grasp_orn = 0

		return pick_point, grasp_orn

	def pca_cluster(self, grayscale_image):

		contours, centers, cnt_radius, bounding_rectangles = self.cluster_detector(grayscale_image)
		# print("centers", centers)
		pca_points = []
		for i, c in enumerate(contours):
			major_pts, minor_pts = self.get_orientation(c, centers[i], cnt_radius[i])
			major_pts = self.make_pca_internal(major_pts)
			minor_pts = self.make_pca_internal(minor_pts)
			pca_points.append([major_pts, minor_pts])

		return pca_points, contours, centers, cnt_radius, bounding_rectangles

	def make_pca_internal(self, std_pts):

		vec = np.array(std_pts[1]) - np.array(std_pts[0])
		u_vec = vec/np.linalg.norm(vec)

		self.INTERNAL_DIST_PCA = 30 #Pixels

		if np.linalg.norm(vec) > self.INTERNAL_DIST_PCA:
			new_pt_1 = std_pts[0] + self.INTERNAL_DIST_PCA*u_vec
			new_pt_2 = std_pts[1] - self.INTERNAL_DIST_PCA*u_vec
		else:
			new_pt_1 = std_pts[0]
			new_pt_2 = std_pts[1]
		return [new_pt_1, new_pt_2]
	# def get_orientation(self, pts, cntr, cnt_radius):
	# 	sz = len(pts)
	# 	data_pts = np.empty((sz, 2), dtype=np.float64)
	# 	# plt.scatter(data_pts[:,0], data_pts[:,1])
	# 	# plt.show()
	# 	for i in range(data_pts.shape[0]):
	# 		data_pts[i, 0] = pts[i, 0, 0]
	# 		data_pts[i, 1] = pts[i, 0, 1]

	# 	pca = PCA(n_components=2)
	# 	pca.fit(data_pts)
	# 	comp = pca.components_
	# 	minor_axis = comp[1]
	# 	major_axis = comp[0]

	# 	line_dist = 2*cnt_radius
	# 	east_line_second_pt = np.array(cntr) + line_dist*np.array(major_axis)
	# 	west_line_second_pt = np.array(cntr) - line_dist*np.array(major_axis)
	# 	north_line_second_pt = np.array(cntr) + line_dist*np.array(minor_axis)
	# 	south_line_second_pt = np.array(cntr) - line_dist*np.array(minor_axis)

	# 	minor_line = LineString([tuple(south_line_second_pt.tolist()), tuple(north_line_second_pt.tolist())])
	# 	major_line = LineString([tuple(east_line_second_pt.tolist()), tuple(west_line_second_pt.tolist())])

	# 	contour_polygon = LineString(data_pts)
	# 	print("contour_polygon: ", contour_polygon)
	# 	print("major_line: ", major_line)
	# 	major_int_points = contour_polygon.intersection(major_line)
	# 	minor_int_points = contour_polygon.intersection(minor_line)
	# 	pdb.set_trace()
	# 	print("minor_int_points: ", minor_int_points.array_interface_base)
	# 	minor_array = np.array(minor_int_points.array_interface()['data'])
	# 	minor_pts = [minor_array[0:2], minor_array[-2:]]

	# 	major_array = np.array(major_int_points.array_interface()['data'])
	# 	major_pts = [major_array[0:2], major_array[-2:]]

	# 	return major_pts, minor_pts
	import numpy as np
	from shapely.geometry import LineString, MultiPoint, Point
	from sklearn.decomposition import PCA

	import numpy as np
	from sklearn.decomposition import PCA

	def get_orientation(self, pts, cntr, cnt_radius):
		sz = len(pts)
		data_pts = np.empty((sz, 2), dtype=np.float64)
		
		for i in range(data_pts.shape[0]):
			data_pts[i, 0] = pts[i, 0, 0]
			data_pts[i, 1] = pts[i, 0, 1]

		pca = PCA(n_components=2)
		pca.fit(data_pts)
		comp = pca.components_
		minor_axis = comp[1]
		major_axis = comp[0]

		line_dist = 2 * cnt_radius
		east_line_second_pt = cntr + line_dist * major_axis
		west_line_second_pt = cntr - line_dist * major_axis
		north_line_second_pt = cntr + line_dist * minor_axis
		south_line_second_pt = cntr - line_dist * minor_axis

		# Calculate intersection points manually
		major_int_points = []
		minor_int_points = []
		for pt in data_pts:
			if np.dot(pt - cntr, major_axis) >= 0:
				major_int_points.append(list(pt))
			else:
				minor_int_points.append(list(pt))

		return major_int_points, minor_int_points



	def check_common_pts(self, axis_points, cnt_pts, cntr):

		CNT_THRESH = 2
		common_pts = []
		common_dists = []
		for ax_pt in axis_points:
			for cnt_pt in cnt_pts:
				if np.linalg.norm(np.array(ax_pt)-np.array(cnt_pt)) <= CNT_THRESH:
					common_pts.append(ax_pt)
					common_dists.append(np.linalg.norm(np.array(ax_pt) - np.array(cntr)))

		return common_pts, common_dists

	def find_min_max(self, dist_pt):
		new_dict = {}
		keys = dist_pt.keys()
		ld = max(keys)
		new_dict[ld] = dist_pt[ld]
		sd = min(keys)
		new_dict[sd] = dist_pt[sd]
		return new_dict

	def find_min_max_2(self, major_int_pts, major_dists):

		max_pt_ind = np.argmax(major_dists)
		first_pt = major_int_pts[max_pt_ind]

		new_major_int_pts = deepcopy(major_int_pts)
		new_major_int_pts.pop(max_pt_ind)
		max_pt_ind = np.argmax(major_dists)
		second_pt = new_major_int_pts[max_pt_ind]

		return [first_pt, second_pt]

	def twoPoints(self, dist_pt):
		points = []
		for k, coord in dist_pt.items():
			x = int(coord[0])
			y = int(coord[1])
			points.append([x, y])
		return points[0], points[1]

	def maxPoints(self, dist_pt):
		keys = dist_pt.keys()
		# print(keys)
		ld = max(keys)
		coord = dist_pt[ld]
		# print(type(coord[0]))
		x = int(coord[0])
		y = int(coord[1])
		return [x, y]

	def drawAxis(self, img, p_, q_, colour, scale, pts, direction):
		p = list(p_)
		q = list(q_)

		angle = m.atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
		hypotenuse = m.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
		# Here we lengthen the arrow by a factor of scale
		q[0] = p[0] - scale * hypotenuse * m.cos(angle)
		q[1] = p[1] - scale * hypotenuse * m.sin(angle)
		cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
		l1 = np.linspace([int(p[0]), int(p[1])], [int(q[0]), int(q[1])])

		# only look for points on the line that fall in our image domain
		if direction == 1:
			left = min([i[0] for i in l1 if (0.0 <= i[0] <= 925 and 0 <= i[1] <= 690)])
			point = [i for i in l1 if i[0] == left]
			l1 = np.linspace((int(point[0][0]), int(point[0][1])), (int(p[0]), int(p[1])))
		elif direction == -1:
			right = max([i[0] for i in l1 if (0 <= i[0] <= 925 and 0 <= i[1] <= 690)])
			point = [i for i in l1 if i[0] == right]
			l1 = np.linspace((int(point[0][0]), int(point[0][1])), (int(p[0]), int(p[1])))
		elif direction == 2:
			up = min([i[1] for i in l1 if (0 <= i[1] <= 690.0 and 0 <= i[0] <= 925)])
			point = [i for i in l1 if i[1] == up]
			l1 = np.linspace((int(point[0][0]), int(point[0][1])), (int(p[0]), int(p[1])))
		else:
			down = max([i[1] for i in l1 if (0 <= i[1] <= 690 and 0 <= i[0] <= 925)])
			point = [i for i in l1 if i[1] == down]
			l1 = np.linspace((int(point[0][0]), int(point[0][1])), (int(p[0]), int(p[1])))

		dist_pt = {}
		for i in range(len(pts)):
			for j in range(len(pts[i])):
				point = pts[i][j]
				for pt in l1:
					# create a range of points that are "valid"
					if int(pt[0]) in range(int(point[0]) - 5, int(point[0]) + 5):
						if int(pt[1]) in range(int(point[1]) - 5, int(point[1]) + 5):
							dist = distance.euclidean(p, point)
							dist_pt[int(dist)] = [point[0], point[1]]
		# create the arrow hooks
		p[0] = q[0] + 9 * m.cos(angle + m.pi / 4)
		p[1] = q[1] + 9 * m.sin(angle + m.pi / 4)
		p[0] = q[0] + 9 * m.cos(angle - m.pi / 4)
		p[1] = q[1] + 9 * m.sin(angle - m.pi / 4)
		return dist_pt

	def nonWhite(self, mask_points, bw_image):

		sub_sampled_mask_points = random.sample(mask_points.tolist(), 500)

		num_mask_pixels = len(sub_sampled_mask_points) #len(np.column_stack(np.nonzero(mask)))
		# print(num_mask_pixels)
		num_white_pixels = 0

		mn = [bw_image[i[0], i[1]]>=200 for i in sub_sampled_mask_points]
		cluster_cloth_points = deepcopy(sub_sampled_mask_points)
		for i in mn:
			if i == True:
				num_white_pixels+=1
				cluster_cloth_points.remove(cluster_cloth_points[i])
		coverage = num_white_pixels/num_mask_pixels
		return coverage, cluster_cloth_points

	def getColorDiversity(self, masked_image, contours_bounding_rectangles):
		max_diversity_scores = []
		max_diversity_points = []
		gripper_width = 150 # TODO figure out the actual gripper width in pixels
		for bounded_contour in contours_bounding_rectangles:
			max_diversity_score = 0
			max_diversity_point = []
			x, y, w, h = bounded_contour
			for b in range(y, h + y - gripper_width, int(gripper_width)):
				for a in range(x, w + x - gripper_width, int(gripper_width)):
					cropped = masked_image[b: b + gripper_width, a: a + gripper_width]
					print ('cropped_shape', cropped.shape)
					cv2.imshow('masked_image', masked_image)
					# cv2.imshow('title', cropped)
					cv2.waitKey()
					colors_x = extcolors.extract_from_image(Image.fromarray(cropped), tolerance = 40, limit = 12)
					dp = (len(colors_x[0]))
					print ('dp', dp)
					if (dp > max_diversity_score):
						max_diversity_score = dp
						max_diversity_point = [a + int(gripper_width/2), b + int(gripper_width/2)]
			max_diversity_scores.append(max_diversity_score)
			max_diversity_points.append(max_diversity_point)
		return max_diversity_points, max_diversity_scores

	def getMaskedImage(self, contours,contour_centers, rgb_image):

		# IPython.embed()
		mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype="uint8")
		final_mask = mask
		for i in range(len(contours)):
			actual_mask = self.getMaskPoints(contours[i], contour_centers[i], rgb_image)[1]

			final_mask = final_mask + actual_mask
		im = cv2.bitwise_or(rgb_image.copy(), rgb_image.copy(),mask=final_mask)

		# cv2.imshow('im', im)
		# cv2.waitKey()
		return im

	def getMaskPoints(self, contour, contour_center, image):
		# IPython.embed()
		im_shape = image.shape
		mask = np.zeros((im_shape[0], im_shape[1]), dtype="uint8")
		c = cv2.drawContours(mask.copy(), contours=np.array(contour, dtype=int), contourIdx=-1, color=255, thickness=-1)
		a, mask, d, e= cv2.floodFill(c, None, [contour_center[0]-0, contour_center[1] - 20], [255,255,255])
		# plt.imshow(mask_points)
		mask_points = np.column_stack(np.nonzero(mask))

		# plt.show()
		return mask_points, mask

	def getCoverage(self, rgb_image, contours, contour_centers, bw_image):

		start_time = timeit.default_timer()
		masks = [self.getMaskPoints(contours[i], contour_centers[i], rgb_image)[0] for i in range(len(contours))]
		total_mask_time = timeit.default_timer() - start_time
		print ('total_mask_time', total_mask_time)

		# # print(masks)
		# coverage = [self.nonWhite(masks[i], bw_image)[0] for i in range(len(masks))]

		all_coverage = []
		all_cluster_cloth_pts = []
		for i in range(len(masks)):
			start_time = timeit.default_timer()
			coverage, cluster_cloth_pts = self.nonWhite(masks[i], bw_image)
			total_nw_time = timeit.default_timer() - start_time
			print ('NW time', total_nw_time)
			print ('')

			all_coverage.append(coverage)
			all_cluster_cloth_pts.append(cluster_cloth_pts)

		# max_coverage = max(all_coverage)
		# index_max_coverage = all_coverage.index(max_coverage)
		# max_mask = masks[index_max_coverage]

		return all_coverage, all_cluster_cloth_pts

	def recordWeight(self, program_start_time):
		port = list(list_ports.comports())
		# for p in port:
		# 	print(p.device)

		weight_read = False
		weight = 0 
		for _ in range(20):
			try:
				ser = serial.Serial('/dev/ttyUSB0', baudrate=9600)  # Replace 'COM1' with your port name
			# 	# Now you can use the 'ser' object for communication
			# except serial.SerialException as e:
			# 	print(f"Error opening serial port: {e}")
			# try:
				# ser = serial.Serial()
				# print(ser)
				# ser.port = "/dev/ttyUSB0"
				# print(ser.port)
				# ser.baudrate = 9600
				# print(ser.baudrate)
				# print("ser.open()",ser.open())
				try:
					ser.open()
				except:
					pass
				packet = ser.readline()
				weight = packet.decode('utf-8').rstrip('\n')
				weight_read = True
				print('Weight: ' + str(weight))
				# print ('counting the weight ////')
			except Exception as e:
				print ('Weight not read: ', e)

			if weight_read:
				break
			else:
				time.sleep(0.5)
		# with open(f'weight_data/weight_data_{program_start_time}.txt', 'a') as f:
		# 	packet = ser.readline()
		# 	current_time = datetime.datetime.now()
		# 	data = (str(current_time) + " " + packet.decode('utf-8').rstrip('\n'))
		# 	print(data)
		# 	f.write(data)

		to_keep = '1234567890.'
		weight_fl = ''
		for i in range(len(weight)):
			if weight[i] in to_keep:
				weight_fl += weight[i]

		weight_fl += '0'

		return float(weight_fl)


	def laundry_net_stack_policy_old(self, rgb_image, cloth_points, cloth_depths, depth_image, bw_image, viz_policy=False, save=False):
		# Get max weight network to predict a pick point.
		max_weight_pick_point, max_weight_pick_orn, max_weight_im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, save=False)
		second_pred_weight = 0
		second_im_pick_pt = [0,0]
		if len(max_weight_im_pick_pt) > 0:
			if save:
				c_im = cv2.circle(rgb_image, [max_weight_im_pick_pt[1], max_weight_im_pick_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
				cv2.imwrite(self.trial_path+'laundry_max_weight_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		if pred_weight < self.WEIGHT_POLICY_THRESH:
			# Stack activated
			print ('Predicted weight is ', pred_weight)
			print ('Predicted weight did not meet the threshold -----')
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
											bw_image, depth_image, pick_method='centroid')
			# Find new pick point
			# New pick point is max weight points when we remove all cloth points around the current max_weight
			if len(clusters) > 0:
				# Avoid grasping at noise
				second_weight_pick_point, second_weight_pick_orn, second_im_pick_pt, second_pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, mask_point=max_weight_im_pick_pt, use_mask=True)
				# Pick at new pick point and place at max_weight pick point.
				second_weight_place_point = max_weight_pick_point

				if len(second_weight_pick_point) > 0:
					self.pick_and_place(second_weight_pick_point, second_weight_place_point, second_weight_pick_orn,
										self.translation_z, use_shake_action=False)

					if save:
						c_im2 = cv2.circle(rgb_image, [second_im_pick_pt[1], second_im_pick_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
						cv2.imwrite(self.trial_path+'laundry_second_weight_rgb.jpg', c_im2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


		# Pick at max_weight pick point and then place in bin. (in any case)
		max_weight_place_point = self.gen_place_point()
		self.pick_and_place(max_weight_pick_point, max_weight_place_point, max_weight_pick_orn,
							self.translation_z, use_shake_action=False, save=save)

		return pred_weight, second_pred_weight, max_weight_im_pick_pt, second_im_pick_pt

	def laundry_net_stack_policy(self, viz_policy=False, save=False, use_fold=False):

		# Initialization
		num_stacks = 0
		while num_stacks < self.MAX_NUM_STACKS:
			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			max_weight_pick_point, max_weight_pick_orn, max_weight_im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, save=False)
			print ('Predicted weight is', pred_weight)

			if pred_weight >= self.WEIGHT_POLICY_THRESH:
				print ('Predicted weight passes the check')
				max_weight_place_point = self.gen_place_point()
				self.pick_and_place(max_weight_pick_point, max_weight_place_point, max_weight_pick_orn,
							self.translation_z, use_shake_action=False, save=save)
				break
			else:
				next_best_pick_pt, next_best_pick_orn, next_best_im_pt, next_best_pred_weight = self.get_next_best_laundry_point(cloth_points, cloth_depths, rgb_image, bw_image, depth_image, viz_policy=viz_policy, mask_point=max_weight_im_pick_pt)

				if len(next_best_pick_pt) == 0:
					# No more stacks possible
					# Execute at max weight
					max_weight_place_point = self.gen_place_point()
					if use_fold:
						self.pick_and_place_controlled(max_weight_pick_point, max_weight_place_point, max_weight_pick_orn,
								self.translation_z, use_shake_action=False, save=save)
					else:
						self.pick_and_place(max_weight_pick_point, max_weight_place_point, max_weight_pick_orn,
								self.translation_z, use_shake_action=False, save=save)
					break
				else:
					# Stack
					next_best_place_pt = deepcopy(max_weight_pick_point)
					self.pick_and_place(next_best_pick_pt, next_best_place_pt, next_best_pick_orn,
										self.translation_z, use_shake_action=False)

					if save:
						c_im2 = cv2.circle(rgb_image, [next_best_im_pt[1], next_best_im_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
						cv2.imwrite(self.trial_path+'laundry_stack_{}_rgb.jpg'.format(num_stacks), c_im2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

			# Move the robot to the check point (to take another image)
			check_point_orn = 0
			self.move_ur5py(self.CHECKPOINT, check_point_orn, self.joint_speed, self.joint_acc)

			num_stacks += 1

	def get_next_best_laundry_point(self, cloth_points, cloth_depths, rgb_image, bw_image, depth_image, viz_policy=False, mask_point=[0,0]):

		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
										bw_image, depth_image, pick_method='centroid')
		# Find new pick point
		# New pick point is max weight points when we remove all cloth points around the current max_weight
		if len(clusters) > 0:
			# Avoid grasping at noise
			pick_point, pick_orn, im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, mask_point=mask_point, use_mask=True)
			# Pick at new pick point and place at max_weight pick point.
		else:
			pick_point = []
			pick_orn = 0
			im_pick_pt = []
			pred_weight = 1e3

		return pick_point, pick_orn, im_pick_pt, pred_weight

	def laundry_net_stack_fold_policy_old(self, rgb_image, cloth_points, cloth_depths, depth_image, bw_image, viz_policy=False, save=False):
		# Get max weight network to predict a pick point.
		max_weight_pick_point, max_weight_pick_orn, max_weight_im_pick_pt, pred_weight = self.laundry_net_policy(rgb_image, depth_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, save=False)
		second_pred_weight = 0
		second_im_pick_pt = [0,0]


		if len(max_weight_im_pick_pt) > 0:

			if save:
				c_im = cv2.circle(rgb_image, [max_weight_im_pick_pt[1], max_weight_im_pick_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
				cv2.imwrite(self.trial_path+'laundry_max_weight_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		if pred_weight < self.WEIGHT_POLICY_THRESH:
			# Stack activated
			print ('Predicted weight is ', pred_weight)
			print ('Predicted weight did not meet the threshold -----')
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
											bw_image, depth_image, pick_method='centroid')
			# Find new pick point
			# New pick point is max weight points when we remove all cloth points around the current max_weight
			if len(clusters) > 0:
				# Avoid grasping at noise
				second_weight_pick_point, second_weight_pick_orn, second_im_pick_pt, second_pred_weight = self.laundry_net_policy(rgb_image, depth_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy, mask_point=max_weight_im_pick_pt, use_mask=True)
				# Pick at new pick point and place at max_weight pick point.
				second_weight_place_point = max_weight_pick_point


				if len(second_weight_pick_point) > 0:
					depth_at_pick_pt = depth_image[second_im_pick_pt[0], second_im_pick_pt[1]]
					pick_z = self.get_pick_depth(depth_at_pick_pt)
					second_weight_pick_point[-1] = pick_z

					self.pick_and_place(second_weight_pick_point, second_weight_place_point, second_weight_pick_orn,
										self.translation_z, use_shake_action=False, depth_at_pick_pt=depth_at_pick_pt)

					if save:
						c_im2 = cv2.circle(rgb_image, [second_im_pick_pt[1], second_im_pick_pt[0]], self.IM_CIRCLE_P_1, (255, 0,0), self.IM_CIRCLE_P_2)
						cv2.imwrite(self.trial_path+'laundry_second_weight_rgb.jpg', c_im2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		# Pick at max_weight pick point and then place in bin. (in any case)

		max_weight_place_point = self.gen_place_point()

		depth_at_pick_pt = 1/depth_image[max_weight_im_pick_pt[0], max_weight_im_pick_pt[1]]
		print ('Depth at pick point', depth_at_pick_pt)
		pick_z = self.get_pick_depth(depth_at_pick_pt)
		max_weight_pick_point[-1] = pick_z
		self.pick_and_place_controlled(max_weight_pick_point, max_weight_place_point, max_weight_pick_orn,
							self.translation_z, use_shake_action=False, save=save, depth_at_pick_pt=depth_at_pick_pt)

		return pred_weight, second_pred_weight, max_weight_im_pick_pt, second_im_pick_pt

	def gen_line_pts(self, im_pt, pick_orn):
		unit_vec = np.array([m.cos(pick_orn), m.sin(pick_orn)])

		st_pt = np.array(im_pt) + unit_vec*self.IM_LINE_WIDTH/2.
		e_pt = np.array(im_pt) - unit_vec*self.IM_LINE_WIDTH/2.

		return np.array([int(st_pt[0]), int(st_pt[1])]), np.array([int(e_pt[0]), int(e_pt[1])])

	def create_grasp_marks(self, p1, rgb_image, index, color, save=True):
		self.IM_LINE_WIDTH = 40
		# if save:
		# 	cv2.imwrite('../Downloads/original_rgb.jpg', rgb_image,  [int(cv2.IMWRITE_JPEG_QUALITY), 90])
		if save:
			c_im = cv2.circle((rgb_image.copy()), [int(p1[0]), int(p1[1])], self.IM_CIRCLE_P_1, color, self.IM_CIRCLE_P_2)
			print(c_im)
			s_pt, e_pt = self.gen_line_pts([int(p1[0]), int(p1[1])], int(p1[2]))
			print(s_pt)
			c_im = cv2.line(c_im, s_pt, e_pt, color, self.IM_CIRCLE_P_2)
			print(c_im)
			cv2.imwrite(f"../Downloads/ws3/drawn_image_{index}.jpg", c_im)
			return c_im

	def inspect_clusters(self, clusters, rgb_image, pick_method='weight', save=False):

		all_fold_lists = []
		all_fold_actions = []

		self.IM_LINE_WIDTH = 40

		if save:
			cv2.imwrite(self.trial_path+'original_rgb.jpg', rgb_image,  [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		# IPython.embed()

		cluster_ind = 0
		for cluster in clusters:
			fold_list = []
			# Get the major pca points
			pca_points = cluster['pca_points']
			major_points = [np.array(pca_points[0], dtype='int'), np.array(pca_points[1], dtype='int')]

			pca_orn_1 = self.get_pca_grasp_orn(cluster['cluster_pts'], major_points[0])
			pca_orn_2 = self.get_pca_grasp_orn(cluster['cluster_pts'], major_points[1])

			if pick_method == 'weight':
				im_pick_pt = cluster['ln_im_pick_pt']
				pick_orn = cluster['ln_pick_orn']
			elif pick_method == 'height':
				im_pick_pt = cluster['max_height_point']
				pick_orn = cluster['max_height_pick_orn']
			elif pick_method == 'volume':
				im_pick_pt = cluster['max_volume_point']
				pick_orn = cluster['max_volume_pick_orn']
			elif pick_method == 'centroid':
				im_pick_pt = cluster['centroid']
				pick_orn = cluster['centroid_pick_orn']
			elif pick_method == 'random':
				im_pick_pt = cluster['cluster_pts'][np.random.randint(len(cluster['cluster_pts']))]
				pick_orn = random.uniform(0, m.pi)
			else:
				print ('Invalid pick method')

			gamma_1 = np.linalg.norm(np.array(im_pick_pt) - np.array(major_points[0]))
			gamma_2 = np.linalg.norm(np.array(im_pick_pt) - np.array(major_points[1]))

			# print ('Gamma 1', gamma_1)
			# print ('Gamma 2', gamma_2)
			im_pick_pt[0] = im_pick_pt[0] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
			im_pick_pt[1] = im_pick_pt[1] + int(np.random.uniform(low=-self.PICK_NOISE, high=self.PICK_NOISE))
			im_pick_pt[0], im_pick_pt[1] = self.im_pt_exceeded(im_pick_pt[0], im_pick_pt[1])

			if save:
				c_im = cv2.circle(rgb_image, [major_points[0][1], major_points[0][0]], self.IM_CIRCLE_P_1, (0, 255, 0), self.IM_CIRCLE_P_2)
				s_pt, e_pt = self.gen_line_pts([major_points[0][1], major_points[0][0]], pca_orn_1)
				c_im = cv2.line(c_im, s_pt, e_pt, (0, 255, 0), self.IM_CIRCLE_P_2)

				c_im_2 = cv2.circle(c_im, [major_points[1][1], major_points[1][0]], self.IM_CIRCLE_P_1, (0, 255, 0), self.IM_CIRCLE_P_2)
				s_pt, e_pt = self.gen_line_pts([major_points[1][1], major_points[1][0]], pca_orn_2)
				c_im_2 = cv2.line(c_im_2, s_pt, e_pt, (0, 255, 0), self.IM_CIRCLE_P_2)

				c_im_3 = cv2.circle(c_im_2, [im_pick_pt[1], im_pick_pt[0]], self.IM_CIRCLE_P_1, (255, 255, 255), self.IM_CIRCLE_P_2)
				s_pt, e_pt = self.gen_line_pts([im_pick_pt[1], im_pick_pt[0]], pick_orn)
				c_im_3 = cv2.line(c_im_3, s_pt, e_pt, (255, 255, 255), self.IM_CIRCLE_P_2)
				if not os.path.isdir(self.trial_path):
					os.mkdir(self.trial_path)
				np.save(self.trial_path+'circ_1_{}'.format(cluster_ind), [major_points[0][1], major_points[0][0], pca_orn_1])
				np.save(self.trial_path+'circ_2_{}'.format(cluster_ind), [major_points[1][1], major_points[1][0], pca_orn_2])
				np.save(self.trial_path+'circ_center_{}'.format(cluster_ind), [im_pick_pt[1], im_pick_pt[0], pick_orn])

				# cv2.imshow('RGB Image', c_im_3)
				# cv2.waitKey()

			fold_action_1 = [major_points[0], im_pick_pt, pick_orn]
			fold_action_2 = [major_points[1], im_pick_pt, pick_orn]

			all_fold_actions.append([fold_action_1, fold_action_2])

			if gamma_1 >= self.FOLD_WIDTH:
				fold_list.append(True)
			else:
				fold_list.append(False)

			if gamma_2 >= self.FOLD_WIDTH:
				fold_list.append(True)
			else:
				fold_list.append(False)

			all_fold_lists.append(fold_list)

			cluster_ind += 1

		if save:
			cv2.imwrite(self.trial_path+'clusters.jpg', c_im_3, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		return all_fold_lists, all_fold_actions

	def fold_cluster(self, cloth_points, fold_action):

		im_pick_pt = fold_action[0]
		im_place_pt = fold_action[1]

		pca_orn = self.get_pca_grasp_orn(cloth_points, im_pick_pt)

		[rw_x_pick, rw_y_pick] = self.image_pt_to_rw_pt([im_pick_pt[1], im_pick_pt[0]])[0]

		[rw_x_place, rw_y_place] = self.image_pt_to_rw_pt([im_place_pt[1], im_place_pt[0]])[0]

		pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
		place_pt = [rw_x_place, rw_y_place, self.z_grasp]

		# Pick at PCA point and place at Centroid
		self.pick_and_place(pick_pt, place_pt, pca_orn, self.translation_z, use_shake_action=False)

	def get_optimal_cluster(self, clusters):


		all_weights = []
		all_volumes = []
		all_heights = []

		if len(clusters) == 0:
			return all_weights, all_volumes, all_heights
		for cluster in clusters:
			cluster_weight = cluster['pred_weight']
			cluster_volume = cluster['max_volume']
			cluster_height = cluster['max_height']
			all_weights.append(cluster_weight)
			all_volumes.append(cluster_volume)
			all_heights.append(cluster_height)


		# ================= Weight ====================
		opt_cluster_ind = np.argmax(all_weights)
		opt_cluster = clusters[opt_cluster_ind]
		opt_weight = opt_cluster['pred_weight']
		opt_im_pick_pt = opt_cluster['ln_im_pick_pt']
		opt_pick_orn = opt_cluster['ln_pick_orn']

		other_clusters = []
		other_cluster_weights = []
		for cluster_ind in range(len(clusters)):
			if cluster_ind != opt_cluster_ind:
				other_clusters.append(clusters[cluster_ind])
				other_cluster_weights.append(clusters[cluster_ind]['pred_weight'])

		weight_params = [opt_weight, opt_im_pick_pt, opt_pick_orn, other_clusters, other_cluster_weights]
		# print("weight_params: ", weight_params)
		# print("opt_weight: ", opt_weight)
		# print("opt_im_pick_pt: ", opt_im_pick_pt)
		# print("opt_pick_orn: ", opt_pick_orn)
		# print("opther clusters: ", other_clusters)
		# print("other_cluster_weights: ", other_cluster_weights)
		# ================= Volume ====================
		opt_cluster_ind = np.argmax(all_volumes)
		opt_cluster = clusters[opt_cluster_ind]
		opt_volume = opt_cluster['max_volume']
		opt_im_pick_pt_v = opt_cluster['max_volume_point']
		opt_pick_orn_v = opt_cluster['max_volume_pick_orn']

		other_clusters_v = []
		other_cluster_volumes = []
		for cluster_ind in range(len(clusters)):
			if cluster_ind != opt_cluster_ind:
				other_clusters_v.append(clusters[cluster_ind])
				other_cluster_volumes.append(clusters[cluster_ind]['max_volume'])

		volume_params = [opt_volume, opt_im_pick_pt_v, opt_pick_orn_v, other_clusters_v, other_cluster_volumes]

		# ================= Height ====================
		opt_cluster_ind = np.argmax(all_heights)
		opt_cluster = clusters[opt_cluster_ind]
		opt_height = opt_cluster['max_height']
		opt_im_pick_pt_h = opt_cluster['max_height_point']
		opt_pick_orn_h = opt_cluster['max_height_pick_orn']

		other_clusters_h = []
		other_cluster_heights = []
		for cluster_ind in range(len(clusters)):
			if cluster_ind != opt_cluster_ind:
				other_clusters_h.append(clusters[cluster_ind])
				other_cluster_heights.append(clusters[cluster_ind]['max_height'])

		height_params = [opt_height, opt_im_pick_pt_h, opt_pick_orn_h, other_clusters_h, other_cluster_heights]

		return weight_params, volume_params, height_params

	def execute_stack_action(self, stack_action):
		im_pick_pt = stack_action[0]
		im_place_pt = stack_action[1]
		pick_orn = stack_action[2]

		[rw_x_pick, rw_y_pick] = self.image_pt_to_rw_pt([im_pick_pt[1], im_pick_pt[0]])[0]
		pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]

		[rw_x_place, rw_y_place] = self.image_pt_to_rw_pt([im_place_pt[1], im_place_pt[0]])[0]
		place_pt = [rw_x_place, rw_y_place, self.z_grasp]

		self.pick_and_place(pick_pt, place_pt, pick_orn, self.translation_z, use_shake_action=False)

	def gen_stack_actions(self, clusters, pick_method='weight'):

		if len(clusters) == 0:
			return [], 0, [], [], 0, []

		stack_actions = []

		# Find the optimal cluster
		# if pick_method != 'random':
		weight_params, height_params, volume_params = self.get_optimal_cluster(clusters)

		if pick_method == 'random':
			i = np.random.randint(len(clusters))
			chosen_cluster = clusters[i]
			cluster_pts = chosen_cluster['cluster_pts']
			r_clusters = deepcopy(clusters)
			r_clusters.pop(i)
			i = np.random.randint(len(cluster_pts))
			opt_param = 0 # TODO
			total_stack_param = 0 # TODO
			opt_im_pick_pt = cluster_pts[i]
			opt_pick_orn = random.uniform(0, m.pi)
			other_clusters = deepcopy(r_clusters)
			other_cluster_params = [0 for i in range(len(other_clusters))]
		elif pick_method == 'weight':
			opt_param = weight_params[0]
			total_stack_param = opt_param
			opt_im_pick_pt = weight_params[1]
			opt_pick_orn = weight_params[2]
			other_clusters = weight_params[3]
			other_cluster_params = weight_params[4]
		elif pick_method == 'height':
			opt_param = height_params[0]
			total_stack_param = opt_param
			opt_im_pick_pt = height_params[1]
			opt_pick_orn = height_params[2]
			other_clusters = height_params[3]
			other_cluster_params = height_params[4]
		elif pick_method == 'volume':
			opt_param = volume_params[0]
			total_stack_param = opt_param
			opt_im_pick_pt = volume_params[1]
			opt_pick_orn = volume_params[2]
			other_clusters = volume_params[3]
			other_cluster_params = volume_params[4]
		else:
			print ('Invalid pick method')

		if len(clusters) > 1:

			cluster_ind = 0
			for cluster_to_stack in other_clusters:
				if pick_method == 'weight':
					im_pick_pt = cluster_to_stack['ln_im_pick_pt']
					pick_orn = cluster_to_stack['ln_pick_orn']
				elif pick_method == 'volume':
					im_pick_pt = cluster_to_stack['max_volume_point']
					pick_orn = cluster_to_stack['max_volume_pick_orn']
				elif pick_method == 'height':
					im_pick_pt = cluster_to_stack['max_height_point']
					pick_orn = cluster_to_stack['max_height_pick_orn']
				elif pick_method == 'random':
					i = np.random.randint(len(cluster_to_stack['cluster_pts']))
					im_pick_pt = cluster_to_stack['cluster_pts'][i]
					pick_orn = random.uniform(0, m.pi)
				else:
					print ('Invalid pick method')

				im_place_pt = deepcopy(opt_im_pick_pt)
				stack_action = [im_pick_pt, im_place_pt, pick_orn]

				total_stack_param += other_cluster_params[cluster_ind]

				if True: #total_stack_param < self.PICK_PARAMETER_THRESH:
					stack_actions.append(stack_action)
				else:
					break

				cluster_ind += 1

		return stack_actions, opt_param, opt_im_pick_pt, opt_pick_orn, other_cluster_params

	def basket_trip(self, fold_action):
		opt_im_pick_pt = fold_action[0]
		opt_pick_orn = fold_action[1]
		rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
		print(rw_pick)

		rw_x_pick = rw_pick[0][0]
		rw_y_pick = rw_pick[0][1]

		pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
		print(pick_pt)
		place_pt = self.gen_place_point()
		self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)

		return None


	# Load SAM model weights
	# how to use      : masks = mask_generator.generate(image)
	# remember to use : image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	def loadSAM(self):
		device_type = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		print('Cuda is available: ' + str(torch.cuda.is_available()))
		print(torch.cuda.device_count())
		#modelsize = "vit_h"

		modelcheckpoint = self.sam_path + self.sam_model_name

		sam = sam_model_registry[self.model_size](checkpoint=modelcheckpoint)
		sam.to(device=device_type)

		mask_generator = SamAutomaticMaskGenerator(sam)
		return mask_generator


	# Helper function, sorts the masks by some metric (either stability_score or area usually)
	def sortMasks(self, masks):
		if len(masks) == 0:
			return
		sorted_masks = sorted(masks, key=(lambda x: x[self.metr]), reverse=True)
		return sorted_masks


	# Helper function, given a segmentation turns it into an official "mask" dictionary; make up your own stability_score and predicted_iou values
	# segment : should be image-sized array of 0's and 1's
	def makeMask(self, segment):
		mask = {}
	
		a = np.nonzero(segment)
		bound_box = [np.min(a[0]), np.min(a[1]), np.max(a[0])-np.min(a[0]), np.max(a[1])-np.min(a[1])]
	
		mask['segmentation'] = segment
		mask['area'] = sum(sum(segment))
		mask['bbox'] = bound_box
		mask['stability_score'] = self.stab_score_default
		mask['predicted_iou'] = self.pred_iou_default
		mask['crop_box'] = bound_box
		mask['point_coords'] = [(bound_box[0] + bound_box[2]/2), (bound_box[1] + bound_box[3]/2)]
	
		return mask


	# Returns an array of the size of the image; each value is the number of segments containing the corresponding pixel
	# basically tries to identify where masks overlap a lot for pruning
	def getSegmentCounts(self, masks):
		img_shape = np.shape(masks[0]['segmentation'])
		seg_count = np.zeros(img_shape)
	
		for mask in masks:
			seg_count += mask['segmentation']

		return seg_count


	# Same as above, but first expands each segment by convolving it with a conv_size x conv_size matrix of ones; 
	# shows how many segments are near the given pixel.
	def getExpandedSegmentCounts(self, masks):
		img_shape = np.shape(masks[0]['segmentation'])
		exp_seg_count = np.zeros(img_shape)
		conv_mat = np.ones([self.conv_size, self.conv_size])
	
		for mask in masks:
			exp_seg_count += (convolve(mask['segmentation'], conv_mat) > 0)

		return exp_seg_count


	# Finds segments which are completely contained within (one or more) other segments ("subset masks")
	# returns vector of length len(masks); 0 means not a subset mask, 1 means the corresponding mask in masks is a subset mask

	def getSubsetMasks(self, masks):
		num_masks = len(masks)
		subset_masks = np.zeros(num_masks)
		seg_count = self.getSegmentCounts(masks)

		for mask_num in range(num_masks):
			seg_count_diff = seg_count - 2*masks[mask_num]['segmentation']
			if np.all(seg_count_diff >= 0):
				subset_masks[mask_num] = 1
			
		return subset_masks


	# Gets a vector of segment areas

	def getMaskAreas(self, masks):
		tot_area = 0
		num_masks = len(masks)
		mask_areas = np.zeros(num_masks)
		
		for mask_num in range(num_masks):
			#print(mask['area'])
			mask_areas[mask_num] = masks[mask_num]['area']
			tot_area += mask_areas[mask_num]
			#print(mask['bbox'])  # bounding box in format x-minvalue, y-minvalue, x-width, y-width
	
		#print(tot_area)
		#print(np.shape(masks[0]['segmentation'])[0]*np.shape(masks[0]['segmentation'])[1])
	
		return mask_areas


	# Finds segments which are contained in the expansions of (one or more) other segments
	# Avoids issue where if A lies within (B union C) it generally contains a pixel on their border which is not in either of them
	# Goes from 0 to conv_limit. Higher the number, the lower the convolution conv_size needed to make it a subset.

	def getExpandedSubsetMasks(self, masks):
		num_masks = len(masks)
		exp_subset_masks = np.zeros(num_masks)


		seg_count = self.getExpandedSegmentCounts(masks) # [NOTE] conv_size was increased by 1 for some reason??

		for mask_num in range(num_masks):
			seg_count_diff = seg_count - 2*masks[mask_num]['segmentation']
			if np.all(seg_count_diff >= 0):
				exp_subset_masks[mask_num] = 1
	
		return exp_subset_masks


	# Removes biggest area unnecessary segment until only necessary segments remain
	# Current standard is to use smaller segments (avoid having one segment = multiple objects)

	def pruneSegments(self, masks):
		masks_nec = self.sortMasks(masks)

		exp_subset_masks = self.getExpandedSubsetMasks(masks_nec)
	
		while exp_subset_masks.any():
			#print(np.max(np.nonzero(exp_subset_masks)))
			if self.max_first:
				del masks_nec[np.max(np.nonzero(exp_subset_masks))]
			else:
				del masks_nec[np.min(np.nonzero(exp_subset_masks))]
			exp_subset_masks = self.getExpandedSubsetMasks(masks_nec)
		
		return masks_nec


	# Gets pixels not contained in any segments. Convolution to expand segments and remove border pixels and leave big regions

	def getHoles(self, masks):
		exp_seg_count = self.getExpandedSegmentCounts(masks)

		return (exp_seg_count == 0)


	# Helper function, takes binary array and labels connected components (2D)

	def labelComponents(self, segments):
		structure = np.ones([3, 3])  
		labeled, ncomponents = label(segments, structure)
		return [labeled, ncomponents]


	# Takes masks and adds hole components to complete them

	def completeMasks(self, masks):
		#masks_completed = []
	
		background = self.getHoles(masks)
	
		[labeled, ncomponents] = self.labelComponents(background)
	
		for i in range(ncomponents):
			component = (labeled == i+1)
			masks.append(self.makeMask(component))

		return



	# Removes masks which are too small to correspond to actual objects in the scene

	def removeSmallMasks(self, masks):
		#return [mask for mask in masks if mask['area'] > thresh]
		big_masks = []

		if len(masks) == 0:
			return big_masks
	
		for mask in masks:
			#print(mask['area'])
			if mask['area'] > self.size_thresh:
				big_masks.append(mask)
			
		return big_masks


	# Takes a set of (raw) masks from SAM
	# Cleans them up (BUT DOES NOT REMOVE BACKGROUND MASKS!)
	# max_first refers to *removing* large ones first -- so max_first=True means keep the small ones

	def cleanMasks(self, masks):

		masks_clean = self.pruneSegments(masks)
		masks_clean = self.removeSmallMasks(masks_clean)
		self.completeMasks(masks_clean)
		masks_clean = self.removeSmallMasks(masks_clean)

		return masks_clean


	# Finds "background points" based on the fact that the surface we're using is white
	# This may need some improvement in order to get it to apply to general surfaces... can we use edge of vision to detect surface colors?
	# 0 = cloth points, 1 = background points

	def getClothMap(self, rgb_image):

		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		(thresh, cloth_map) = cv2.threshold(grayscale_image, self.im_thresh, self.maxval, cv2.THRESH_BINARY)

		cloth_map = cv2.dilate(cloth_map, kernel=np.ones((self.kernel_size, self.kernel_size)))

		# print ('cloth depths', cloth_depths)
		return cloth_map


	# Takes an image and removes holes using two convolutions
	def removeHoles(self, cloth_map):
		conv_mat = np.ones([self.hole_remove_conv, self.hole_remove_conv])

		cloth_map_new = (convolve(1-cloth_map, conv_mat) > 0)
		cloth_map_new = (convolve(1-cloth_map_new, conv_mat) > 0)

		return cloth_map_new


	# Takes masks and removes those corresponding to the background (as given by cloth_map)

	def removeBackgroundMasks(self, masks, cloth_map):
		masks_clean = []

		for mask in masks:
			#mask_area = mask['area']
			mask_seg = mask['segmentation']
			mask_on_cloth_fraction = np.sum(np.multiply(1-cloth_map, mask_seg))/np.sum(mask_seg)
			#print(mask_on_cloth_fraction)
			if mask_on_cloth_fraction > self.cloth_thresh:
				masks_clean.append(mask)

		return masks_clean


	# Takes an image, mask_generator, and parameters
	# Generates masks and cleans them

	def getCleanSegmentation(self, image):
		masks_raw = self.mask_generator.generate(image)
		masks = masks_raw

		if self.show_steps:
			self.showMasksOnImage(masks, image)

		masks_clean = self.cleanMasks(masks)

		if self.show_steps:
			self.showMasksOnImage(masks_clean, image)

		cloth_map = self.getClothMap(image)
		cloth_map = self.removeHoles(cloth_map)

		#if self.show_steps:
		#	plt.imshow(cloth_map)
		#	plt.show()

		masks_clean = self.removeBackgroundMasks(masks_clean, cloth_map)

		if self.show_steps:
			self.showMasksOnImage(masks_clean, image)

		return masks_clean, masks_raw, cloth_map





	def showMasks(self, masks):
		if len(masks) == 0:
			return
		#sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
		ax = plt.gca()
		ax.set_autoscale_on(False)
		for mask in masks:
			#print(type(mask))
			if type(mask) is dict:
				#print(type(mask))
				ma = mask['segmentation']
			else:
				ma = mask
			
			img = np.ones((ma.shape[0],ma.shape[1],3))
			color_mask = np.random.random((1,3)).tolist()[0]
			for i in range(3):
				img[:,:,i] = color_mask[i]
			#transparency = 1
			np.dstack((img, ma*self.transparency))
			#print(np.shape(np.dstack((img, ma*self.transparency))))
			ax.imshow(np.dstack((img, ma*self.transparency)))



	def showGraspEllipses(self, grasps, img_dims):

		if len(grasps) == 0:
			return

		ax = plt.gca()
		ax.set_autoscale_on(False)
		for grasp in grasps:
			grasp_pt = [grasp[1], grasp[0]]
			grasp_orn = grasp[2]

			#rot_mat = np.array([[m.cos(grasp_orn), -m.sin(grasp_orn)], [m.sin(grasp_orn), m.cos(grasp_orn)]])
			grasp_mask = np.zeros(img_dims[0:2])

			k = img_dims[0]
			l = img_dims[1]

			for i in range(k):
				for j in range(l):
					dist_to_pt = self.rotatedNorm((np.array([i,j])-np.array(grasp_pt)), grasp_orn)
					if dist_to_pt < self.dist_baseline_thresh:
						grasp_mask[i,j] = 1

			img = np.ones((grasp_mask.shape[0],grasp_mask.shape[1],3))
			#color_mask = np.random.random((1,3)).tolist()[0]
			
			for i in range(3):
				img[:,:,i] = 1
			#transparency = 1

			#print("Grasp mask shape: " + np.shape(grasp_mask))
			np.dstack((img, grasp_mask*self.transparency))
			#print(np.shape(np.dstack((img, grasp_mask*self.transparency))))
			ax.imshow(np.dstack((img, grasp_mask*self.transparency)))



	def showMasksOnImage(self, masks, image):
		plt.figure(figsize=(12,9))
		plt.imshow(image)
		self.showMasks(masks)
		plt.axis('off')
		plt.show()


	def showMasksAndGraspPtsOnImage(self, masks, image, grasps):
		plt.figure(figsize=(12,9))
		if len(grasps) > 0:
			cand_grasps_to_show = np.array(grasps).T
			plt.scatter(cand_grasps_to_show[0], cand_grasps_to_show[1], marker='o', color="red")
		plt.imshow(image)
		self.showMasks(masks)
		plt.axis('off')
		plt.show()

	def showMasksAndGraspsFullOnImage(self, masks, image, grasps):
		plt.figure(figsize=(12,9))
		if len(grasps) > 0:
			cand_grasps_to_show = np.array(grasps.T)
			plt.scatter(cand_grasps_to_show[0], cand_grasps_to_show[1], marker='o', color="red")
		plt.imshow(image)

		print(grasps)
		
		self.showMasks(masks)
		self.showGraspEllipses(grasps, np.shape(image))

		plt.axis('off')
		plt.show()


	def rotateZAxis(self, pick_orn):
		rzangle = pick_orn

		rot_mat = RigidTransform.rotation_from_axis_angle((0,m.pi,0)) @ RigidTransform.z_axis_rotation(rzangle)
		axisang = RigidTransform(rotation=rot_mat).axis_angle

		return axisang


	def getExpandedSegsNoBack(self, masks, cloth_map):
	
		num_masks = len(masks)
		expanded_segs = []
	
		conv_mat = np.zeros([2*self.distance_thresh+1, 2*self.distance_thresh+1])
		for i in range(2*self.distance_thresh+1):
			for j in range(2*self.distance_thresh+1):
				if np.sqrt(i**2 + j**2) <= self.distance_thresh:
					conv_mat[i+round(self.distance_thresh/2),j+round(self.distance_thresh/2)] = 1

		for mask_num in range(num_masks):
			mask_seg = masks[mask_num]['segmentation']
			expanded_segs.append((convolve(mask_seg, conv_mat) > 0)*(1-cloth_map))
		
		if len(expanded_segs) > 0:
			expanded_segs_tens = np.stack(expanded_segs)
		else:
			expanded_segs_tens = expanded_segs

		return expanded_segs, expanded_segs_tens


	def getMaximalSets(self, expanded_segs_tens):
		maximal_sets = []
		
		img_height = np.shape(expanded_segs_tens)[1]
		img_length = np.shape(expanded_segs_tens)[2]
		
		for i in range(img_height):
			for j in range(img_length):
				candidate_set = expanded_segs_tens[:,i,j]
				to_add = True
				
				#recompute_maximal = True
				
				for max_set in maximal_sets:
					if np.all(max_set >= candidate_set):
						to_add = False
						recompute_maximal = False
						break
				
				#if recompute_maximal:
				#    maximal_sets = [max_set for max_set in maximal_sets if not np.all(max_set <= candidate_set)]
				
				if to_add:
					maximal_sets = [max_set for max_set in maximal_sets if not np.all(max_set <= candidate_set)]
					maximal_sets.append(candidate_set)
					
		
		maximal_sets = np.array(maximal_sets)
		maximal_sets = maximal_sets.astype(int)
		
		return maximal_sets

	def getMaximalSetMasks(self, maximal_sets, expanded_segs_tens, cloth_map):
		num_max_sets = len(maximal_sets)
		maximal_set_masks = []
		represent_midpoints = []
		
		img_height = np.shape(expanded_segs_tens)[1]
		img_length = np.shape(expanded_segs_tens)[2]
		
		for set_num in range(num_max_sets):
			to_add = np.zeros([img_height, img_length])
			num_added = 0
			#power_of_two_tracker = 1
			#to_add_rep_mids_next = [0,0]
			
			for i in range(img_height):
				for j in range(img_length):
					if np.all(expanded_segs_tens[:, i, j] == maximal_sets[set_num]):
						to_add[i,j] = 1
						num_added += 1
						ran_var = np.random.uniform()*num_added
						if ran_var <= 1:
							to_add_rep_mids = [j,i]
						#if num_added % power_of_two_tracker == 0:
							#print(num_added)
							#power_of_two_tracker = power_of_two_tracker*2
							#to_add_rep_mids = to_add_rep_mids_next
							#to_add_rep_mids_next = [j,i]
			
			maximal_set_masks.append(to_add)
			#if cloth_map[to_add_rep_mids[0], to_add_rep_mids[1]] == 0:
			represent_midpoints.append(to_add_rep_mids)
			
		return maximal_set_masks, represent_midpoints













	def getCandidateGraspPts(self, image):
		masks, masks_raw, cloth_map = self.getCleanSegmentation(image)
		expanded_segs, expanded_segs_tens = self.getExpandedSegsNoBack(masks, cloth_map)

		if self.show_steps:
			self.showMasksOnImage(expanded_segs, image)

		maximal_sets = self.getMaximalSets(expanded_segs_tens)
		max_set_masks, cand_grasp_pts = self.getMaximalSetMasks(maximal_sets, expanded_segs_tens, cloth_map)

		#print(cand_grasp_pts)

		if self.show_steps:
			self.showMasksAndGraspPtsOnImage(max_set_masks, image, cand_grasp_pts)
			

		return masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts


	def graspNeighborhood(self, image, grasp_pt):
		img_height = np.shape(image)[0]
		img_length = np.shape(image)[1]

		xrange_min = max(0,grasp_pt[0]-self.nbr_width)
		xrange_max = min(grasp_pt[0]+self.nbr_width, img_length)

		yrange_min = max(0,grasp_pt[1]-self.nbr_width)
		yrange_max = min(grasp_pt[1]+self.nbr_width, img_height)

		#print(np.shape(image))
		#print([xrange_min, xrange_max, yrange_min, yrange_max])
		grasp_nbrhd = image[yrange_min:yrange_max, xrange_min:xrange_max, :]

		return grasp_nbrhd



	def validityChecker(self, bef_img, aft_img):
		bef_img_ext = np.zeros([40,40,3])
		aft_img_ext = np.zeros([40,40,3])

		#bef_img_ext[0:np.shape(bef_img)[0], 0:np.shape(bef_img)[1], :] = bef_img
		#aft_img_ext[0:np.shape(aft_img)[0], 0:np.shape(aft_img)[1], :] = aft_img

		abs_diff = np.sqrt(np.sum(np.multiply(bef_img - aft_img, bef_img - aft_img)))
		print('Absolute diff: ' + str(abs_diff))
		#avg_abs_diff = abs_diff/max(np.sqrt(np.shape(bef_img)[0]*np.shape(bef_img)[1]), np.sqrt(np.shape(aft_img)[0]*np.shape(aft_img)[1]))
		avg_abs_diff = abs_diff/np.sqrt(np.shape(bef_img)[0]*np.shape(bef_img)[1])
		print('Avg diff: ' + str(avg_abs_diff))

		if avg_abs_diff < self.diff_thresh:
			return True
		else:
			return False


	def rotatedNorm(self, pt, orn):
		orn_fixed = orn+m.pi/2
		rot_mat = np.array([[m.cos(orn_fixed), -m.sin(orn_fixed)], [m.sin(orn_fixed), m.cos(orn_fixed)]])
		pt_array = np.array(pt)

		rot_pt = np.matmul(rot_mat, pt_array)

		rot_norm = np.sqrt((rot_pt[0])**2 + (self.rat**2)*((rot_pt[1])**2))

		return rot_norm


	def findMinDistDumb(self, mask, grasp_pt, grasp_orn):
		k = np.shape(mask)[0]
		l = np.shape(mask)[1]
	
		min_dist = max(k,l)
	
		for i in range(k):
			for j in range(l):
				if mask[i,j] != 0:
					dist_to_pt = self.rotatedNorm((np.array([i,j])-np.array(grasp_pt)), grasp_orn)
					min_dist = min(dist_to_pt, min_dist) 
	
		return min_dist


	def ellipseSample(self, orn):

		orn_fixed = orn+m.pi/2
		rot_mat = np.array([[m.cos(orn_fixed), m.sin(orn_fixed)], [-m.sin(orn_fixed), m.cos(orn_fixed)]])

		x_full = 2*np.random.rand()-1
		y_full = 2*np.random.rand()-1

		while(x_full**2 + y_full**2 > 1):
			x_full = 2*np.random.rand()-1
			y_full = 2*np.random.rand()-1

		y_full = y_full/self.rat
		pt = np.array([x_full, y_full])

		pt = pt*self.dist_baseline_thresh

		pt_rot = np.matmul(rot_mat, pt)

		return pt_rot

	
	def evaluatePtOrnRandom(self, masks, grasp_pt, grasp_orn):
		masks_are_dict = False
		
		if type(masks[0]) is dict:
			masks_shape = np.shape(masks[0]['segmentation'])
			masks_are_dict = True
		else:
			masks_shape = np.shape(masks[0])

		num_masks = len(masks)
		mask_hits = np.zeros(num_masks)

		for _ in range(self.sample_num):
			pt_rand = self.ellipseSample(grasp_orn)
			pt_sampled = np.around(grasp_pt + pt_rand)

			if pt_sampled[0] >= masks_shape[0] or pt_sampled[0] < 0:
				continue

			if pt_sampled[1] >= masks_shape[1] or pt_sampled[1] < 0:
				continue

			for i in range(num_masks):
				if masks_are_dict:
					m = masks[i]['segmentation']
				else:
					m = masks[i]
				

				if m[round(pt_sampled[0]),round(pt_sampled[1])] == 1:
					mask_hits[i] += 1

		log_mask_hits = np.log(mask_hits+1)
		orn_score = np.sum(log_mask_hits)
		#print('Cand pt: ' + str(grasp_pt) + '; Orn: ' + str(grasp_orn) + '; Score: ' + str(orn_score))

		return mask_hits, orn_score


	def evaluatePtOrn(self, masks, grasp_pt, grasp_orn):
		mask_captures = np.zeros([len(masks)])
		
		for mask_index in range(len(masks)):
			mask = masks[mask_index]
			if type(mask) is dict:
				m = mask['segmentation']
			else:
				m = mask

			dist_grasp_mask = self.findMinDistDumb(m, grasp_pt, grasp_orn)

			if dist_grasp_mask < self.dist_baseline_thresh:
				mask_captures[mask_index] = 1
		
		tot_cap_num = np.sum(mask_captures)

		return mask_captures, tot_cap_num


	def getBestGraspOrns(self, masks, cand_grasp_pts, is_greedy=True):
		orns_to_try = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity
		num_grasp_pts = len(cand_grasp_pts)

		cand_captures = []
		best_orn_in_tot = []
		orn_scores = []

		for grasp_index in range(num_grasp_pts):

			tot_max = 0
			orn_captures = []
			best_orn_index=0
			orn_score_to_add = []

			for orn_index in range(self.orn_granularity):
				mask_captures, tot_cap_num = self.evaluatePtOrnRandom(masks, cand_grasp_pts[grasp_index], orns_to_try[orn_index])
				orn_captures.append(mask_captures)
				orn_score_to_add.append(tot_cap_num)

				if tot_max < tot_cap_num:
					tot_max = tot_cap_num
					best_orn_index = orn_index

			orn_scores.append(orn_score_to_add)
			best_orn_in_tot.append(orns_to_try[best_orn_index])

			cand_captures.append(orn_captures)

		if is_greedy:
			return best_orn_in_tot, orn_scores

		else:
			return #TODO


	def getBestGraspOrnsRandom(self, masks, cand_grasp_pts, is_greedy=True):
		orns_to_try = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity
		num_grasp_pts = len(cand_grasp_pts)

		cand_hits = []
		best_orn_in_tot = []
		orn_scores = []

		for grasp_index in range(num_grasp_pts):

			tot_max = 0
			orn_hits = []
			best_orn_index=0
			orn_score_to_add = []

			for orn_index in range(self.orn_granularity):
				mask_hits, orn_score = self.evaluatePtOrnRandom(masks, cand_grasp_pts[grasp_index], orns_to_try[orn_index])
				orn_hits.append(mask_hits)
				orn_score_to_add.append(orn_score)

				if tot_max < orn_score:
					tot_max = orn_score
					best_orn_index = orn_index

			orn_scores.append(orn_score_to_add)
			best_orn_in_tot.append(orns_to_try[best_orn_index])

			cand_hits.append(orn_hits)

		if is_greedy:
			return best_orn_in_tot, orn_scores

		else:
			return #TODO	


	def evaluateOrientationAdHoc(self, masks, grasp_pt, grasp_orn, scale = 0.1):
			num_masks = len(masks)
			
			if num_masks == 0:
				return 0
			
			masks_are_dict = False
			ma = masks[0]

			if type(masks[0]) is dict:
				masks_are_dict = True
				ma = masks[0]['segmentation']

			#print(ma)

			img_dims = np.shape(ma)
			#print(img_dims)
			#img_dims = [img_dims[0], img_dims[1]]
			
			grasp_mask = np.zeros(img_dims)

			k = img_dims[0]
			l = img_dims[1]

			k_min = max(grasp_pt[1]-self.dist_baseline_thresh,0)
			k_max = min(grasp_pt[1]+self.dist_baseline_thresh,k)

			l_min = max(grasp_pt[0]-self.dist_baseline_thresh,0)
			l_max = min(grasp_pt[0]+self.dist_baseline_thresh,l)



			for i in range(k_min,k_max):
				for j in range(l_min,l_max):
					dist_to_pt = self.rotatedNorm((np.array([i,j])-np.array([grasp_pt[1],grasp_pt[0]])), grasp_orn)
					if dist_to_pt < self.dist_baseline_thresh:
						grasp_mask[i,j] = 1


			mask_hits = []

			for mask_num in range(num_masks):
				if masks_are_dict:
					ma = masks[mask_num]['segmentation']
				else:
					ma = masks[mask_num]

				#print(np.shape(ma))
				#print(np.shape(grasp_mask))

				#plt.imshow(ma + grasp_mask*0.5)
				#plt.show()

				#plt.imshow(np.multiply(ma, grasp_mask))
				#plt.show()

				hit_count = np.sum(np.sum(np.multiply(ma, grasp_mask)))
				mask_hits.append(hit_count)

			mask_hits = np.array(mask_hits)

			print('Mask hits array:')
			print(mask_hits)

			log_mask_hits = np.log(mask_hits*scale + 1)
			orn_score = np.sum(log_mask_hits)
			print('Score:')
			print(orn_score)

			return mask_hits, orn_score


	def getBestOrns(self, masks, cand_grasp_pts, is_greedy=True):
		orns_to_try = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity
		num_grasp_pts = len(cand_grasp_pts)

		cand_hits = []
		orns_greedy = []
		orn_scores = []

		for grasp_index in range(num_grasp_pts):

			orn_score_max_at_pt = 0
			best_orn_index=0
			
			orn_hits = []
			orn_score_to_add = []

			for orn_index in range(self.orn_granularity):
				mask_hits, orn_score = self.evaluateOrientationAdHoc(masks, cand_grasp_pts[grasp_index], orns_to_try[orn_index])
				orn_hits.append(mask_hits)
				orn_score_to_add.append(orn_score)

				if orn_score_max_at_pt < orn_score:
					orn_score_max_at_pt = orn_score
					best_orn_index = orn_index

			orn_scores.append(orn_score_to_add)
			orns_greedy.append(orns_to_try[best_orn_index])

			cand_hits.append(orn_hits)

		if is_greedy:
			return orns_greedy, orn_scores


	def predictCaptures(self, grasp_pt, grasp_orn, masks, predictor='baseline'):

		if predictor=='baseline':
			num_masks = len(masks)
			
			if num_masks == 0:
				return 0
			
			masks_are_dict = False
			ma = masks[0]

			if type(masks[0]) is dict:
				masks_are_dict = True
				ma = masks[0]['segmentation']

			#print(ma)

			img_dims = np.shape(ma)
			#print(img_dims)
			#img_dims = [img_dims[0], img_dims[1]]
			
			grasp_mask = np.zeros(img_dims)

			k = img_dims[0]
			l = img_dims[1]

			k_min = max(grasp_pt[1]-self.dist_baseline_thresh,0)
			k_max = min(grasp_pt[1]+self.dist_baseline_thresh,k)

			l_min = max(grasp_pt[0]-self.dist_baseline_thresh,0)
			l_max = min(grasp_pt[0]+self.dist_baseline_thresh,l)



			for i in range(k_min,k_max):
				for j in range(l_min,l_max):
					dist_to_pt = self.rotatedNorm((np.array([i,j])-np.array([grasp_pt[1],grasp_pt[0]])), grasp_orn)
					if dist_to_pt < self.dist_baseline_thresh:
						grasp_mask[i,j] = 1


			mask_hits = []

			for mask_num in range(num_masks):
				if masks_are_dict:
					ma = masks[mask_num]['segmentation']
				else:
					ma = masks[mask_num]

				#print(np.shape(ma))
				#print(np.shape(grasp_mask))

				#plt.imshow(ma + grasp_mask*0.5)
				#plt.show()

				#plt.imshow(np.multiply(ma, grasp_mask))
				#plt.show()

				hit_count = np.sum(np.sum(np.multiply(ma, grasp_mask)))
				mask_hits.append(hit_count)

			mask_hits = np.array(mask_hits)
			this_mask_capture = mask_hits/(mask_hits + self.baseline_prob_scaling_factor)

		elif predictor == 'NN':
			TODO

		return this_mask_capture


	def getMaskCaptures(self, image, masks, cand_pts, predictor='baseline'):
		
		num_pts = len(cand_pts)
		num_orns = self.orn_granularity
		num_masks = len(masks)

		cand_orns = m.pi*np.array(range(num_orns))/num_orns

		mask_captures = np.zeros([num_pts, num_orns, num_masks])

		for pt_index in range(num_pts):

			print("Point " +  str(pt_index+1) + " out of " + str(num_pts))

			for orn_index in range(num_orns):
				print("Orientation " +  str(orn_index+1) + " out of " + str(num_orns))
				if predictor=='baseline':
					m_cs = self.predictCaptures(cand_pts[pt_index], cand_orns[orn_index], masks, predictor)
				elif predictor=='NN':

					image_rescaled = color_image = cv2.resize(image, (575, 444))

					masks_rescaled = []
					for i in range(len(masks)):
						mask_seg = np.array(masks[i]['segmentation']*255)
						mask_seg = mask_seg.astype(np.float)
						#mask_seg_three_channel = np.dstack()

						masks_rescaled_to_add = cv2.resize(mask_seg, dsize=(575, 444))
						masks_rescaled_to_add = np.around(masks_rescaled_to_add/255)
						masks_rescaled_to_add = masks_rescaled_to_add.astype(int)

						#print("Mask " + str(i+1) + " out of " + str(len(masks)))

						#print(np.max(masks_rescaled_to_add))
						#print(np.min(masks_rescaled_to_add))
						masks_rescaled.append(masks_rescaled_to_add)

					self.proc_data.load_img_masks(image_rescaled, masks_rescaled)

					cand_pts_rescaled = cand_pts.copy()
					cand_pts_rescaled[:,0] = np.around(cand_pts[:,0]*(575/330))
					cand_pts_rescaled[:,1] = np.around(cand_pts[:,1]*(444/240))


					self.proc_data.load_cand_pts_orns(cand_pts_rescaled, self.orn_granularity)

					masked_pile_images_, masked_pile_feat_, masked_image_arrs_, masked_image_feats_ = self.proc_data.process_(self.resnet, pt_index, orn_index, self.device)

					#transformation = transforms.Compose([transforms.Resize((224,224)),
									 #transforms.ToTensor()])

					m_cs = np.zeros(num_masks)

					#img_pile = transformation(masked_pile_images_)
					for mask_index in range(num_masks):
						#img_mask = transformation(img_mask_FIND)

						#if not np.any(img_mask):
						#img_pile, img_mask = img_pile.to(self.device), img_mask.to(self.device)

						#img = self.resnet(img_pile.unsqueeze(0))
						#mask = self.resnet(img_mask.unsqueeze(0))

						output = self.sia_net(torch.tensor(masked_pile_feat_).cuda(), torch.tensor(masked_image_feats_[mask_index]).cuda())
						m_cs[mask_index] = torch.sigmoid(output)

				mask_captures[pt_index, orn_index, :] = m_cs

		return mask_captures
	




	
	# mask_captures is a 3D tensor with the following dimensions
	# 1. num_pts --- states which candidate point is being referred to
	# 2. num_orns --- states which orientation is being referred to
	# 3. num_masks --- states which mask is being referred to
	# BOTTOM LINE: mask_captures[pt_index, orn_index, mask_index] is 
	#	the probability that the mask is captured by a grasp at that point and orientation
	def getBestOrnsIdeal(self, mask_captures, masks, cand_pts, is_greedy=False):

		num_pts = len(cand_pts)
		num_orns = self.orn_granularity
		num_masks = len(masks)

		#best_grasp_combo_score = 0 
		prob_escape = np.ones(num_masks)

		best_orn_indices = [0]*num_pts

		if is_greedy:
			for pt_index in range(num_pts):
				orn_capture_sum = np.sum(mask_captures[pt_index, :, :], axis=1)
				best_orn_index = np.argmax(orn_capture_sum)
				best_orn_indices[pt_index] = best_orn_index
		else:
			combo_list = [np.array(range(num_orns))]*num_pts
			most_captures = 0
			for orn_selection in product(*combo_list):
				prob_escape = np.ones(num_masks)
				for pt_index in range(num_pts):

					#print('Mask captures shape is:')
					#print(np.shape(mask_captures))

					#print(pt_index, orn_selection[pt_index])

					prob_escape = np.multiply(prob_escape, 1 - mask_captures[pt_index, int(orn_selection[pt_index]), :])
					prob_capture = 1 - prob_escape
					expected_captures = np.sum(prob_capture)
					if expected_captures > most_captures:
						most_captures = expected_captures
						best_orn_indices = orn_selection

		return best_orn_indices



			














	def graspAtOptPt(self, opt_im_pick_pt, opt_pick_orn):

		# Grasp at optimal param point
		# rw_x_pick, rw_y_pick = self.depth_lookup_table[opt_im_pick_pt[0], opt_im_pick_pt[1]]
		#print('Converting to robot coordinates...')
		#print([opt_im_pick_pt[1], opt_im_pick_pt[0]])
		rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
		#print(rw_pick)

		rw_x_pick = rw_pick[0][0]
		rw_y_pick = rw_pick[0][1]

		#print([rw_x_pick, rw_y_pick])
		pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
		place_pt = self.gen_place_point()

		init_action_weight = self.recordWeight(0)
		self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)



	def evaluateGraspNN(self, pos, orn):
		testloader = DataLoader(test, batch_size= 32, shuffle=True)



	def laundry_seg_and_height_algo(self, save=False, multipick=False, predictor='baseline', height_thresh = 1.37):

		#self.mask_generator = self.loadSAM()
		
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop

		self.num_attempts = 0

		self.trips = 0

		num_clusters = 1

		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
		# subprocess.call(['mkdir', '-p', self.trial_path])

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		new_compute_seq = []
		compute_time_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0
		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
									bw_image, depth_image)
		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			weight_params, height_params, volume_params = self.get_optimal_cluster(clusters)
			print("height: ", height_params[0])
			if (height_params[0] < height_thresh):
				
				print("Height not exceeded: doing segmentations")
				
				
				if save:
					ts_before_compute = timeit.default_timer()
				
				
				masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
				cand_grasp_pts = np.array(cand_grasp_pts)

				sort_order = np.argsort(cand_grasp_pts[:,0])
				cand_grasp_pts = cand_grasp_pts[sort_order,:]

				grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

				mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
				best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
				best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

				cand_pts_seq[self.num_attempts]=cand_grasp_pts
				cand_orns_seq[self.num_attempts]=best_orns

				grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)
				
				orig_nbrhd = []

				for c_g_p_index in range(len(cand_grasp_pts)):
					curr_grasp_pt = cand_grasp_pts[c_g_p_index]
					orig_nbrhd.append(self.graspNeighborhood(rgb_image, curr_grasp_pt))

				if save:
					ts_after_compute = timeit.default_timer()
					compute_time_seq.append(ts_after_compute-ts_before_compute)


				for c_g_p_index in range(len(cand_grasp_pts)):
					curr_grasp_pt = cand_grasp_pts[c_g_p_index]
					grasp_nbrhd = self.graspNeighborhood(new_image, curr_grasp_pt)
					print(curr_grasp_pt)
					opt_pick_orn = best_orns[0][c_g_p_index]
					if opt_pick_orn < -m.pi:
						opt_pick_orn += m.pi
					if opt_pick_orn > m.pi:
						opt_pick_orn += -m.pi
					if self.validityChecker(orig_nbrhd[c_g_p_index], grasp_nbrhd):
						conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
						cloth_map = self.getClothMap(new_image)
						cloth_map_new = 1-self.removeHoles((1-cloth_map))
						cloth_map_new = self.removeHoles(cloth_map)
						if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
							print('Executing grasp...')
							grasp_executed=True
							self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
							self.trips += 1
							if save:
								time.sleep(3)
								weight = self.recordWeight(0)
								weight_seq.append(weight)
								now_ts = timeit.default_timer()
								time_seq.append(now_ts-start_ts)
								rgb_image_seq.append(new_image)
								depth_image_seq.append(depth_image)
								masks_seq.append(masks)
								masks_raw_seq.append(masks_raw)
								pick_seq.append([curr_grasp_pt, opt_pick_orn])
								if new_compute:
									new_compute_seq.append(1)
								else:
									new_compute_seq.append(0)
								new_compute = False
								bw_image_seq.append(cloth_map)
					rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
					new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
			else:
				print('Height exceeded: picking high point')

				opt_im_pick_pt = height_params[1]
				opt_pick_orn = height_params[2]

				pick_pt_depth = depth_image[opt_im_pick_pt[0], opt_im_pick_pt[1]] * 1000
				if self.use_pick_pt_depth:
					#print("depth: " + str(pick_pt_depth))
					#print("height: " + str(800 - pick_pt_depth))
					#print("pick point in image:" + str(opt_im_pick_pt))
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]], pick_pt_depth)
				else:
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
				
				
				rw_x_pick = rw_pick[0][0]
				rw_y_pick = rw_pick[0][1]
				pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
				place_pt = self.gen_place_point()

				if self.show_steps:
					plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
					plt.imshow(rgb_image)
					plt.show()
				
				self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)
				self.wrist_unwinder()

				if save:
					time.sleep(3)
					weight = self.recordWeight(0)
					weight_seq.append(weight)
					
					curr_time = timeit.default_timer()
					time_seq.append(curr_time - start_ts)
					bw_image_seq.append(bw_image)
					rgb_image_seq.append(rgb_image)
					depth_image_seq.append(depth_image)
					pick_seq.append([opt_im_pick_pt, opt_pick_orn])
				self.trips += 1
				#continue

			rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
			new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]

			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1
			
		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq




	def laundry_seg_and_vol_algo(self, save=False, multipick=False, predictor='baseline', height_thresh = 1.37):

		self.mask_generator = self.loadSAM()
		
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop

		self.num_attempts = 0

		self.trips = 0

		num_clusters = 1

		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
		# subprocess.call(['mkdir', '-p', self.trial_path])

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		new_compute_seq = []
		compute_time_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0
		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
									bw_image, depth_image)
		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			weight_params, height_params, volume_params = self.get_optimal_cluster(clusters)
			print("height: ", height_params[0])
			if (height_params[0] < height_thresh):
				
				print("Height not exceeded: doing segmentations")
				
				
				if save:
					ts_before_compute = timeit.default_timer()
				
				
				masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
				cand_grasp_pts = np.array(cand_grasp_pts)

				sort_order = np.argsort(cand_grasp_pts[:,0])
				cand_grasp_pts = cand_grasp_pts[sort_order,:]

				grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

				mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
				best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
				best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

				cand_pts_seq[self.num_attempts]=cand_grasp_pts
				cand_orns_seq[self.num_attempts]=best_orns

				grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)
				
				orig_nbrhd = []

				for c_g_p_index in range(len(cand_grasp_pts)):
					curr_grasp_pt = cand_grasp_pts[c_g_p_index]
					orig_nbrhd.append(self.graspNeighborhood(rgb_image, curr_grasp_pt))

				if save:
					ts_after_compute = timeit.default_timer()
					compute_time_seq.append(ts_after_compute-ts_before_compute)


				for c_g_p_index in range(len(cand_grasp_pts)):
					curr_grasp_pt = cand_grasp_pts[c_g_p_index]
					grasp_nbrhd = self.graspNeighborhood(new_image, curr_grasp_pt)
					print(curr_grasp_pt)
					opt_pick_orn = best_orns[0][c_g_p_index]
					if opt_pick_orn < -m.pi:
						opt_pick_orn += m.pi
					if opt_pick_orn > m.pi:
						opt_pick_orn += -m.pi
					if self.validityChecker(orig_nbrhd[c_g_p_index], grasp_nbrhd):
						conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
						cloth_map = self.getClothMap(new_image)
						cloth_map_new = 1-self.removeHoles((1-cloth_map))
						cloth_map_new = self.removeHoles(cloth_map)
						if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
							print('Executing grasp...')
							grasp_executed=True
							self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
							self.trips += 1
							if save:
								time.sleep(3)
								weight = self.recordWeight(0)
								weight_seq.append(weight)
								now_ts = timeit.default_timer()
								time_seq.append(now_ts-start_ts)
								rgb_image_seq.append(new_image)
								depth_image_seq.append(depth_image)
								masks_seq.append(masks)
								masks_raw_seq.append(masks_raw)
								pick_seq.append([curr_grasp_pt, opt_pick_orn])
								if new_compute:
									new_compute_seq.append(1)
								else:
									new_compute_seq.append(0)
								new_compute = False
								bw_image_seq.append(cloth_map)
					rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
					new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
			else:
				print('Height exceeded: picking to maximize volume')

				opt_im_pick_pt = volume_params[1]
				opt_pick_orn = volume_params[2]


				pick_pt_depth = depth_image[opt_im_pick_pt[0], opt_im_pick_pt[1]] * 1000
				if self.use_pick_pt_depth:
					#print("depth: " + str(pick_pt_depth))
					#print("height: " + str(800 - pick_pt_depth))
					#print("pick point in image:" + str(opt_im_pick_pt))
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]], pick_pt_depth)
				else:
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
				

				rw_x_pick = rw_pick[0][0]
				rw_y_pick = rw_pick[0][1]
				pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
				place_pt = self.gen_place_point()

				if self.show_steps:
					plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
					plt.imshow(rgb_image)
					plt.show()
				
				self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)
				self.wrist_unwinder()

				if save:
					time.sleep(3)
					weight = self.recordWeight(0)
					weight_seq.append(weight)
					
					curr_time = timeit.default_timer()
					time_seq.append(curr_time - start_ts)
					bw_image_seq.append(bw_image)
					rgb_image_seq.append(rgb_image)
					depth_image_seq.append(depth_image)
					pick_seq.append([opt_im_pick_pt, opt_pick_orn])
				self.trips += 1
				#continue

			rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
			new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]

			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1
			
		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq







	def laundry_seg_algo_sequences_long(self, save=True, multipick=False, predictor='baseline'):
		self.mask_generator = self.loadSAM()
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop
		self.num_attempts = 0
		self.trips = 0
		num_clusters = 1
		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		cand_pts_len_seq = []
		new_compute_seq = []
		compute_time_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			
			if save:
				ts_before_compute = timeit.default_timer()
			masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
		
			cand_grasp_pts = np.array(cand_grasp_pts)

			sort_order = np.argsort(cand_grasp_pts[:,0])
			cand_grasp_pts = cand_grasp_pts[sort_order,:]
			cand_grasp_pts = np.flip(cand_grasp_pts, 0)
			

			grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

			mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
			
			best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
			best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

			if save:
				ts_after_compute = timeit.default_timer()
				compute_time = ts_after_compute - ts_before_compute

			cand_pts_seq[self.num_attempts]=cand_grasp_pts
			cand_orns_seq[self.num_attempts]=best_orns

			cand_pts_len_seq.append(len(cand_grasp_pts))

			print("Grasping at image coords...")
			print(cand_grasp_pts)


			cand_grasp_transformed = []
			for i in range(len(cand_grasp_pts)):
				cand_grasp_transformed.append(self.image_pt_to_rw_pt([cand_grasp_pts[i][0], cand_grasp_pts[i][1]])) #[BOOKMARK]

			#print("Grasping at rw coords...")
			#print(cand_grasp_transformed)

			

			#cand_grasp_pts = np.concatenate((cand_grasp_pts, [[self.x_home, self.y_home]]), axis=0)

			cand_grasp_transformed.append(np.array([[self.x_home, self.y_home]]))
			print("Grasping at rw coords...")
			print(cand_grasp_transformed)

			#cand_grasp_transformed = np.flip(cand_grasp_transformed, 1)

			grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)

			if self.debug_motion_plan:
				pdb.set_trace()
			self.pick_and_place_sequence(cand_grasp_transformed, best_orns, self.translation_z)
		
			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

			if save:
				rgb_image_seq.append(rgb_image)
				depth_image_seq.append(depth_image)
				bw_image_seq.append(bw_image)
				#pick_seq.append([cand_pts_seq[-2], best_orns[-1]])
				masks_seq.append(masks)
				masks_raw_seq.append(masks_raw)
				compute_time_seq.append(compute_time)

			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq




	# MASK CAPTURE MAT tells you which masks each candidate point is likely to select
	# INDICES TO GRASP is a boolean vector saying whether that point should be in the sequence to select
	# heuristic 1: a sequence should not have the same segment grasped (a lot) by multiple grasps
	# heuristic 2: always start from the back, that way if you drop something it might fall on something else and make your life easier
	# THIS IS GREEDY -- always prefers furthest-back point it can take
	def sequence_selector_volume_limit(self, mask_capture_mat, masks, depth_image, max_volume_threshold=250):
		indices_to_grasp = [True]
		overall_capture_mask = np.zeros(len(mask_capture_mat[0]))
		max_overlap_threshold = max(1.25*np.max(mask_capture_mat), 1)

		height_image = 0.8 - np.array(depth_image)

		mask_vols = []

		for i in range(len(masks)):
			if type(masks[i]) is dict:
				ma = masks[i]['segmentation']
			else:
				ma = masks[i]
			
			mask_vols.append(np.sum(np.multiply(ma, height_image)))

		mask_vols = np.array(mask_vols)

		for i in range(len(mask_capture_mat)-1):
			new_overall_capture_mask = overall_capture_mask + mask_capture_mat[i+1]

			overall_predicted_capture_volume = np.matmul(mask_vols, new_overall_capture_mask.T)

			print(overall_predicted_capture_volume)

			if np.max(new_overall_capture_mask) < max_overlap_threshold and overall_predicted_capture_volume < max_volume_threshold:
				overall_capture_mask = new_overall_capture_mask
				indices_to_grasp.append(True)
			else:
				indices_to_grasp.append(False)


		return indices_to_grasp




	def laundry_hybrid_seg_height_sequences(self, save=True, multipick=False, predictor='baseline'):
		#self.mask_generator = self.loadSAM()
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop
		self.num_attempts = 0
		self.trips = 0
		num_clusters = 1
		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		cand_pts_len_seq = []
		new_compute_seq = []
		compute_time_seq = []
		num_rearrange_seq = []
		self.num_rearrangements = 0

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0

		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
		num_clusters = len(clusters)

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			
			weight_params, height_params, volume_params = self.get_optimal_cluster(clusters)
			print("height: ", height_params[0])
			if (height_params[0] < self.height_thresh):
				print("Height not exceeded: doing segmentations")

				if save:
					ts_before_compute = timeit.default_timer()
				masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
			
				cand_grasp_pts = np.array(cand_grasp_pts)

				sort_order = np.argsort(cand_grasp_pts[:,0])
				cand_grasp_pts = cand_grasp_pts[sort_order,:]
				cand_grasp_pts = np.flip(cand_grasp_pts, 0)
				

				grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

				mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
				
				best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
				best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

				#print(len(cand_grasp_pts))
				print(best_orns)
				
				mask_capture_mat = np.zeros([len(cand_grasp_pts), len(masks)])

				for i in range(len(cand_grasp_pts)):
					mask_capture_mat[i] = mask_capture_tens[i][best_orns_index[i]]

				sequence_to_try = self.sequence_selector_area_limit(mask_capture_mat, cand_grasp_pts, masks)

				
				self.num_rearrangements += len(sequence_to_try)-1

				grasps_to_try = []
				orns_to_try = []
				for i in range(len(cand_grasp_pts)):
					if sequence_to_try[i]:
						grasps_to_try.append(cand_grasp_pts[i])
						orns_to_try.append(best_orns[0][i])

				print(orns_to_try)

				if save:
					ts_after_compute = timeit.default_timer()
					compute_time = ts_after_compute - ts_before_compute

					cand_pts_seq[self.num_attempts]=cand_grasp_pts
					cand_orns_seq[self.num_attempts]=best_orns

					cand_pts_len_seq.append(len(cand_pts_seq))

				print("Grasping at image coords...")
				print(grasps_to_try)


				grasps_transformed = []
				for i in range(len(grasps_to_try)):
					grasps_transformed.append(self.image_pt_to_rw_pt([grasps_to_try[i][0], grasps_to_try[i][1]])) #[BOOKMARK]

				#print("Grasping at rw coords...")
				#print(cand_grasp_transformed)

				

				#cand_grasp_pts = np.concatenate((cand_grasp_pts, [[self.x_home, self.y_home]]), axis=0)

				#grasps_transformed.append(np.array([[self.x_home, self.y_home]]))
				print("Grasping at rw coords...")
				print(grasps_transformed)

				#cand_grasp_transformed = np.flip(cand_grasp_transformed, 1)

				#grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)

				if self.debug_motion_plan:
					pdb.set_trace()
				self.pick_and_place_sequence(grasps_transformed, [orns_to_try[:-1]], self.rearrangement_translation_z)

				self.pick_and_place([grasps_transformed[-1][0][0], grasps_transformed[-1][0][1], self.z_grasp], [self.x_home, self.y_home, self.z_home], orns_to_try[-1], (self.translation_z+self.rearrangement_translation_z)/2)
				time.sleep(3)

				weight = self.recordWeight(0)

				
				

				if save:
					rgb_image_seq.append(rgb_image)
					depth_image_seq.append(depth_image)
					bw_image_seq.append(bw_image)
					pick_seq.append([grasps_to_try, orns_to_try])
					masks_seq.append(masks)
					masks_raw_seq.append(masks_raw)
					compute_time_seq.append(compute_time)
					curr_time = timeit.default_timer() - start_ts
					time_seq.append(curr_time)
					weight_seq.append(weight)
					num_rearrange_seq.append(len(sequence_to_try)-1)

				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
				rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
				clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
				num_clusters = len(clusters)
				self.num_attempts += 1

			else:
				print("Height exceeded: picking high point")

				opt_im_pick_pt = height_params[1]
				opt_pick_orn = height_params[2]

				pick_pt_depth = depth_image[opt_im_pick_pt[0], opt_im_pick_pt[1]] * 1000
				if self.use_pick_pt_depth:
					#print("depth: " + str(pick_pt_depth))
					#print("height: " + str(800 - pick_pt_depth))
					#print("pick point in image:" + str(opt_im_pick_pt))
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]], pick_pt_depth)
				else:
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
				
				
				rw_x_pick = rw_pick[0][0]
				rw_y_pick = rw_pick[0][1]
				pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
				place_pt = self.gen_place_point()

				if self.show_steps:
					plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
					plt.imshow(rgb_image)
					plt.show()
				
				self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)
				self.wrist_unwinder()
				time.sleep(3)
				weight = self.recordWeight(0)


				if save:
					pick_seq.append([opt_im_pick_pt[0], opt_im_pick_pt[1], opt_pick_orn])
					weight_seq.append(weight)
					curr_time = timeit.default_timer()
					time_seq.append(curr_time - start_ts)





				if save:
					rgb_image_seq.append(rgb_image)
					depth_image_seq.append(depth_image)
					bw_image_seq.append(bw_image)
					#pick_seq.append([cand_pts_seq[-2], best_orns[-1]])
					#masks_seq.append(masks)
					#masks_raw_seq.append(masks_raw)
					#compute_time_seq.append(compute_time)
					weight_seq.append(weight)

			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq, num_rearrange_seq





	# MASK CAPTURE MAT tells you which masks each candidate point is likely to select
	# INDICES TO GRASP is a boolean vector saying whether that point should be in the sequence to select
	# heuristic 1: a sequence should not have the same segment grasped (a lot) by multiple grasps
	# heuristic 2: always start from the back, that way if you drop something it might fall on something else and make your life easier
	# THIS IS GREEDY -- always prefers furthest-back point it can take
	def sequence_selector_area_limit(self, mask_capture_mat, cand_grasp_pts, masks, max_area_threshold=10000):
		indices_to_grasp = [True]
		overall_capture_mask = np.zeros(len(mask_capture_mat[0]))
		max_overlap_threshold = max(1.25*np.max(mask_capture_mat), 1)

		mask_areas = []

		for i in range(len(masks)):
			if type(masks[i]) is dict:
				ma = masks[i]['segmentation']
			else:
				ma = masks[i]
			
			mask_areas.append(np.sum(ma))

		mask_areas = np.array(mask_areas)

		for i in range(len(mask_capture_mat)-1):
			new_overall_capture_mask = overall_capture_mask + mask_capture_mat[i+1]

			overall_predicted_capture_area = np.matmul(mask_areas, new_overall_capture_mask.T)

			print(overall_predicted_capture_area)

			if np.max(new_overall_capture_mask) < max_overlap_threshold and overall_predicted_capture_area + 500*np.sum(new_overall_capture_mask) < max_area_threshold:
				if cand_grasp_pts[i][1] < 220 and cand_grasp_pts[i][1] > 20:
					if cand_grasp_pts[i][0] < 310 and cand_grasp_pts[i][0] > 20:
						overall_capture_mask = new_overall_capture_mask
						indices_to_grasp.append(True)
					else:
						indices_to_grasp.append(False)
				else:
					indices_to_grasp.append(False)
			else:
				indices_to_grasp.append(False)


		return indices_to_grasp




	def laundry_seg_seq_selected(self, save=True, multipick=False, predictor='baseline'):
		self.mask_generator = self.loadSAM()
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop
		self.num_attempts = 0
		self.trips = 0
		num_clusters = 1
		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		cand_pts_len_seq = []
		new_compute_seq = []
		compute_time_seq = []
		num_rearrange_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0
		self.num_rearrangements=0

		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
		num_clusters = len(clusters)

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			#grasp_executed = False
			#new_compute = True
			
			#weight_params, height_params, volume_params = self.get_optimal_cluster(clusters)

			#print("Height not exceeded: doing segmentations")

			if save:
				ts_before_compute = timeit.default_timer()
			masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
		
			print('Image dimensions are: ')
			print(np.shape(masks[0]['segmentation']))

			cand_grasp_pts = np.array(cand_grasp_pts)

			sort_order = np.argsort(cand_grasp_pts[:,0])
			cand_grasp_pts = cand_grasp_pts[sort_order,:]
			cand_grasp_pts = np.flip(cand_grasp_pts, 0)
			
			print('Considering points:')
			print(cand_grasp_pts)

			grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

			mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
			
			best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
			best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

			#print(len(cand_grasp_pts))
			print(best_orns)
			
			mask_capture_mat = np.zeros([len(cand_grasp_pts), len(masks)])

			for i in range(len(cand_grasp_pts)):
				mask_capture_mat[i] = mask_capture_tens[i][best_orns_index[i]]

			sequence_to_try = self.sequence_selector_area_limit(mask_capture_mat, cand_grasp_pts, masks)

			
			self.num_rearrangements += len(sequence_to_try)-1

			grasps_to_try = []
			orns_to_try = []
			for i in range(len(cand_grasp_pts)):
				if sequence_to_try[i]:
					grasps_to_try.append(cand_grasp_pts[i])
					orns_to_try.append(best_orns[0][i])

			print(orns_to_try)

			if save:
				ts_after_compute = timeit.default_timer()
				compute_time = ts_after_compute - ts_before_compute

				cand_pts_seq[self.num_attempts]=cand_grasp_pts
				cand_orns_seq[self.num_attempts]=best_orns

				cand_pts_len_seq.append(len(cand_pts_seq))

			print("Grasping at image coords...")
			print(grasps_to_try)


			grasps_transformed = []
			for i in range(len(grasps_to_try)):
				grasps_transformed.append(self.image_pt_to_rw_pt([grasps_to_try[i][0], grasps_to_try[i][1]])) #[BOOKMARK]

			#print("Grasping at rw coords...")
			#print(cand_grasp_transformed)

			

			#cand_grasp_pts = np.concatenate((cand_grasp_pts, [[self.x_home, self.y_home]]), axis=0)

			#grasps_transformed.append(np.array([[self.x_home, self.y_home]]))
			print("Grasping at rw coords...")
			print(grasps_transformed)

			#cand_grasp_transformed = np.flip(cand_grasp_transformed, 1)

			#grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)

			if self.debug_motion_plan:
				pdb.set_trace()
			self.pick_and_place_sequence(grasps_transformed, [orns_to_try[:-1]], self.rearrangement_translation_z)

			self.pick_and_place([grasps_transformed[-1][0][0], grasps_transformed[-1][0][1], self.z_grasp], [self.x_home, self.y_home, self.z_home], orns_to_try[-1], (self.translation_z+self.rearrangement_translation_z)/2)
			time.sleep(3)

			weight = self.recordWeight(0)

			
			

			if save:
				rgb_image_seq.append(rgb_image)
				depth_image_seq.append(depth_image)
				bw_image_seq.append(bw_image)
				pick_seq.append([grasps_to_try, orns_to_try])
				masks_seq.append(masks)
				masks_raw_seq.append(masks_raw)
				compute_time_seq.append(compute_time)
				curr_time = timeit.default_timer() - start_ts
				time_seq.append(curr_time)
				weight_seq.append(weight)
				num_rearrange_seq.append(len(sequence_to_try)-1)

			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, cand_pts_len_seq, new_compute_seq, compute_time_seq, num_rearrange_seq







	def laundry_seg_algo_with_consolidation(self, save=False, multipick=False, predictor='baseline'):
		self.mask_generator = self.loadSAM()
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop
		self.num_attempts = 0
		self.trips = 0
		num_clusters = 1
		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		new_compute_seq = []
		compute_time_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			
			if save:
				ts_before_compute = timeit.default_timer()
			masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
		
			cand_grasp_pts = np.array(cand_grasp_pts)

			sort_order = np.argsort(cand_grasp_pts[:,0])
			cand_grasp_pts = cand_grasp_pts[sort_order,:]

			grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

			mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
			
			best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
			best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

			cand_pts_seq[self.num_attempts]=cand_grasp_pts
			cand_orns_seq[self.num_attempts]=best_orns

			grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)

			orig_nbrhd = []

			for c_g_p_index in range(len(cand_grasp_pts)):
				curr_grasp_pt = cand_grasp_pts[c_g_p_index]
				orig_nbrhd.append(self.graspNeighborhood(rgb_image, curr_grasp_pt))

			if save:
				ts_after_compute = timeit.default_timer()
				compute_time_seq.append(ts_after_compute-ts_before_compute)

			for c_g_p_index in range(len(cand_grasp_pts)):
				curr_grasp_pt = cand_grasp_pts[c_g_p_index]
				grasp_nbrhd = self.graspNeighborhood(new_image, curr_grasp_pt)
				print(curr_grasp_pt)
				opt_pick_orn = best_orns[0][c_g_p_index]
				if opt_pick_orn < -m.pi:
					opt_pick_orn += m.pi
				if opt_pick_orn > m.pi:
					opt_pick_orn += -m.pi

				if self.validityChecker(orig_nbrhd[c_g_p_index], grasp_nbrhd):
					conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
					cloth_map = self.getClothMap(new_image)
					cloth_map_new = 1 - self.removeHoles((1 - cloth_map))

					cloth_map_new = self.removeHoles(cloth_map)

					if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
						print('Executing grasp...')
						grasp_executed = True
						self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
						self.trips += 1

						if save:
							time.sleep(3)
							weight = self.recordWeight(0)
							weight_seq.append(weight)
							now_ts = timeit.default_timer()
							time_seq.append(now_ts - start_ts)
							rgb_image_seq.append(new_image)
							depth_image_seq.append(depth_image)

							masks_seq.append(masks)

							masks_raw_seq.append(masks_raw)

							pick_seq.append([curr_grasp_pt, opt_pick_orn])
							if new_compute:
								new_compute_seq.append(1)
							else:
								new_compute_seq.append(0)
							new_compute = False
							bw_image_seq.append(cloth_map)

						# Transfer clothes to another grasp point if there are two viable points
						if len(cand_grasp_pts) >= 2:
							transfer_index = 1 - c_g_p_index  # Index of the other viable grasp point
							transfer_grasp_pt = cand_grasp_pts[transfer_index]
							transfer_grasp_pt = np.insert(transfer_grasp_pt, 2, self.z_grasp)
							curr_grasp_pt = np.insert(curr_grasp_pt, 2, self.z_grasp)
							print(curr_grasp_pt)
							# curr_grasp_pt.append(self.translation_z)
							print('Transferring clothes to another grasp point...')
							self.pick_and_place(curr_grasp_pt, transfer_grasp_pt, opt_pick_orn, self.translation_z, use_shake_action=False)
							# grasp_executed = True

					rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
					# Use x_crop and y_crop to get workspace image
					new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
		# color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

			# for c_g_p_index in range(len(cand_grasp_pts)):
			# 	curr_grasp_pt = cand_grasp_pts[c_g_p_index]
			# 	grasp_nbrhd = self.graspNeighborhood(new_image, curr_grasp_pt)
			# 	print(curr_grasp_pt)
			# 	opt_pick_orn = best_orns[0][c_g_p_index]
			# 	if opt_pick_orn < -m.pi:
			# 		opt_pick_orn += m.pi
			# 	if opt_pick_orn > m.pi:
			# 		opt_pick_orn += -m.pi


				# if self.validityChecker(orig_nbrhd[c_g_p_index], grasp_nbrhd):
				# 	conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
				# 	cloth_map = self.getClothMap(new_image)
				# 	cloth_map_new = 1-self.removeHoles((1-cloth_map))

				# 	cloth_map_new = self.removeHoles(cloth_map)

				# 	if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
				# 		print('Executing grasp...')
				# 		grasp_executed=True
				# 		self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
				# 		self.trips += 1
						
				# 		if save:
				# 			time.sleep(3)
				# 			weight = self.recordWeight(0)
				# 			weight_seq.append(weight)
				# 			now_ts = timeit.default_timer()
				# 			time_seq.append(now_ts-start_ts)
				# 			rgb_image_seq.append(new_image)
				# 			depth_image_seq.append(depth_image)
							
				# 			masks_seq.append(masks)
					
				# 			masks_raw_seq.append(masks_raw)

				# 			pick_seq.append([curr_grasp_pt, opt_pick_orn])
				# 			if new_compute:
				# 				new_compute_seq.append(1)
				# 			else:
				# 				new_compute_seq.append(0)
				# 			new_compute = False
				# 			bw_image_seq.append(cloth_map)
						
				# rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)
				# # Use x_crop and y_crop to get workspace image
				# new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
				#color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		
			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)
			self.num_attempts += 1

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq




	def laundry_seg_algo(self, save=False, multipick=False, predictor='baseline'):

		#self.mask_generator = self.loadSAM()
		
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop

		self.num_attempts = 0

		self.trips = 0

		num_clusters = 1

		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
		# subprocess.call(['mkdir', '-p', self.trial_path])

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		new_image = rgb_image

		if self.show_steps:
			plt.imshow(rgb_image)
			plt.axis('off')
			plt.show()

			plt.imshow(depth_image)
			plt.axis('off')
			plt.show()

		weight_seq = []
		time_seq = []
		rgb_image_seq = []
		bw_image_seq = []
		depth_image_seq = []
		pick_seq = []
		masks_seq = []
		masks_raw_seq = []
		cand_pts_seq = dict()
		cand_orns_seq = dict()
		new_compute_seq = []
		compute_time_seq = []

		if save:
			weight = self.recordWeight(0)
			weight_seq.append(weight)
			start_ts = timeit.default_timer()
			time_seq.append(0)

		self.num_attempts=0

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:
			grasp_executed = False
			new_compute = True
			
			if save:
				ts_before_compute = timeit.default_timer()
			masks, masks_raw, cloth_map, max_set_masks, cand_grasp_pts = self.getCandidateGraspPts(rgb_image)
			

			#if save:
			#	masks_seq.append(masks)
				
			#	masks_raw_seq.append(masks_raw)

			
			cand_grasp_pts = np.array(cand_grasp_pts)

			sort_order = np.argsort(cand_grasp_pts[:,0])
			cand_grasp_pts = cand_grasp_pts[sort_order,:]
			#self.showMasksAndGraspPtsOnImage(masks, rgb_image, cand_grasp_pts)

			grasp_orns = np.array(range(self.orn_granularity))*m.pi/self.orn_granularity

			print("Getting mask captures...")
			mask_capture_tens = self.getMaskCaptures(rgb_image, masks, cand_grasp_pts, predictor=predictor)
				####best_orns, orn_scores = self.getBestOrns(masks, cand_grasp_pts)
				#print('Best orn index length: ' + str(len(best_orn_index)))
				#print(best_orn_index)


			best_orns_index = self.getBestOrnsIdeal(mask_capture_tens, masks, cand_grasp_pts, is_greedy=True)
			best_orns = np.array([best_orns_index])*m.pi/self.orn_granularity

			cand_pts_seq[self.num_attempts]=cand_grasp_pts
			cand_orns_seq[self.num_attempts]=best_orns

			grasps_to_execute = np.concatenate((cand_grasp_pts, best_orns.T), axis=1)
			#print(orn_scores)
			#self.showMasksAndGraspsFullOnImage(masks, rgb_image, grasps_to_execute)

			#print('Orientations chosen...')

			orig_nbrhd = []

			for c_g_p_index in range(len(cand_grasp_pts)):
				curr_grasp_pt = cand_grasp_pts[c_g_p_index]
				#print(curr_grasp_pt)
				orig_nbrhd.append(self.graspNeighborhood(rgb_image, curr_grasp_pt))

			if save:
				ts_after_compute = timeit.default_timer()
				compute_time_seq.append(ts_after_compute-ts_before_compute)


			#grasp_executed=False

			for c_g_p_index in range(len(cand_grasp_pts)):
				curr_grasp_pt = cand_grasp_pts[c_g_p_index]
				#[UPDATE RGB IMAGE!]
				
				grasp_nbrhd = self.graspNeighborhood(new_image, curr_grasp_pt)
				#plt.imshow(grasp_nbrhd)
				#plt.show()

				#print(best_orns)
				print(curr_grasp_pt)
				opt_pick_orn = best_orns[0][c_g_p_index]
				#print(opt_pick_orn)
				if opt_pick_orn < -m.pi:
					opt_pick_orn += m.pi
				if opt_pick_orn > m.pi:
					opt_pick_orn += -m.pi
				#opt_pick_orn = min(opt_pick_orn, m.pi/2)
				#opt_pick_orn = max(opt_pick_orn, -m.pi/2)


				#if not grasp_executed:
					# conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
					# cloth_map = self.getClothMap(new_image)

					# #plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					# #plt.imshow(cloth_map)
					# #plt.show()

					# cloth_map_new = 1-self.removeHoles((1-cloth_map))

					# #plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					# #plt.imshow(cloth_map_new)
					# #plt.show()

					# cloth_map_new = self.removeHoles(cloth_map)

					# #plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					# #plt.imshow(cloth_map_new)
					# #plt.show()

					# #cloth_map_exp = (convolve(cloth_map_new, conv_mat) > 0)

					# #plt.scatter(cand_grasp_pts[:,0], cand_grasp_pts[:,1], c='blue', marker='o')
					# #plt.imshow(cloth_map_new)
					# #plt.show()

					# print("Curr grasp point is...")
					# print([curr_grasp_pt[0], curr_grasp_pt[1]])

					# #print(np.shape(cloth_map_new))
					# if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
					# #if True:
					# 	print('Executing grasp...')
					# 	grasp_executed=True
					# 	self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
					# 	self.trips += 1
						
					# 	if save:
					# 		time.sleep(3)
					# 		weight = self.recordWeight(0)
					# 		weight_seq.append(weight)
					# 		now_ts = timeit.default_timer()
					# 		time_seq.append(now_ts-start_ts)
					# 		rgb_image_seq.append(new_image)
					# 		depth_image_seq.append(depth_image)
							
					# 		masks_seq.append(masks)
					
					# 		masks_raw_seq.append(masks_raw)

					# 		pick_seq.append([curr_grasp_pt, opt_pick_orn])
					# 		if new_compute:
					# 			new_compute_seq.append(1)
					# 		else:
					# 			new_compute_seq.append(0)
					# 		new_compute = False
					# 		#cloth_map = self.getClothMap(new_image)
					# 		bw_image_seq.append(cloth_map)

				if self.validityChecker(orig_nbrhd[c_g_p_index], grasp_nbrhd):
					
					#print(cand_grasp_pts)

					#plt.scatter(cand_grasp_pts[:,0], cand_grasp_pts[:,1], c='blue', marker='o')
					#plt.imshow(new_image)
					#plt.show()

					conv_mat = np.ones([self.KERNEL_SIZE, self.KERNEL_SIZE])
					cloth_map = self.getClothMap(new_image)

					#plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					#plt.imshow(cloth_map)
					#plt.show()

					cloth_map_new = 1-self.removeHoles((1-cloth_map))

					#plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					#plt.imshow(cloth_map_new)
					#plt.show()

					cloth_map_new = self.removeHoles(cloth_map)

					#plt.scatter(curr_grasp_pt[0], curr_grasp_pt[1], c='red', marker='o')
					#plt.imshow(cloth_map_new)
					#plt.show()

					#cloth_map_exp = (convolve(cloth_map_new, conv_mat) > 0)

					#plt.scatter(cand_grasp_pts[:,0], cand_grasp_pts[:,1], c='blue', marker='o')
					#plt.imshow(cloth_map_new)
					#plt.show()

					#print(np.shape(cloth_map_new))
					if cloth_map_new[curr_grasp_pt[1], curr_grasp_pt[0]] == 0:
					#if True:
						print('Executing grasp...')
						grasp_executed=True
						self.graspAtOptPt([curr_grasp_pt[1], curr_grasp_pt[0]], opt_pick_orn)
						self.trips += 1
						
						if save:
							time.sleep(3)
							weight = self.recordWeight(0)
							weight_seq.append(weight)
							now_ts = timeit.default_timer()
							time_seq.append(now_ts-start_ts)
							rgb_image_seq.append(new_image)
							depth_image_seq.append(depth_image)
							
							masks_seq.append(masks)
					
							masks_raw_seq.append(masks_raw)

							pick_seq.append([curr_grasp_pt, opt_pick_orn])
							if new_compute:
								new_compute_seq.append(1)
							else:
								new_compute_seq.append(0)
							new_compute = False
							#cloth_map = self.getClothMap(new_image)
							bw_image_seq.append(cloth_map)
						

				rs_color_image, rs_scaled_depth_image, aligned_depth_frame = get_rs_image(self.pipeline, self.align, self.depth_scale, use_depth=False)

				# Use x_crop and y_crop to get workspace image
				new_image = rs_color_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]
				#color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		
			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image, bw_image, depth_image)
			num_clusters = len(clusters)

			self.num_attempts += 1

		#cand_orns_seq = np.array(cand_orns_seq)
		#cand_pts_seq = np.array(cand_pts_seq)

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq


	def laundry_new_algo(self, viz_policy=False, use_stack=False, use_fold=False, pick_method='random', save=True, multipick=False):

		self.WEIGHT_POLICY_THRESH = 100
		self.VOLUME_POLICY_THRESH = 100
		self.HEIGHT_POLICY_THRESH = 100
		self.CENTROID_POLICY_THRESH = 100

		if pick_method == 'weight':
			self.PICK_PARAMETER_THRESH = self.WEIGHT_POLICY_THRESH
		elif pick_method == 'height':
			self.PICK_PARAMETER_THRESH = self.HEIGHT_POLICY_THRESH
		elif pick_method == 'volume':
			self.PICK_PARAMETER_THRESH = self.VOLUME_POLICY_THRESH
		elif pick_method == 'centroid':
			self.PICK_PARAMETER_THRESH = self.CENTROID_POLICY_THRESH
		elif pick_method == 'random':
			self.PICK_PARAMETER_THRESH = 100
		elif pick_method == 'segmentation':
			self.mask_generator = self.loadSAM()
		elif pick_method == 'segmentation_and_height':			
			self.mask_generator = self.loadSAM()
			self.PICK_PARAMETER_THRESH = self.HEIGHT_POLICY_THRESH
		elif pick_method == "segmentation_and_consolidation":
			self.mask_generator = self.loadSAM()
			self.PICK_PARAMETER_THRESH = self.HEIGHT_POLICY_THRESH
		else:
			print ('Pick method is invalid')

		self.PICK_PARAMETER_THRESH = 0
		# cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		# # Get all clusters
		# clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
		# 						bw_image, depth_image)
		# num_clusters = len(clusters)

		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop

		self.num_attempts = 0

		self.trips = 0
		num_fold_actions = 0
		num_stack_actions = 0

		num_clusters = 1

		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
		# subprocess.call(['mkdir', '-p', self.trial_path])

		weight_seq = [0] 
		time_seq = [0] 
		rgb_image_seq = [] 
		bw_image_seq = [] 
		pick_seq = []
		depth_image_seq = []

		start_time = timeit.default_timer()

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:

			# Update state
			if pick_method == 'weight':
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False, is_weight=True)
				rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
				# Get all clusters
				clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
										bw_image, depth_image)
				num_clusters = len(clusters)
			else:
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
				rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
				# Get all clusters
				clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
										bw_image, depth_image)
				num_clusters = len(clusters)

			# print ('Number of cloth points', len(cloth_points))

			if num_clusters == 0:
				break

			if use_fold:
				print ('Attempting fold actions')
				all_fold_lists, all_fold_actions = self.inspect_clusters(clusters, rgb_image, pick_method=pick_method, save=save)
				pdb.set_trace()
				print("all_fold_lists: ", all_fold_lists)
				print("all_fold_actions: ", all_fold_actions)
				for cluster_ind in range(num_clusters):
					cluster = clusters[cluster_ind]
					num_folded = 0

					if pick_method == 'weight':
						pick_parameter = cluster['pred_weight']
					elif pick_method == 'height':
						pick_parameter = cluster['max_height']
					elif pick_method == 'volume':
						pick_parameter = cluster['max_volume']
					elif pick_method == 'centroid':
						pick_parameter = 1000
					elif pick_method == 'random':
						pick_parameter = 0 #TODO
					else:
						print ('Invalid pick method')

					fold_list = all_fold_lists[cluster_ind]
					fold_actions = all_fold_actions[cluster_ind]

					if fold_list[0]:
						print ('Executing first fold')
						self.fold_cluster(cloth_points, fold_actions[0])
						folded_clothes = True
						num_fold_actions += 1
						num_folded += 1


					if fold_list[1]:
						print ('Executing second fold')
						self.fold_cluster(cloth_points, fold_actions[1])
						folded_clothes = True
						num_folded += 1
						num_fold_actions += 1


					if num_folded > 0:
						# Place point is same for either fold action.
						# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
						# subprocess.call(['mkdir', '-p', self.trial_path])
						# init_action_weight = self.recordWeight(0)
						self.basket_trip([fold_actions[0][1], fold_actions[0][2]])
						# time.sleep(3)
						# final_action_weight = self.recordWeight(0)

						# np.save(self.trial_path+'init_action_weight', init_action_weight)
						# np.save(self.trial_path+'final_action_weight', final_action_weight)

						self.trips += 1
						print ('Should continue from for loop')
						continue

				if num_folded > 0:
					print ('Continue after at least a single fold')
					continue

			if use_stack:
				pdb.set_trace()
				print ('Attempting stack actions')
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
				# Get all clusters
				clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
										bw_image, depth_image)
				num_clusters = len(clusters)
				if num_clusters <= 1:
					break
				stack_actions, opt_param, opt_im_pick_pt, opt_pick_orn, other_cluster_params = self.gen_stack_actions(clusters, pick_method=pick_method)
				did_stack = False
				if len(stack_actions) > 0:
					c_im = cv2.circle(rgb_image, [opt_im_pick_pt[1], opt_im_pick_pt[0]], self.IM_CIRCLE_P_1, (0, 255, 0), self.IM_CIRCLE_P_2)
					np.save(self.trial_path+'opt_stack_circ', [opt_im_pick_pt[1], opt_im_pick_pt[0], opt_pick_orn])
				cluster_ind = 0

				stack_index = 0
				for stack_action in stack_actions:
					did_stack = True
					stack_param = other_cluster_params[cluster_ind]
					print ('Executing stack action ....')
					# print ('Stack action', stack_action)
					# print ('Stack param', stack_param)
					cluster_ind += 1
					if save:
						c_im_2 = cv2.circle(c_im, [stack_action[0][1], stack_action[0][0]], self.IM_CIRCLE_P_1, (0, 255, 0), 4)
						np.save(self.trial_path+'stack_circ_{}'.format(stack_index), [stack_action[0][1], stack_action[0][0], stack_action[2]])

						# cv2.imshow('Stack action', c_im_2)
						# cv2.waitKey()

					self.execute_stack_action(stack_action)
					# num_rearrangement_actions += 1
					num_stack_actions += 1
					stack_index += 1


				if did_stack:
					if save:
						cv2.imwrite(self.trial_path+'stacks.jpg', c_im_2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

					rw_x_pick, rw_y_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
					pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
					place_pt = self.gen_place_point()

					# init_action_weight = self.recordWeight(0)

					self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)

					# time.sleep(3)
					# final_action_weight = self.recordWeight(0)
					# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
					# subprocess.call(['mkdir', '-p', self.trial_path])
					# np.save(self.trial_path+'init_action_weight', init_action_weight)
					# np.save(self.trial_path+'final_action_weight', final_action_weight)

					self.trips += 1
					# print ('Continue from stack action')
					continue

			min_height = 50
			if not use_fold and not use_stack:
				print ('Attempting no fold and no stack pick')
				weight_params, volume_params, height_params = self.get_optimal_cluster(clusters)
				if pick_method == 'weight':
					print('Selecting max weight point')
					opt_im_pick_pt = weight_params[1]
					opt_pick_orn = weight_params[2]
				elif pick_method == 'height':
					print('Selecting high point')
					opt_im_pick_pt = height_params[1]
					opt_pick_orn = height_params[2]
				elif pick_method == 'volume':
					print('Selecting max volume point')
					opt_im_pick_pt = volume_params[1]
					opt_pick_orn = volume_params[2]
				elif pick_method == 'segmentation_and_height':
					if height_params[0] > min_height:
						opt_im_pick_pt = height_params[1]
						opt_pick_orn = height_params[2]
					else:
						# TODO
						weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq, masks_seq, masks_raw_seq, cand_pts_seq, cand_orns_seq, new_compute_seq, compute_time_seq = self.laundry_seg_algo() 
				# elif pick_method == 'segmentation_and_consolidation':

				elif pick_method == 'random':
					x = clusters[0]['cluster_pts']
					#print(len(x))
					
					opt_im_pick_pt = x[np.random.randint(len(x))]
					#print(opt_im_pick_pt)
					opt_pick_orn = random.uniform(-m.pi/2, m.pi/2)


					if self.show_steps:
						plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
						#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
						plt.imshow(rgb_image)
						plt.show()

					#plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#plt.imshow(bw_image)
					#plt.show()

				else:
					print ('Invalid pick method')
					pick_seq.append([opt_im_pick_pt, opt_pick_orn])
					

					#c_im = cv2.circle(rgb_image, [opt_im_pick_pt[1], opt_im_pick_pt[0]], self.IM_CIRCLE_P_1, (0, 255, 0), 4)
					# cv2.imshow('Pick method {}'.format(pick_method), c_im)
					# cv2.waitKey()
					#np.save(self.trial_path+'no_stack_no_fold', [opt_im_pick_pt[1], opt_im_pick_pt[0], opt_pick_orn])
					#cv2.imwrite(self.trial_path+'direct_pick.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

				# Grasp at optimal param point
				# rw_x_pick, rw_y_pick = self.depth_lookup_table[opt_im_pick_pt[0], opt_im_pick_pt[1]]
				#print(opt_im_pick_pt)

				# pdb.set_trace()
				print("Optimal pick point in image: " + str(opt_im_pick_pt))
				pick_pt_depth = depth_image[int(opt_im_pick_pt[0]), int(opt_im_pick_pt[1])] * 1000
				if self.use_pick_pt_depth:
					print("depth: " + str(pick_pt_depth))
					print("height: " + str(800 - pick_pt_depth))
					print("pick point in image:" + str(opt_im_pick_pt))
					rw_pick = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]], pick_pt_depth)
				else:
					rw_pick = self.ir.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])

				rw_x_pick = rw_pick[0][0]
				rw_y_pick = rw_pick[0][1]

				if self.show_steps:
					plt.plot((444/240)*opt_im_pick_pt[1], (575/300)*opt_im_pick_pt[0], marker='o', color="red")
					#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
					plt.imshow(rgb_image)
					plt.show()

				

				#print([rw_x_pick, rw_y_pick])
				pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
				place_pt = self.gen_place_point()

				
				self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)
				self.wrist_unwinder()

				if save:
					time.sleep(3)
					weight = self.recordWeight(0)
					weight_seq.append(weight)
					
					curr_time = timeit.default_timer()
					time_seq.append(curr_time - start_time)
					bw_image_seq.append(bw_image)
					rgb_image_seq.append(rgb_image)
					depth_image_seq.append(depth_image)
					pick_seq.append([opt_im_pick_pt, opt_pick_orn])
					

				# time.sleep(3)
				# final_action_weight = self.recordWeight(0)
				# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
				# subprocess.call(['mkdir', '-p', self.trial_path])
				# np.save(self.trial_path+'pick_pt'.format(self.trips), pick_pt)
				# np.save(self.trial_path+'place_pt'.format(self.trips), place_pt)
				# np.save(self.trial_path+'init_action_weight'.format(self.trips), init_action_weight)
				# np.save(self.trial_path+'final_action_weight'.format(self.trips), final_action_weight)

				self.trips += 1
				# print ('Continue from no fold and no stack')
				continue
			# if use_fold and use_stack and stack_actions == []
			self.num_attempts += 1

		# sample_total_time = timeit.default_timer() - start_time

		print ('Total number of trips', self.trips)
		print ('Total number of fold actions', num_fold_actions)
		print ('Total number of stack actions', num_stack_actions)
		self.num_rearrangements = num_stack_actions + num_fold_actions
		# weight_seq, time_seq, rgb_image_seq, bw_image_seq, pick_seq

		return weight_seq, time_seq, rgb_image_seq, bw_image_seq, depth_image_seq, pick_seq
			


	def laundry_algo(self, viz_policy=False, use_stack=True, use_fold=True, pick_method='weight', save=False, multipick=False):

		self.WEIGHT_POLICY_THRESH = 100
		self.VOLUME_POLICY_THRESH = 100
		self.HEIGHT_POLICY_THRESH = 100
		self.CENTROID_POLICY_THRESH = 100

		if pick_method == 'weight':
			self.PICK_PARAMETER_THRESH = self.WEIGHT_POLICY_THRESH
		elif pick_method == 'height':
			self.PICK_PARAMETER_THRESH = self.HEIGHT_POLICY_THRESH
		elif pick_method == 'volume':
			self.PICK_PARAMETER_THRESH = self.VOLUME_POLICY_THRESH
		elif pick_method == 'centroid':
			self.PICK_PARAMETER_THRESH = self.CENTROID_POLICY_THRESH
		elif pick_method == 'random':
			self.PICK_PARAMETER_THRESH = 100
		elif pick_method == 'segmentation':
			self.mask_generator = self.loadSAM()
		else:
			print ('Pick method is invalid')

		self.PICK_PARAMETER_THRESH = 0
		# cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		# # Get all clusters
		# clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
		# 						bw_image, depth_image)
		# num_clusters = len(clusters)
		MAX_NUM_ATTEMPTS = 100 # Avoid infinite loop
		self.num_attempts = 0

		self.trips = 0
		num_fold_actions = 0
		num_stack_actions = 0

		num_clusters = 1

		self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
		# subprocess.call(['mkdir', '-p', self.trial_path])

		

		while num_clusters > 0 and self.num_attempts < MAX_NUM_ATTEMPTS:

			# Update state
			cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
			rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
			# Get all clusters
			clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
									bw_image, depth_image)
			num_clusters = len(clusters)

			# print ('Number of cloth points', len(cloth_points))

			if num_clusters == 0:
				break

			if use_fold:
				print ('Attempting fold actions')
				all_fold_lists, all_fold_actions = self.inspect_clusters(clusters, rgb_image, pick_method=pick_method, save=save)

				for cluster_ind in range(num_clusters):
					cluster = clusters[cluster_ind]
					num_folded = 0

					if pick_method == 'weight':
						pick_parameter = cluster['pred_weight']
					elif pick_method == 'height':
						pick_parameter = cluster['max_height']
					elif pick_method == 'volume':
						pick_parameter = cluster['max_volume']
					elif pick_method == 'centroid':
						pick_parameter = 1000
					elif pick_method == 'random':
						pick_parameter = 0 #TODO
					else:
						print ('Invalid pick method')

					fold_list = all_fold_lists[cluster_ind]
					fold_actions = all_fold_actions[cluster_ind]

					if fold_list[0]:
						print ('Executing first fold')
						self.fold_cluster(cloth_points, fold_actions[0])
						folded_clothes = True
						num_fold_actions += 1
						num_folded += 1


					if fold_list[1]:
						print ('Executing second fold')
						self.fold_cluster(cloth_points, fold_actions[1])
						folded_clothes = True
						num_folded += 1
						num_fold_actions += 1


					if num_folded > 0:
						# Place point is same for either fold action.
						# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
						# subprocess.call(['mkdir', '-p', self.trial_path])
						# init_action_weight = self.recordWeight(0)
						self.basket_trip([fold_actions[0][1], fold_actions[0][0]])
						# time.sleep(3)
						# final_action_weight = self.recordWeight(0)

						# np.save(self.trial_path+'init_action_weight', init_action_weight)
						# np.save(self.trial_path+'final_action_weight', final_action_weight)

						self.trips += 1
						print ('Should continue from for loop')
						continue

				if num_folded > 0:
					print ('Continue after at least a single fold')
					continue

			if use_stack:
				print ('Attempting stack actions')
				cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
				# Get all clusters
				clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
										bw_image, depth_image)
				num_clusters = len(clusters)
				if num_clusters == 0:
					break
				stack_actions, opt_param, opt_im_pick_pt, opt_pick_orn, other_cluster_params = self.gen_stack_actions(clusters, pick_method=pick_method)
				did_stack = False
				if len(stack_actions) > 0:
					c_im = cv2.circle(rgb_image, [opt_im_pick_pt[1], opt_im_pick_pt[0]], self.IM_CIRCLE_P_1, (0, 255, 0), self.IM_CIRCLE_P_2)
					np.save(self.trial_path+'opt_stack_circ', [opt_im_pick_pt[1], opt_im_pick_pt[0], opt_pick_orn])
				cluster_ind = 0

				stack_index = 0
				for stack_action in stack_actions:
					did_stack = True
					stack_param = other_cluster_params[cluster_ind]
					print ('Executing stack action ....')
					# print ('Stack action', stack_action)
					# print ('Stack param', stack_param)
					cluster_ind += 1
					if save:
						c_im_2 = cv2.circle(c_im, [stack_action[0][1], stack_action[0][0]], self.IM_CIRCLE_P_1, (0, 255, 0), 4)
						np.save(self.trial_path+'stack_circ_{}'.format(stack_index), [stack_action[0][1], stack_action[0][0], stack_action[2]])

						# cv2.imshow('Stack action', c_im_2)
						# cv2.waitKey()

					self.execute_stack_action(stack_action)
					# num_rearrangement_actions += 1
					num_stack_actions += 1
					stack_index += 1


				if did_stack:
					if save:
						cv2.imwrite(self.trial_path+'stacks.jpg', c_im_2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

					[rw_x_pick, rw_y_pick] = self.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])
					pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
					place_pt = self.gen_place_point()

					# init_action_weight = self.recordWeight(0)

					self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)

					# time.sleep(3)
					# final_action_weight = self.recordWeight(0)
					# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
					# subprocess.call(['mkdir', '-p', self.trial_path])
					# np.save(self.trial_path+'init_action_weight', init_action_weight)
					# np.save(self.trial_path+'final_action_weight', final_action_weight)

					self.trips += 1
					# print ('Continue from stack action')
					continue


			if not use_fold and not use_stack:
				print ('Attempting no fold and no stack pick')
				weight_params, volume_params, height_params = self.get_optimal_cluster(clusters)
				if pick_method == 'weight':
					opt_im_pick_pt = weight_params[1]
					opt_pick_orn = weight_params[2]
				elif pick_method == 'height':
					opt_im_pick_pt = height_params[1]
					opt_pick_orn = height_params[2]
				elif pick_method == 'volume':
					opt_im_pick_pt = volume_params[1]
					opt_pick_orn = volume_params[2]
				elif pick_method == 'random':
					x = clusters[0]['cluster_pts']
					#print(len(x))
					
					opt_im_pick_pt = x[np.random.randint(len(x))]
					#print(opt_im_pick_pt)
					opt_pick_orn = random.uniform(0, m.pi)

					plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
					plt.imshow(rgb_image)
					plt.show()

					#plt.plot(opt_im_pick_pt[1], opt_im_pick_pt[0], marker='o', color="red")
					#plt.imshow(bw_image)
					#plt.show()

				else:
					print ('Invalid pick method')

				if save:
					c_im = cv2.circle(rgb_image, [opt_im_pick_pt[1], opt_im_pick_pt[0]], self.IM_CIRCLE_P_1, (0, 255, 0), 4)
					# cv2.imshow('Pick method {}'.format(pick_method), c_im)
					# cv2.waitKey()
					np.save(self.trial_path+'no_stack_no_fold', [opt_im_pick_pt[1], opt_im_pick_pt[0], opt_pick_orn])
					cv2.imwrite(self.trial_path+'direct_pick.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

				# Grasp at optimal param point
				# rw_x_pick, rw_y_pick = self.depth_lookup_table[opt_im_pick_pt[0], opt_im_pick_pt[1]]
				rw_pick = self.ir.image_pt_to_rw_pt([opt_im_pick_pt[1], opt_im_pick_pt[0]])

				rw_x_pick = rw_pick[0][0]
				rw_y_pick = rw_pick[0][1]

				#print([rw_x_pick, rw_y_pick])
				pick_pt = [rw_x_pick, rw_y_pick, self.z_grasp]
				place_pt = self.gen_place_point()

				init_action_weight = self.recordWeight(0)
				self.pick_and_place(pick_pt, place_pt, opt_pick_orn, self.translation_z, use_shake_action=False)

				# time.sleep(3)
				# final_action_weight = self.recordWeight(0)
				# self.trial_path = self.data_path + 'trial_{}/'.format(self.trips)
				# subprocess.call(['mkdir', '-p', self.trial_path])
				# np.save(self.trial_path+'pick_pt'.format(self.trips), pick_pt)
				# np.save(self.trial_path+'place_pt'.format(self.trips), place_pt)
				# np.save(self.trial_path+'init_action_weight'.format(self.trips), init_action_weight)
				# np.save(self.trial_path+'final_action_weight'.format(self.trips), final_action_weight)

				self.trips += 1
				# print ('Continue from no fold and no stack')
				continue

			self.num_attempts += 1

		# sample_total_time = timeit.default_timer() - start_time

		print ('Total number of trips', self.trips)
		print ('Total number of fold actions', num_fold_actions)
		print ('Total number of stack actions', num_stack_actions)

		return self.trips, num_fold_actions, num_stack_actions

	def laundry_net_stack_fold_policy(self, viz_policy=False):

		self.laundry_net_stack_policy(viz_policy=viz_policy, use_fold=True)

	def laundry_net_fold_policy(self, viz_policy=False, save=False):

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)

		pick_point, pick_orn, im_pt, pred_weight = self.laundry_net_policy(rgb_image, cloth_points, cloth_depths, depth_image, viz_policy=viz_policy)
		place_pt = self.gen_place_point()
		self.pick_and_place_controlled(pick_point, place_pt, pick_orn, self.translation_z, save=save)

		return pick_point, pick_orn, im_pt, pred_weight

	def get_pick_depth(self, depth_at_pick_pt):

		table_surface_z = self.z_grasp
		cam_to_point = depth_at_pick_pt

		h_delta = self.CAM_TO_TABLE_SURFACE - cam_to_point
		pick_start_pt = table_surface_z + h_delta
		pick_z = pick_start_pt - self.MAX_HEAP_DEFORMATION

		# Clip the depth values
		if pick_z < self.z_grasp:
			pick_z = self.z_grasp

		if pick_z > self.translation_z:
			pick_z = self.translation_z

		return self.z_grasp

	def pick_and_place_controlled(self, pick_point, place_point, grasp_orn, translation_z, use_shake_action=False, save=False, depth_at_pick_pt=1.1):

		# Open gripper

		self.s_gripper.sendall(b'SET POS 0\n')
		time.sleep(self.open_gripper_time)

		x_pick, y_pick, z_pick = pick_point
		x_place, y_place, z_place = place_point

		# Above pick point
		xyz_above_pick = [x_pick, y_pick, translation_z]
		orn_above_pick = grasp_orn

		# Pick point
		xyz_pick = [x_pick, y_pick, z_pick]
		orn_pick = grasp_orn

		# Above place point
		xyz_above_place = [x_place, y_place, translation_z]
		orn_above_place = grasp_orn

		# Place point
		xyz_place = [x_place, y_place, z_place]
		orn_place = grasp_orn

		self.move_ur5py(xyz_above_pick, orn_above_pick, self.joint_speed, self.joint_acc)
		self.move_ur5py(xyz_pick, orn_pick, self.joint_speed, self.joint_acc)

		# Close gripper
		self.s_gripper.sendall(b'SET POS 255\n')
		time.sleep(self.close_gripper_time)

		self.move_ur5py(xyz_above_pick, orn_above_pick, self.joint_speed, self.joint_acc)

		if use_shake_action:
			shake_point = deepcopy(xyz_above_pick)
			shake_orn = grasp_orn
			self.shake_action(shake_point, shake_orn, num_shake_actions=3)


		# Execute second phase only if check point conditions are satisfied

		self.long_cloth_policy(orn_above_pick, save=save)

		return None

	def create_window(self, rgb_image, window_width, window_height, viz_method=False):
		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		(thresh, bw_image) = cv2.threshold(grayscale_image, self.IM_THRESH_LOW, self.IM_THRESH_HIGH, cv2.THRESH_BINARY)

		if viz_method:
			cv2.imshow('passed before window ', bw_image)
			cv2.waitKey()
		for im_x in range(0, window_width):
			for im_y in range(0, window_height):
				bw_image[im_y, im_x] = 255
		(thresh, bw_image) = cv2.threshold(bw_image, self.IM_THRESH_LOW, self.IM_THRESH_HIGH, cv2.THRESH_BINARY_INV)

		if viz_method:
			cv2.imshow('passed in ', bw_image)
			cv2.waitKey()

		return bw_image

	def closest_contour_to_edge(self, rgb_image, bw_image, width, height, viz_method=False):
		# check how many contours are in that window
		contours, centers, cnt_radius, bounding_rectangles = self.cluster_detector(bw_image, window=True)

		# # Get contour center
		# IPython.embed()

		if len(contours) == 0:
			return None

		if len(contours)!=0:
			valid_cnt_centers = []
			CNT_CENTER_LIMIT = 575 - self.PROXIMITY_THRESHOLD/2.
			PIXEL_THRESH = 3
			for cnt_center in centers:
				# print ('edge eps', abs(cnt_center[0] - CNT_CENTER_LIMIT))
				if abs(cnt_center[0] - CNT_CENTER_LIMIT) < PIXEL_THRESH:
					# Valid contour
					valid_cnt_centers.append(cnt_center)

			if len(valid_cnt_centers) == 0:
				return None

		# choose the contour whose center is closed to the middle edge
		edge_center = [width, height/2]
		dist_list = []
		for i in centers:
			dist = distance.euclidean(edge_center, i)
			dist_list.append([dist, i])
		dist_list.sort(key = lambda x: x[0])
		closest_contour_center = dist_list[0][1]

		if viz_method:
			# print("=============CENTERS=============", centers)
			for i in range(len(centers)):
				cv2.circle(rgb_image, centers[i], self.IM_CIRCLE_P_1, (100, 0, 50), self.IM_CIRCLE_P_2)
				cv2.rectangle(rgb_image, (bounding_rectangles[i][0], bounding_rectangles[i][1]), (bounding_rectangles[i][2] + bounding_rectangles[i][0], bounding_rectangles[i][3] + bounding_rectangles[i][1]), (10*i,255,10*i), 1)
				cv2.imshow('Window Clusters', rgb_image)
				cv2.waitKey()
			cv2.circle(rgb_image, closest_contour_center, self.IM_CIRCLE_P_1, (100, 0, 50), self.IM_CIRCLE_P_2)
			cv2.imshow('Edge cluster Center', rgb_image)
			cv2.waitKey()
		# return that contours center
		return closest_contour_center

	def check_trailing_cloth(self, rgb_image, proximity_threshold = None, viz_method=False, save=False):
		# find window
		(image_height, image_width, channels) = rgb_image.shape
		if (proximity_threshold is None):
			proximity_threshold = self.PROXIMITY_THRESHOLD
		window_width = image_width - proximity_threshold
		bw_image = self.create_window(rgb_image, window_width, image_height, viz_method=viz_method)
		# find the contour's center whose trailing cloth it belongs to
		closest_contour_center = self.closest_contour_to_edge(rgb_image, bw_image, image_width, image_height)
		if closest_contour_center is None:
			return None, [], []
		grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
		# find the pca_points of that contour
		pca_points, contours, centers, cnt_radius, bounding_rectangles = self.pca_cluster(grayscale_image)

		if len(contours)>0:

			dist_list = []
			for i in range(len(centers)):
				cv2.circle(rgb_image, centers[i], self.IM_CIRCLE_P_1, (100, 0, 50), self.IM_CIRCLE_P_2)
				dist = distance.euclidean(closest_contour_center, centers[i])
				dist_list.append([dist, centers[i], pca_points[i]])
			dist_list.sort(key = lambda x: x[0])

			if len(dist_list) > 0:
				corresponding_cluster_center = dist_list[0][1]
				corresponding_cluster_pca_pts = dist_list[0][2]
			else:
				corresponding_cluster_center = []
				corresponding_cluster_pca_pts = []

			if viz_method:
				cv2.circle(rgb_image, corresponding_cluster_center, 5, (255, 0, 255), self.IM_CIRCLE_P_2)

			if save and len(corresponding_cluster_center) >0:
				circ = cv2.circle(rgb_image, corresponding_cluster_center, 5, (255, 0, 255), self.IM_CIRCLE_P_2)
				cv2.imwrite(self.trial_path+'laundry_trail_rgb.jpg', circ, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

			# find the furthest pca point from the center
			gamma = 120
			valid_pca_grasps = []
			for i in corresponding_cluster_pca_pts:
				end_1 = i[0].tolist()
				end_2 = i[1].tolist()
				dist_from_center_1 = distance.euclidean(closest_contour_center, end_1)
				dist_from_center_2 = distance.euclidean(closest_contour_center, end_2)
				valid_pca_grasps.append([dist_from_center_1, i[0]])
				valid_pca_grasps.append([dist_from_center_2, i[1]])
			valid_pca_grasps.sort(key = lambda x: x[0], reverse=True)
			dist_furthest_pca_pt = valid_pca_grasps[0][0]

			furthest_pca_pt = valid_pca_grasps[0][1]

			if dist_furthest_pca_pt >= gamma:
				# furthest_pca_pt = valid_pca_grasps[0][1]
				pt = furthest_pca_pt.tolist()
				if viz_method:
					cv2.circle(rgb_image, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), self.IM_CIRCLE_P_2)

				if save:
					# Add center and pca
					circ = cv2.circle(rgb_image, corresponding_cluster_center, 5, (255, 0, 255), self.IM_CIRCLE_P_2)
					pca_circ = cv2.circle(circ, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), self.IM_CIRCLE_P_2)
					cv2.imwrite(self.trial_path+'laundry_trail_pca_rgb.jpg', pca_circ, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

			else:
				# furthest_pca_pt = valid_pca_grasps
				pass

			if viz_method:
				cv2.imshow('FINAL', rgb_image)
				cv2.waitKey()

			# if save:
			# 	circ = cv2.circle(rgb_image, corresponding_cluster_center, 5, (255, 0, 255), self.IM_CIRCLE_P_2)
			# 	cv2.imwrite(self.trial_path+'laundry_trail_rgb.jpg', circ, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
		else:
			corresponding_cluster_center = None
			furthest_pca_pt = []
			dist_furthest_pca_pt = []

		return corresponding_cluster_center, furthest_pca_pt, dist_furthest_pca_pt

	def test_window(self):
		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		self.match_edge_contour(rgb_image)

	def long_cloth_policy(self, curr_orn, save=False):

		self.move_ur5py(self.CHECKPOINT, curr_orn, self.joint_speed, self.joint_acc)

		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		#
		# try:
		# 	plt.scatter(np.array(cloth_points)[:,0], np.array(cloth_points)[:,1])
		# 	plt.show()
		# except:
		# 	print ('could not plot ')

		cluster_centroid_im_pt, furthest_pca_im_pt, dist_furthest_pca_pt = self.check_trailing_cloth(rgb_image, save=save, viz_method=False)

		bin_place_point = self.gen_place_point()

		# if trailing cloth found
		if cluster_centroid_im_pt is not None:
			# place cloth at centroid
			print ('We have a cluster!')
			cluster_centroid_im_pt_switched = [int(cluster_centroid_im_pt[1]), int(cluster_centroid_im_pt[0])]
			rw_pick = self.ir.image_pt_to_rw_pt(int(cluster_centroid_im_pt_switched[0]), int(cluster_centroid_im_pt_switched[1]))
			rw_x, rw_y = rw_pick[0][0], rw_pick[0][1]
			centroid_pt = [rw_x, rw_y, self.z_grasp]
			grasp_centroid_orn = self.get_pca_grasp_orn(cloth_points, cluster_centroid_im_pt)


			# Move above drop point
			above_centroid_pt = deepcopy(centroid_pt)
			above_centroid_pt[-1] = self.translation_z

			self.drop_at_point(above_centroid_pt, curr_orn)

			# self.move_ur5py(above_centroid_pt, curr_orn, self.joint_speed, self.joint_acc)

			# check if need to grab pca point
			print ('Furthest pca point dist', dist_furthest_pca_pt)

			if dist_furthest_pca_pt >= self.ROBOT_MAX_HEIGHT_PIXELS:
				furthest_pca_im_pt_switched = [int(furthest_pca_im_pt[1]), int(furthest_pca_im_pt[0])]
				rw_pick = self.ir.image_pt_to_rw_pt([int(furthest_pca_im_pt_switched[0]), int(furthest_pca_im_pt_switched[1])])
				rw_x, rw_y = rw_pick[0][0], rw_pick[0][1]
				pca_pick_point = [rw_x, rw_y, self.z_grasp]

				grasp_pca_orn = self.get_pca_grasp_orn(cloth_points, furthest_pca_im_pt)

				# Pick at PCA point and place at Centroid
				self.pick_and_place(pca_pick_point, centroid_pt, grasp_pca_orn, self.translation_z, use_shake_action=False)
				# Pick at centroid point and continue to bin
				# self.pick_and_place(centroid_pt, bin_place_point, grasp_centroid_orn, self.translation_z, use_shake_action=False)

			self.pick_and_place(centroid_pt, bin_place_point, grasp_centroid_orn, self.translation_z, use_shake_action=False)

		else:
			# proceed to drop in bin
			self.drop_at_point(bin_place_point, curr_orn)

	def max_volume_policy(self, cloth_points, cloth_depths, rgb_image, bw_image, depth_image, pick_method='max_height', viz_policy=False, save=False):

		# Pick method is centroid here, could also be max_height
		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
								bw_image, depth_image, pick_method=pick_method)

		volumes = []
		for cluster in clusters:
			cluster_volume = cluster['area']*cluster['avg_height']
			volumes.append(cluster_volume)

		max_ind = np.argmax(volumes)
		best_cluster = clusters[max_ind]


		# Get best cluster props
		pick_point = best_cluster['pick_point']
		pick_orn = best_cluster['pick_orn']
		pick_im_pt = best_cluster['pick_im_pt']

		if viz_policy:
			c_im = cv2.circle(rgb_image, [pick_im_pt[1], pick_im_pt[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			cv2.imshow('MAX volume', c_im)
			cv2.waitKey()

		if save:
			c_im = cv2.circle(rgb_image, [pick_im_pt[1], pick_im_pt[0]], self.IM_CIRCLE_P_1, (0,0,255), self.IM_CIRCLE_P_2)
			cv2.imwrite(self.trial_path+'max_volume_policy_rgb.jpg', c_im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

		return pick_point, pick_orn, pick_im_pt

	def pick_experiments(self, policy='random', experiment_name='no_rearrangement', save=False):
		total_weight = self.recordWeight(0)

		# IPython.embed()
		NUM_SAMPLE_RUNS = 100
		print ('Method:', policy)
		start_pt = 2
		end_pt = start_pt + NUM_SAMPLE_RUNS

		all_total_times = []
		for sample_num in range(start_pt, end_pt):
			print ('Method: ', policy)
			print ('Sample number: ', sample_num)
			ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
			self.data_path = './{}/method_{}/sample_{}/'.format(experiment_name, policy, ts)
			# subprocess.call(['rm', '-rf', self.data_path])
			subprocess.call(['mkdir', '-p', self.data_path])

			print ('Check basket')

			# IPython.embed()

			initial_weight = self.recordWeight(0)

			# Scene reset
			self.reset_scene()
			self.move_above_place_point()
			self.shuffle_placed_clothes_v2()

			sample_start_time = timeit.default_timer()

			if experiment_name == 'data_collection':
				self.clear_table(policy=policy, sample_num=sample_num)
			elif experiment_name == 'no_rearrangement':
				time_sequence, weight_sequence, num_trials = self.no_rearrangement_sample(policy=policy, ts=ts)
			elif experiment_name == 'rearrangement':
				self.laundry_algo(pick_method=policy, save=save)
			else:
				print ('Experiment is undefined!')
				return None
			sample_total_time = timeit.default_timer() - sample_start_time

			time.sleep(3)
			final_weight = self.recordWeight(0)

			np.save(self.data_path+'total_weight', total_weight)

			np.save(self.data_path+'initial_weight', initial_weight)
			np.save(self.data_path+'final_weight', final_weight)
			np.save(self.data_path+'sample_total_time', sample_total_time)
			np.save(self.data_path+'time_sequence', time_sequence)
			np.save(self.data_path+'weight_sequence', weight_sequence)

			print ('Initial load weight', initial_weight)
			print ('Final load weight', final_weight)
			print ('Sample total time', sample_total_time)
			all_total_times.append(sample_total_time)

		return all_total_times

	def test_long_cloth(self):
		orn_val = 0
		self.long_cloth_policy(orn_val)

	def test_points(self):
		# Update state
		cloth_points, cloth_depths, rgb_image, depth_image, bw_image = self.get_state(use_depth=self.depth_flag, viz_image=False)
		# Get all clusters
		clusters = self.get_clusters(cloth_points, cloth_depths, rgb_image,
								bw_image, depth_image)
		num_clusters = len(clusters)

		sets = 'gen' #'generalization'
		pick_methods = ['weight', 'volume', 'height']
		sample_num = 2

		for pick_method in pick_methods:
			save = True
			rgb_to_use = deepcopy(rgb_image)
			self.trial_path = './inspect_results_{}/method_{}/sample_{}/'.format(sets, pick_method, sample_num)
			subprocess.call(['mkdir', '-p', self.trial_path])
			all_fold_lists, all_fold_actions = self.inspect_clusters(clusters, rgb_to_use, pick_method=pick_method, save=save)

		return None

	def test_images(self, trial):
		path = '../Downloads/rearrangement_training_1/rearrangement/method_weight/sample_3/'+ trial
		rgb_image = cv2.imread(path + 'original_rgb.jpg')
		import os
		circ_center = [f for f in os.listdir(path=path) if f.startswith("circ_center")]
		circ_pca = [f for f in os.listdir(path=path) if (f.startswith("circ_") and (not f.startswith('circ_center')))]
		# print(circ_center, circ_pca)
		# circ_points = circ_center + circ_pca
		index0 = 0
		for file in circ_pca:
			p1 = np.load(path + file)
			save = True
			# rgb_to_use = deepcopy(rgb_image)
			color = (0, 255, 0)
			rgb_image = self.create_grasp_marks(p1, rgb_image, index0, color, save=save)
			index0+=1

		index = 0
		for file in circ_center:
			p1 = np.load(path + file)
			save = True
			# rgb_to_use = deepcopy(rgb_image)
			color = (255, 255, 255)
			rgb_image = self.create_grasp_marks(p1, rgb_image, index, color, save=save)
			index+=1
		# self.create_grasp_marks()
		return None

	def demo(self):
		self.joint_speed = 2.0
		self.joint_acc = 2.0
		self.data_path = './demo/'
		#self.laundry_algo(pick_method='segmentation', use_fold=False, use_stack=False, save=False)
		#self.laundry_algo(pick_method='random', save=False)
		self.laundry_algo(pick_method='random', use_stack=False, use_fold=False, save=False)

		return None

	def seg_demo(self):
		self.joint_speed = 2.0
		self.joint_acc = 2.0
		self.data_path = './demo/'
		#self.laundry_algo(pick_method='segmentation', use_fold=False, use_stack=False, save=False)
		#self.laundry_algo(pick_method='random', save=False)
		self.mask_generator = self.loadSAM()
		self.laundry_seg_algo()

		return None

	def process_data_main(self, img, masks, grasp_pts, grasp_orns, size_ = 224, pt_orn_all_combos=True):

		pt_num = len(grasp_pts)
		orn_num = len(grasp_orns)
		mask_num = len(masks)

		img_dims = np.shape(img)[0:2]

		# RETURNED DATA: a 4D tensor, pt_num x orn_num x size_ x size
		# Entry: [chopped_img, [chopped_mask_1, chopped_mask_2, ...]]
		# if mask k doesn't appear in the chopped region, chopped_mask_k is None

		img_mask_pairs = []

		if pt_orn_all_combos:
			for pt_index in range(pt_num):
				to_add_pt = []

				for orn_index in range(orn_num):
					to_add_orn = []
					img_chop = self.proc_data.extend_translate_rotate_cut(self, img, grasp_pts[pt_index], grasp_orns[orn_index], size_)
					mask_chop_seq = []
					mask_chop_indices = []
					for mask_index in range(mask_num):
						grasp_x_min = round(max(grasp_pts[pt_index][0] - m.sqrt(2)*size_, 0))
						grasp_x_max = round(min(grasp_pts[pt_index][0] - m.sqrt(2)*size_, img_dims[0]))
						grasp_y_min = round(max(grasp_pts[pt_index][0] - m.sqrt(2)*size_, 0))
						grasp_y_max = round(min(grasp_pts[pt_index][0] - m.sqrt(2)*size_, img_dims[0]))

						if type(masks[mask_index]) == dict:
							ma = masks[mask_index]['segmentation']
						else:
							ma = masks[mask_index]

						

						if np.sum(np.sum(ma[grasp_x_min:grasp_x_max][grasp_y_min:grasp_y_max])) > 0:
							mask_chop = self.proc_data.extend_translate_rotate_cut(self, ma, grasp_pts[pt_index], grasp_orns[orn_index], size_)
							mask_chop = np.around(mask_chop)

							if np.sum(mask_chop) > 0:
								mask_chop_seq.append(mask_chop)
								mask_chop_indices.append(mask_index)
					
					grasp_pt_orn_labels = [img_chop, mask_chop_seq, mask_chop_indices]
					to_add_orn.append(grasp_pt_orn_labels)
				
				to_add_pt.append(to_add_orn)

			img_mask_pairs.append(to_add_pt)






















def main():
	dmog = DMog()
	#policy = sys.argv[1]
	#dmog.demo()
	#dmog.seg_demo()
	dmog.go_home()
	dmog.full_experiment_pipeline(policy='segmentation', rearrangement='sequence_selected', predictor='baseline', num_samples=1, save=True)

	# no_rearrangement

	#dmog.test_points()
	# dmog.test_images('trial_0/')
	# Rearrangement experiments
	# print("rearrangement Experiments with Random")
	# dmog.pick_experiments(policy='random', experiment_name='rearrangement', save=True)
	# # print ('rearrangement Experiments with Weight')
	# dmog.pick_experiments(policy='weight', experiment_name='rearrangement', save=True)
	# print ('rearrangement Experiments with Volume')
	# dmog.pick_experiments(policy='volume', experiment_name='rearrangement', save=True)
	# print ('rearrangement Experiments with Height')
	# dmog.pick_experiments(policy='height', experiment_name='rearrangement', save=True)

	# # No rearrangement experiments
	# print ('No rearrangement Experiments with Weight')
	# # dmog.pick_experiments(policy='weight', experiment_name='no_rearrangement', save=True)
	# print ('No rearrangement Experiments with Volume')
	# dmog.pick_experiments(policy='volume', experiment_name='no_rearrangement', save=True)
	# print ('No rearrangement Experiments with Height')
	# dmog.pick_experiments(policy='height', experiment_name='no_rearrangement', save=True)

	#print ('No rearrangement Experiments with Random')
	#dmog.pick_experiments(policy='random', experiment_name='no_rearrangement', save=True)

	#print ('No rearrangement Experiments with Segmentation')
	#dmog.pick_experiments(policy='segmentation', experiment_name='no_rearrangement_seg', save=True)

	print ('Yayyyyyyy done with experiments')
	# IPython.embed()

if __name__ == "__main__":
	main()
