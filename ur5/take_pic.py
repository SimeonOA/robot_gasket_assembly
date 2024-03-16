from utils import *
import cv2
import numpy as np

overhead_cam_id = 22008760 # overhead camera
front_eval_cam_id = 20812520 # front eval camera
side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id)
front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth = setup_zed_camera(front_eval_cam_id)

import matplotlib.pyplot as plt
N = 0
PICK_MODE = 'binary'

f_name = f'evaluation_images/trapezoid/overhead_{N}_{PICK_MODE}.png'
overhead_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
plt.imsave(f_name, overhead_img)
f_name = f'evaluation_images/trapezoid/front_{N}_{PICK_MODE}.png'
front_img = get_zed_img(front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth)
front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
plt.imsave(f_name, front_img)
