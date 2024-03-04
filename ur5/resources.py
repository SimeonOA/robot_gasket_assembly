import cv2
import numpy as np

# [minY, maxY, minX, maxX]
# only works well if you are resizing the image to be 640, by 480
# CROP_REGION = [120, 360, 130, 460]
# only works if you're using the original 1280 by 720 image
# CROP_REGION = [136, 600, 321, 940] 
CROP_REGION = [250, 843, 286, 1122]
# CROP_REGION = [193, 508, 308, 723]
# # curved_template_mask = cv2.imread('template_masks/processed_new_curved_mask.jpg')
# # # curved_template_mask = cv2.imread('template_masks/master_curved_fill_template.png')
# # straight_template_mask = cv2.imread('template_masks/master_straight_channel_template.png')
# # trapezoid_template_mask = cv2.imread('template_masks/master_trapezoid_channel_template.png')
# curved_template_mask = cv2.imread('templates_crop_master/master_curved_channel_template.png')
# # curved_template_mask = cv2.imread('templates_crop_master/master_curved_fill_template.png')
# straight_template_mask = cv2.imread('templates_crop_master/master_straight_channel_template.png')
# trapezoid_template_mask = cv2.imread('templates_crop_master/master_trapezoid_channel_template.png')
# curved_template_mask = cv2.imread('current_masks/fixed_curved_channel.png')
curved_template_mask = cv2.imread('current_masks/curved_channel_mask.png')
curved_template_mask_align = cv2.imread('current_masks/curved_channel_mask_align.png')
straight_template_mask = cv2.imread('current_masks/master_straight_channel_template.png')
# straight_template_mask = cv2.imread('current_masks/my_straight_mask.png')
straight_template_mask_align = cv2.imread('current_masks/my_straight_mask2.png')
trapezoid_template_mask = cv2.imread('current_masks/fixed_trapezoid_channel.png')
trapezoid_template_skeleton = cv2.imread('template_masks/trapezoid_skeleton.png')

MIDPOINT_THRESHOLD = 10