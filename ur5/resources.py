import cv2

MIDPOINT_THRESHOLD = 10
#             [minY, maxY, minX, maxX]
CROP_REGION = [250, 843, 286, 1122]
curved_template_mask = cv2.imread('template_masks/curved_channel_mask.png')
curved_template_mask_align = cv2.imread('template_masks/curved_channel_mask_align.png')
straight_template_mask = cv2.imread('template_masks/master_straight_channel_template.png')
straight_template_mask_align = cv2.imread('template_masks/straight_mask.png')
trapezoid_template_mask = cv2.imread('template_masks/trapezoid_channel.png')
trapezoid_template_skeleton =  cv2.imread('template_masks/trapezoid_skeleton.png')