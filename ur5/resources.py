import cv2
import os
curr_dir = os.path.dirname(__file__)
START_IDX = 20
END_IDX = -START_IDX - 1
NUM_PTS_PRESS = 8
NUM_PTS = 4
START_SIDE = 'left'
TOTAL_PICK_PLACE = 5
TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
# height of the templates in meters
TEMPLATE_HEIGHT = {'curved':0.0200, 'straight':0.0127, 'trapezoid':0.0200}
# first elem is curved width/height, second elem is straight width/height, third elem is trapezoid width/height
TEMPLATE_RECTS = [(23.129, 6.6156774806225),(26.5, 2.75), (12, 5.75)]
TEMPLATE_RATIOS = [max(t)/min(t) for t in TEMPLATE_RECTS]
#             [minY, maxY, minX, maxX]
# TODO: fill these in with your values!
CROP_REGION = [250, 843, 286, 1122]
curved_template_mask = cv2.imread(os.path.join(curr_dir,'template_masks/curved_channel_mask.png'))
curved_template_mask_align = cv2.imread(os.path.join(curr_dir,'template_masks/curved_channel_mask_align.png'))
straight_template_mask = cv2.imread(os.path.join(curr_dir,'template_masks/master_straight_channel_template.png'))
straight_template_mask_align = cv2.imread(os.path.join(curr_dir,'template_masks/straight_mask.png'))
trapezoid_template_mask = cv2.imread(os.path.join(curr_dir,'template_masks/trapezoid_channel.png'))
trapezoid_template_skeleton =  cv2.imread(os.path.join(curr_dir,'template_masks/trapezoid_skeleton.png'))