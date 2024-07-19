import matplotlib.pyplot as plt
import cv2
import numpy as np
from resources import *

# Edit these to fit your needs
GREEN_HSV = np.array([[35, 95, 92], [113, 255, 255]])
CABLE_HSV = np.array([[0, 0, 255], [115, 76, 255]])

def get_contours(edges, cnt_type='external'):
    if cnt_type=='all':
        cnts = sorted(cv2.findContours(edges.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea, reverse=True)
    if cnt_type=='external':
        cnts = sorted(cv2.findContours(edges.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea, reverse=True)
    return cnts

def get_bbox(contour):
    rect = cv2.minAreaRect(contour)
    center, size, theta = rect[0], rect[1], rect[2]
    return rect, center, size, theta

def post_process_mask(binary_map):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    val = 255
    for i in range(0, nlabels - 1):
        if areas[i] >= 8000:
            result[labels == i + 1] = val
    # Visualization:
    # plt.imshow(result, cmap='gray')
    # plt.show()
    return result

def get_hsv_mask(image, lower, upper):
    # output mask values: array([  0, 255], dtype=uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def get_cable(image):
    crop_image = image[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    masked_cable = get_hsv_mask(crop_image, CABLE_HSV[0], CABLE_HSV[1])
    masked_cable = post_process_mask(masked_cable)
    
    sorted_cnts = get_contours(masked_cable, 'external')
    best_cnts = [sorted_cnts[0]]
    best_mask = None
    for i, cnt in enumerate(best_cnts):
        # Visualization:
        # cnt_rgb = cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
        # plt.imshow(cv2.drawContours(image.copy(), [cnt_rgb], -1, 255, 3))
        # plt.title('check cable contour dimensions')
        # plt.show()
        mask = np.zeros_like(crop_image, dtype=np.uint8)
        _ = cv2.drawContours(mask, [cnt], -1, 255, 3)
        mask = mask.sum(axis=-1)
        if i == len(best_cnts) - 1:
            best_mask = mask
    return best_cnts[-1], best_mask

def get_channel(image):
    crop_image = image[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    masked_bg = get_hsv_mask(crop_image, GREEN_HSV[0], GREEN_HSV[1])
    masked_cable = get_hsv_mask(crop_image, CABLE_HSV[0], CABLE_HSV[1])
    masked_cable = post_process_mask(masked_cable)
    masked_channel = cv2.bitwise_and(1-masked_bg, 1-masked_cable)
    masked_channel = post_process_mask(masked_channel)
    
    sorted_cnts = get_contours(masked_channel, 'external')
    matched_template = None
    best_cnt = None
    max_channel_density = 0
    min_cnt_val = np.inf
    matched_results = []

    for i, cnt in enumerate(sorted_cnts):
        rect, center, size, theta = get_bbox(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = size[0]*size[1]
        if area < 1000:
            continue
        ratio = max(size)/min(size)
        dists = [np.abs(t-ratio) for t in TEMPLATE_RATIOS]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        minX_box, maxX_box = np.min(box[:,0]), np.max(box[:,0])
        minY_box, maxY_box = np.min(box[:,1]), np.max(box[:,1])
        channel_density = crop_image[minY_box:maxY_box, minX_box:maxX_box].sum()

        box_rgb = box + np.array([CROP_REGION[2], CROP_REGION[0]])
        scale_y, scale_x = max(box_rgb[:,1]) - min(box_rgb[:,1]), max(box_rgb[:,0]) - min(box_rgb[:,0])
        true_size = (scale_x, scale_y)
        if min_dist < min_cnt_val and channel_density > max_channel_density:
            matched_results = [rect, center, size, theta, box_rgb]
            min_cnt_val = min_dist
            matched_template = TEMPLATES[min_idx]
            best_cnt = cnt
            min_cnt_idx = i
            max_channel_density = channel_density
        # Visualization:
        # plt.imshow(cv2.drawContours(image.copy(), [box_rgb], -1, 255, 3))
        # plt.title('check channel contour dimensions in get_channel')
        # plt.show()
    return matched_template, matched_results, best_cnt