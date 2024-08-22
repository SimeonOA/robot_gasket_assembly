import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from resources import *
from utils import *

# Edit these to fit your needs
GREEN_HSV = np.array([[35, 95, 92], [113, 255, 255]])
CABLE_HSV = np.array([[0, 0, 255], [115, 76, 255]])

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_path', type=str, default='images/cable_detection.png')
argparser.add_argument('--blur_radius', type=int, default=5)
argparser.add_argument('--sigma', type=int, default=0)
argparser.add_argument('--dilate_size_channel', type=int, default=2)
argparser.add_argument('--dilate_size_rope', type=int, default=20)
argparser.add_argument('--canny_threshold_channel', type=tuple, default=(100,255))
argparser.add_argument('--canny_threshold_rope', type=tuple, default=(0,255))
argparser.add_argument('--visualize', default=False, action='store_true')
argparser.add_argument('--robot', default=False, action='store_true')
argparser.add_argument('--curved_template_cnt_path', type=str, default='templates/curved_template_full_cnt.npy')
argparser.add_argument('--straight_template_cnt_path', type=str, default='templates/straight_template_full_cnt.npy')
argparser.add_argument('--trapezoid_template_cnt_path', type=str, default='templates/trapezoid_template_full_cnt.npy')

def make_img_gray(img_path, img = None):
    if img is None:
        img = cv.imread(img_path)
    orig_img = img
    crop_img = orig_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    gray = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
    return gray, crop_img

def get_edges(gray_img, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,255)):
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dst = cv.GaussianBlur(gray_img,(blur_radius, blur_radius),sigma)
    edges = cv.Canny(dst,canny_threshold[0],canny_threshold[1])
    edges = cv.dilate(edges, dilate_kernel)
    return edges

def get_contours(edges, cnt_type='external'):
    if cnt_type=='all':
        cnts = sorted(cv.findContours(edges.astype('uint8'), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea, reverse=True)
    if cnt_type=='external':
        cnts = sorted(cv.findContours(edges.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2], key=cv.contourArea, reverse=True)
    return cnts

def get_bbox(contour):
    rect = cv2.minAreaRect(contour)
    center, size, theta = rect[0], rect[1], rect[2]
    return rect, center, size, theta

def post_process_mask(binary_map, viz=False):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    val = 255
    for i in range(0, nlabels - 1):
        if areas[i] >= 8000:
            result[labels == i + 1] = val
    if viz:
        plt.imshow(result, cmap='gray')
        plt.show()
    return result

def get_hsv_mask(image, lower, upper):
    # output mask values: array([  0, 255], dtype=uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def detect_cable(rgb_img, args):
    cable_cnt, cable_mask_hollow  = get_cable(img = rgb_img, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize)
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_mask_binary = np.zeros(rgb_img.shape, np.uint8)
    cv2.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
    cable_mask_binary = (cable_mask_binary.sum(axis=2)/255).astype('uint8')
    cable_mask_binary = cv2.morphologyEx(cable_mask_binary,cv2.MORPH_CLOSE,np.ones((5,5), np.uint8))
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    assert len(cable_endpoints) == 2
    return cable_skeleton, cable_length, cable_endpoints, cable_mask_binary

def get_cable(image, viz=False):
    crop_image = image[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    masked_cable = get_hsv_mask(crop_image, CABLE_HSV[0], CABLE_HSV[1])
    masked_cable = post_process_mask(masked_cable, viz)
    
    sorted_cnts = get_contours(masked_cable, 'external')
    best_cnts = [sorted_cnts[0]]
    best_mask = None
    for i, cnt in enumerate(best_cnts):
        if viz:
            cnt_rgb = cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
            plt.imshow(cv2.drawContours(image.copy(), [cnt_rgb], -1, 255, 3))
            plt.title('check cable contour dimensions')
            plt.show()
        mask = np.zeros_like(crop_image, dtype=np.uint8)
        _ = cv2.drawContours(mask, [cnt], -1, 255, 3)
        mask = mask.sum(axis=-1)
        if i == len(best_cnts) - 1:
            best_mask = mask
    return best_cnts[-1], best_mask

def detect_channel(rgb_img, viz=False):
    matched_template, matched_results, channel_cnt = get_channel(rgb_img, viz)
    if matched_template == 'curved':
        template_mask = curved_template_mask_align
    elif matched_template == 'straight':
        template_mask = straight_template_mask_align
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask
    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    channel_cnt_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    _ = cv2.drawContours(channel_cnt_mask, [channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])],-1, [255,255,255], -1)
    channel_skeleton = skeletonize(aligned_channel_mask)
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    return channel_skeleton, channel_length, channel_endpoints, matched_template, aligned_channel_mask, channel_cnt_mask, channel_cnt

def get_channel(image, viz=False):
    crop_image = image[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    masked_bg = get_hsv_mask(crop_image, GREEN_HSV[0], GREEN_HSV[1])
    masked_cable = get_hsv_mask(crop_image, CABLE_HSV[0], CABLE_HSV[1])
    masked_cable = post_process_mask(masked_cable, viz)
    masked_channel = cv2.bitwise_and(1-masked_bg, 1-masked_cable)
    masked_channel = post_process_mask(masked_channel, viz)
    
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
        if viz:
            plt.imshow(cv2.drawContours(image.copy(), [box_rgb], -1, 255, 3))
            plt.title('check channel contour dimensions in get_channel')
            plt.show()
    return matched_template, matched_results, best_cnt

def get_img_crops(img, cnts):
    crops, crop_cnts = [], []
    for cnt in cnts:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        x, y = [b[0] for b in box], [b[1] for b in box]
        minX, minY = min(x), min(y)
        maxX, maxY = max(x), max(y)
        crop = img[minY:maxY, minX:maxX]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crops.append(crop)
        crop_cnts.append(cnt - np.array([minX,minY]))
    return crops, crop_cnts


def get_channel_rope_cnts(crops, cnt_crop_frame, sorted_cnts, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,200)):
    len_cropped_cnts = []
    for idx, (cnt, crop) in enumerate(zip(cnt_crop_frame, crops)):
        if idx >= 5:
            break
        mask = np.zeros_like(crop, dtype=np.uint8)
        _ = cv.drawContours(mask, [cnt],-1, 255, -1)
        crop_edges = get_edges(cv.cvtColor(mask, cv.COLOR_RGB2GRAY), blur_radius, sigma, dilate_size, canny_threshold)
        cropped_cnts = get_contours(crop_edges, 'all')
        if len(cropped_cnts) == 0:
            continue
        print('number of shapes = ', len(cropped_cnts))
        len_cropped_cnts.append(len(cropped_cnts))
    best_cnts = [sorted_cnts[i] for i in np.argsort(len_cropped_cnts)[:2]]
    return best_cnts


def get_cable(img = None, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,255), viz=False):
    rgb_img  = img.copy()
    crop_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    if viz:
        plt.imshow(crop_img)
        plt.title("cropped image")
        plt.show()
    orig_gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
    gray_img = cv.threshold(orig_gray_img.copy(), 240, 255, cv.THRESH_BINARY_INV)[1]
    
    edges = get_edges(gray_img, blur_radius, sigma, dilate_size, canny_threshold)
    sorted_cnts = get_contours(edges, 'external')
    best_cnts = [sorted_cnts[0]]
    best_mask = None
    for i, cnt in enumerate(best_cnts):
        if viz:
            cnt_rgb = cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
            plt.imshow(cv.drawContours(rgb_img.copy(), [cnt_rgb], -1, 255, 3))
            plt.title('check cable contour dimensions')
            plt.show()
            print(get_bbox(cnt))
        mask = np.zeros_like(crop_img, dtype=np.uint8)
        _ = cv.drawContours(mask, [cnt], -1, 255, 3)
        mask = cv.dilate(mask, np.ones((dilate_size,dilate_size), np.uint8), iterations=1)
        mask = mask.sum(axis=-1)
        if i == len(best_cnts) - 1:
            best_mask = mask
    return best_cnts[-1], best_mask

def rotate_image(image, angle):
    # Rotate image by a specified angle in degrees
    rows, cols = image.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv.warpAffine(image, M, (cols, rows))
    return rotated_image

def get_bboxes(contours, frame):
    # bbox_data = []
    size_info = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        size_info.append(rect[1])
    return size_info

def get_bbox(contour):
    rect = cv.minAreaRect(contour)
    center, size, theta = rect[0], rect[1], rect[2]
    return rect, center, size, theta

def crop_image_with_mask(mask, image):
    # Find the coordinates of the non-zero (True) elements in the mask
    mask = mask[:,:,0]
    coordinates = np.argwhere(mask)

    # Get the bounding box of the non-zero elements
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    # Crop the image based on the bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    return cropped_image

def translate_mask(channel_mask, center, img):
    x, y = center
    x,y = round(x), round(y)
    mask_center = (channel_mask.shape[1] // 2, channel_mask.shape[0] // 2)
    x_position = x - mask_center[0]
    y_position = y - mask_center[1]
    image_with_mask = np.zeros_like(img)
    image_with_mask[y_position:y_position+channel_mask.shape[0], x_position:x_position+channel_mask.shape[1]] = channel_mask
    return image_with_mask

def center_mask(mask, image):
    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = image.shape[:2]
    x_translation = (image_width - mask_width) // 2
    y_translation = (image_height - mask_height) // 2

    # Create an empty image of the same size as the input image
    translated_image = np.zeros_like(image)

    # Apply the translation to the mask and place it in the center of the image
    translated_image[y_translation:y_translation+mask_height, x_translation:x_translation+mask_width] = mask
    return translated_image

def get_bbox_crop_mask(crop_mask):
    edges = get_edges(crop_mask)
    cnts = get_contours(edges)
    bbox = get_bbox(cnts[-1])
    rect1 = bbox[0]
    box = cv.boxPoints(rect1)
    box = np.int0(box)
    # plt.scatter(bbox[1][0], bbox[1][1], c='g')
    # plt.imshow(crop_mask)
    # plt.scatter(crop_mask.shape[1]//2, crop_mask.shape[0]//2, c='b')
    # plt.imshow(cv.drawContours(crop_mask.copy(), [box], -1, 255, 3))
    # plt.show()
    return bbox

#### IDEA: for getting relative scale of the cropped template compared the dim of cropped template to dim of cropped
# contour that we associated with our template 

def estimate_scale(vert_set0, vert_set1):
    vert_set0 = vert_set0 - np.average(vert_set0, axis=0)
    vert_set1 = vert_set1 - np.average(vert_set1, axis=0)
    closestedpoint = np.argmin(np.linalg.norm(vert_set0 - vert_set1[0], axis=1))
    vert_set0 = np.vstack([vert_set0[closestedpoint:], vert_set0[:closestedpoint]])
    xscale = np.linalg.lstsq(vert_set0[:,0].reshape((-1,1)), vert_set1[:,0])[0]
    yscale = np.linalg.lstsq(vert_set0[:,1].reshape((-1,1)), vert_set1[:,1])[0]
    return xscale[0], yscale[0]

#  NOTE: Modified to return index
def best_fit_template(all_masks, img, matched_cnt):
    matched_mask = np.zeros_like(img, dtype=np.uint8)
    matched_cnt = matched_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cv.drawContours(matched_mask, [matched_cnt], -1, 255, -1)
    matched_mask = matched_mask.sum(axis=-1)
    # plt.imshow(matched_mask, cmap='gray')
    # plt.show()
    most_ones = -np.inf
    best_mask = None
    best_idx = None
    for idx, template_mask in enumerate(all_masks):
        # plt.imshow(template_mask, cmap='gray')
        # plt.show()
        overlap = np.bitwise_and(template_mask, matched_mask)
        # plt.imshow(overlap, cmap='gray')
        # plt.show()
        if overlap.sum() > most_ones:
            most_ones = overlap.sum()
            best_mask = template_mask
            best_idx = idx
    return (best_mask, best_idx)

def align_channel(template_mask, matched_results, img, matched_cnt, matched_template, viz=False):
    for k,v in TEMPLATES.items():
        if v == matched_template:
           matched_template = k
           break 
    
    rect, center, size, theta, box = matched_results
    rotation_angles = [theta, -theta, 90-theta, 90+theta, 180-theta, 180+theta, 270+theta, 270-theta]
    scale_x, scale_y = int(size[0]), int(size[1])


    # plt.imshow(template_mask, cmap='gray')
    # plt.title('Padded template mask')
    # plt.show()

    if np.abs(scale_x - template_mask.shape[1]) < np.abs(scale_y - template_mask.shape[1]):
        scaled_template_mask = cv2.resize(template_mask, (scale_x, scale_y), interpolation= cv2.INTER_LINEAR)
    else:
        scaled_template_mask = cv2.resize(template_mask, (scale_y, scale_x), interpolation= cv2.INTER_LINEAR)
    # plt.imshow(scaled_template_mask)
    # plt.title('Scaled template mask')
    # plt.show()
    padded_scaled_template_mask = center_mask(scaled_template_mask, img)
    # plt.imshow(padded_scaled_template_mask, cmap='gray')
    # plt.title('Padded template mask')
    # plt.show()
    all_masks = []
    for rot in rotation_angles:
        print('rot = ', rot)
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
        y,x = np.where(rotated_scaled_template_mask[:,:,0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask


        all_masks.append(shifted_rotated_scaled_template_mask)

    best_template, best_idx = best_fit_template(all_masks, img, matched_cnt)

    if matched_template == 0:
        template_mask = curved_template_mask
        if np.abs(scale_x - template_mask.shape[1]) < np.abs(scale_y - template_mask.shape[1]):
            scaled_template_mask = cv2.resize(template_mask, (scale_x, scale_y), interpolation= cv2.INTER_LINEAR)
        else:
            scaled_template_mask = cv2.resize(template_mask, (scale_y, scale_x), interpolation= cv2.INTER_LINEAR)
        padded_scaled_template_mask = center_mask(scaled_template_mask, img)
        rot = rotation_angles[best_idx]
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
        y,x = np.where(rotated_scaled_template_mask[:,:,0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask
        best_template = shifted_rotated_scaled_template_mask
    elif matched_template == 1:
        template_mask = straight_template_mask
        if np.abs(scale_x - template_mask.shape[1]) < np.abs(scale_y - template_mask.shape[1]):
            scaled_template_mask = cv2.resize(template_mask, (scale_x, scale_y), interpolation= cv2.INTER_LINEAR)
        else:
            scaled_template_mask = cv2.resize(template_mask, (scale_y, scale_x), interpolation= cv2.INTER_LINEAR)
        padded_scaled_template_mask = center_mask(scaled_template_mask, img)
        rot = rotation_angles[best_idx]
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
        y,x = np.where(rotated_scaled_template_mask[:,:,0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask
        best_template = shifted_rotated_scaled_template_mask
    elif matched_template == 2:
        template_mask = trapezoid_template_skeleton
        if np.abs(scale_x - template_mask.shape[1]) < np.abs(scale_y - template_mask.shape[1]):
            scaled_template_mask = cv2.resize(template_mask, (scale_x, scale_y), interpolation= cv2.INTER_LINEAR)
        else:
            scaled_template_mask = cv2.resize(template_mask, (scale_y, scale_x), interpolation= cv2.INTER_LINEAR)
        padded_scaled_template_mask = center_mask(scaled_template_mask, img)
        rot = rotation_angles[best_idx]
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
        y,x = np.where(rotated_scaled_template_mask[:,:,1] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask
        best_template = shifted_rotated_scaled_template_mask
    if viz:
        plt.imshow(best_template)
        plt.imshow(img, alpha=0.5)
        plt.scatter(center[0]+CROP_REGION[2], center[1]+CROP_REGION[0], c='r')
        plt.title('Best template')
        plt.show()
    return best_template

def get_true_center_curved_bbox(curved_bbox, gray_img):
    (x,y), (width, height), angle = curved_bbox[1:]
    rect = ((x, y), (width, height), angle)
    box_points = cv.boxPoints(rect)
    box_points = np.int0(box_points)

    # Calculate the lengths of all four sides of the rotated bounding box
    side_lengths = [np.linalg.norm(box_points[i] - box_points[(i + 1) % 4]) for i in range(4)]

    # Identify the long side and short side of the bounding box
    long_side_index = np.argmax(side_lengths)
    short_side_index = np.argmin(side_lengths)

    # Find the mid-point along the long side
    point1, point2 = box_points[long_side_index], box_points[(long_side_index + 1) % 4]
    midpoint_long_side = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

    # Find the point that is 80% along the short side
    short_side_endpoint = box_points[short_side_index]
    short_side_length = side_lengths[short_side_index]
    point_80_percent_short_side = (
        int(short_side_endpoint[0] - 0.8 * short_side_length * np.cos(np.radians(angle))),
        int(short_side_endpoint[1] - 0.8 * short_side_length * np.sin(np.radians(angle)))
    )
    point_20_percent_short_side = (
        int(short_side_endpoint[0] - 0.2 * short_side_length * np.cos(np.radians(angle))),
        int(short_side_endpoint[1] - 0.2 * short_side_length * np.sin(np.radians(angle)))
    )

    # Interpolate between the mid-point along the long side and the point 80% along the short side
    final_point1 = (
        int(0.5 * (midpoint_long_side[0] + point_80_percent_short_side[0])),
        int(0.5 * (midpoint_long_side[1] + point_80_percent_short_side[1]))
    )
    final_point2 = (
        int(0.5 * (midpoint_long_side[0] + point_20_percent_short_side[0])),
        int(0.5 * (midpoint_long_side[1] + point_20_percent_short_side[1]))
    )
    if gray_img[final_point2[1]][final_point2[0]] < 50:
        return final_point2
    if gray_img[final_point1[1]][final_point2[0]] <  50:
        return final_point1

def get_cable_endpoint_in_channel(cable_endpoints, aligned_channel_mask):
    for idx, cable_endpoint in enumerate(cable_endpoints):
        if aligned_channel_mask[cable_endpoint[0]][cable_endpoint[1]][0] > 0:
            if idx == 0:
                return cable_endpoints[0], cable_endpoints[1]
            return cable_endpoints[1], cable_endpoints[0]
    # need a robust way to deal with the case where neither endpoint is in the channel due to bad
    # segmentation
    return cable_endpoints[0], cable_endpoints[1]

    