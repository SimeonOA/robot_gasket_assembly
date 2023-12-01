import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
from matplotlib.path import Path
import json
import argparse
from PIL import Image
from resources import CROP_REGION, curved_template_mask, straight_template_mask, trapezoid_template_mask

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_path', type=str, default='imgs/curved7.png')
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

# curved_template_mask = cv.imread('/home/r2d2/robot_cable_insertion/franka/templates_crop_master/master_curved_channel_template.png')

# trapezoid_template_mask = cv.imread('/home/r2d2/R2D2/cable_insertion/templates_crop_master/processed_fuzzy_trap_mask2.png')
# print(trapezoid_template_mask)

TEMPLATES = {0:'curved', 1:'straight', 2:'trapezoid'}
# first elem is currved width/height, second elem is straight width/height, third elem is trapezoid width/height
TEMPLATE_RECTS = [(587.4852905273438, 168.0382080078125),(2.75, 26.5), (12, 5.75)]
TEMPLATE_RATIOS = [max(t)/min(t) for t in TEMPLATE_RECTS]

# DEPTH_IMG, RGB_IMG = get_rgb_get_depth()
# RGB_IMG = RGB_IMG[:,:,:3]
# RGB_IMG = cv.cvtColor(RGB_IMG, cv.COLOR_BGR2RGB) 
# plt.imshow(RGB_IMG)
# plt.show()
DEPTH_IMG = np.zeros((720,1280,3))
RGB_IMG = np.zeros((720,1280,3))

def make_img_gray(img_path, img = None):
    if img is None:
        img = cv.imread(img_path)
    # orig_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    orig_img = img
    # plt.imshow(orig_img)
    # plt.show()
    crop_img = orig_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    # plt.imshow(crop_img)
    # plt.show()
    # ## REMOVE GLARE
    # seed = (10, 10)  # Use the top left corner as a "background" seed color (assume pixel [10,10] is not in an object).
    # # Use floodFill for filling the background with black color
    # cv.floodFill(img, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
    # plt.imshow(img)
    # plt.show()
    gray = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    return gray, crop_img


def get_edges(gray_img, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,255)):
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dst = cv.GaussianBlur(gray_img,(blur_radius, blur_radius),sigma)
    edges = cv.Canny(dst,canny_threshold[0],canny_threshold[1])
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    edges = cv.dilate(edges, dilate_kernel)
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    return edges


def get_contours(edges, cnt_type='external'):
    if cnt_type=='all':
        cnts = sorted(cv.findContours(edges.astype('uint8'), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea, reverse=True)
    if cnt_type=='external':
        cnts = sorted(cv.findContours(edges.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2], key=cv.contourArea, reverse=True)
    return cnts

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
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        crop_edges = get_edges(cv.cvtColor(mask, cv.COLOR_RGB2GRAY), blur_radius, sigma, dilate_size, canny_threshold)
        cropped_cnts = get_contours(crop_edges, 'all')
        if len(cropped_cnts) == 0:
            continue
        print('number of shapes = ', len(cropped_cnts))
        len_cropped_cnts.append(len(cropped_cnts))
    best_cnts = [sorted_cnts[i] for i in np.argsort(len_cropped_cnts)[:2]]
    return best_cnts


def get_cable(img = None, img_path=None, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,255), viz=False):
    if img is not None:
        rgb_img  = img.copy()
        # gray_img, crop_img= make_img_gray(None, img)
        crop_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
        orig_gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
        gray_img = cv.threshold(orig_gray_img.copy(), 120, 255, cv.THRESH_BINARY_INV)[1]
    elif img_path is None:
        if DEPTH_IMG is None or RGB_IMG is None:
            _, rgb_img = get_rgb_get_depth()
        else:
            rgb_img = RGB_IMG.copy()
        rgb_img = rgb_img[:,:,:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)
        crop_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
        # plt.imshow(crop_img)
        # plt.show()
        gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
        # plt.imshow(gray_img, cmap='gray')
        # plt.show()
        gray_img = cv.threshold(gray_img.copy(), 250, 255, cv.THRESH_BINARY)[1]
        # plt.imshow(gray_img, cmap='gray')
        # plt.show()
    else:
        gray_img, rgb_img = make_img_gray(img_path)
    
    edges = get_edges(gray_img, blur_radius, sigma, dilate_size, canny_threshold)
    sorted_cnts = get_contours(edges, 'external')
    best_cnts = [sorted_cnts[0]]
    best_mask = None
    for i, cnt in enumerate(best_cnts):
        cnt_rgb = cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
        # plt.imshow(cv.drawContours(rgb_img.copy(), [cnt_rgb], -1, 255, 3))
        # plt.title('check cable contour dimensions')
        # plt.show()
        print(get_bbox(cnt))
        mask = np.zeros_like(crop_img, dtype=np.uint8)
        _ = cv.drawContours(mask, [cnt], -1, 255, 3)
        mask = cv.dilate(mask, np.ones((dilate_size,dilate_size), np.uint8), iterations=1)
        mask = mask.sum(axis=-1)
        if i == len(best_cnts) - 1:
            best_mask = mask
        # plt.imshow(mask, cmap='gray')
        # plt.show()
    return best_cnts[-1], best_mask



def get_channel( img=None,
                cable_mask = None,img_path = None, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100,255), viz=False):
    if img is not None:
        rgb_img = img.copy()
        # gray_img, crop_img  = make_img_gray(None, img)
        crop_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
        orig_gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
        gray_img = cv.threshold(orig_gray_img.copy(), 60, 255, cv.THRESH_BINARY_INV)[1]
        # plt.imshow(gray_img)
        # plt.show()

    elif img_path is None:
        if DEPTH_IMG is None or RGB_IMG is None:
            _, rgb_img = get_rgb_get_depth()
        else:
            rgb_img = RGB_IMG.copy()
        rgb_img = rgb_img[:,:,:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2RGB)
        crop_img = rgb_img[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
        # plt.imshow(crop_img)
        # plt.show()
        orig_gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
        # gray_img = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
        # plt.imshow(orig_gray_img, cmap='gray')
        # plt.show()
        # gray_img = orig_gray_img.copy()
        gray_img = cv.threshold(orig_gray_img.copy(), 100, 255, cv.THRESH_BINARY_INV)[1]
        # plt.imshow(gray_img, cmap='gray')
        # plt.show()
    else:
        gray_img, rgb_img = make_img_gray(img_path)
    
    edges = get_edges(gray_img, blur_radius, sigma, dilate_size, canny_threshold)

    # # assumes a good mask of the cable and removes it from the edges of this image so our channel is better
    # cable_mask = cable_mask[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]
    # cable_mask = cv.dilate(cable_mask, np.ones((7,7), np.uint8))
    # plt.imshow(edges)
    # plt.imshow(cable_mask, alpha=0.5)
    # plt.show()
    # edges = edges * np.invert(cable_mask[:,:,0])
    # plt.imshow(edges)
    # plt.show()

    sorted_cnts = get_contours(edges)[:2]
    # sorted_cnts = [get_contours(edges)[0]]
    matched_template = None
    best_cnt = None
    min_cnt_idx = None
    max_channel_density = 0
    min_cnt_val = np.inf
    matched_results = []
    for i, cnt in enumerate(sorted_cnts):
        # plt.imshow(cv.drawContours(rgb_img.copy(), [cnt], -1, 255, 3))
        # plt.show()
        # print(get_bbox(cnt))
        rect, center, size, theta = get_bbox(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print('size: ', size)
        area = size[0]*size[1]
        if area < 1000:
            continue
        ratio = max(size)/min(size)
        dists = [np.abs(t-ratio) for t in TEMPLATE_RATIOS]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        minX_box, maxX_box = np.min(box[:,0]), np.max(box[:,0])
        minY_box, maxY_box = np.min(box[:,1]), np.max(box[:,1])
        print(minX_box, maxX_box, minY_box, maxY_box)
        print(gray_img.shape)
        # plt.imshow(gray_img[minY_box:maxY_box, minX_box:maxX_box], cmap='gray')
        # plt.show()
        channel_density = gray_img[minY_box:maxY_box, minX_box:maxX_box].sum()

        # matched_template = TEMPLATES[min_idx]

        # print(box, box.shape)
        # cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

        print(TEMPLATES[min_idx])
        # plt.imshow(gray_img, cmap='gray')
        # plt.scatter(center[0], center[1], c='r')
        # plt.show()
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
        # print(scale_x, scale_y)
        # plt.imshow(cv.drawContours(rgb_img.copy(), [box_rgb], -1, 255, 3))
        # plt.title('check channel contour dimensions')
        # plt.show()

    print(matched_template, min_cnt_val, min_cnt_idx, matched_results[-1])
    
    return matched_template, matched_results, best_cnt, rgb_img


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

def best_fit_template(all_masks, img, matched_cnt):
    matched_mask = np.zeros_like(img, dtype=np.uint8)
    matched_cnt = matched_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cv.drawContours(matched_mask, [matched_cnt], -1, 255, -1)
    matched_mask = matched_mask.sum(axis=-1)
    # plt.imshow(matched_mask, cmap='gray')
    # plt.show()
    most_ones = -np.inf
    best_mask = None
    for template_mask in all_masks:
        # plt.imshow(template_mask, cmap='gray')
        # plt.show()
        overlap = np.bitwise_and(template_mask, matched_mask)
        # plt.imshow(overlap, cmap='gray')
        # plt.show()
        if overlap.sum() > most_ones:
            most_ones = overlap.sum()
            best_mask = template_mask
    return best_mask

def align_channel(template_mask, matched_results, img, matched_cnt, matched_template):
    for k,v in TEMPLATES.items():
        if v == matched_template:
           matched_template = k
           break 
    # plt.imshow(template_mask, cmap='gray')
    # plt.title('Original template mask')
    # plt.show()
    rect, center, size, theta, box = matched_results
    rotation_angles = [theta, -theta, 90-theta, 180-theta, 180+theta]
    # print(box)
    # print('theta = ', theta)
    # print('size = ', size)
    scale_x, scale_y = box[:,0].max() - box[:,0].min(), box[:,1].max() - box[:,1].min()
    # print(template_mask.shape)

    if matched_template == 0:
        x_dim, y_dim, _ = template_mask.shape
        if x_dim < y_dim:
            pad_x1 = int(3/14 * x_dim)
            pad_x2 = int(2/14 * x_dim)
            pad_y = int(2/26 * y_dim)
            mask = np.zeros((x_dim + pad_x1 + pad_x2, y_dim + pad_y*2, 3), dtype=np.uint8)
            start_x = pad_x1
            end_x = pad_x1 + x_dim
            start_y = pad_y
            end_y = pad_y + y_dim
            mask[start_x:end_x,start_y:end_y] = template_mask
        else:
            pad_x = int(2/26 * x_dim)
            pad_y1 = int(3/14 * y_dim)
            pad_y2 = int(2/14 * y_dim)
            mask = np.zeros((x_dim + pad_x*2, y_dim + pad_y1 + pad_y2, 3), dtype=np.uint8)
            start_x = pad_x
            end_x = pad_x + x_dim
            start_y = pad_y1
            end_y = pad_y1 + y_dim
            mask[start_x:end_x,start_y:end_y] = template_mask
        dilate_kernel = np.ones((10,10), np.uint8)
        mask = cv.dilate(mask, dilate_kernel, iterations=1)
        template_mask = mask

    if matched_template == 2:
        x_dim, y_dim, _ = template_mask.shape
        if x_dim > y_dim:
            pad_x1 = int(1.7/27.5 * x_dim)
            pad_x2 = int(2/27.5 * x_dim)
            pad_y1 = int(0.5/4.625 * y_dim)
            # pad_y2 = int(1/4.625 * y_dim)
            pad_y2 = int(0.5/4.625 * y_dim)
            mask = np.zeros((x_dim + pad_x1 + pad_x2, y_dim + pad_y1 + pad_y2, 3), dtype=np.uint8)
            start_x = pad_x1
            end_x = pad_x1 + x_dim
            start_y = pad_y1
            end_y = pad_y1 + y_dim
        else:
            pad_x1 = int(0.5/4.625 * x_dim)
            # pad_x2 = int(1/4.625 * x_dim)
            pad_x2 = int(0.5/4.625 * x_dim)
            pad_y1 = int(1.7/27.5 * y_dim)
            pad_y2 = int(2/27.5 * y_dim)
            mask = np.zeros((x_dim + pad_x1 + pad_x2, y_dim + pad_y1 + pad_y2, 3), dtype=np.uint8)
            start_x = pad_x1
            end_x = pad_x1 + x_dim
            start_y = pad_y1
            end_y = pad_y1 + y_dim
        # mask = np.zeros((x_dim + 2*pad_x, y_dim + 2*pad_y, 3), dtype=np.uint8)
        # start_x = pad_x
        # end_x = pad_x + x_dim
        # start_y = pad_y
        # end_y = pad_y + y_dim
        mask[start_x:end_x,start_y:end_y] = template_mask
        # dilate_kernel = np.ones((5,5), np.uint8)
        # mask = cv.dilate(mask, dilate_kernel, iterations=1)
        template_mask = mask
    
    x_dim, y_dim, _ = template_mask.shape
    template_mask_ratio = max(x_dim, y_dim)/min(x_dim, y_dim)
    match_ratio = max(scale_x,scale_y)/min(scale_x,scale_y)
    # if matched_template != 2:
    if template_mask_ratio > match_ratio:
        if y_dim > x_dim:
            new_x_dim = int(y_dim/match_ratio)
            mask = np.zeros((new_x_dim, y_dim, 3), dtype=np.uint8)
            # print(mask.shape, template_mask.shape, TEMPLATE_RATIOS[1])
            pad_dim = new_x_dim - x_dim
            mask[pad_dim//2:pad_dim//2+x_dim,:] = template_mask
        else:
            new_y_dim = int(x_dim/match_ratio)
            mask = np.zeros((x_dim, new_y_dim, 3), dtype=np.uint8)
            # print(mask.shape, template_mask.shape, TEMPLATE_RATIOS[1])
            pad_dim = new_y_dim - y_dim
            mask[:,pad_dim//2:pad_dim//2+y_dim] = template_mask
    else:
        if y_dim > x_dim:
            new_y_dim = int(x_dim * match_ratio)
            mask = np.zeros((x_dim, new_y_dim, 3), dtype=np.uint8)
            # print(mask.shape, template_mask.shape, TEMPLATE_RATIOS[1])
            pad_dim = new_y_dim - y_dim
            mask[:,pad_dim//2:pad_dim//2+y_dim] = template_mask
        else:
            new_x_dim = int(y_dim * match_ratio)
            mask = np.zeros((new_x_dim, y_dim, 3), dtype=np.uint8)
            # print(mask.shape, template_mask.shape, TEMPLATE_RATIOS[1])
            pad_dim = new_x_dim - x_dim
            mask[pad_dim//2:pad_dim//2+x_dim,:] = template_mask
    # else:
    #     if template_mask_ratio != match_ratio:
    #         if scale_y > scale_x:
    #             new_x_dim, new_y_dim = int(2*scale_y*match_ratio), int(2*scale_x)
    #         else:
    #             new_x_dim, new_y_dim = int(2*scale_y), int(2*scale_x*match_ratio)
    #         mask = np.zeros((new_x_dim, new_y_dim, 3), dtype=np.uint8)
    #         start_x = new_x_dim//2 - x_dim//2
    #         end_x = new_x_dim//2 + (x_dim - x_dim//2)
    #         start_y = new_y_dim//2 - y_dim//2
    #         end_y = new_y_dim//2 + (y_dim - y_dim//2)
    #         mask[start_x:end_x, start_y:end_y] = template_mask
    #         plt.imshow(mask)
    #         plt.show()


    # print(mask.shape)
    template_mask = mask
    # plt.imshow(template_mask, cmap='gray')
    # plt.title('Padded template mask')
    # plt.show()

    scaled_template_mask = cv.resize(template_mask, (scale_x, scale_y), interpolation= cv.INTER_LINEAR)
    print(scaled_template_mask.shape)
    # plt.imshow(scaled_template_mask)
    # plt.title('Scaled template mask')
    # plt.show()
    padded_scaled_template_mask = center_mask(scaled_template_mask, img)
    # plt.imshow(padded_scaled_template_mask, cmap='gray')
    # plt.title('Padded template mask')
    # plt.show()
    all_masks = []
    for rot in rotation_angles:
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        # plt.imshow(rotated_scaled_template_mask)
        # plt.title('Rotation angle = ' + str(rot))
        # plt.show()
        shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
        y,x = np.where(rotated_scaled_template_mask[:,:,0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask
        #shifted_rotated_scaled_template_mask = translate_mask(rotated_scaled_template_mask, (shift_x, shift_y), img)
        # plt.imshow(shifted_rotated_scaled_template_mask)
        # plt.imshow(img, alpha=0.5)
        # plt.title('Shifted, rotated, and scaled template mask')
        # plt.show()


        all_masks.append(shifted_rotated_scaled_template_mask)

    best_template = best_fit_template(all_masks, img, matched_cnt)

    # plt.imshow(best_template)
    # plt.imshow(img, alpha=0.5)
    # plt.scatter(center[0]+CROP_REGION[2], center[1]+CROP_REGION[0], c='r')
    # plt.title('Best template')
    # plt.show()
    
    return best_fit_template(all_masks, img, matched_cnt)


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

def get_closest_channel_endpoint(cable_endpoint, channel_endpoints):
    min_dist = np.inf
    closest_channel_endpoint = None
    for channel_endpoint in channel_endpoints:
        dist = np.linalg.norm(np.array(channel_endpoint) - np.array(cable_endpoint))
        if dist < min_dist:
            min_dist = dist
            closest_channel_endpoint = channel_endpoint
    if closest_channel_endpoint == None:
        print('neither endpoint was found in the channel')
    farthest_channel_endpoint = channel_endpoints[0] if channel_endpoints[1] == closest_channel_endpoint else channel_endpoints[1]
    return closest_channel_endpoint, farthest_channel_endpoint

if __name__ == '__main__':
    DEPTH_IMG, RGB_IMG = get_rgb_get_depth()
    RGB_IMG = RGB_IMG[:,:,:3]
    RGB_IMG = cv.cvtColor(RGB_IMG, cv.COLOR_BGR2RGB) 
    plt.imshow(RGB_IMG)
    plt.title('RGB image')
    plt.show()
    args = argparser.parse_args()
    # template_cnts = [np.load(args.curved_template_cnt_path), np.load(args.straight_template_cnt_path), np.load(args.trapezoid_template_cnt_path)]
    # get_bboxes(template_cnts, cv.imread(args.img_path))
    if not args.robot:
        matched_template, matched_results, best_cnts = get_channel(img_path=args.img_path, blur_radius=args.blur_radius, 
                    sigma=args.sigma, dilate_size=args.dilate_size_channel, canny_threshold=args.canny_threshold, viz=args.visualize)
    else:
        cable_cnt, cable_mask_hollow  = get_cable(img_path=None, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_rope, 
                  canny_threshold=args.canny_threshold_rope, viz=args.visualize)
        # get_channel(img_path=None, blur_radius=args.blur_radius, sigma=args.sigma, dilate_size=args.dilate_size_channel,canny_threshold=args.canny_threshold_rope, viz=args.visualize)
        matched_template, matched_results, channel_cnt, rgb_img = get_channel(img_path=None, blur_radius=args.blur_radius, sigma=args.sigma, 
                                                     dilate_size=args.dilate_size_channel, canny_threshold=args.canny_threshold_channel, viz=args.visualize)

    # cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    # cable_mask_binary = np.zeros_like(RGB_IMG[:,:,:3])
    # cv.drawContours(cable_mask_binary, [cable_cnt], -1, (255,255,255), -1)
    # cable_skeleton = skeletonize(cable_mask_binary)
    # plt.imshow(cable_skeleton)
    # plt.show()
    # cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    # plt.scatter(x=[i[1] for i in cable_endpoints], y=[i[0] for i in cable_endpoints])
    # plt.imshow(RGB_IMG)
    # plt.show()

    if matched_template == 'curved':
        template_mask = curved_template_mask
    elif matched_template == 'straight':
        template_mask = straight_template_mask
    elif matched_template == 'trapezoid':
        template_mask = trapezoid_template_mask
    aligned_channel_mask = align_channel(template_mask, matched_results, rgb_img, channel_cnt, matched_template) 
    # making it 3 channels
    aligned_channel_mask = cv.merge((aligned_channel_mask, aligned_channel_mask, aligned_channel_mask))
    aligned_channel_mask = aligned_channel_mask.astype('uint8')
    plt.imshow(rgb_img)
    plt.imshow(aligned_channel_mask, alpha=0.5)
    plt.show()
    channel_skeleton = skeletonize(aligned_channel_mask)
    plt.imshow(channel_skeleton)
    plt.imshow(rgb_img, alpha=0.5)
    plt.title('overlayed channel skeleton and rgb image')
    plt.show()
    channel_length, channel_endpoints = find_length_and_endpoints(channel_skeleton)
    # # need to figure which cable endpoint is in the channel and the closeseted channel endpoint and sort from there
    # # check which cable endpoint is in the channel then pick the channel endpoint that is closest to that cable endpoint
    # # then sort from there
    # cable_endpoint_in = get_cable_endpoint_in_channel(cable_endpoints, aligned_channel_mask)
    # channel_endpoint_in = get_closest_channel_endpoint(cable_endpoint_in, channel_endpoints)
    # breakpoint()
    # sorted_channel_pts = sort_skeleton_pts(channel_skeleton, channel_endpoint_in)
    # sorted_cable_pts = sort_skeleton_pts(cable_skeleton, cable_endpoint_in)

    # plt.scatter(x=channel_endpoint_in[1], y=channel_endpoint_in[0], c='r')
    # plt.scatter(x=cable_endpoint_in[1], y=cable_endpoint_in[0], c='b')
    # plt.imshow(RGB_IMG)
    # plt.show()

    