from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sensing.utils_binary import *

img_path = '/home/r2d2/Robot-Cable-Insertion_TRI-Demo/depth_rgb_img.png'
point_coords = np.array([[799, 199], [826, 296], [808,296], [827, 246], [923, 232]])
point_labels = np.array([1, 1, 1, 0, 0])

# img_path = '/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask1.png'
# point_coords = np.array([[700,179],[750,158],[907, 111], [583, 246],])
# point_labels = np.array([1,1,0, 0])


def get_mask():
    img = cv2.imread(img_path)

    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(img)

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, ious, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)

    # print(ious)

    for mask in masks:
        plt.imshow(img)
        plt.imshow(mask, cmap='jet', alpha=0.5)
        plt.show()


    # cv2.imwrite("sam_mask.png", masks[-1])
    # np.save("sam_mask.npy", masks[-1])

    # better_mask = np.zeros_like(masks[-1])

    rgb_mask = cv2.cvtColor(masks[-1].astype('float32'), cv2.COLOR_GRAY2RGB)

    kernel_size = 10

    better_mask = cv2.dilate(rgb_mask, np.ones((kernel_size,kernel_size), np.uint8), iterations=1)

    plt.imshow(better_mask)
    plt.show()

    return better_mask

    # for mask in masks:
    #     plt.imshow(mask)
    #     plt.show()

    # print(masks.shape)

    # import json
    # with open('masks.json', 'w') as f:
    #     json.dump(masks, f)

    # print(len(masks))
    # for mask in masks:
    #     plt.imshow(mask['segmentation'])
    #     plt.show()

    # channel_mean, cable_mean = 161.4, 66.7
    # channel_seg, cable_seg = None, None
    # eps = 10


    # for mask in masks:
    #     # print(mask['bbox'])
    #     x,y,w,h = mask['bbox']
    #     if img[y:y+h,x:x+w].mean() >= cable_mean-eps and img[y:y+h,x:x+w].mean() <= cable_mean+eps:
    #         cable_seg = mask['segmentation']
    #     if img[y:y+h,x:x+w].mean() >= channel_mean-eps and img[y:y+h,x:x+w].mean() <= channel_mean+eps:
    #         channel_seg = mask['segmentation']


    # plt.imshow(cable_seg)
    # plt.show()
    # print(channel_seg)
    # plt.imshow(channel_seg)
    # plt.show()

    # mask[0]['segmentation']

    # for mask in masks:
    #     print(mask.shape)

    # print(masks[0].shape)
    # print(masks.min(), masks.max())

def get_channel_sorted_list(skeleton_img_channel, endpoint):
    endpoint = (endpoint[1], endpoint[0])
    sorted_pixel_list = sort_skeleton_pts(skeleton_img_channel, endpoint)
    for i, pt in enumerate(sorted_pixel_list):
        sorted_pixel_list[i] = (pt[1], pt[0])
    # sorted_pixel_list = filter_points(sorted_pixel_list, point_cloud)
    return sorted_pixel_list

channel_mask = get_mask()
skeleton_channel_img = skeletonize_img(channel_mask)
total_channel_pixels, channel_endpoints = find_length_and_endpoints(skeleton_channel_img) 
channel_end1, channel_end2 = channel_endpoints[0], channel_endpoints[-1]
sorted_channel_pixels = get_channel_sorted_list(skeleton_channel_img, channel_end1)
channel_end1, channel_end2 = sorted_channel_pixels[0], sorted_channel_pixels[-1]