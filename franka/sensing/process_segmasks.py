
from tqdm import tqdm
from skimage.morphology import thin
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import random

# from torchvision import models
# from simnetneedles.vision.mask_smoother import smooth_mask
from PIL import Image
import cv2
from sensing.utils import *


def custom_processing(task):
    if task == "cable_segmasks":

        def crop_segmask(mask, image):
            mask[600:] = 0
            mask[:, :200] = 0
            mask[450:, 1000:] = 0
            return mask

        return crop_segmask
    else:
        return lambda x, y: x


def smooth_mask(mask, remove_small_artifacts=False, hsv=None):
    """
    mask: a binary image
    returns: a filtered segmask smoothing feathered images
    """
    paintval = np.max(mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1
    )  # CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS
    smoothedmask = np.zeros(mask.shape, dtype=mask.dtype)
    cv2.drawContours(smoothedmask, contours, -1, float(paintval), -1)
    smoothedmask = thin(smoothedmask, max_iter=1).astype(np.uint8)

    if remove_small_artifacts:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            smoothedmask, connectivity=8
        )
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 200

        # your answer image
        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if (sizes[i] >= min_size) or (
                hsv is not None
                and hsv[output == i + 1][:, 1].mean() > 120
                and sizes[i] >= 100
            ):
                smoothedmask[output == i + 1] = 255

    return smoothedmask


def get_segmasks(img_left, img_right, plot=False):
    hsv_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2HSV)
    hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)

    lower1 = np.array([22, 33, 182])
    upper1 = np.array([45, 128, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([150, 50, 100])
    upper2 = np.array([180, 255, 255])

    lower_mask = cv2.inRange(hsv_left, lower1, upper1)
    upper_mask = cv2.inRange(hsv_left, lower2, upper2)
    mask_left = lower_mask + upper_mask

    lower_mask = cv2.inRange(hsv_right, lower1, upper1)
    upper_mask = cv2.inRange(hsv_right, lower2, upper2)
    mask_right = lower_mask + upper_mask

    mask_left = smooth_mask(mask_left, True, hsv_left)
    mask_right = smooth_mask(mask_right, True, hsv_right)

    if plot:
        _, axs = plt.subplots(3, 2)
        axs[0, 0].imshow(img_left)
        axs[0, 1].imshow(img_right)
        axs[1, 0].imshow(hsv_left)
        axs[1, 1].imshow(hsv_right)
        axs[2, 0].imshow(mask_left)
        axs[2, 1].imshow(mask_right)
        plt.show()
    return mask_left, mask_right, img_left, img_right


def main():
    data_dir = osp.join("data", "cloth_images")
    right_images = [
        f for f in os.listdir(data_dir) if ("imager_" in f and "uv" not in f)
    ]

    data_outdir = data_dir + "_processed"

    if not osp.exists(data_outdir):
        os.mkdir(data_outdir)
    # random.shuffle(right_images)

    custom_processor = custom_processing("cable_endpoints")

    for r_img_name in tqdm(right_images):

        l_img_name = r_img_name.replace("imager", "imagel")
        r_uv_img_name = r_img_name.replace("imager", "imager_uv")
        l_uv_img_name = l_img_name.replace("imagel", "imagel_uv")
        r_mask_name = r_img_name.replace("imager", "maskr")
        l_mask_name = l_img_name.replace("imagel", "maskl")

        r_img = np.array(Image.open(osp.join(data_dir, r_img_name)))
        r_uv_img = np.array(Image.open(osp.join(data_dir, r_uv_img_name)))
        l_uv_img = np.array(Image.open(osp.join(data_dir, l_uv_img_name)))
        l_img = np.array(Image.open(osp.join(data_dir, l_img_name)))
        r_mask = np.array(Image.open(osp.join(data_dir, r_mask_name)))
        l_mask = np.array(Image.open(osp.join(data_dir, l_mask_name)))

        # # # l_mask, r_mask, _, _ = get_segmasks(l_uv_img, r_uv_img, False)
        # l_mask = smooth_mask(l_mask, remove_small_artifacts=True)
        # r_mask = smooth_mask(r_mask, remove_small_artifacts=True)

        # l_mask = custom_processor(l_mask, l_uv_img) > 0
        # r_mask = custom_processor(r_mask, r_uv_img) > 0

        np.save(osp.join(data_outdir, l_mask_name.replace(".png", ".npy")), l_mask)
        np.save(osp.join(data_outdir, r_mask_name.replace(".png", ".npy")), r_mask)
        Image.fromarray(l_mask).save(osp.join(data_outdir, l_mask_name))
        Image.fromarray(r_mask).save(osp.join(data_outdir, r_mask_name))
        Image.fromarray(r_img / 255.0).save(osp.join(data_outdir, r_img_name))
        Image.fromarray(l_img / 255.0).save(osp.join(data_outdir, l_img_name))
        Image.fromarray(l_uv_img).save(osp.join(data_outdir, l_uv_img_name))
        Image.fromarray(r_uv_img).save(osp.join(data_outdir, r_uv_img_name))

        # _,axs=plt.subplots(3,2)
        # # l_img[l_mask>.1,1]=255
        # # r_img[r_mask > .1, 1] = 255
        # axs[0,0].imshow(l_img)
        # axs[0,1].imshow(r_img)
        # axs[1,0].imshow(l_uv_img)
        # axs[1,1].imshow(r_uv_img)
        # axs[2, 0].imshow(l_mask)
        # axs[2, 1].imshow(r_mask)
        # plt.show()

        l_img[l_mask > 0.1] = 0
        r_img[r_mask > 0.1] = 0
        l_img[l_mask > 0.1, 1] = 255
        r_img[r_mask > 0.1, 1] = 255
        im = Image.fromarray(l_img)
        im.save(osp.join(data_outdir, l_mask_name.replace(".png", "overlayed.png")))
        im = Image.fromarray(r_img)
        im.save(osp.join(data_outdir, r_mask_name.replace(".png", "overlayed.png")))



def get_rope(im, color_bounds, plot=False, dilate = True, dilate_erode_num=[16, 13]):
    """
    Assume im is an RGB image
    """
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    mask = np.zeros_like(im)[:, :, 0]
    for lower1, upper1 in color_bounds: # this allows multiple lower and upper bounds and union them
        partial_mask = cv2.inRange(hsv, lower1, upper1)
        mask = np.maximum(mask, partial_mask)
    
    mask[mask > 0] = 1
    # mask = smooth_mask(mask)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, None, iterations=1)
    
    # Do some clean up
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    # import pdb;pdb.set_trace()
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 850
    uv_seg_lst = []
    # import pdb; pdb.set_trace()
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        seg_temp = np.zeros_like(mask)
        if sizes[i] >= min_size:
            seg_temp[output == i + 1] = 1
            # skip erosion here since the seg mask is already pretty small
            # for i in range(1):
            #     # remove some from border for safer grasps
            #     eroded = cv2.erode(seg_temp, np.ones((3, 3)))
            #     seg_temp &= eroded
            # # dilate/erode to make the segmentation clearer
            # seg_temp = cv2.dilate(seg_temp, None, iterations=4)
            if dilate:
                seg_temp = cv2.dilate(seg_temp, None, iterations=dilate_erode_num[0]) # original=10
                
                # close holes to make it solid rectangle
                kernel = np.ones((5,5),np.uint8)
                seg_temp = cv2.morphologyEx(seg_temp, cv2.MORPH_CLOSE, kernel)
            # else:
            #     seg_temp = cv2.dilate(seg_temp, None, iterations=2)
            #     seg_temp = cv2.erode(seg_temp, None, iterations=2)
            uv_seg_lst.append(seg_temp) # A list of binary masks for each component 
    # import pdb;pdb.set_trace()
    # combine the masks
    mask = np.zeros_like(mask)
    for uv_seg in uv_seg_lst:
        mask = np.logical_or(mask, uv_seg).astype(np.uint8)


    # After expansion to connect the components, detect the segments again
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 4000
    uv_seg_lst = []
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        seg_temp = np.zeros_like(mask)
        if sizes[i] >= min_size:
            seg_temp[output == i + 1] = 1

            # skip erosion here since the seg mask is already pretty small
            # for i in range(1):
            #     # remove some from border for safer grasps
            #     eroded = cv2.erode(seg_temp, np.ones((3, 3)))
            #     seg_temp &= eroded
            # # dilate/erode to make the segmentation clearer
            seg_temp = cv2.erode(seg_temp, None, iterations=dilate_erode_num[1]) # original=7
            # seg_temp = cv2.dilate(seg_temp, None, iterations=2)
            
            # close holes to make it solid rectangle
            kernel = np.ones((7,7),np.uint8)
            seg_temp = cv2.morphologyEx(seg_temp, cv2.MORPH_CLOSE, kernel)

            uv_seg_lst.append(seg_temp) # A list of binary masks for each component 
    


    # clean up the bag segmentation to remove noises
    uv_boundary_lst = [] # A list of binary masks for the boundary of each component 
    for uv_seg in uv_seg_lst:
        eroded = cv2.erode(uv_seg, np.ones((3, 3)))
        uv_boundary_lst.append(uv_seg-eroded)
    
        
    # combine the masks
    mask = np.zeros_like(mask)
    for uv_seg in uv_seg_lst:
        mask = np.logical_or(mask, uv_seg).astype(np.uint8)

    if plot:
        _, axs = plt.subplots(2, 1)
        axs[0].imshow(im)
        axs[1].imshow(mask)
        plt.show()
        
    return mask




def get_channel(img, color_bounds, plot=False):
    """
    Assume img is an RGB image
    """
    # crop image (220,100), (1100,450)
    img_copy = img.copy()
    offset_x = 380
    offset_y = 100
    img = img[offset_y:450, offset_x:1100]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.zeros_like(img)[:, :, 0]
    for lower1, upper1 in color_bounds: # this allows multiple lower and upper bounds and union them
        partial_mask = cv2.inRange(hsv, lower1, upper1)
        mask = np.maximum(mask, partial_mask)
    
    # mask = smooth_mask(mask)
    mask = cv2.dilate(mask, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)
    
    # Do some clean up
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1000
    uv_seg_lst = []
    # import pdb; pdb.set_trace()
    # for every component in the image, you keep it only if it's above min_size
    seg_temp = np.zeros_like(mask)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            seg_temp[output == i + 1] = 1
    
    contours, _ = cv2.findContours(seg_temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # breakpoint()

    # take the first contour
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,255,255),2)
    # breakpoint()
    print(box) # shape (4, 2)

    # draw contour
    # img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

    # # draw the bounding rectangle
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    # # display the image with bounding rectangle drawn on it
    # cv2.imshow("Bounding Rectangle", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    output = np.zeros_like(img_copy)
    # cv2.rectangle(output, (int(box[0]+offset_x), int(box[1]+offset_y)), (int(box[2]+offset_x), int(box[3]+offset_y)), (255, 255, 255), -1)
    box_copy = box.copy()
    box_copy[:, 0] += offset_x
    box_copy[:, 1] += offset_y
    
    # the left side of the channel
    left = box_copy[np.argsort(box_copy[:, 0])[:2]]
    right = box_copy[np.argsort(box_copy[:, 0])[2:]]
    middle_left = left.mean(axis=0)
    middle_right = right.mean(axis=0)
    upper_left = left[np.argmin(left[:, 1])]
    upper_right = right[np.argmin(right[:, 1])]
    lower_left = left[np.argmax(left[:, 1])]
    lower_right = right[np.argmax(right[:, 1])]
    upper_left = ((upper_left + 3 * middle_left) / 4).astype(int)
    upper_right = ((upper_right + 3 * middle_right) / 4).astype(int)
    lower_left = ((lower_left + 3 * middle_left) / 4).astype(int)
    lower_right = ((lower_right + 3 * middle_right) / 4).astype(int)
    box_copy = np.array([upper_left, upper_right, lower_right, lower_left])
    cv2.drawContours(output, [box_copy], 0, (255, 255, 0), -1)
    
    
    if plot:
        _, axs = plt.subplots(3, 1)
        axs[0].imshow(img_copy)
        axs[1].imshow(mask)
        axs[1].scatter(box[:, 0], box[:, 1], c='r', s=40)
        axs[2].imshow(output)
        plt.show()
        
    output_mask = np.zeros_like(img_copy)[:, :, 0]
    output_mask[output[:, :, 0] > 0] = 1
    return output_mask

if __name__ == "__main__":
    # main()
    
    for i in range(1, 7):
        img = cv2.imread(f"colors/color{i}.png")
        # img = cv2.imread(f"problem_segments_im/rope_mask{i}.png")
        # img = cv2.imread(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_channel_error368.png")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(img)[:, :, 0]
        # cable_lower = np.array([22, 22, 182])
        cable_lower = np.array([22, 35, 200])
        cable_upper = np.array([45, 128, 255])
        rope_mask_img = get_mask(img_rgb, color_bounds=[(cable_lower, cable_upper)], plot=True, dilate=True)

        # skeleton_img = skeletonize_img(rope_mask_img)
        # length_rope, endpoints_rope = find_length_and_endpoints(skeleton_img)
        # print(length_rope)
        # print(endpoints_rope)
        # channel_lower = np.array([11, 16, 29])
        # channel_upper = np.array([179, 100, 140]) # [179, 60, 150]
        # channel_mask_img = get_channel(img_rgb, color_bounds=[(channel_lower, channel_upper)], plot=True)
        # skeleton_img_channel = skeletonize_img(channel_mask_img)
        # length, endpoints = find_length_and_endpoints(skeleton_img_channel)
        # sorted_endpoints = sort_skeleton_pts(skeleton_img_channel, endpoints[0])
        # print(length)
        # print("END points: ",endpoints)
        # # print(sorted_endpoints)
        
    