from new_shape_match import get_channel
from shape_match import get_cable
import argparse
from skimage.morphology import skeletonize, thin
from skimage.transform import probabilistic_hough_line
from resources import (
    CROP_REGION,
    curved_template_mask,
    straight_template_mask,
    trapezoid_template_mask,
    straight_template_mask_align,
)
import matplotlib.pyplot as plt
import cv2
import numpy as np

TEMPLATES = {0: "curved", 1: "straight", 2: "trapezoid"}

argparser = argparse.ArgumentParser()
argparser.add_argument("--img_path", type=str, default="zed_images/curved_1.png")
argparser.add_argument("--blur_radius", type=int, default=5)
argparser.add_argument("--sigma", type=int, default=0)
argparser.add_argument("--dilate_size_channel", type=int, default=2)
argparser.add_argument("--canny_threshold_channel", type=tuple, default=(100, 255))
argparser.add_argument("--visualize", default=False, action="store_true")


def center_mask(mask, image):
    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = image.shape[:2]
    x_translation = (image_width - mask_width) // 2
    y_translation = (image_height - mask_height) // 2

    # Create an empty image of the same size as the input image
    translated_image = np.zeros_like(image)

    # Apply the translation to the mask and place it in the center of the image
    translated_image[
        y_translation : y_translation + mask_height,
        x_translation : x_translation + mask_width,
    ] = mask
    return translated_image


def rotate_image(image, angle):
    # Rotate image by a specified angle in degrees
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image


def best_fit_template(all_masks, img, matched_cnt):
    matched_mask = np.zeros_like(img, dtype=np.uint8)
    matched_cnt = matched_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cv2.drawContours(matched_mask, [matched_cnt], -1, 255, -1)
    matched_mask = matched_mask.sum(axis=-1)
    # plt.imshow(matched_mask, cmap="gray")
    # plt.show()
    most_ones = -np.inf
    best_mask = None
    best_idx = None
    for i, template_mask in enumerate(all_masks):
        """plt.subplot(1,2,1)
        plt.imshow(template_mask, cmap='gray')
        # plt.show()
        plt.subplot(1,2,2)"""
        overlap = np.bitwise_and(template_mask, matched_mask)
        """plt.imshow(overlap, cmap='gray')
        plt.show()
        print('overlap sum = ', overlap.sum())"""
        if overlap.sum() > most_ones:
            most_ones = overlap.sum()
            best_mask = template_mask
            best_idx = i
    """print('best idx = ', i)"""
    return best_mask, best_idx, matched_mask


def align_channel(template_mask, matched_results, img, matched_cnt, matched_template):
    for k, v in TEMPLATES.items():
        if v == matched_template:
            matched_template = k
            break

    rect, center, size, theta, box = matched_results
    rotation_angles = [
        theta,
        -theta,
        90 - theta,
        90 + theta,
        180 - theta,
        180 + theta,
        270 + theta,
        270 - theta,
    ]
    scale_x, scale_y = int(size[0]), int(size[1])

    """plt.imshow(template_mask, cmap='gray')
    plt.title('Padded template mask')
    plt.show()"""

    if np.abs(scale_x - template_mask.shape[1]) < np.abs(
        scale_y - template_mask.shape[1]
    ):
        scaled_template_mask = cv2.resize(
            template_mask, (scale_x, scale_y), interpolation=cv2.INTER_LINEAR
        )
    else:
        scaled_template_mask = cv2.resize(
            template_mask, (scale_y, scale_x), interpolation=cv2.INTER_LINEAR
        )
    """plt.imshow(scaled_template_mask)
    plt.title('Scaled template mask')
    plt.show()"""
    padded_scaled_template_mask = center_mask(scaled_template_mask, img)
    """plt.imshow(padded_scaled_template_mask, cmap='gray')
    plt.title('Padded template mask')
    plt.show()"""
    all_masks = []
    for rot in rotation_angles:
        """print('rot = ', rot)"""
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(
            center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1] // 2
        ), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0] // 2)
        y, x = np.where(rotated_scaled_template_mask[:, :, 0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask

        all_masks.append(shifted_rotated_scaled_template_mask)

    best_template, best_idx = best_fit_template(all_masks, img, matched_cnt)

    if matched_template == 1:
        template_mask = straight_template_mask
        if np.abs(scale_x - template_mask.shape[1]) < np.abs(
            scale_y - template_mask.shape[1]
        ):
            scaled_template_mask = cv2.resize(
                template_mask, (scale_x, scale_y), interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_template_mask = cv2.resize(
                template_mask, (scale_y, scale_x), interpolation=cv2.INTER_LINEAR
            )
        padded_scaled_template_mask = center_mask(scaled_template_mask, img)
        rot = rotation_angles[best_idx]
        rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
        shift_x, shift_y = int(
            center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1] // 2
        ), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0] // 2)
        y, x = np.where(rotated_scaled_template_mask[:, :, 0] > 0)
        mask = np.zeros_like(img).sum(axis=-1)
        mask[y + shift_y, x + shift_x] = 255
        shifted_rotated_scaled_template_mask = mask
        best_template = shifted_rotated_scaled_template_mask

    """plt.imshow(best_template)
    plt.imshow(img, alpha=0.5)
    plt.scatter(center[0]+CROP_REGION[2], center[1]+CROP_REGION[0], c='r')
    plt.title('Best template')
    plt.show()"""

    return best_template


if __name__ == "__main__":
    args = argparser.parse_args()
    image = cv2.imread(args.img_path)
    matched_template, matched_results, channel_cnt = get_channel(image)
    # breakpoint()

    if matched_template == "curved":
        template_mask = curved_template_mask
    elif matched_template == "straight":
        template_mask = straight_template_mask_align
    elif matched_template == "trapezoid":
        template_mask = trapezoid_template_mask

    """channel_cnt = channel_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    p = cv2.arcLength(channel_cnt, True) # cnt is the rect Contours
    appr = cv2.approxPolyDP(channel_cnt, 0.02*p, True) # appr contains the 4 points

    appr = sorted(appr, key=lambda c: c[0][0])

    from scipy.spatial.distance import cdist

    if len(appr) == 5:
        temp = np.array(appr).reshape(-1,2)
        pairwise_dist = cdist(temp, temp) + np.eye(temp.shape[0])*1e6
        nearest_neigh_loc = np.array(np.where(pairwise_dist == np.min(pairwise_dist)))
        assert np.array_equal(nearest_neigh_loc, nearest_neigh_loc.T)
        temp = np.array([temp[nearest_neigh_loc[0][0]], temp[nearest_neigh_loc[0][1]]])
        cnt = channel_cnt.reshape(-1,2)
        keep = [x for x in cnt if x[0] >= min(temp[0][0], temp[1][0]) and x[0] <= max(temp[0][0],temp[1][0]) and x[1] >= min(temp[0][1],temp[1][1]) and x[1] <= max(temp[0][1],temp[1][1])]
        interp_idx = np.argmax(cdist(temp[:2],keep).sum(axis=0))
        new_pt = keep[interp_idx].reshape(-1,2)
        appr = [x for i, x in enumerate(appr) if i not in nearest_neigh_loc[0]]
        appr.insert(nearest_neigh_loc[0][0], new_pt)

    #pa = top lef point
    #pb = bottom left point
    #pc = top right point
    #pd = bottom right point
    # breakpoint()

    pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
    pc, pd = sorted(appr[2:4], key=lambda c: c[0][1])

    def classify_corners_new(channel_skeleton_corners):
        dist0 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[1])
        dist1 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[2])
        dist2 = np.linalg.norm(channel_skeleton_corners[1]-channel_skeleton_corners[3])
        dist3 = np.linalg.norm(channel_skeleton_corners[2]-channel_skeleton_corners[3])
        breakpoint()
        max_dist = max([dist0,dist1, dist2, dist3])
        # long_cornerX and med_cornerX are the corners on the same side of the trapezoid with one being a corner for the long side and the other being the corner for the medium side
        if max_dist == dist0:
            long_corner0 = channel_skeleton_corners[0]
            long_corner1 = channel_skeleton_corners[1]
            med_corner0 = channel_skeleton_corners[2]
            med_corner1 = channel_skeleton_corners[3]
        elif max_dist == dist1:
            long_corner0 = channel_skeleton_corners[0]
            long_corner1 = channel_skeleton_corners[2]
            med_corner0 = channel_skeleton_corners[1]
            med_corner1 = channel_skeleton_corners[3]
        elif max_dist == dist2:
            long_corner0 = channel_skeleton_corners[1]
            long_corner1 = channel_skeleton_corners[3]
            med_corner0 = channel_skeleton_corners[0]
            med_corner1 = channel_skeleton_corners[2]
        else:
            long_corner0 = channel_skeleton_corners[2]
            long_corner1 = channel_skeleton_corners[3]
            med_corner0 = channel_skeleton_corners[0]
            med_corner1 = channel_skeleton_corners[1]
        
        return long_corner0, long_corner1, med_corner0, med_corner1
    
    def classify_corners(channel_skeleton_corners):
        dist0 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[1])
        dist1 = np.linalg.norm(channel_skeleton_corners[1]-channel_skeleton_corners[2])
        dist2 = np.linalg.norm(channel_skeleton_corners[0]-channel_skeleton_corners[3])
        max_dist = max([dist0,dist1, dist2])
        # long_cornerX and med_cornerX are the corners on the same side of the trapezoid with one being a corner for the long side and the other being the corner for the medium side
        if max_dist == dist0:
            long_corner0 = channel_skeleton_corners[0]
            long_corner1 = channel_skeleton_corners[1]
            med_corner0 = channel_skeleton_corners[3]
            med_corner1 = channel_skeleton_corners[2]
        elif max_dist == dist1:
            long_corner0 = channel_skeleton_corners[1]
            long_corner1 = channel_skeleton_corners[2]
            med_corner0 = channel_skeleton_corners[0]
            med_corner1 = channel_skeleton_corners[3]
        else:
            long_corner0 = channel_skeleton_corners[0]
            long_corner1 = channel_skeleton_corners[3]
            med_corner0 = channel_skeleton_corners[1]
            med_corner1 = channel_skeleton_corners[2]
        
        return long_corner0, long_corner1, med_corner0, med_corner1
    
    long_corner0, long_corner1, med_corner0, med_corner1 = classify_corners_new([pa, pb,pc,pd])
    plt.subplot(1,2,1)
    plt.imshow(cv2.drawContours(image.copy(), [channel_cnt], -1, 255, 3))
    plt.scatter(long_corner0[0,0], long_corner0[0,1], c='r')
    plt.scatter(long_corner1[0,0], long_corner1[0,1], c='b')
    plt.scatter(med_corner0[0,0], med_corner0[0,1], c='g')
    plt.scatter(med_corner1[0,0], med_corner1[0,1], c='k')
    plt.title('check channel contour dimensions in get_channel')
    # plt.show()


    plt.subplot(1,2,2)
    plt.imshow(cv2.drawContours(image.copy(), [channel_cnt], -1, 255, 3))
    plt.scatter(pa[0,0], pa[0,1], c='r')
    plt.scatter(pb[0,0], pb[0,1], c='b')
    plt.scatter(pc[0,0], pc[0,1], c='g')
    plt.scatter(pd[0,0], pd[0,1], c='k')
    plt.title('check channel contour dimensions in get_channel')
    plt.show()"""

    aligned_channel_mask = align_channel(
        template_mask, matched_results, image, channel_cnt, matched_template
    )
    aligned_channel_mask = cv2.merge(
        (aligned_channel_mask, aligned_channel_mask, aligned_channel_mask)
    )
    aligned_channel_mask = aligned_channel_mask.astype("uint8")
    plt.imshow(image)
    plt.imshow(aligned_channel_mask, alpha=0.5)
    plt.show()
    channel_skeleton = skeletonize(aligned_channel_mask)
    # plt.subplot(1,2,1)
    plt.imshow(channel_skeleton)
    plt.imshow(image, alpha=0.5)
    plt.title("overlayed channel skeleton and rgb image")
    plt.show()

    # breakpoint()
    """from skimage.draw import line as get_line_pixels
    aligned_channel_mask_2d = np.mean(aligned_channel_mask, axis=2)
    lines = probabilistic_hough_line(aligned_channel_mask_2d, line_length=10)
    cleaned = np.zeros_like(image)
    for ((r0, c0), (r1, c1)) in lines:
        rr, cc = get_line_pixels(r0, c0, r1, c1)
        cleaned[cc, rr] = 255
    plt.subplot(1,2,2)
    plt.imshow(cleaned)
    # plt.imshow(channel_skeleton)
    plt.imshow(image, alpha=0.5)
    plt.title('NEW overlayed channel skeleton and rgb image')
    plt.show()"""

    # aligned_channel_mask_2d = np.mean(aligned_channel_mask, axis=2)
    # channel_thinned = thin(aligned_channel_mask_2d, max_num_iter=25)
    # plt.subplot(1,2,2)
    # plt.imshow(channel_thinned)
    # plt.imshow(image, alpha=0.5)
    # plt.title('overlayed channel thinned and rgb image')
    # plt.show()

    cable_cnt, _ = get_cable(image)
    cable_cnt = cable_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    cable_contour = cv2.drawContours(
        np.zeros(image.shape, np.uint8), [cable_cnt], -1, 255, 3
    )
    blue_channel_mask = cv2.merge(
        (
            aligned_channel_mask[:, :, 0] * 0,
            aligned_channel_mask[:, :, 0],
            aligned_channel_mask[:, :, 0] * 0,
        )
    )

    cable_mask_binary = np.zeros(image.shape, np.uint8)
    cv2.drawContours(cable_mask_binary, [cable_cnt], -1, 255, -1)
    cable_mask_binary = (cable_mask_binary.sum(axis=2) / 255).astype("uint8")
    cable_mask_binary = cv2.morphologyEx(
        cable_mask_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    red_cable_mask = cv2.merge(
        (cable_mask_binary * 0, cable_mask_binary * 255, cable_mask_binary * 0)
    )
    masks = red_cable_mask + blue_channel_mask
    mask_img = cv2.addWeighted(image, 0.7, masks, 0.7, 0)
    plt.imshow(mask_img)
    plt.show()

    cable_skeleton = skeletonize(cable_mask_binary)

    red_cable_skeleton = cv2.merge(
        (cable_skeleton * 0, cable_skeleton * 255, cable_skeleton * 0)
    )
    blue_channel_skeleton = channel_skeleton  # cv2.merge((channel_skeleton[:,:,1]*0, channel_skeleton[:,:,1]*255, channel_skeleton[:,:,1]*0))
    skeletons = red_cable_skeleton + blue_channel_skeleton
    skeleton_img = cv2.addWeighted(image, 0.3, skeletons.astype(np.uint8), 1, 0)
    plt.imshow(skeleton_img)
    plt.show()

    breakpoint()

    cv2.imwrite("/zed_image/trapezoid_skeleton_overlay.png", skeleton_img)
    cv2.imwrite("/zed_image/trapezoid_mask_overlay.png", mask_img)
