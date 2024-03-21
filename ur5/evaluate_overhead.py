import cv2
import numpy as np
import matplotlib.pyplot as plt
from resources import (
    CROP_REGION,
    curved_template_mask,
    straight_template_mask,
    trapezoid_template_mask,
    straight_template_mask_align,
)
from new_shape_match import (
    get_channel,
    get_contours,
    get_bbox,
    TEMPLATE_RATIOS,
    TEMPLATES,
)

# from shape_match import get_cable
# from test_stuff import align_channel
from test_stuff import center_mask, rotate_image, best_fit_template
from shape_match import get_channel as get_channel_old

# CROP_REGION = np.array([])


def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    print(np.unique(overlap))
    print(overlap.sum())
    plt.imshow(overlap, cmap="gray")
    plt.show()
    union = (pred_mask + gt_mask) > 0  # Logical OR
    print(np.unique(union))
    print(union.sum())
    plt.imshow(union, cmap="gray")
    plt.show()
    iou = overlap.sum() / float(union.sum())
    return iou


def calculate_rel_inter(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    rel_inter = overlap.sum() / float(gt_mask.sum())
    return rel_inter


def align_channel(
    template_mask, matched_results, img, matched_cnt, matched_template, thinning=None
):
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
    # scale_x, scale_y = int(size[0]), int(size[1])
    REF_SCALE_X, REF_SCALE_Y = 737, 75
    if np.abs(size[0] - REF_SCALE_X) < np.abs(size[0] - REF_SCALE_Y) and np.abs(
        size[1] - REF_SCALE_X
    ) > np.abs(size[1] - REF_SCALE_Y):
        scale_x = REF_SCALE_X
        scale_y = REF_SCALE_Y
    elif np.abs(size[0] - REF_SCALE_X) > np.abs(size[0] - REF_SCALE_Y) and np.abs(
        size[1] - REF_SCALE_X
    ) < np.abs(size[1] - REF_SCALE_Y):
        scale_x = REF_SCALE_Y
        scale_y = REF_SCALE_X
    # print('scale_x = ', scale_x)
    # print('scale_y = ', scale_y)

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
        if thinning is None:
            template_mask = straight_template_mask
        else:
            x, y, _ = np.where(template_mask > 0)
            height = x.max() - x.min() + 1
            new_height = thinning * height
            diff = height - new_height
            template_mask = np.copy(straight_template_mask)
            template_mask[: int(x.min() + diff // 2), :, :] = 0
            template_mask[int(x.max() - diff // 2) :, :, :] = 0
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
    # else:
    #     if thinning is not None:
    #         rot = rotation_angles[best_idx]
    #         rotated_scaled_template_mask = rotate_image(padded_scaled_template_mask, rot)
    #         shift_x, shift_y = int(center[0] + CROP_REGION[2] - rotated_scaled_template_mask.shape[1]//2), int(center[1] + CROP_REGION[0] - rotated_scaled_template_mask.shape[0]//2)
    #         y,x = np.where(rotated_scaled_template_mask[:,:,0] > 0)
    #         thinned_mask = np.zeros_like(img).sum(axis=-1)
    #         thinned_mask[y + shift_y, x + shift_x] = 255
    #         return best_template, thinned_mask

    return best_template, rot


def get_edges(
    gray_img, blur_radius=5, sigma=0, dilate_size=10, canny_threshold=(100, 255)
):
    dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dst = cv2.GaussianBlur(gray_img, (blur_radius, blur_radius), sigma)
    edges = cv2.Canny(dst, canny_threshold[0], canny_threshold[1])
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    edges = cv2.dilate(edges, dilate_kernel)
    # plt.imshow(edges, cmap='gray')
    # plt.show()
    return edges


def get_cable(
    img, blur_radius=5, sigma=0, dilate_size=2, canny_threshold=(100, 255), viz=False
):
    rgb_img = img.copy()
    # gray_img, crop_img= make_img_gray(None, img)
    crop_img = rgb_img[CROP_REGION[0] : CROP_REGION[1], CROP_REGION[2] : CROP_REGION[3]]
    # plt.imshow(crop_img)
    # plt.show()
    orig_gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(orig_gray_img)
    # plt.title('cropped image')
    # plt.show()
    gray_img = cv2.threshold(orig_gray_img.copy(), 240, 255, cv2.THRESH_BINARY_INV)[1]
    # plt.imshow(gray_img)
    # plt.show()

    edges = get_edges(gray_img, blur_radius, sigma, dilate_size, canny_threshold)
    sorted_cnts = get_contours(edges, "external")
    best_cnts = [sorted_cnts[0]]
    # a = input('press enter: ')
    # plt.imshow(cv2.drawContours(rgb_img.copy(), [best_cnts[-1]], -1, 255, 3))
    # # plt.title('check cable contour dimensions')
    # # # a = input('press enter')
    # print('best_cnts[-1] = ', best_cnts[-1])
    # plt.imshow(cv2.drawContours(rgb_img.copy(), [best_cnts[-1] + np.array([CROP_REGION[2], CROP_REGION[0]])], -1, 255, 3))
    # print('best_cnts[-1] + np.array([CROP_REGION[2], CROP_REGION[0]]) = ', best_cnts[-1] + np.array([CROP_REGION[2], CROP_REGION[0]]))
    # plt.title('check cable contour dimensions')
    """best_mask = None
    for i, cnt in enumerate(best_cnts):
        cnt_rgb = cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
        plt.imshow(cv2.drawContours(rgb_img.copy(), [cnt_rgb], -1, 255, 3))
        plt.title('check cable contour dimensions')
        plt.show()
        print(get_bbox(cnt))
        mask = np.zeros_like(crop_img, dtype=np.uint8)
        _ = cv2.drawContours(mask, [cnt], -1, 255, 3)
        mask = cv2.dilate(mask, np.ones((dilate_size,dilate_size), np.uint8), iterations=1)
        mask = mask.sum(axis=-1)
        if i == len(best_cnts) - 1:
            best_mask = mask
    
    # return best_cnts[-1], best_mask"""

    best_cnt = best_cnts[-1]
    # rect, center, size, theta = get_bbox(best_cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # box_rgb = box + np.array([CROP_REGION[2], CROP_REGION[0]])
    # best_mask = cv2.drawContours(np.zeros_like(rgb_img), [box_rgb], -1, (255,255,255), -1)
    best_cnt = best_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    best_mask = cv2.drawContours(
        np.zeros_like(rgb_img), [best_cnt], -1, (255, 255, 255), -1
    )
    best_mask = best_mask.mean(axis=-1)
    print((best_mask > 0).sum())
    print(best_mask.shape)
    print(np.unique(best_mask))
    # plt.imshow(best_mask, cmap='gray')
    # plt.title('check cable contour dimensions')
    # plt.show()

    return best_cnt, best_mask


def get_cable_contour(masked_cable, image, crop_image):
    # NOTE: CHANGED
    sorted_cnts = get_contours(masked_cable, "external")
    # sorted_cnts = [get_contours(edges)[0]]
    matched_template = None
    best_cnt = None
    min_cnt_idx = None
    max_channel_density = 0
    min_cnt_val = np.inf
    matched_results = []
    for i, cnt in enumerate(sorted_cnts):
        rect, center, size, theta = get_bbox(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = size[0] * size[1]
        if area < 1000:
            continue
        ratio = max(size) / min(size)
        dists = [np.abs(t - ratio) for t in TEMPLATE_RATIOS]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        minX_box, maxX_box = np.min(box[:, 0]), np.max(box[:, 0])
        minY_box, maxY_box = np.min(box[:, 1]), np.max(box[:, 1])
        channel_density = crop_image[minY_box:maxY_box, minX_box:maxX_box].sum()

        box_rgb = box + np.array([CROP_REGION[2], CROP_REGION[0]])
        scale_y, scale_x = max(box_rgb[:, 1]) - min(box_rgb[:, 1]), max(
            box_rgb[:, 0]
        ) - min(box_rgb[:, 0])
        true_size = (scale_x, scale_y)
        if min_dist < min_cnt_val and channel_density > max_channel_density:
            matched_results = [rect, center, size, theta, box_rgb]
            min_cnt_val = min_dist
            matched_template = TEMPLATES[min_idx]
            best_cnt = cnt
            min_cnt_idx = i
            max_channel_density = channel_density

    # rect, center, size, theta = get_bbox(best_cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # box_rgb = box + np.array([CROP_REGION[2], CROP_REGION[0]])
    # best_mask = cv2.drawContours(np.zeros_like(image), [box_rgb], -1, (255,255,255), -1)
    best_cnt = best_cnt + np.array([CROP_REGION[2], CROP_REGION[0]])
    best_mask = cv2.drawContours(
        np.zeros_like(image), [best_cnt], -1, (255, 255, 255), -1
    )
    best_mask = best_mask.mean(axis=-1)
    print((best_mask > 0).sum())
    print(best_mask.shape)
    print(np.unique(best_mask))
    # plt.imshow(best_mask, cmap='gray')
    # plt.title('check channel contour dimensions')
    # plt.show()

    return best_cnt, best_mask


FRONT_CHANNEL_HSV = np.array([[0, 0, 0], [179, 255, 44]])
OVERHEAD_CABLE_HSV = np.array([[0, 0, 179], [179, 31, 255]])

# overhead_ref_path = 'results/straight/reference_gt/overhead_view_reference.png'
overhead_ref_path = "../overhead_1_uni.png"
# overhead_ref_path = 'results/straight/confirmed/overhead_0_binary.png'
# overhead_ref_path = 'results/straight/confirmed/overhead_1_uni.png'


"""front_view = cv2.imread(front_ref_path)
crop_front_view = front_view[FRONT_CROP_REGION[0]:FRONT_CROP_REGION[1], FRONT_CROP_REGION[2]:FRONT_CROP_REGION[3]]
plt.imshow(crop_front_view)
plt.show()

print(crop_front_view.shape)
pick_hsv(crop_front_view)"""

overhead_view = cv2.imread(overhead_ref_path)
overhead_view = cv2.cvtColor(overhead_view, cv2.COLOR_BGR2RGB)
# plt.imshow(overhead_view)
# plt.show()
# overhead_view = cv2.resize(overhead_view, (0, 0), fx=6, fy=6)
crop_overhead_view = overhead_view[
    CROP_REGION[0] : CROP_REGION[1], CROP_REGION[2] : CROP_REGION[3]
]
# plt.imshow(crop_overhead_view)
# plt.show()

_, cable_mask = get_cable(overhead_view)
print(cable_mask.shape)
print((cable_mask > 0).sum())
# plt.imshow(cable_mask, cmap='gray')
# plt.show()


# Added arg to force matching template
# 0:'curved', 1:'straight', 2:'trapezoid'
matched_template, matched_results, channel_cnt = get_channel(overhead_view, 1)

if matched_template == "curved":
    template_mask = curved_template_mask
elif matched_template == "straight":
    template_mask = straight_template_mask_align
elif matched_template == "trapezoid":
    template_mask = trapezoid_template_mask

aligned_channel_mask, rot = align_channel(
    template_mask,
    matched_results,
    overhead_view,
    channel_cnt,
    matched_template,
    thinning=None,
)
print(aligned_channel_mask.shape)
print((aligned_channel_mask > 0).sum())
# plt.subplot(1,2,1)
# plt.imshow(aligned_channel_mask, cmap='gray')
# plt.imshow(overhead_view, alpha=0.5)
# plt.title('original aligned channel mask')
# plt.show()
# thinning with skimage
# plt.subplot(1,2,2)
# plt.imshow(channel_thinned, cmap='gray')
# plt.imshow(rotate_image(crop_overhead_view, -rot))
# plt.imshow(template_mask, alpha=0.5)
# plt.title('thinned aligned channel mask')
# plt.show()

# plt.imshow(cable_mask, alpha=0.5)
# plt.imshow(overhead_view, alpha=0.5)

aligned_channel_mask[aligned_channel_mask > 0] = 1
# channel_thinned[channel_thinned>0] = 1
cable_mask[cable_mask > 0] = 1
plt.imshow(rotate_image(template_mask.astype("float32"), rot))
plt.title("comparing masks")
plt.show()
print(calculate_iou(aligned_channel_mask, cable_mask))
# print(calculate_iou(channel_thinned, cable_mask))
# breakpoint()
# print(calculate_rel_inter(aligned_channel_mask, cable_mask))
