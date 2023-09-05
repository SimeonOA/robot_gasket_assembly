from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
# from sensing.utils_binary import *
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import distance_transform_edt
# from DSE-skeleton-pruning-master.dsepruning import skel_pruning_DSE
from plantcv import plantcv as pcv

# img_path = '/Users/karimel-refai/robot_cable_insertion/franka/imgs/straight10_mask (1).png'
img_path = '/Users/karimel-refai/robot_cable_insertion/franka/imgs/straight2_mask.png'
# img_path = '/Users/karimel-refai/robot_cable_insertion/franka/imgs/trapezoid8_mask.png'

point_coords_fore = np.array(
            [
            [
                  576.0,
                  277.0
            ],
            [
                  687.0,
                  241.0
            ],
            [
                  848.0,
                  218.0
            ],
            [
                  711.0,
                  245.0
            ],
            [
                  579.0,
                  276.0
            ],
            [
                  642.0,
                  262.0
            ],
            [
                  687.0,
                  248.0
            ],
            [
                  553.0,
                  283.0
            ],
            [
                  474.0,
                  299.0
            ],
            [
                  822.0,
                  220.0
            ]
      ])
point_coords_back = np.array([
            [
                  532.0,
                  644.0
            ],
            [
                  268.0,
                  287.0
            ],
            [
                  1007.0,
                  287.0
            ],
            [
                  190.0,
                  438.0
            ],
            [
                  999.0,
                  473.0
            ],
            [
                  883.0,
                  408.0
            ],
            [
                  1266.0,
                  275.0
            ],
            [
                  745.0,
                  318.0
            ],
            [
                  304.0,
                  76.0
            ],
            [
                  432.0,
                  42.0
            ]
      ])
# breakpoint()
point_labels = np.concatenate((np.ones(len(point_coords_fore)), np.zeros(len(point_coords_back))))
point_coords = np.concatenate((point_coords_fore, point_coords_back))

# img_path = '/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask1.png'
# point_coords = np.array([[700,179],[750,158],[907, 111], [583, 246],])
# point_labels = np.array([1,1,0, 0])


def get_mask():
    img = cv2.imread(img_path)
    skeleton = skeletonize_img(img)
    # dist = distance_transform_edt(img, return_indices=False, return_distances=True)
    # skeleton_prune = skel_pruning_DSE(skeleton, dist, 100)
    nonzero_pts = cv2.findNonZero(np.float32(skeleton))
    # how many pixels we set the size to be before we prune it
    # set it to be around 15% of the total number of pixels
    size_prune = round(len(nonzero_pts)*0.80)
    if len(skeleton.shape) == 3:
        skeleton = skeleton[:,:,1]
        skeleton = skeleton.astype(np.uint8)
    skeleton = skeleton.astype(np.uint8)
    skeleton_prune, _, _ = pcv.morphology.prune(skel_img=skeleton, size=70)
    plt.imshow(skeleton_prune)
    plt.show()
    find_length_and_endpoints(skeleton_prune)
    img_overlay = cv2.imread('/Users/karimel-refai/robot_cable_insertion/franka/imgs/straight10.png')

    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(img)

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, ious, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)

    # print(ious)

    i = 0
    for idx, mask in enumerate(masks):
        print(ious[idx])
        plt.imshow(img_overlay)
        plt.imshow(mask, cmap='jet', alpha=0.7)
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

# def get_channel_sorted_list(skeleton_img_channel, endpoint):
#     endpoint = (endpoint[1], endpoint[0])
#     sorted_pixel_list = sort_skeleton_pts(skeleton_img_channel, endpoint)
#     for i, pt in enumerate(sorted_pixel_list):
#         sorted_pixel_list[i] = (pt[1], pt[0])
#     # sorted_pixel_list = filter_points(sorted_pixel_list, point_cloud)
#     return sorted_pixel_list

def skeletonize_img(img):
    img[img[:,:,2] < 250] = 0
    img[img[:,:,2] > 250] = 1
    img = img[:,:,0]
    gray = img*255
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    image = cv2.threshold(gray,30,1,cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    # blurred_image = gaussian_filter(image, sigma=1)

    # dilate the image to just eliminate risk of holes causes severances in the skeleton
    dilated_img = image.copy()
    kernel = np.ones((5,5), np.uint8)
    cv2.dilate(image, kernel, dst=dilated_img, iterations=1)
    # perform skeletonization
    skeleton = skeletonize(dilated_img)

    #find candidates who 

    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    return skeleton

def find_length_and_endpoints(skeleton_img):
    # breakpoint()
    # instead of doing DFS just do BFS and the last 2 points to have which end up having no non-visited neighbor 

    #### IDEA: do DFS but have a left and right DFS with distances for one being negative and the other being positive 
    # done because skeleton from sam mask has 3 channels for some reason
    # only valid channel is the green one
    if len(skeleton_img.shape) == 3:
        skeleton_img = skeleton_img[:,:,1]
        skeleton_img = skeleton_img.astype(bool)
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))

    if nonzero_pts is None:
        nonzero_pts = [[[0,0]]]
    total_length = len(nonzero_pts)
    # pdb.set_trace()
    start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])
    # run dfs from this start_pt, when we encounter a point with no more non-visited neighbors that is an endpoint
    endpoints = []
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    q = [start_pt]
    dist_q = [0]
    # tells us if the first thing we look at is actually an endpoint
    initial_endpoint = False
    # carry out floodfill
    q = [start_pt]
    paths = [[]]
    # carry out floodfill
    IS_LOOP = False
    IS_LOOP_VISIT = False
    # ENTIRE_VISITED = [False] * total_length
    def dfs(q, dist_q, visited, paths, start_pixel, increment_amt):
        '''
        q: queue with next point on skeleton for one direction
        dist_q: queue with distance from start point to next point for one direction
        visited: queue with visited points for only one direction
        increment_amt: counter that indicates direction +/- 1
        '''

        # is_loop = ENTIRE_VISITED[start_pixel + increment_amt*len(visited)]
        # if is_loop:
        #     return is_loop


        while len(q) > 0:
            next_loc = q.pop()
            distance = dist_q.pop()
            # paths[-1].append(next_loc)
            # if next_loc in visited:
            #     return True # we have a loop
            # print("distance: ",distance)
            visited.add(next_loc)
            counter = 0
            count_visited = []
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                # print("count_visited ", count_visited)
                if (test_loc in visited):
                    count_visited.append(test_loc)
                    if len(count_visited) >= 3:
                        plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
                        plt.scatter(x = [j[1] for j in count_visited], y=[i[0] for i in count_visited],c='m')
                        plt.scatter(x = [j[1] for j in visited], y=[i[0] for i in visited],c='r')
                        plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
                        plt.scatter(x=next_loc[1], y=next_loc[0], c='b')
                        plt.imshow(skeleton_img)
                        plt.show()
                        # return True
                    continue
                if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                    continue
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    counter += 1
                    #length_checker += 1
                    q.append(test_loc)
                    dist_q.append(distance+increment_amt)
            # this means we haven't added anyone else to the q so we "should" be at an endpoint
            if counter == 0:
                endpoints.append([next_loc, distance])
            # if next_loc == start_pt and counter == 1:
            #     endpoints.append([next_loc, distance])
            #     initial_endpoint = True
        return False # we don't have a loop
    counter = 0
    length_checker = 0
    increment_amt = 1
    visited = set([start_pt])
    for n in NEIGHS:
        test_loc = (start_pt[0]+n[0], start_pt[1]+n[1])
        # one of the neighbors is valued at one so we can dfs across it
        if skeleton_img[test_loc[0]][test_loc[1]] == True:
            counter += 1
            q = [test_loc]
            dist_q = [0]
            start_pixel = test_loc[0]*len(skeleton_img[0]) + test_loc[1]
            # breakpoint()
            IS_LOOP = dfs(q, dist_q, visited, paths, start_pixel, increment_amt)
            # if IS_LOOP:
            #     break
            # the first time our distance will be incrementing but the second time
            # , i.e. when dfs'ing the opposite direction our distance will be negative to differentiate both paths
            increment_amt = -1
    # we only have one neighbor therefore we must be an endpoint
    # only works for the start point so we need to check
    if counter == 1:
        distance = 0
        endpoints.insert(0, [start_pt, distance])
        initial_endpoint = True

    # pdb.set_trace()
    final_endpoints = []
    # largest = second_largest = None
    # for pt, distance in endpoints:
    #     if largest is None or distance > endpoints[largest][1]:
    #         second_largest = largest
    #         largest = endpoints.index([pt, distance])
    #     elif second_largest is None or distance > endpoints[second_largest][1]:
    #         second_largest = endpoints.index([pt, distance])
    # if initial_endpoint:
    #     final_endpoints = [endpoints[0][0], endpoints[largest][0]]
    # else:
    #     final_endpoints = [endpoints[second_largest][0], endpoints[largest][0]]
    
    largest_pos = largest_neg = None
    print("IS LOOP:", IS_LOOP)
    print("IS LOOP VISIT: ", IS_LOOP_VISIT)
    if IS_LOOP:
        final_endpoints = [endpoints[0][0], (0,0)] # MAYBE RAND INDEX LATER
    else:
        for pt, distance in endpoints:
            if largest_pos is None or distance > endpoints[largest_pos][1]:
                largest_pos = endpoints.index([pt, distance])
            elif largest_neg is None or distance < endpoints[largest_neg][1]:
                largest_neg = endpoints.index([pt, distance])
        if initial_endpoint:
            final_endpoints = [endpoints[0][0], endpoints[largest_pos][0]]
        else:
            final_endpoints = [endpoints[largest_neg][0], endpoints[largest_pos][0]]
    breakpoint()
    print("num endpoints = ", len(endpoints))
    print("endpoins are: ", endpoints)
    branch_endpoints = endpoints.copy()
    branch_endpoints = [x[0] for x in branch_endpoints]
    for final_endpoint in final_endpoints:
        branch_endpoints.remove(final_endpoint)
    pruned_skeleton = prune_branches(branch_endpoints, skeleton_img.copy())

    plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.title("final endpoints")
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 

    print("the final endpoints are: ", final_endpoints)
    breakpoint()
    # pdb.set_trace()
    # display results

    print("the total length is ", total_length)
    return total_length, final_endpoints

def test_skeletons():
    mask = cv2.imread(img_path)
    # Compute the medial axis (skeleton) and the distance transform
    # skel, distance = medial_axis(mask, return_distance=True)

    # Compare with other skeletonization algorithms
    skeleton = skeletonize(mask)
    skeleton_lee = skeletonize(mask, method='lee')

    # Distance to the background for pixels of the skeleton
    # dist_on_skel = distance * skel

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(mask, cmap=plt.cm.gray)
    ax[0].set_title('original')
    ax[0].axis('off')

    # ax[1].imshow(dist_on_skel, cmap='magma')
    # ax[1].contour(mask, [0.5], colors='w')
    # ax[1].set_title('medial_axis')
    # ax[1].axis('off')

    ax[2].imshow(skeleton, cmap=plt.cm.gray)
    ax[2].set_title('skeletonize')
    ax[2].axis('off')

    ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
    ax[3].set_title("skeletonize (Lee 94)")
    ax[3].axis('off')

    fig.tight_layout()
    plt.show()
# remove the branches by following them until we reach the main branch while simultaneously setting all those pixels to false
def prune_branches(branch_endpoints, skeleton_img):
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    # all of the pixels that we passed by
    past_pixels_lst = []
    save_pixels = []
    for branch_endpoint in branch_endpoints:
        if branch_endpoint in save_pixels:
            continue
        q = [branch_endpoint]
        while len(q) > 0:
            next_loc = q.pop()
            visited.add(next_loc)
            count = 0
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    past_pixels_lst.append(test_loc)
                    count += 1
                    if test_loc not in visited:
                        q.append(test_loc)
            # means we've reached the main branch
            if count >= 3:
                breakpoint()
                # need to remove these cause they're probably actually a part of the main branch
                # need to also figure out a way to remove branch_endpoints from the q if they

# THIS IS IMPORTANT READ THIS LINE!!!!! #################################################################


            # need to also figure out a way to remove branch_endpoints from the q if they are a part of the 3/4 neighbors that our count found
            # thinking of having those points be in a remove list and then doing .remove from the q instead of trying to eliminate the last x cause the same point will appear twice
            # could also have 

            
# READ THE PARAGRAPH IN BETWEEN!!!!! ########################################################################################################
                # save_pixels.extend()
                q = q[:-count]
                # saved_pixels = past_pixels_lst[count:]
                plt.scatter(x=[i[1] for i in past_pixels_lst], y=[i[0] for i in past_pixels_lst], c='g')
                past_pixels_lst = past_pixels_lst[:-count]
                for past_pixel in past_pixels_lst:
                    skeleton_img[past_pixel[0]][past_pixel[1]] = False
                # plt.scatter(x=[i[1] for i in saved_pixels], y=[i[0] for i in saved_pixels], c='g')
                plt.scatter(x=[i[1] for i in past_pixels_lst], y=[i[0] for i in past_pixels_lst], c='b')
                plt.scatter(x=[i[1] for i in q], y=[i[0] for i in q], c='r')
                plt.scatter(x=[i[1] for i in branch_endpoints], y=[i[0] for i in branch_endpoints], c='y')
                plt.scatter(x=branch_endpoint[1], y=branch_endpoint[0], c='w')
                plt.scatter(x=next_loc[1], y=next_loc[0], c='m')
                plt.imshow(skeleton_img)
                plt.show()
                break
        # sets all of those passed pixels to False
    # breakpoint()
    plt.title("pruned skeleton")
    plt.imshow(skeleton_img)
    plt.show()
    return skeleton_img

# test_skeletons()
channel_mask = get_mask()




# skeleton_channel_img = skeletonize_img(channel_mask)
# total_channel_pixels, channel_endpoints = find_length_and_endpoints(skeleton_channel_img) 
# channel_end1, channel_end2 = channel_endpoints[0], channel_endpoints[-1]
# sorted_channel_pixels = get_channel_sorted_list(skeleton_channel_img, channel_end1)
# channel_end1, channel_end2 = sorted_channel_pixels[0], sorted_channel_pixels[-1]

# def bfs(skeleton_img):
#     if len(skeleton_img.shape) == 3:
#         skeleton_img = skeleton_img[:,:,1]
#         skeleton_img = skeleton_img.astype(bool)
#     nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))
#     NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
#     start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])
#     # run dfs from this start_pt, when we encounter a point with no more non-visited neighbors that is an endpoint
#     endpoints = []
#     visited = set()
#     q = [start_pt]
#     dist_q = [0]
#     initial_endpoint = False
#     q = [start_pt]
#     IS_LOOP = False
#     while len(q) > 0:




def dfs_with_all_paths(image, start_pixel, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    x, y = start_pixel

    visited.add(start_pixel)
    path.append(start_pixel)

    width, height = len(image[0]), len(image)

    # Define possible moves: up, down, left, right
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1,-1), (-1,1), (1,-1),(1,1)]

    valid_paths = []  # Store all valid paths found

    counter = 0
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy

        # Check if the new position is within the image bounds
        if 0 <= new_x < width and 0 <= new_y < height:
            new_pixel = (new_y, new_x)

            if image[new_y][new_x] == True and new_pixel not in visited:
                counter += 1
                # Recursively explore the unvisited neighboring pixel
                new_paths = dfs_with_all_paths(image, new_pixel, visited.copy(), path.copy())
                valid_paths.extend(new_paths)

    # If the current pixel is the destination, add the current path to valid paths
    if counter == 0:
        valid_paths.append(path)

    return valid_paths

# this version relies on getting all of the paths and then filtering them out based on the ones that are the farthest from the start point
def find_length_and_endpoints_modified(skeleton_img):
    if len(skeleton_img.shape) == 3:
        skeleton_img = skeleton_img[:,:,1]
        skeleton_img = skeleton_img.astype(bool)
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))
    if nonzero_pts is None:
        nonzero_pts = [[[0,0]]]
    start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])
    x,y = start_pt
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1,-1), (-1,1), (1,-1),(1,1)]
    width, height = len(skeleton_img[0]), len(skeleton_img)
    visited = set()

    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < width and 0 <= new_y < height:
            new_pixel = (new_y, new_x)

            if skeleton_img[new_y][new_x] == True and new_pixel not in visited:
# everything below this line is stuff idk!!!
                return 
    return