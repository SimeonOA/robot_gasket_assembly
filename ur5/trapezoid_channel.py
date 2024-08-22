from utils import *
from resources import *
from shape_match import detect_cable
import matplotlib.pyplot as plt
import math
import numpy as np

def pick_and_place_trap(robot, cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac, idx, num_points, channel_mask=None, last=False, 
                               descending=False, channel_idx1_used=False, channel_idx2_used=False, last_trap_side=False, hybrid=False):
    curr_cable_end = None
    cable_skeleton = skeletonize(cable_mask_binary)
    cable_length, cable_endpoints = find_length_and_endpoints(cable_skeleton)
    sorted_cable_pts = get_sorted_cable_pts(cable_endpoints, cable_skeleton)
    
    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
    if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
        corner1 = pair[0]
        corner2 = pair[1]
    else:
        corner1 = pair[0]
        corner2 = len(sorted_channel_pts)-1-pair[1]

    curr_dist = int(np.abs(corner1-corner2)*idx/num_points)
    if last_trap_side:
        if idx == 0:
            if prev_frac < 0.5:
                cable_idx = 0
            else:
                cable_idx = -1
        elif idx == 1:
            if prev_frac < 0.5:
                cable_idx = int(prev_frac * len(sorted_cable_pts))//2
            else:
                cable_idx = len(sorted_cable_pts) - int((1-prev_frac) * len(sorted_cable_pts))//2
    else:
        if descending:
            cable_idx = int(prev_frac*len(sorted_cable_pts)) - curr_dist
        else:
            cable_idx = int(prev_frac*len(sorted_cable_pts)) + curr_dist

    if idx/num_points != 0.5:
        channel_idx1 = (corner1+corner2)//2 - curr_dist
        channel_idx2 = (corner1+corner2)//2 + curr_dist
        if last_trap_side:
            if np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx1])) < np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx2])):
                channel_idx = channel_idx1
            else:
                channel_idx = channel_idx2
        else:
            if not channel_idx1_used and not channel_idx2_used:
                if np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx1])) < np.linalg.norm(np.array(sorted_cable_pts[cable_idx]) - np.array(sorted_channel_pts[channel_idx2])):
                    channel_idx = channel_idx1
                    channel_idx1_used = True
                    channel_idx2_used = False
                else:
                    channel_idx = channel_idx2
                    channel_idx1_used = False
                    channel_idx2_used = True
            else:
                assert (channel_idx1_used or channel_idx2_used) and not (channel_idx1_used and channel_idx2_used)
                if channel_idx1_used:
                    channel_idx = channel_idx2
                    channel_idx1_used = True
                    channel_idx2_used = True
                else:
                    channel_idx = channel_idx1
                    channel_idx1_used = True
                    channel_idx2_used = True
    else:
        if descending:
            channel_idx = corner1
        else:
            channel_idx = corner2
    pick_pt = sorted_cable_pts[cable_idx] 
    place_pt = sorted_channel_pts[channel_idx]

    swapped_sorted_cable_pts = [(pt[1], pt[0]) for pt in sorted_cable_pts]
    swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
    
    if channel_mask is not None and channel_mask[pick_pt[0]][pick_pt[1]][0] != 0:
        on_channel = True
    else:
        on_channel = False
    pick_pose = robot.get_rw_pose((pick_pt[1], pick_pt[0]), swapped_sorted_cable_pts, 15, matched_template='trapezoid', is_channel_pt=on_channel)
    if hybrid:
        place_pose = robot.get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, matched_template='trapezoid', is_channel_pt=True)
    else:
        place_pose = robot.get_rw_pose((place_pt[1], place_pt[0]), swapped_sorted_channel_pts, 15, matched_template='trapezoid', is_channel_pt=True)
    robot.pick_and_place(pick_pose, place_pose, on_channel)
    robot.go_home()
    if last:
        return prev_frac - np.abs(corner1 - corner2)/(2*len(sorted_channel_pts)) if descending else prev_frac + np.abs(corner1 - corner2)/(2 *len(sorted_channel_pts)), channel_idx1_used, channel_idx2_used
    else:
        return prev_frac, channel_idx1_used, channel_idx2_used

def trapezoid_actuation(channel_cnt, channel_cnt_mask, channel_skeleton, VIZ, overhead_cam, args, PICK_MODE, robot, matched_template):
    corners = get_corners(channel_cnt)
    channel_skeleton_corners = match_corners_to_skeleton(corners, channel_skeleton)
    long_corner0, long_corner1, med_corner0, med_corner1 = classify_corners(channel_skeleton_corners)
    channel_start_pt = long_corner0
    if VIZ:
        plt.title("correct long and med corner pts")
        plt.imshow(channel_skeleton)
        plt.scatter(long_corner0[1], long_corner0[0], c='m')
        plt.scatter(long_corner1[1], long_corner1[0], c='y')
        plt.scatter(med_corner0[1], med_corner0[0], c='c')
        plt.scatter(med_corner1[1], med_corner1[0], c='k')
        plt.scatter(channel_start_pt[1], channel_start_pt[0], c='r')
        plt.show()
    rgb_img = overhead_cam.get_zed_img()
    cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
    sorted_cable_pts, sorted_channel_pts = get_sorted_pts(cable_endpoints, channel_start_pt, cable_skeleton, channel_skeleton, is_trapezoid=True)
    if START_SIDE == 'left':
        if sorted_cable_pts[-1][1] < 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] < 555:
            sorted_channel_pts = sorted_channel_pts[::-1]
    else:
        if sorted_cable_pts[-1][1] >= 555:
            sorted_cable_pts = sorted_cable_pts[::-1]
        if sorted_channel_pts[-1][1] >= 555:
            sorted_channel_pts = sorted_channel_pts[::-1]
    # gets the indices of where each corner is in the sorted channel skeleton
    sorted_channel_pts = np.array(sorted_channel_pts).tolist()
    channel_skeleton_corners = np.array(channel_skeleton_corners).tolist()
    # we always want to sort the channel pts such that long_corner0 is idx 0 and long_corner1_idx < med_corner1_idx < med_corner0_idx
    long_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner0[0] and x[1] == long_corner0[1]][0]
    long_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner1[0] and x[1] == long_corner1[1]][0]
    med_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner0[0] and x[1] == med_corner0[1]][0]
    med_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner1[0] and x[1] == med_corner1[1]][0]

    if np.abs(long_corner0_idx - long_corner1_idx) > len(sorted_channel_pts)/2:
        sorted_channel_pts = sorted_channel_pts[::-1]
        long_corner0_idx = 0
        long_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == long_corner1[0] and x[1] == long_corner1[1]][0]
        med_corner0_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner0[0] and x[1] == med_corner0[1]][0]
        med_corner1_idx = [i for i,x in enumerate(sorted_channel_pts) if x[0] == med_corner1[0] and x[1] == med_corner1[1]][0]
    pairs = [[long_corner0_idx, long_corner1_idx], [long_corner1_idx, med_corner1_idx], [med_corner1_idx, med_corner0_idx], [med_corner0_idx, long_corner0_idx]]

    if PICK_MODE == 'uni':
        prev_frac = 0
        all_num_pts = [8, 4, 4, 4]
        for pair_idx, pair in enumerate(pairs):
            num_pts = all_num_pts[pair_idx]
            for idx in range(num_pts):
                rgb_img = overhead_cam.get_zed_img()
                cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
                prev_frac = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac, idx, num_pts, channel_cnt_mask)
                robot.go_home()

            if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
                place_pts = np.linspace(pair[0], pair[1], num_pts).astype(int)
            else:
                place_pts = np.linspace(pair[0], len(sorted_channel_pts)-1-pair[1], num_pts).astype(int)
            for idx in range(num_pts):
                channel_idx = place_pts[idx]
                press_idx(robot, sorted_channel_pts, channel_idx, trap=True)
            
            place_pt1 = sorted_channel_pts[pair[0]]
            place_pt2 = sorted_channel_pts[pair[1]]
            # needs to be swapped as this is how it is expected for the robot
            swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
            side_len = np.abs(pair[0]-pair[1])
            slide_start_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template, True)
            slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template, True)
            robot.slide_linear(slide_end_pose, slide_start_pose)
            robot.go_home()
    
    if PICK_MODE == 'binary':
        prev_frac1 = 0.5
        prev_frac2 = 0.5

        # long side
        pair = pairs[0]
        def process_step(step_num, prev_frac, cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, mode, channel_cnt_mask,     descending=False, last=False, channel_idx1_used=None, channel_idx2_used=None):
            print(f'STEP {step_num}')
            return pick_and_place_trap(
                cable_mask_binary, cable_endpoints, sorted_channel_pts, pair,
                prev_frac, mode, NUM_PTS, channel_cnt_mask,
                descending=descending, last=last, 
                channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used
            )

        for step_num, (mode, descending, last) in enumerate([
            (0, False, False), (2, False, False), (2, True, False),
            (3, False, False), (3, True, False), (1, False, True), (1, True, True)
        ], start=1):
            rgb_img = overhead_cam.get_zed_img()
            cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
            
            if step_num in [3, 5, 7]:
                prev_frac2, _, _ = process_step(step_num, prev_frac2, cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, mode, channel_cnt_mask, descending, last, channel_idx1_used, channel_idx2_used)
            else:
                prev_frac1, channel_idx1_used, channel_idx2_used = process_step(step_num, prev_frac1, cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, mode, channel_cnt_mask, descending, last)

        if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
            corner1 = pair[0]
            corner2 = pair[1]
        else:
            corner1 = pair[0]
            corner2 = len(sorted_channel_pts)-1-pair[1]
        
        nums = [4, 2, 6, 1, 7, 3, 5]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/8)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template, True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template, True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template, True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        # medium sides
        pair1 = pairs[1]
        pair2 = pairs[3]
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair1, prev_frac1, 0, 4, channel_cnt_mask)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair2, prev_frac2, 0, 4, channel_cnt_mask, descending=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, channel_idx1_used1, channel_idx2_used1 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair1, prev_frac1, 1, 4, channel_cnt_mask)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair2, prev_frac2, 1, 4, channel_cnt_mask,
                                                                                        descending=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair1, prev_frac1, 1, 4, channel_cnt_mask, last=True,
                                                channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair2, prev_frac2, 1, 4, channel_cnt_mask, descending=True, last=True,
                                                        channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2)
        
        if np.abs(pair1[0] - pair1[1]) < len(sorted_channel_pts)//2:
            corner1 = pair1[0]
            corner2 = pair1[1]
        else:
            corner1 = pair1[0]
            corner2 = len(sorted_channel_pts)-1-pair1[1]
        nums = [2, 1, 3]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        if np.abs(pair2[0] - pair2[1]) < len(sorted_channel_pts)//2:
            corner1 = pair2[0]
            corner2 = pair2[1]
        else:
            corner1 = pair2[0]
            corner2 = len(sorted_channel_pts)-1-pair2[1]
        nums = [2, 1, 3]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        print('LAST SIDE -- SHORT SIDE')
        # short side
        pair = pairs[2]
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, channel_idx1_used1, channel_idx2_used1  = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 0, 4, channel_cnt_mask, last_trap_side=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 0, 4, channel_cnt_mask, descending=True, last_trap_side=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 1, 4, channel_cnt_mask, last_trap_side=True,
                                                        channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 1, 4, channel_cnt_mask, descending=True, 
                                                        last_trap_side=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2)
        
        if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
            corner1 = pair[0]
            corner2 = pair[1]
        else:
            corner1 = pair[0]
            corner2 = len(sorted_channel_pts)-1-pair[1]
        nums = [2, 1, 3]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]
        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()
    
    if PICK_MODE == 'hybrid':
        prev_frac1 = 0.5
        prev_frac2 = 0.5
        # long side
        pair = pairs[0]
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)

        print('STEP 1')
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 0, 4, channel_cnt_mask, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        print('STEP 2')
        prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 1, 4, channel_cnt_mask, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        print('STEP 3')
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 1, 4, channel_cnt_mask, 
                                                descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        print('STEP 4')
        prev_frac1, channel_idx1_used, channel_idx2_used = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 2, 4, channel_cnt_mask, last=True, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        print('STEP 5')
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 2, 4, channel_cnt_mask, 
                                                        descending=True, channel_idx1_used=channel_idx1_used, channel_idx2_used=channel_idx2_used, last=True, hybrid=True)

        if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
            corner1 = pair[0]
            corner2 = pair[1]
        else:
            corner1 = pair[0]
            corner2 = len(sorted_channel_pts)-1-pair[1]
        
        nums = [2, 1, 3, 0, 4]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_mid_pose, slide_end_pose)
        robot.go_home()

        # medium sides
        pair1 = pairs[1]
        pair2 = pairs[3]
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, channel_idx1_used1, channel_idx2_used1 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair1, prev_frac1, 0, 4, channel_cnt_mask, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair2, prev_frac2, 0, 4, channel_cnt_mask, descending=True, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair1, prev_frac1, 2, 4, channel_cnt_mask, last=True,
                                                                                        channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair2, prev_frac2, 2, 4, channel_cnt_mask,
                                                                                        descending=True, last=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2, hybrid=True)
        
        if np.abs(pair1[0] - pair1[1]) < len(sorted_channel_pts)//2:
            corner1 = pair1[0]
            corner2 = pair1[1]
        else:
            corner1 = pair1[0]
            corner2 = len(sorted_channel_pts)-1-pair1[1]
        nums = [2, 4]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot,sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[corner1]
        place_pt2 = sorted_channel_pts[corner2]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_start_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_start_pose, slide_end_pose)
        robot.go_home()

        if np.abs(pair2[0] - pair2[1]) < len(sorted_channel_pts)//2:
            corner1 = pair2[0]
            corner2 = pair2[1]
        else:
            corner1 = pair2[0]
            corner2 = len(sorted_channel_pts)-1-pair2[1]
        
        nums = [2, 4]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot, sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[corner1]
        place_pt2 = sorted_channel_pts[corner2]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_start_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_start_pose, slide_end_pose)
        robot.go_home()

        print('LAST SIDE -- SHORT SIDE')
        # short side
        pair = pairs[2]
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, channel_idx1_used1, channel_idx2_used1  = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 1, 4, channel_cnt_mask, last_trap_side=True, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac2, channel_idx1_used2, channel_idx2_used2 = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 1, 4, channel_cnt_mask, descending=True, last_trap_side=True, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        prev_frac1, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac1, 0, 4, channel_cnt_mask, last_trap_side=True,
                                                        channel_idx1_used=channel_idx1_used1, channel_idx2_used=channel_idx2_used1, hybrid=True)
        
        rgb_img = overhead_cam.get_zed_img()
        cable_skeleton, cable_length, cable_endpoints, cable_mask_binary = detect_cable(rgb_img, args)
        print('CHECK ISSUE!!')
        prev_frac2, _, _ = pick_and_place_trap(cable_mask_binary, cable_endpoints, sorted_channel_pts, pair, prev_frac2, 0, 4, channel_cnt_mask, descending=True, 
                                                        last_trap_side=True, channel_idx1_used=channel_idx1_used2, channel_idx2_used=channel_idx2_used2, hybrid=True)
        
        if np.abs(pair[0] - pair[1]) < len(sorted_channel_pts)//2:
            corner1 = pair[0]
            corner2 = pair[1]
        else:
            corner1 = pair[0]
            corner2 = len(sorted_channel_pts)-1-pair[1]
        nums = [1, 3, 2]
        for num in nums:
            channel_idx = corner1 + int(np.abs(corner1-corner2)*num/4)
            press_idx(robot, sorted_channel_pts, channel_idx, trap=True)
        
        place_pt1 = sorted_channel_pts[(corner1+corner2)//2]
        place_pt2 = sorted_channel_pts[corner1]

        swapped_sorted_channel_pts = [(pt[1], pt[0]) for pt in sorted_channel_pts]
        side_len = np.abs(corner1-corner2)
        slide_mid_pose = robot.get_rw_pose((place_pt1[1], place_pt1[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_end_pose, slide_mid_pose)
        robot.go_home()

        place_pt2 = sorted_channel_pts[corner2]
        slide_end_pose = robot.get_rw_pose((place_pt2[1], place_pt2[0]), swapped_sorted_channel_pts[::-1], side_len, matched_template,True)
        robot.slide_linear(slide_end_pose, slide_mid_pose)
        robot.go_home()