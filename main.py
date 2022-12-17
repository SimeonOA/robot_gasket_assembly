from interface_rws import Interface
from yumirws.yumi import YuMiArm, YuMi
from push import push_action_endpoints
import cv2
from scipy.ndimage.filters import gaussian_filter
from grasp import Grasp, GraspSelector
from tcps import *
from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage, CameraIntrinsics, Point, PointCloud
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from rotate import rotate_from_pointcloud, rotate
import time
import os
import sys
cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0, cable)
behavior_cloning_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../multi-fidelity-behavior-cloning"
sys.path.insert(0, behavior_cloning_path)

DISPLAY = False
TWO_ENDS = False
PUSH_DETECT = True


def click_points(img):
    # left click mouse for pick point, right click for place point
    fig, ax = plt.subplots()
    ax.imshow(img.color.data)
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    pick, place = None, None

    def onclick(event):
        xind, yind = int(event.xdata), int(event.ydata)
        coords = (xind, yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
        nonlocal pick, place
        point = iface.T_PHOXI_BASE*points_3d[lin_ind]
        print("Clicked point in world coords: ", point)
        if (point.z > .5):
            print("Clicked point with no depth info!")
            return
        if (event.button == 1):
            pick = coords
        elif (event.button == 3):
            place = coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return pick, place


# SPEED=(.5,6*np.pi)
SPEED = (.025, 0.3*np.pi)
# iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
#                   ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
iface = Interface("1703005", METAL_GRIPPER.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
                  METAL_GRIPPER.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)


def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)


def take_action(pick, place, angle):

    # handle the actual grabbing motion
    # l_grasp=None
    # r_grasp=None
    # single grasp
    #grasp = g.single_grasp(pick,.007,iface.R_TCP)
    # g.col_interface.visualize_grasps([grasp.pose],iface.R_TCP)
    #wrist = grasp.pose*iface.R_TCP.inverse()
    print("grabbing with left arm")
    # r_grasp=grasp
    # r_grasp.pose.from_frame=YK.r_tcp_frame
    # iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.set_speed((.1, 1))
    #yumi_left = iface.y.left
    # print(yumi_left.get_pose())

    iface.go_delta(l_trans=[0, 0, 0.2])  # lift
    # if angle != 0:
    #    pick[2] = pick[2] + .05
    iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
                                                 from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # if angle != 0:
    #    rotate(angle, iface)
    #    iface.go_delta(l_trans=[0, 0, -.05])
    # iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]),
    #                                            from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))

    iface.close_grippers()
    time.sleep(3)
    iface.go_delta(l_trans=[0, 0, 0.1])  # lift
    time.sleep(1)
    iface.set_speed((.1, 1))
    delta = [place[i] - pick[i] for i in range(3)]
    change_height = 0
    delta[2] = delta[2] + change_height
    iface.go_delta(l_trans=delta)
    time.sleep(1)
    iface.go_delta(l_trans=[0, 0, -0.06])
    time.sleep(3)
    iface.open_grippers()
    iface.home()
    iface.sync()
    # iface.set_speed(SPEED)
    iface.set_speed((.1, 1))
    time.sleep(2)
    iface.open_grippers()


def take_action_2(pick, pick_2, place, place_2):

    print("grabbing with left arm")
    # GRIP LEFT
    iface.set_speed((.1, 1))
    iface.go_delta(l_trans=[0, 0, 0.2])  # lift
    iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
                                                 from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # GRIP RIGHT
    iface.go_delta(r_trans=[0, 0, 0.23])  # lift
    time.sleep(3)
    iface.go_cartesian(r_targets=[RigidTransform(translation=pick_2, rotation=Interface.GRIP_DOWN_R,
                                                 from_frame=YK.r_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    iface.close_grippers()
    time.sleep(3)
    # LIFT AND MOVE LEFT
    iface.go_delta(l_trans=[0, 0, 0.18])  # lift
    time.sleep(1)
    delta = [place[i] - pick[i] for i in range(3)]
    change_height = 0
    delta[2] = delta[2] + change_height
    iface.go_delta(l_trans=delta)
    # LIFT AND MOVE RIGHT
    iface.go_delta(r_trans=[0, 0, 0.09])  # lift
    time.sleep(1)
    delta = [place_2[i] - pick_2[i] for i in range(3)]
    change_height = 0
    delta[2] = delta[2] + change_height
    # Re-write go-delta because previous was error!
    l_delta, r_delta = None, None
    r_trans = delta
    if r_trans is not None:
        r_cur = iface.y.right.get_pose()
        r_delta = RigidTransform(
            translation=r_trans, from_frame=r_cur.to_frame, to_frame=r_cur.to_frame)
        r_new = r_delta*r_cur
    if r_delta is not None:
        iface.y.right.goto_pose(r_new, speed=iface.speed)
    # DROP BOTH
    time.sleep(2)
    iface.go_delta(l_trans=[0, 0, -0.12])
    #iface.go_delta(r_trans=[0, 0, -0.015])
    time.sleep(3)
    iface.open_grippers()
    time.sleep(2)
    iface.go_delta(l_trans=[0, 0, 0.1])
    iface.go_delta(r_trans=[0, 0, 0.1])
    iface.home()
    iface.sync()
    time.sleep(2)


while True:
    iface.home()
    iface.open_grippers()
    iface.sync()
    # set up a simple interface for clicking on a point in the image
    img = iface.take_image()

    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    # NEW --------------------------------------------------------------------------------

    # ----------------------Find brightest pixel for segment_cable
    if DISPLAY:
        plt.imshow(img.color.data, interpolation="nearest")
        plt.show()
    three_mat_color = img.color.data
    pixel_r = 0
    pixel_c = 0
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    lower = 0
    upper = 190
    delete_later = []
    max_score = 0
    max_scoring_loc = (0, 0)
    highlight_upper = 256
    highlight_lower = 254
    for r in range(len(three_mat_color)):
        for c in range(len(three_mat_color[r])):
            if (highlight_lower < three_mat_color[r][c][0] <= highlight_upper and 210 < three_mat_color[r][c][1] <= highlight_upper and 210 < three_mat_color[r][c][2] <= highlight_upper):
                curr_score = 0
                for add in range(1, 10):
                    if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[max(0, r-add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[r][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[max(0, r-add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                if (curr_score > max_score):
                    max_scoring_loc = (c, r)
                    max_score = curr_score

            if (lower < three_mat_color[r][c][1] < upper):
                delete_later += [(c, r)]
    copy_color = copy.deepcopy(three_mat_color)
    remove_glare_tolerance = 170
    remove_glare_tolerance_2 = 120
    for r in range(len(copy_color)):
        for c in range(len(copy_color[r])):
            dist = dist = np.linalg.norm(
                np.array([r-max_scoring_loc[1], c-max_scoring_loc[0]]))
            if dist < remove_glare_tolerance:
                copy_color[r][c][0] = 0.0
                copy_color[r][c][1] = 0.0
                copy_color[r][c][2] = 0.0
            if dist < remove_glare_tolerance_2:
                delete_later += [(c, r)]
    highlight_lower = 200
    highlight_upper = 255
    max_score = 0
    max_scoring_loc = (0, 0)
    for r in range(len(copy_color)):
        for c in range(len(copy_color[r])):
            if (highlight_lower < copy_color[r][c][0] <= highlight_upper and 210 < copy_color[r][c][1] <= highlight_upper and 210 < copy_color[r][c][2] <= highlight_upper):
                curr_score = 0
                for add in range(1, 13):
                    if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[max(0, r-add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[r][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[r][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[min(len(copy_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[max(0, r-add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if (highlight_lower < copy_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                if (curr_score > max_score):
                    max_scoring_loc = (c, r)
                    max_score = curr_score
    if DISPLAY:
        plt.imshow(copy_color, interpolation="nearest")
        plt.show()
    loc = max_scoring_loc
    print("Starting segmenet_cable pt: "+str(max_scoring_loc))
    # ----------------------Segment
    rope_cloud, _ = g.segment_cable(loc)
    # ----------------------Remove block

    new_transf = iface.T_PHOXI_BASE.inverse()
    transformed_rope_cloud = new_transf.apply(rope_cloud)
    di = iface.cam.intrinsics.project_to_image(
        transformed_rope_cloud, round_px=False)
    if DISPLAY:
        plt.imshow(di._image_data(), interpolation="nearest")
        plt.show()

    di_data = di._image_data()
    for delete in delete_later:
        di_data[delete[1]][delete[0]] = [float(0), float(0), float(0)]

    mask = np.zeros((len(di_data), len(di_data[0])))
    loc_list = [loc]

    # modified segment_cable code to build a mask for the cable

    # pick the brightest rgb point in the depth image
    # increment in each direction for it's neighbors looking to see if it meets the thresholded rgb value
    # if not, continue
    # if yes set it's x,y position to the mask matrix with the value 1
    # add that value to the visited list so that we don't go back to it again

    new_di_data = np.zeros((len(di_data), len(di_data[0])))
    xdata = []
    ydata = []

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            new_di_data[r][c] = di_data[r][c][0]
            if (new_di_data[r][c] > 0):
                xdata += [c]
                ydata += [r]

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (new_di_data[r][c] != 0):
                curr_edges = 0
                for add in range(1, 8):
                    if (new_di_data[min(len(new_di_data)-add, r+add)][c] != 0):
                        curr_edges += 1
                    if (new_di_data[max(0, r-add)][c] != 0):
                        curr_edges += 1
                    if (new_di_data[r][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if (new_di_data[r][max(0, c-add)] != 0):
                        curr_edges += 1
                    if (new_di_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if (new_di_data[min(len(new_di_data)-add, r+add)][max(0, c-add)] != 0):
                        curr_edges += 1
                    if (new_di_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if (new_di_data[max(0, r-add)][max(0, c-add)] != 0):
                        curr_edges += 1
                if (curr_edges < 11):
                    new_di_data[r][c] = 0.0

    new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    if DISPLAY:
        plt.imshow(new_di._image_data(), interpolation="nearest")
        plt.show()

    new_di_data = gaussian_filter(new_di_data, sigma=1)

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    save_loc = (0, 0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
                save_loc = (c, r)
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    compress_factor = 30
    rows_comp = int(math.floor(len(di_data)/compress_factor))
    cols_comp = int(math.floor(len(di_data[0])/compress_factor))
    compressed_map = np.zeros((rows_comp, cols_comp))

    for r in range(rows_comp):
        if r != 0:
            r = float(r) - 0.5
        for c in range(cols_comp):
            if c != 0:
                c = float(c) - 0.5
            for add in range(1, 5):
                if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(c*compress_factor)] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(max(0, r*compress_factor-add))][int(c*compress_factor)] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(r*compress_factor)][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(r*compress_factor)][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(max(0, r*compress_factor-add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if (new_di_data[int(max(0, r*compress_factor-add))][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
    max_edges = 0
    test_locs = (0, 0)
    for r in range(len(compressed_map)):
        for c in range(len(compressed_map[r])):
            if (compressed_map[r][c] != 0):
                curr_edges = 0
                for add in range(1, 2):
                    if (compressed_map[min(len(compressed_map)-add, r+add)][c] == 0):
                        curr_edges += 1
                    if (compressed_map[max(0, r-add)][c] == 0):
                        curr_edges += 1
                    if (compressed_map[r][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if (compressed_map[r][max(0, c-add)] == 0):
                        curr_edges += 1
                    if (compressed_map[min(len(compressed_map)-add, r+add)][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if (compressed_map[min(len(compressed_map)-add, r+add)][max(0, c-add)] == 0):
                        curr_edges += 1
                    if (compressed_map[max(0, r-add)][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if (compressed_map[max(0, r-add)][max(0, c-add)] == 0):
                        curr_edges += 1
                if (curr_edges > max_edges):
                    test_loc = (c, r)
                    max_edges = curr_edges
    print(test_loc)
    print("scaled: " +
          str((test_loc[0]*compress_factor, test_loc[1]*compress_factor)))
    all_solns = []
    tightness = 0
    while (True):
        all_solns = []
        for r in range(len(compressed_map)):
            for c in range(len(compressed_map[r])):
                if (compressed_map[r][c] != 0):
                    curr_edges = 0
                    for add in range(1, 2):
                        if (compressed_map[min(len(compressed_map)-add, r+add)][c] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][c] == 0):
                            curr_edges += 1
                        if (compressed_map[r][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[r][max(0, c-add)] == 0):
                            curr_edges += 1
                        if (compressed_map[min(len(compressed_map)-add, r+add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[min(len(compressed_map)-add, r+add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if (compressed_map[max(0, r-add)][max(0, c-add)] == 0):
                            curr_edges += 1
                    if (max_edges-tightness <= curr_edges <= max_edges+tightness):
                        all_solns += [(c, r)]
        print("ALL SOLUTIONS TIGHTNESS "+str(tightness) + ": "+str(all_solns))
        if (len(all_solns) >= 2):
            min_x = 100000
            max_x = 0
            for soln in all_solns:
                if soln[0] < min_x:
                    min_x = soln[0]
                if soln[0] > max_x:
                    max_x = soln[0]
            if (max_x-min_x) > 2:
                break
        tightness += 1

    origin_x = 0
    origin_y = 0
    min_dist = 100000000
    max_dist = 0
    min_all_solns = (0, 0)
    max_all_solns = (0, 0)
    for soln in all_solns:
        dist = np.linalg.norm(
            np.array([origin_x-soln[1], origin_y-soln[0]]))
        if dist < min_dist:
            min_dist = dist
            min_all_solns = soln
        if TWO_ENDS:
            if dist > max_dist:
                max_dist = dist
                max_all_solns = soln

    scaled_test_loc = [min_all_solns[0]*compress_factor,
                       min_all_solns[1]*compress_factor]
    scaled_test_loc_2 = []
    if TWO_ENDS:
        scaled_test_loc_2 = [max_all_solns[0]*compress_factor,
                             max_all_solns[1]*compress_factor]
    if (scaled_test_loc[0] != 0):
        scaled_test_loc[0] = scaled_test_loc[0] - int(compress_factor/2)
    if (scaled_test_loc[1] != 0):
        scaled_test_loc[1] = scaled_test_loc[1] - int(compress_factor/2)
    if TWO_ENDS:
        if (scaled_test_loc_2[0] != 0):
            scaled_test_loc_2[0] = scaled_test_loc_2[0] - \
                int(compress_factor/2)
        if (scaled_test_loc_2[1] != 0):
            scaled_test_loc_2[1] = scaled_test_loc_2[1] - \
                int(compress_factor/2)
    if DISPLAY:
        plt.imshow(compressed_map, interpolation="nearest")
        plt.show()
    min_dist = 10000
    min_dist_2 = 1000
    candidate_rope_loc = (0, 0)
    candidate_rope_loc_2 = (0, 0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if (di_data[r][c][0] != 0):
                dist = np.linalg.norm(
                    np.array([r-scaled_test_loc[1], c-scaled_test_loc[0]]))
                if (dist < min_dist):
                    candidate_rope_loc = (c, r)
                    min_dist = dist
                if TWO_ENDS:
                    dist_2 = np.linalg.norm(
                        np.array([r-scaled_test_loc_2[1], c-scaled_test_loc_2[0]]))
                    if (dist_2 < min_dist_2):
                        candidate_rope_loc_2 = (c, r)
                        min_dist_2 = dist_2
    min_loc = candidate_rope_loc
    min_loc_2 = (0, 0)
    if TWO_ENDS:
        min_loc_2 = candidate_rope_loc_2
    print("FITTED POINT: " + str(min_loc))
    if TWO_ENDS:
        print("FITTED POINT OF OTHER END: " + str(min_loc_2))
    if DISPLAY:
        plt.imshow(new_di_data, interpolation="nearest")
        plt.show()

    # ----------------------FIND END OF CHANNEL
    lower = 254
    upper = 256
    channel_start = (0, 0)
    max_edges = 0

    for r in range(len(three_mat_color)):
        for c in range(len(three_mat_color[r])):
            if (lower < three_mat_color[r][c][0] < upper):
                curr_edges = 0
                for add in range(1, 11):
                    if (lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[max(0, r-add)][c][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[r][max(0, c-add)][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if (lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] < upper):
                        curr_edges += 1
                if (curr_edges > max_edges):
                    max_edges = curr_edges
                    channel_start = (c, r)
    print("CHANNEL_START: "+str(channel_start))
    channel_cloud, _ = g.segment_channel(channel_start)
    transformed_channel_cloud = new_transf.apply(channel_cloud)
    image_channel = iface.cam.intrinsics.project_to_image(
        transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
    image_channel_data = image_channel._image_data()
    copy_channel_data = copy.deepcopy(image_channel_data)
    lower = 80
    upper = 255

    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if (new_di_data[r][c] != 0):
                image_channel_data[r][c][0] = 0.0
                image_channel_data[r][c][1] = 0.0
                image_channel_data[r][c][2] = 0.0

    # Finish Thresholding, now find corner to place
    if DISPLAY:
        plt.imshow(copy_channel_data, interpolation="nearest")
        plt.show()
    for r in range(len(copy_channel_data)):
        for c in range(len(copy_channel_data[r])):
            if copy_channel_data[r][c][0] != 0:
                copy_channel_data[r][c][0] = 255.0
                copy_channel_data[r][c][1] = 255.0
                copy_channel_data[r][c][2] = 255.0
    img_skeleton = cv2.cvtColor(copy_channel_data, cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(img_skeleton, 10, 0.01, 200)
    print("OPEN CV2 FOUND FEATURES: ", features)
    endpoints = [x[0] for x in features]

    closest_to_origin = (0, 0)
    furthest_from_origin = (0, 0)
    min_dist = 10000000
    max_dist = 0
    for endpoint in endpoints:
        dist = np.linalg.norm(np.array([endpoint[0], endpoint[1]-400]))
        if dist < min_dist:
            min_dist = dist
            closest_to_origin = endpoint
    for endpoint in endpoints:
        dist = np.linalg.norm(
            np.array([closest_to_origin[0]-endpoint[0], closest_to_origin[1]-endpoint[1]]))
        if dist > max_dist:
            max_dist = dist
            furthest_from_origin = endpoint
    endpoints = [closest_to_origin, furthest_from_origin]
    print("ENDPOINTS SELECTED: " + str(endpoints))
    if DISPLAY:
        plt.imshow(copy_channel_data, interpolation="nearest")
        plt.show()
    # ----------------------FIND END OF CHANNEL
    pick = min_loc
    pick_2 = (0, 0)
    if TWO_ENDS:
        pick_2 = min_loc_2
    # Use estimation
    place = (0, 0)
    place_2 = (0, 0)
    # Use left side
    if (endpoints[0][0] < endpoints[1][0]):
        place = endpoints[0]
        if TWO_ENDS:
            place_2 = endpoints[1]
    else:
        place = endpoints[1]
        if TWO_ENDS:
            place_2 = endpoints[0]
    print("ACTUAL PLACE: "+str(place))
    if TWO_ENDS:
        print("ACTUAL PLACE 2: "+str(place_2))
    assert pick is not None and place is not None

    # Convert Pick, Place, and Pick 2 to world coordinates:
    xind, yind = pick
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    xind2, yind2 = place
    lin_ind2 = int(img.depth.ij_to_linear(np.array(xind2), np.array(yind2)))

    xind3, yind3 = pick_2
    lin_ind3 = 0
    if TWO_ENDS:
        lin_ind3 = int(img.depth.ij_to_linear(
            np.array(xind3), np.array(yind3)))

    xind4, yind4 = place_2
    lin_ind4 = 0
    if TWO_ENDS:
        lin_ind4 = int(img.depth.ij_to_linear(
            np.array(xind4), np.array(yind4)))

    points_3d = iface.cam.intrinsics.deproject(img.depth)
    point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    point = [p for p in point]
    point[2] -= 0.0028  # manually adjust height a tiny bit
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]
    point_2 = None
    if TWO_ENDS:
        point_2 = iface.T_PHOXI_BASE*points_3d[lin_ind3]
        point_2 = [p for p in point_2]
        point_2[2] -= 0.003  # manually adjust height a tiny bit
    # Convert Channel End and Channel start to world coordinates
    xind, yind = place
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    place_point_2 = (0, 0)
    if TWO_ENDS:
        place_point_2 = iface.T_PHOXI_BASE*points_3d[lin_ind4]

    xind, yind = endpoints[0]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    endpoint_1_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

    xind, yind = endpoints[1]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    endpoint_2_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

    new_place_point_data = np.array(
        [place_point.y, place_point.x, place_point.z])
    new_place_point = Point(new_place_point_data, frame=place_point.frame)

    new_endpoint_1_point_data = np.array(
        [endpoint_1_point.y, endpoint_1_point.x, endpoint_1_point.z])
    new_endpoint_1_point = Point(
        new_endpoint_1_point_data, frame=endpoint_1_point.frame)

    new_endpoint_2_point_data = np.array(
        [endpoint_2_point.y, endpoint_2_point.x, endpoint_2_point.z])
    new_endpoint_2_point = Point(
        new_endpoint_2_point_data, frame=endpoint_2_point.frame)

    # FIND THE PRINCIPLE AXIS FOR THE GRIPPER AND ROTATE, THEN TAKE ACTION

    rope_cloud_data_x = rope_cloud.x_coords
    rope_cloud_data_y = rope_cloud.y_coords
    rope_cloud_data_z = rope_cloud.z_coords
    tolerance = 0.03
    new_rope_cloud_data_x = []
    new_rope_cloud_data_y = []
    new_rope_cloud_data_z = []
    for count in range(len(rope_cloud_data_x)):
        dist = np.linalg.norm(np.array(
            [rope_cloud_data_x[count]-point[0], rope_cloud_data_y[count]-point[1]]))
        if dist < tolerance:
            new_rope_cloud_data_x += [rope_cloud_data_x[count]]
            new_rope_cloud_data_y += [rope_cloud_data_y[count]]
            new_rope_cloud_data_z += [rope_cloud_data_z[count]]
    new_rope_cloud_data_x = np.array(new_rope_cloud_data_x)
    new_rope_cloud_data_y = np.array(new_rope_cloud_data_y)
    new_rope_cloud_data_z = np.array(new_rope_cloud_data_z)

    new_rope_cloud = PointCloud(np.array(
        [new_rope_cloud_data_x, new_rope_cloud_data_y, new_rope_cloud_data_z]), frame=rope_cloud.frame)

    transformed_new_rope_cloud = new_transf.apply(new_rope_cloud)
    di_2 = iface.cam.intrinsics.project_to_image(
        transformed_new_rope_cloud, round_px=False)
    if DISPLAY:
        plt.imshow(di_2._image_data(), interpolation="nearest")
        plt.show()
    # START PICK AND PLACE ___________________________________
    if not TWO_ENDS:
        take_action(point, place_point, 0)
        # print("skip")
    else:
        take_action_2(point, point_2, place_point, place_point_2)

    # PACKING __________________________________________________
    if not PUSH_DETECT:
        push_action_endpoints(
            new_place_point, [new_endpoint_1_point, new_endpoint_2_point], iface)
    else:
        while (True):
            push_action_endpoints(
                new_place_point, [new_endpoint_1_point, new_endpoint_2_point], iface, False)
            img = iface.take_image()
            depth = img.depth.data
            print(depth)
            start = np.array([endpoints[0][0], endpoints[0][1]])
            end = np.array([endpoints[1][0], endpoints[1][1]])
            move_vector = (end-start)/np.linalg.norm(end-start)
            current = copy.deepcopy(start)
            loop_again = False
            tolerance = 0.0042
            interval_scaling = 4
            print("START: ", start)
            print("END: ", end)
            print("MOVE VECTOR: ", move_vector)
            for count in range(210):
                curr_depth = depth[int(math.floor(current[1]))][int(
                    math.floor(current[0]))]
                depth_lower = depth[int(math.floor(
                    current[1]+9))][int(math.floor(current[0]))]
                print("CURRENT POINT: ", [int(math.floor(current[0])), int(math.floor(
                    current[1]))], " CURRENT DEPTH: ", curr_depth, " DEPTH_LOWER: ", depth_lower)
                if (curr_depth != 0 and depth_lower != 0 and (abs(depth_lower - curr_depth) > tolerance)):
                    loop_again = True
                    print("EXCEEDED DEPTH TOLERANCE!")
                current[0] += move_vector[0]*interval_scaling
                current[1] += move_vector[1]*interval_scaling
            if DISPLAY:
                plt.imshow(img.depth.data, interpolation="nearest")
                plt.show()
            if not loop_again:
                break

    break

print("Done with script, can end")
