# from ../../cable_untangling.interface_rws import Interface
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
# from ../../cable_untangling.tcps import *
# from ../../cable_untangling.grasp import Grasp,GraspSelector
import time
import os
import sys
cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0, cable)
behavior_cloning_path = os.path.dirname(os.path.abspath(
    __file__)) + "/../../multi-fidelity-behavior-cloning"
sys.path.insert(0, behavior_cloning_path)
#from analysis import CornerPullingBCPolicy

DISPLAY = True


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
        if(point.z > .5):
            print("Clicked point with no depth info!")
            return
        if(event.button == 1):
            pick = coords
        elif(event.button == 3):
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


def take_action(pick, place):

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
    iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
                                                 from_frame=YK.l_tcp_frame, to_frame='base_link')], nwiggles=(10, 10), rot=(.0, .0))
    # iface.go_cartesian(r_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
    #     from_frame = YK.r_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
    iface.close_grippers()
    time.sleep(3)
    iface.go_delta(l_trans=[0, 0, 0.1])  # lift
    time.sleep(1)
    iface.set_speed((.1, 1))
    delta = [place[i] - pick[i] for i in range(3)]
    change_height = 0.002
    delta[2] = delta[2] + change_height
    iface.go_delta(l_trans=delta)
    #iface.go_delta(l_trans=[0, 0, -0.06])
    iface.open_grippers()
    iface.home()
    iface.sync()
    # iface.set_speed(SPEED)
    iface.set_speed((.1, 1))
    time.sleep(2)
    iface.open_grippers()


#policy = CornerPullingBCPolicy()
while True:
    #q = input("Enter to home arms, anything else to quit\n")
    # if not q=='':break
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
            if(highlight_lower < three_mat_color[r][c][0] <= highlight_upper and 210 < three_mat_color[r][c][1] <= highlight_upper and 210 < three_mat_color[r][c][2] <= highlight_upper):
                curr_score = 0
                for add in range(1, 10):
                    if(highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[max(0, r-add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[r][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[min(len(three_mat_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[max(0, r-add)][min(len(three_mat_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                if(curr_score > max_score):
                    max_scoring_loc = (c, r)
                    max_score = curr_score

            if(lower < three_mat_color[r][c][1] < upper):
                delete_later += [(c, r)]
            # if(c == 500):
            #    print("X: " + str(c)+" Y: "+str(r)+" R: "+str(three_mat_color[r][c][0]) + " G: "+str(three_mat_color[r][c][1]) + " B:" +str(three_mat_color[r][c][2]) + " AVG: ")
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
                delete_later+= [(c, r)]
    highlight_lower = 200
    highlight_upper = 255
    max_score = 0
    max_scoring_loc = (0,0)
    for r in range(len(copy_color)):
        for c in range(len(copy_color[r])):
            if(highlight_lower < copy_color[r][c][0] <= highlight_upper and 210 < copy_color[r][c][1] <= highlight_upper and 210 < copy_color[r][c][2] <= highlight_upper):
                curr_score = 0
                for add in range(1, 13):
                    if(highlight_lower < copy_color[min(len(copy_color)-add, r+add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[max(0, r-add)][c][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[r][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[r][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[min(len(copy_color)-add, r+add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[min(len(copy_color)-add, r+add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[max(0, r-add)][min(len(copy_color[0])-add, c+add)][0] <= highlight_upper):
                        curr_score += 1
                    if(highlight_lower < copy_color[max(0, r-add)][max(0, c-add)][0] <= highlight_upper):
                        curr_score += 1
                if(curr_score > max_score):
                    max_scoring_loc = (c, r)
                    max_score = curr_score
    if DISPLAY:
        plt.imshow(copy_color, interpolation="nearest")
        plt.show()
    loc = max_scoring_loc
    print("Starting segmenet_cable pt: "+str(max_scoring_loc))
    # print(loc)
    # print(delete_later)
    # print(delete_later)
    # ----------------------Segment
    rope_cloud, _ = g.segment_cable(loc)
    # print(rope_cloud.data)
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

    #mask = np.ones((len(di_data),len(di_data[0])))
    new_di_data = np.zeros((len(di_data), len(di_data[0])))
    xdata = []
    ydata = []

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            # if(mask[r][c] == 1):
            #    new_di_data[r][c] = di_data[r][c][0] # This actually changes the depth data so, use di_data if you need the depth
            new_di_data[r][c] = di_data[r][c][0]
            if (new_di_data[r][c] > 0):
                xdata += [c]
                ydata += [r]

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                curr_edges = 0
                for add in range(1, 8):
                    if(new_di_data[min(len(new_di_data)-add, r+add)][c] != 0):
                        curr_edges += 1
                    if(new_di_data[max(0, r-add)][c] != 0):
                        curr_edges += 1
                    if(new_di_data[r][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if(new_di_data[r][max(0, c-add)] != 0):
                        curr_edges += 1
                    if(new_di_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if(new_di_data[min(len(new_di_data)-add, r+add)][max(0, c-add)] != 0):
                        curr_edges += 1
                    if(new_di_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)] != 0):
                        curr_edges += 1
                    if(new_di_data[max(0, r-add)][max(0, c-add)] != 0):
                        curr_edges += 1
                if(curr_edges < 11):
                    new_di_data[r][c] = 0.0

    new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    if DISPLAY:
        plt.imshow(new_di._image_data(), interpolation="nearest")
        plt.show()
    # plt.savefig("Isolated_Cable")
    # new_di.save("Isolated_Cable.png")

    # Simeon: Why four times?

    new_di_data = gaussian_filter(new_di_data, sigma=1)

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    save_loc = (0, 0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
                save_loc = (c, r)
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    compress_factor = 30
    # print(int(math.floor(len(di_data)/compress_factor)))
    # print(int(math.floor(len(di_data[0])/compress_factor)))
    rows_comp = int(math.floor(len(di_data)/compress_factor))
    cols_comp = int(math.floor(len(di_data[0])/compress_factor))
    compressed_map = np.zeros((rows_comp, cols_comp))

    for r in range(rows_comp):
        if r != 0:
                    r = float(r) - 0.5
        for c in range(cols_comp):
            if c != 0:
                    c = float(c) - 0.5
            for add in range(1, 8):
                if(new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(c*compress_factor)] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(max(0, r*compress_factor-add))][int(c*compress_factor)] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(r*compress_factor)][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(r*compress_factor)][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(min(len(new_di_data)-add, r*compress_factor+add))][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(max(0, r*compress_factor-add))][int(min(len(new_di_data[0])-add, c*compress_factor+add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
                if(new_di_data[int(max(0, r*compress_factor-add))][int(max(0, c*compress_factor-add))] != 0):
                    compressed_map[int(r)][int(c)] = 255
                    break
    max_edges = 0
    test_locs = (0, 0)
    for r in range(len(compressed_map)):
        for c in range(len(compressed_map[r])):
            if(compressed_map[r][c] != 0):
                curr_edges = 0
                for add in range(1, 2):
                    if(compressed_map[min(len(compressed_map)-add, r+add)][c] == 0):
                        curr_edges += 1
                    if(compressed_map[max(0, r-add)][c] == 0):
                        curr_edges += 1
                    if(compressed_map[r][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if(compressed_map[r][max(0, c-add)] == 0):
                        curr_edges += 1
                    if(compressed_map[min(len(compressed_map)-add, r+add)][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if(compressed_map[min(len(compressed_map)-add, r+add)][max(0, c-add)] == 0):
                        curr_edges += 1
                    if(compressed_map[max(0, r-add)][min(len(compressed_map[0])-add, c+add)] == 0):
                        curr_edges += 1
                    if(compressed_map[max(0, r-add)][max(0, c-add)] == 0):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    test_loc = (c, r)
                    max_edges = curr_edges
    print(test_loc)
    print("scaled: " +
          str((test_loc[0]*compress_factor, test_loc[1]*compress_factor)))
    all_solns = []
    tightness = 0
    while(True):
        all_solns = []
        for r in range(len(compressed_map)):
            for c in range(len(compressed_map[r])):
                if(compressed_map[r][c] != 0):
                    curr_edges = 0
                    for add in range(1, 2):
                        if(compressed_map[min(len(compressed_map)-add, r+add)][c] == 0):
                            curr_edges += 1
                        if(compressed_map[max(0, r-add)][c] == 0):
                            curr_edges += 1
                        if(compressed_map[r][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(compressed_map[r][max(0, c-add)] == 0):
                            curr_edges += 1
                        if(compressed_map[min(len(compressed_map)-add, r+add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(compressed_map[min(len(compressed_map)-add, r+add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        if(compressed_map[max(0, r-add)][min(len(compressed_map[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(compressed_map[max(0, r-add)][max(0, c-add)] == 0):
                            curr_edges += 1
                    if(max_edges-tightness <= curr_edges <= max_edges+tightness):
                        all_solns += [(c, r)]
        print("ALL SOLUTIONS TIGHTNESS "+str(tightness)+ ": "+str(all_solns))
        #if(4 <= len(all_solns)):
        #    break
        if(len(all_solns) >= 2):
            min_y = 100000
            max_y = 0
            for soln in all_solns:
                if soln[1] < min_y:
                    min_y = soln[1]
                if soln[1] > max_y:
                    max_y = soln[1]
            if (max_y-min_y) > 2:
                break
        else:
            tightness += 1
    origin_x = len(compressed_map)/4
    origin_y = len(compressed_map[0])/2
    min_dist = 100000000
    min_all_solns = (0,0)
    for soln in all_solns:
        dist = np.linalg.norm(
                    np.array([origin_x-soln[1], origin_y-soln[0]]))
        if dist < min_dist:
            min_dist = dist
            min_all_solns = soln
    #if(3 <= len(all_solns_tight) <= 4):

    # for soln in all_solns:
    
    scaled_test_loc = [min_all_solns[0]*compress_factor,
                       min_all_solns[1]*compress_factor]
    if(scaled_test_loc[0] != 0):
        scaled_test_loc[0] = scaled_test_loc[0] - int(compress_factor/2)
    if(scaled_test_loc[1] != 0):
        scaled_test_loc[1] = scaled_test_loc[1] - int(compress_factor/2)
    if DISPLAY:
        plt.imshow(compressed_map, interpolation="nearest")
        plt.show()
    min_dist = 10000
    candidate_rope_loc = (0, 0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(di_data[r][c][0] != 0):
                dist = np.linalg.norm(
                    np.array([r-scaled_test_loc[1], c-scaled_test_loc[0]]))
                if (dist < min_dist):
                    candidate_rope_loc = (c, r)
                    min_dist = dist
    min_loc = candidate_rope_loc
    print("FITTED POINT: " + str(min_loc))

    if DISPLAY:
        plt.imshow(new_di_data, interpolation="nearest")
        plt.show()
    #new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    #plt.imshow(new_di._image_data(), interpolation="nearest")

    # new_di.save("edge_detection_t.png")

    #fig2 = plt.figure()
    # print(min_loc)

    # ----------------------FIND END OF CHANNEL
    lower = 254
    upper = 256
    channel_start = (0, 0)
    max_edges = 0

    for r in range(len(three_mat_color)):
        for c in range(len(three_mat_color[r])):
            if(lower < three_mat_color[r][c][0] < upper):
                curr_edges = 0
                for add in range(1, 11):
                    if(lower < three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[max(0, r-add)][c][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[r][max(0, c-add)][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] < upper):
                        curr_edges += 1
                    if(lower < three_mat_color[max(0, r-add)][max(0, c-add)][0] < upper):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    max_edges = curr_edges
                    channel_start = (c, r)
    print("CHANNEL_START: "+str(channel_start))
    channel_cloud, _ = g.segment_channel(channel_start)
    transformed_channel_cloud = new_transf.apply(channel_cloud)
    image_channel = iface.cam.intrinsics.project_to_image(
        transformed_channel_cloud, round_px=False)  # should this be transformed_channel_cloud?
    image_channel_data = image_channel._image_data()
    copy_channel_data = copy.deepcopy(image_channel_data)
    #if DISPLAY:
    #    plt.imshow(image_channel_data, interpolation="nearest")
    #    plt.show()
    figure = plt.figure()
    plt.savefig("Point_Cloud_Channel.png")
    # Threshold pointcloud
    lower = 80
    upper = 255

    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(new_di_data[r][c] != 0):
                image_channel_data[r][c][0] = 0.0
                image_channel_data[r][c][1] = 0.0
                image_channel_data[r][c][2] = 0.0

    # Finish Thresholding, now find corner to place
    max_edges = 0
    best_location = ()
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(image_channel_data[r][c][0] != 0):
                curr_edges = 0
                for add in range(1, 5):
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][max(0, c-add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][max(0, c-add)][0] == 0):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    best_location = (c, r)
                    max_edges = curr_edges
    print(best_location)
    dist_tolerance = 400
    """
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(np.linalg.norm(np.array([r-best_location[1],c-best_location[0]])) < dist_tolerance):
                image_channel_data[r][c] = 0
    """
    max_edges = 8
    best_locations = []
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(image_channel_data[r][c][0] != 0):
                curr_edges = 0
                for add in range(1, 3):
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][max(0, c-add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][max(0, c-add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][max(0, c-add)][0] == 0):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    best_locations += [(c, r)]
                    #max_edges = curr_edges
    print(best_locations)
    min_dist = 0
    for loc in best_locations:
        if(np.linalg.norm(np.array([loc[1]-best_location[1], loc[0]-best_location[0]])) > min_dist):
            min_dist = np.linalg.norm(
                np.array([loc[1]-best_location[1], loc[0]-best_location[0]]))
            best_location = loc
    print("CHANNEL PLACE: "+str(best_location))
    #if DISPLAY:
    #    plt.imshow(image_channel_data, interpolation="nearest")
    #    plt.show()
    #img_skeleton = np.array(image_channel_data)
    if DISPLAY:
        plt.imshow(copy_channel_data, interpolation="nearest")
        plt.show()
    img_skeleton = cv2.cvtColor(copy_channel_data, cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(img_skeleton, 2, 0.01, 200)
    for (x, y) in features[:, 0].astype("int0"):
        cv2.circle(img_skeleton, (x, y), 27, 127, -1)
    print(features)
    #if DISPLAY:
    #    plt.imshow(img_skeleton)
    endpoints = [x[0] for x in features]

    # plt.savefig("Channel_Remove_Rope.png")
    #plt.imshow(img.color.data, interpolation="nearest")
    # plt.show()
    # ----------------------FIND END OF CHANNEL

    #q = input("EXIT OUT \n")
    # NEW ---------------------------------------------------------------------------------
    # pick,place=click_points(img) #left is pick point, right is place point
    pick = min_loc

    # Use estimation
    place = best_location
    # Use left side
    if(endpoints[0][0] < endpoints[1][0]):
        place = endpoints[0]
    else:
        place = endpoints[1]
    print("ACTUAL PLACE: "+str(place))
    # place = FILL IN HERE
    # VAINAVI: will need to observe and crop image most likely
    #action = policy.get_action(img.color._data)
    #pick, place = act_to_kps(action)
    # breakpoint()
    assert pick is not None and place is not None

    # Convert to world coordinates:
    xind, yind = pick
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    xind2, yind2 = place
    lin_ind2 = int(img.depth.ij_to_linear(np.array(xind2), np.array(yind2)))

    points_3d = iface.cam.intrinsics.deproject(img.depth)
    point = iface.T_PHOXI_BASE*points_3d[lin_ind]
    point = [p for p in point]
    # print(point)
    point[2] -= 0.0025 # manually adjust height a tiny bit
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]

    # point conversion
    xind, yind = place
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

    xind, yind = endpoints[0]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    endpoint_1_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

    xind, yind = endpoints[1]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind), np.array(yind)))
    endpoint_2_point = iface.T_PHOXI_BASE*points_3d[lin_ind]

    print("TESTING")
    #print(place_point)
    #print(endpoint_1_point)
    #print(endpoint_2_point)

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
    """
    rope_cloud_data_x = rope_cloud.x_coords
    rope_cloud_data_y = rope_cloud.y_coords
    rope_cloud_data_z = rope_cloud.z_coords
    tolerance = 0.23
    new_rope_cloud_data_x = []
    new_rope_cloud_data_y = []
    new_rope_cloud_data_z = []
    for count in range(len(rope_cloud_data_x)):
        dist = np.linalg.norm(np.array([rope_cloud_data_x[count]-point[0], rope_cloud_data_x[count]-point[1]]))
        #print("CURR DIST: "+str(dist))
        if dist < tolerance:
            new_rope_cloud_data_x += [rope_cloud_data_x[count]]
            new_rope_cloud_data_y += [rope_cloud_data_y[count]]
            new_rope_cloud_data_z += [rope_cloud_data_z[count]]
    new_rope_cloud_data_x = np.array(new_rope_cloud_data_x)
    new_rope_cloud_data_y = np.array(new_rope_cloud_data_y)
    new_rope_cloud_data_z = np.array(new_rope_cloud_data_z)

    new_rope_cloud = PointCloud(np.array([new_rope_cloud_data_x,new_rope_cloud_data_y,new_rope_cloud_data_z]), frame=rope_cloud.frame)

    transformed_new_rope_cloud = new_transf.apply(new_rope_cloud)
    di_2 = iface.cam.intrinsics.project_to_image(
        transformed_new_rope_cloud, round_px=False)
    if DISPLAY:
        plt.imshow(di_2._image_data(), interpolation="nearest")
        plt.show()
    principle_axis = g.princ_axis(new_rope_cloud)
    print("PRINCIPLE AXIS")
    print(principle_axis)
    R = np.matrix([[0, -1, 0],[1,0,0], [0,0,1]])
    principle_axis = np.matmul(R,np.vstack(principle_axis))
    print(principle_axis)
    print(principle_axis.item((0,0)))
    principle_axis = np.array([principle_axis.item((0,0)),principle_axis.item((1,0)),principle_axis.item((2,0))])
    print("ROTATED AXIS: " + str(principle_axis))
    #principle_axis[1] = 0.5
    grasp_poses = g.generate_grasps(principle_axis, np.array(
        [-0.5810662, -1.34913424,  0.73567095]), iface.L_TCP, point[2])
    grasp_poses_filter = g.filter_unreachable(grasp_poses, iface.L_TCP)
    grasp_pose = g.select_single_grasp(grasp_poses_filter, iface.L_TCP)
    if grasp_pose is None:
       
        count = 0
        print("SELECTED THE "+str(count)+"'th POSSIBLE GRASP")
        while(count < len(grasp_poses)):
            grasp_pose = grasp_poses[count]
            rot_Grasp = Grasp(grasp_pose)
            try:
                iface.grasp(rot_Grasp, None)
                break
            except:
                count+=1
        #raise GraspException("No collision free grasps found")
        # at the end, sanity check that z axis is still facing negative
    #if(grasp_pose.rotation[:, 2].dot([0, 0, 1]) > 0):
    #    print("Warning: upward gripper grasp returned")
    #    #raise Exception("Grasp calculated returned a gripper orientation with the gripper pointing upwards")
    
    """
    # START PICK AND PLACE ___________________________________
    take_action(point, place_point)

    # PACKING __________________________________________________

    push_action_endpoints(
        new_place_point, [new_endpoint_1_point, new_endpoint_2_point], iface)
    break
    # g.close()
print("Done with script, can end")
