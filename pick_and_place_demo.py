#from ../../cable_untangling.interface_rws import Interface
from autolab_core import RigidTransform,RgbdImage,DepthImage,ColorImage, CameraIntrinsics
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from ../../cable_untangling.tcps import *
#from ../../cable_untangling.grasp import Grasp,GraspSelector
import time
import os
import sys
cable = os.path.dirname(os.path.abspath(__file__)) + "/../../cable_untangling"
sys.path.insert(0,cable)
from interface_rws import Interface 
from tcps import *
from grasp import Grasp, GraspSelector
from scipy.ndimage.filters import gaussian_filter
import cv2
import push
behavior_cloning_path = os.path.dirname(os.path.abspath(__file__)) + "/../../multi-fidelity-behavior-cloning"
sys.path.insert(0, behavior_cloning_path)
#from analysis import CornerPullingBCPolicy


def click_points(img):
    # left click mouse for pick point, right click for place point
    fig, ax = plt.subplots()
    ax.imshow(img.color.data)
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    pick,place = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal pick,place
        point=iface.T_PHOXI_BASE*points_3d[lin_ind]
        print("Clicked point in world coords: ",point)
        if(point.z>.5):
            print("Clicked point with no depth info!")
            return
        if(event.button==1):
            pick=coords
        elif(event.button==3):
            place=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return pick,place
#SPEED=(.5,6*np.pi)
SPEED=(.025,0.3*np.pi)
iface=Interface("1703005",METAL_GRIPPER.as_frames(YK.l_tcp_frame,YK.l_tip_frame),
    ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame),speed=SPEED)

def act_to_kps(act):
    x, y, dx, dy = act
    x, y, dx, dy = int(x*224), int(y*224), int(dx*224), int(dy*224)
    return (x, y), (x+dx, y+dy)

def take_action(pick, place):

    #handle the actual grabbing motion
    #l_grasp=None
    #r_grasp=None
    #single grasp
    #grasp = g.single_grasp(pick,.007,iface.R_TCP)
    # g.col_interface.visualize_grasps([grasp.pose],iface.R_TCP)
    #wrist = grasp.pose*iface.R_TCP.inverse()
    print("grabbing with left arm")
    #r_grasp=grasp
    #r_grasp.pose.from_frame=YK.r_tcp_frame
    #iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
      from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
    # iface.go_cartesian(r_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
    #     from_frame = YK.r_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
    iface.set_speed((.25,5))
    iface.close_grippers()
    time.sleep(3)
    iface.go_delta(l_trans=[0,0,0.1]) # lift
    time.sleep(1)
    delta = [place[i] - pick[i] for i in range(3)]
    iface.go_delta(l_trans=delta)
    iface.open_grippers()
    iface.home()
    iface.sync()
    #iface.set_speed(SPEED)
    iface.set_speed((.25,5))
    time.sleep(2)
    iface.open_grippers()
#policy = CornerPullingBCPolicy()
while True:
    q = input("Enter to home arms, anything else to quit\n")
    if not q=='':break
    iface.home()
    iface.open_grippers()
    iface.sync()
    #set up a simple interface for clicking on a point in the image
    img=iface.take_image()

    g = GraspSelector(img,iface.cam.intrinsics,iface.T_PHOXI_BASE)
    #NEW --------------------------------------------------------------------------------
    #----------------------Find brightest pixel for segment_cable
    three_mat_color = img.color.data
    pixel_r = 0
    pixel_c = 0
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    lower = 0
    upper = 190
    delete_later = []
    max_score = 0
    max_scoring_loc = (0,0)
    for r in range(len(three_mat_color)):
        for c  in range(len(three_mat_color[r])):
            if(three_mat_color[r][c][0] == 255 and three_mat_color[r][c][1] == 255 and three_mat_color[r][c][2] == 255):
                curr_score = 0
                for add in range(1,8):
                    if(three_mat_color[min(len(three_mat_color)-add, r+add)][c][0] == 255):
                        curr_score += 1
                    if(three_mat_color[max(0, r-add)][c][0] == 255):
                        curr_score += 1
                    if(three_mat_color[r][min(len(three_mat_color[0])-add, c+add)][0] == 255):
                        curr_score += 1
                    if(three_mat_color[r][max(0, c-add)][0] == 255):
                        curr_score += 1
                    if(three_mat_color[min(len(three_mat_color)-add, r+add)][min(len(three_mat_color[0])-add, c+add)][0] == 255):
                        curr_score += 1
                    if(three_mat_color[min(len(three_mat_color)-add, r+add)][max(0, c-add)][0] == 255):
                        curr_score += 1
                    if(three_mat_color[max(0, r-add)][min(len(three_mat_color[0])-add, c+add)][0] == 255):
                        curr_score += 1
                    if(three_mat_color[max(0, r-add)][max(0, c-add)][0] == 255):
                        curr_score += 1
                if(curr_score>max_score):
                    max_scoring_loc=(c,r)
                    max_score=curr_score
            
            if(lower <  three_mat_color[r][c][1] < upper):
                delete_later += [(c,r)]
            #if(c == 500):
            #    print("X: " + str(c)+" Y: "+str(r)+" R: "+str(three_mat_color[r][c][0]) + " G: "+str(three_mat_color[r][c][1]) + " B:" +str(three_mat_color[r][c][2]) + " AVG: ")
    loc = max_scoring_loc
    print("Starting segmenet_cable pt: "+str(max_scoring_loc))
    #print(loc)
    #print(delete_later)
    #print(delete_later)
    #----------------------Segment
    rope_cloud,_ = g.segment_cable(loc)
    #print(rope_cloud.data)
    #----------------------Remove block

    new_transf = iface.T_PHOXI_BASE.inverse()
    transformed_rope_cloud = new_transf.apply(rope_cloud)
    di = iface.cam.intrinsics.project_to_image(transformed_rope_cloud, round_px = False)
    plt.imshow(di._image_data(), interpolation="nearest")
    plt.show()
    
    
    di_data = di._image_data()
    for delete in delete_later:
         di_data[delete[1]][delete[0]] = [float(0),float(0),float(0)]
    
    mask = np.zeros((len(di_data),len(di_data[0])))
    loc_list = [loc]

    # modified segment_cable code to build a mask for the cable

    #pick the brightest rgb point in the depth image
    #increment in each direction for it's neighbors looking to see if it meets the thresholded rgb value
    # if not, continue 
    # if yes set it's x,y position to the mask matrix with the value 1
    # add that value to the visited list so that we don't go back to it again



    #mask = np.ones((len(di_data),len(di_data[0])))
    new_di_data = np.zeros((len(di_data),len(di_data[0])))
    xdata = []
    ydata = []

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            #if(mask[r][c] == 1):
            #    new_di_data[r][c] = di_data[r][c][0] # This actually changes the depth data so, use di_data if you need the depth
            new_di_data[r][c] = di_data[r][c][0]
            if (new_di_data[r][c] > 0):
                xdata += [c]
                ydata += [r]

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c]!= 0):
                curr_edges = 0
                for add in range(1,8):
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
                    new_di_data[r][c]=0.0

    new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    plt.imshow(new_di._image_data(), interpolation="nearest")
    plt.show()
    #plt.savefig("Isolated_Cable")
    #new_di.save("Isolated_Cable.png")

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

    save_loc = (0,0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
                save_loc = (c,r)
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    compress_factor = 55
    #print(int(math.floor(len(di_data)/compress_factor)))
    #print(int(math.floor(len(di_data[0])/compress_factor)))
    rows_comp = int(math.floor(len(di_data)/compress_factor))
    cols_comp = int(math.floor(len(di_data[0])/compress_factor))
    compressed_map = np.zeros((rows_comp,cols_comp))
    for r in range(rows_comp):
        for c in range(cols_comp):
            for add in range(1,5):
                    if(new_di_data[min(len(new_di_data)-add, r*compress_factor+add)][c*compress_factor] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[max(0, r*compress_factor-add)][c*compress_factor] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[r*compress_factor][min(len(new_di_data[0])-add, c*compress_factor+add)] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[r*compress_factor][max(0, c*compress_factor-add)] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[min(len(new_di_data)-add, r*compress_factor+add)][min(len(new_di_data[0])-add, c*compress_factor+add)] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[min(len(new_di_data)-add, r*compress_factor+add)][max(0, c*compress_factor-add)] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[max(0, r*compress_factor-add)][min(len(new_di_data[0])-add, c*compress_factor+add)] != 0):
                        compressed_map[r][c] = 255
                        break
                    if(new_di_data[max(0, r*compress_factor-add)][max(0, c*compress_factor-add)] != 0):
                        compressed_map[r][c] = 255
                        break
    max_edges = 0
    test_locs = (0,0)
    for r in range(len(compressed_map)):
        for c in range(len(compressed_map[r])):
            if(compressed_map[r][c]!= 0):
                curr_edges = 0
                for add in range(1,2):
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
                    test_loc=(c,r)
                    max_edges=curr_edges
    print(test_loc)
    print("scaled: "+str((test_loc[0]*compress_factor,test_loc[1]*compress_factor)))
    all_solns = []
    for r in range(len(compressed_map)):
        for c in range(len(compressed_map[r])):
            if(compressed_map[r][c]!= 0):
                curr_edges = 0
                for add in range(1,2):
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
                if(curr_edges == max_edges or curr_edges == max_edges-1 or curr_edges == max_edges-2):
                    all_solns+=[(c,r)]
    ##rint("ALL SOLUTIONS: "+str(all_solns))
    #for soln in all_solns:
    scaled_test_loc = (test_loc[0]*compress_factor,test_loc[1]*compress_factor)
    plt.imshow(compressed_map, interpolation="nearest")
    plt.show() 
    min_dist = 10000
    candidate_rope_loc = (0,0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(di_data[r][c][0] != 0):
                dist = np.linalg.norm(np.array([r-scaled_test_loc[1],c-scaled_test_loc[0]]))
                if (dist < min_dist):
                    candidate_rope_loc = (c,r)
                    min_dist = dist
    min_loc = candidate_rope_loc
    print("FITTED POINT: " + str(min_loc))

    """
    #TEST FILL METHOD
    print(save_loc)
    queue = []
    queue += [save_loc]
    curr_loc = (0,0)
    dx = [-1, +1, 0, 0]
    dy = [0, 0 , +1, -1]
    rows = len(new_di_data)
    cols = len(new_di_data[0])
    visited = [[False]*cols for i in range(rows)]

    potential_ends = []
    while(queue):
        curr_loc = queue.pop()
        r = curr_loc[1]
        c = curr_loc[0]
        count = 0
        for neighbor in range(0,4):
                rr = r + dx[neighbor]
                cc = c + dy[neighbor]
                if (rr < 0 or cc < 0):
                    continue
                if (rr >= rows or cc >= cols):
                    continue
                if visited[rr][cc]:
                    count+=1
                    continue
                if new_di_data[rr][cc] == 0:
                    continue
                queue.append((cc,rr))
                visited[rr][cc] = True
        if(count == 4):
            potential_ends+= [curr_loc]       
    potential_ends += [curr_loc]
    print("POTENTIAL LOCS " +str(potential_ends))

    #--------------------
    
    min_locs = []
    min_loc = (0,0)
    max_edges = 33
    min_edges = 27
    while(True):
        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if(new_di_data[r][c]!= 0):
                    curr_edges = 0
                    curr_white_edges = 0
                    for add in range(1,9):
                        if(new_di_data[min(len(new_di_data)-add, r+add)][c] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[max(0, r-add)][c] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[r][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[r][max(0, c-add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[min(len(new_di_data)-add, r+add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                        if(new_di_data[max(0, r-add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        else:
                            curr_white_edges += 1
                    #if(curr_white_edges < min_edges):
                    #    print("removed "+str((c,r)))
                    #    new_di_data[r][c] = 0.0
                    if(curr_edges > max_edges):
                        #max_edges = curr_edges
                        min_locs+=[(c,r,curr_edges)]
                        min_loc = (c,r)
        print("DEBUG: "+str(max_edges) + str(min_locs))
        if(len(min_locs) == 0):
            min_locs = []
            max_edges = 10
        if(1 < len(min_locs) <= 100):
            print("TEHIOPUHPIOSUHTGPIOUSADGFPOHGUSUGHSPOIDUPIOSUDUSDHGPSDGPIOSDGHPUDSHIUSHHDG")
            break
        else:
            min_locs = []
            max_edges +=1
    #print(min_locs)
    #print(min_loc)
    # REMOVE NOISE #
    for loc in min_locs:
        for add in range(1,2):
            r = loc[1]
            c = loc[0]
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
        if(curr_edges < 4):
            min_locs.remove(loc)
            print("removed "+str(loc))
    print("potential endpoints")        
    print(min_locs)
    min_dist_to_center = 10000000 
    for loc in min_locs:
        dist = np.linalg.norm(np.array([loc[1]-400,loc[0]-450]))
        if dist<min_dist_to_center:
            min_dist_to_center=dist
            min_loc = loc
    #print(min_loc)
    min_loc = (min_loc[0],min_loc[1])
    min_dist = 10000
    candidate_rope_loc = (0,0)
    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(di_data[r][c][0] != 0):
                dist = np.linalg.norm(np.array([r-min_loc[1],c-min_loc[0]]))
                if (dist < min_dist):
                    candidate_rope_loc = (c,r)
                    min_dist = dist
    min_loc = candidate_rope_loc
    print("FITTED POINT: " + str(min_loc))
    """
    plt.imshow(new_di_data, interpolation="nearest")
    plt.show()   
    plt.imshow(img.color.data, interpolation="nearest")
    plt.show()  
    #new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    #plt.imshow(new_di._image_data(), interpolation="nearest")

    #new_di.save("edge_detection_t.png")
    
    #fig2 = plt.figure()
    #print(min_loc)
    
    #----------------------FIND END OF CHANNEL
    lower = 80
    upper = 90
    channel_start = (0,0)
    max_edges = 0

    for r in range(len(three_mat_color)):
        for c in range(len(three_mat_color[r])):
            if(lower< three_mat_color[r][c][0]<upper):
                curr_edges = 0
                for add in range(1,5):
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
                    channel_start = (c,r)
    #print(channel_start)
    channel_cloud,_ = g.segment_channel(channel_start)
    transformed_channel_cloud = new_transf.apply(channel_cloud)
    image_channel = iface.cam.intrinsics.project_to_image(transformed_channel_cloud, round_px = False) #should this be transformed_channel_cloud?
    image_channel_data = image_channel._image_data()
    copy_channel_data = np.copy(image_channel_data)
    plt.imshow(image_channel_data, interpolation="nearest")
    plt.show()
    figure = plt.figure()
    plt.savefig("Point_Cloud_Channel.png")
    #Threshold pointcloud
    lower = 80
    upper = 255
    
    for r in range(len(image_channel_data)):
        for c  in range(len(image_channel_data[r])):
            if(new_di_data[r][c] != 0):
                image_channel_data[r][c][0] = 0.0
                image_channel_data[r][c][1] = 0.0
                image_channel_data[r][c][2] = 0.0

    #Finish Thresholding, now find corner to place
    max_edges = 0
    best_location = ()
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(image_channel_data[r][c][0] != 0 ):
                curr_edges = 0
                for add in range(1,5):
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[r][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    best_location = (c,r)
                    max_edges = curr_edges
    print(best_location)
    dist_tolerance = 400
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(np.linalg.norm(np.array([r-best_location[1],c-best_location[0]])) < dist_tolerance):
                image_channel_data[r][c] = 0
    max_edges = 8
    best_locations = []
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(image_channel_data[r][c][0] != 0 ):
                curr_edges = 0
                for add in range(1,3):
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][c][0] == 0):
                        curr_edges += 1
                    if(image_channel_data[r][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[r][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[min(len(new_di_data)-add, r+add)][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)][0]  == 0):
                        curr_edges += 1
                    if(image_channel_data[max(0, r-add)][max(0, c-add)][0]  == 0):
                        curr_edges += 1
                if(curr_edges > max_edges):
                    best_locations += [(c,r)]
                    #max_edges = curr_edges
    print(best_locations)
    min_dist = 0
    for loc in best_locations:
        if(np.linalg.norm(np.array([loc[1]-best_location[1],loc[0]-best_location[0]])) > min_dist):
                min_dist = np.linalg.norm(np.array([loc[1]-best_location[1],loc[0]-best_location[0]]))
                best_location = loc
    print("CHANNEL PLACE: "+str(best_location))
    plt.imshow(image_channel_data, interpolation="nearest")
    plt.show()
    #img_skeleton = np.array(image_channel_data)
    plt.imshow(copy_channel_data, interpolation="nearest")
    plt.show()
    img_skeleton = cv2.cvtColor(copy_channel_data,cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(img_skeleton, 2, 0.01, 200)
    for (x,y) in features[:,0].astype("int0"):
        cv2.circle(img_skeleton,(x,y),27,127,-1)
    print(features)
    plt.imshow(img_skeleton)
    endpoints = [x[0] for x in features]
    
    #plt.savefig("Channel_Remove_Rope.png")
    #plt.imshow(img.color.data, interpolation="nearest")
    #plt.show()
    #----------------------FIND END OF CHANNEL

    #q = input("EXIT OUT \n")
    #NEW ---------------------------------------------------------------------------------
    #pick,place=click_points(img) #left is pick point, right is place point
    pick = min_loc
    place = best_location
    #place = FILL IN HERE
    # VAINAVI: will need to observe and crop image most likely
    #action = policy.get_action(img.color._data)
    #pick, place = act_to_kps(action)
    # breakpoint()
    assert pick is not None and place is not None

    # Convert to world coordinates:
    xind, yind = pick
    lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
    xind2, yind2 = place
    lin_ind2 = int(img.depth.ij_to_linear(np.array(xind2),np.array(yind2)))

    points_3d = iface.cam.intrinsics.deproject(img.depth)
    point=iface.T_PHOXI_BASE*points_3d[lin_ind]
    point = [p for p in point]
    #print(point)
    #point[2] += 0.005 # manually adjust height a tiny bit
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]
    take_action(point, place_point)




    #PACKING __________________________________________________
    xind, yind = place
    lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
    place_point=iface.T_PHOXI_BASE*points_3d[lin_ind]

    xind, yind = endpoints[0]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
    endpoint_1_point =iface.T_PHOXI_BASE*points_3d[lin_ind]

    xind, yind = endpoints[1]
    lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
    endpoint_2_point =iface.T_PHOXI_BASE*points_3d[lin_ind]

    print(place_point)
    print(endpoint_1_point)
    print(endpoint_2_point)
    push.push_action_endpoints(place_point, [endpoint_1_point,endpoint_2_point])
    break
    #g.close()
print("Done with script, can end")

