#from ../../cable_untangling.interface_rws import Interface
from autolab_core import RigidTransform,RgbdImage,DepthImage,ColorImage, CameraIntrinsics
import numpy as np
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
    iface.set_speed(SPEED)
    time.sleep(2)
    iface.open_grippers()

#policy = CornerPullingBCPolicy()
while True:
    #q = input("Enter to home arms, anything else to quit\n")
    #if not q=='':break
    #iface.home()
    #iface.open_grippers()
    iface.sync()
    #set up a simple interface for clicking on a point in the image
    img=iface.take_image()
    #print(iface.T_PHOXI_BASE)
    g = GraspSelector(img,iface.cam.intrinsics,iface.T_PHOXI_BASE)
    #NEW --------------------------------------------------------------------------------
    #----------------------Find brightest pixel for segment_cable
    three_mat_color = img.color.data
    pixel_r = 0
    pixel_c = 0
    points_3d = iface.cam.intrinsics.deproject(img.depth)
    lower = 0
    upper = 215
    delete_later = []
    for r in range(len(three_mat_color)):
        for c  in range(len(three_mat_color[r])):
            if(three_mat_color[r][c][0] == 255 and three_mat_color[r][c][1] == 255 and three_mat_color[r][c][2] == 255):
                pixel_r = r
                pixel_c = c
            
            if(lower <  three_mat_color[r][c][1] < upper):
                delete_later += [(c,r)]
            #if(c == 500):
            #    print("X: " + str(c)+" Y: "+str(r)+" R: "+str(three_mat_color[r][c][0]) + " G: "+str(three_mat_color[r][c][1]) + " B:" +str(three_mat_color[r][c][2]) + " AVG: ")
    loc = (pixel_c,pixel_r)
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
                for add in range(1,2):
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
                    new_di_data[r][c]=0.0

    new_di = DepthImage(new_di_data.astype(np.float32), frame=di.frame)
    plt.imshow(new_di._image_data(), interpolation="nearest")
    plt.show()
    #plt.savefig("Isolated_Cable")
    
    new_di.save("Isolated_Cable.png")
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

    for r in range(len(new_di_data)):
        for c in range(len(new_di_data[r])):
            if(new_di_data[r][c] != 0):
                new_di_data[r][c] = 255
    new_di_data = gaussian_filter(new_di_data, sigma=1)

    

    min_locs = []
    min_loc = (0,0)
    max_edges = 30
    while(True):
        for r in range(len(new_di_data)):
            for c in range(len(new_di_data[r])):
                if(new_di_data[r][c]!= 0):
                    curr_edges = 0
                    for add in range(1,8):
                        if(new_di_data[min(len(new_di_data)-add, r+add)][c] == 0):
                            curr_edges += 1
                        if(new_di_data[max(0, r-add)][c] == 0):
                            curr_edges += 1
                        if(new_di_data[r][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(new_di_data[r][max(0, c-add)] == 0):
                            curr_edges += 1
                        if(new_di_data[min(len(new_di_data)-add, r+add)][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(new_di_data[min(len(new_di_data)-add, r+add)][max(0, c-add)] == 0):
                            curr_edges += 1
                        if(new_di_data[max(0, r-add)][min(len(new_di_data[0])-add, c+add)] == 0):
                            curr_edges += 1
                        if(new_di_data[max(0, r-add)][max(0, c-add)] == 0):
                            curr_edges += 1
                    if(curr_edges > max_edges):
                        #max_edges = curr_edges
                        min_locs+=[(c,r,curr_edges)]
                        min_loc = (c,r)
        if(len(min_locs) <= 10):
            break
        else:
            min_locs = []
            max_edges +=1
    print(min_locs)
    print(min_loc)
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
    
    min_dist_to_center = 10000000 
    for loc in min_locs:
        dist = np.linalg.norm(np.array([loc[1]-400,loc[0]-450]))
        if dist<min_dist_to_center:
            min_dist_to_center=dist
            min_loc = loc
    print(min_loc)
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
    plt.imshow(new_di_data, interpolation="nearest")
    plt.show()
    figure = plt.figure()
    plt.savefig("Guassian_Post_Process.png")     
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
    print(channel_start)
    channel_cloud,_ = g.segment_channel(channel_start)
    transformed_channel_cloud = new_transf.apply(channel_cloud)
    image_channel = iface.cam.intrinsics.project_to_image(transformed_rope_cloud, round_px = False)
    image_channel_data = image_channel._image_data()
    
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
    potential_locations = []
    for r in range(len(image_channel_data)):
        for c in range(len(image_channel_data[r])):
            if(image_channel_data[r][c][0] != 0 ):
                curr_edges = 0
                for add in range(1,4):
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
                    potential_locations += [(c,r)]
                    max_edges = curr_edges
    print(potential_locations)
    plt.imshow(image_channel_data, interpolation="nearest")
    plt.show()
    plt.savefig("Channel_Remove_Rope.png")
    plt.imshow(img.color.data, interpolation="nearest")
    plt.show()

    #----------------------FIND END OF CHANNEL

    q = input("EXIT OUT \n")
    #NEW ---------------------------------------------------------------------------------
    pick,place=click_points(img) #left is pick point, right is place point
    pick = min_locs
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
    point[2] -= 0.005 # manually adjust height a tiny bit
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]

    take_action(point, place_point)
    #g.close()
print("Done with script, can end")

