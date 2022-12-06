#from ../../cable_untangling.interface_rws import Interface
from autolab_core import RigidTransform,RgbdImage,DepthImage,ColorImage, CameraIntrinsics
import numpy as np
import matplotlib.pyplot as plt
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
    print("grabbing with right arm")
    #r_grasp=grasp
    #r_grasp.pose.from_frame=YK.r_tcp_frame
    #iface.grasp(l_grasp=l_grasp,r_grasp=r_grasp)
    iface.go_cartesian(l_targets=[RigidTransform(translation=pick, rotation=Interface.GRIP_DOWN_R,
        from_frame = YK.l_tcp_frame, to_frame = 'base_link')], nwiggles=(10,10),rot=(.0,.0))
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
    q = input("Enter to home arms, anything else to quit\n")
    if not q=='':break
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
    lower = 20
    upper = 90.0
    delete_later = []
    #print(three_mat_color[635][231][0])
    #print(three_mat_color[635][231][1])
    #print(three_mat_color[635][231][2])
    for r in range(len(three_mat_color)):
        for c  in range(len(three_mat_color[r])):
            if(three_mat_color[r][c][0] == 255 and three_mat_color[r][c][1] == 255 and three_mat_color[r][c][2] == 255):
                pixel_r = r
                pixel_c = c
            if(lower < (three_mat_color[r][c][0] + three_mat_color[r][c][1] + three_mat_color[r][c][2])/3 < upper):
                delete_later += [(c,r)]
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

    di.save("edge_detection.png")
    
    di_data = di._image_data()
    #print(di_data)
    
    # for r in range(len(di_data)):
    #    for c  in range(len(di_data[r])):
    #        for delete in delete_later:
    #            if(r == delete[1] and c == delete[0]):
    #                 #print("X: "+str(c) +" Y: "+str(r)+ " DEPTH: " + str(di_data[r][c]))
    #                di_data[r][c] = [0,0,0]

    # for delete in delete_later:
    #     #print(di_data[delete[1]][delete[0]])
    #     #print(di_data[delete[1]][delete[0]])
    #     di_data[delete[1]][delete[0]] = [0,0,0]
    #di_image = iface.cam.intrinsics.deproject_to_image(di)
    #plt.imshow(di_data, interpolation="nearest")
    #fig2 = plt.figure()
    plt.imshow(di_data, interpolation="nearest")
    plt.show()
    
    #----------------------Find end of rope


    q = input("EXIT OUT \n")
    #NEW ---------------------------------------------------------------------------------
    pick,place=click_points(img) #left is pick point, right is place point
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
    point[2] -= 0.05 # manually adjust height a tiny bit
    place_point = iface.T_PHOXI_BASE*points_3d[lin_ind2]

    take_action(point, place_point)
    #g.close()
print("Done with script, can end")

