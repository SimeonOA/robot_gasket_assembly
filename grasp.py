from autolab_core import RigidTransform,PointCloud,RgbdImage,ColorImage,DepthImage
from autolab_core.image import BINARY_IM_DEFAULT_THRESH
from numpy.lib.histograms import histogram_bin_edges
from collision import CollisionInterface
from queue import Empty
from yumiplanning.yumi_kinematics import YuMiKinematics
from yumiplanning.yumi_planner import Planner
import numpy as np
from multiprocessing import Queue,Process
from random import shuffle
import math
class GraspException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
'''
Class for describing a grasp pose on a cable. used as a nice wrapper class for
containing useful information about the grasop
'''
class Grasp:
    def __init__(self,target_pose,pregrasp_dist=.075,gripper_pos=.010,grip_open_dist=.05,
                speed=(.15,np.pi)):
        '''
        target_pose: end effector pose (tcp frame)
        pregrasp_dist: how far back to move the arm before trying to grab
        gripper_poas: the value to send the servos. the actual width will be 2* this number
        speed: speed in (mm/s,deg/s) at which to carry out this grasp. speed
                only applies after pre-grasp is reached
        '''
        self.pose=target_pose
        self.pregrasp_dist=pregrasp_dist
        self.gripper_pos=gripper_pos
        self.speed=speed
        self.grip_open_dist=grip_open_dist
        
    def compute_pregrasp(self):
        '''
        Computes a pose which is translated distance away from the grasp along the gripper axis
        '''
        T=RigidTransform(translation=[0,0,-self.pregrasp_dist],
                from_frame=self.pose.from_frame,to_frame=self.pose.from_frame)
        return self.pose*T

    def compute_gripopen(self):
        '''
        Computes a pose which is translated distance away from the grasp along the gripper axis
        '''
        T=RigidTransform(translation=[0,0,-self.grip_open_dist],
                from_frame=self.pose.from_frame,to_frame=self.pose.from_frame)
        return self.pose*T

def cross(l1,l2):
    '''returns the set cartesian product'''
    L=[]
    for l in l1:
        for ll in l2:
            L.append((l,ll))
    return L


class GraspSelector:
    col_interface=CollisionInterface()
    def __init__(self,rgbd_image,intrinsics,T_CAM_BASE):
        self.img = rgbd_image
        self.depth=rgbd_image.depth
        self.color=rgbd_image.color
        self.intr = intrinsics
        self.points_3d = T_CAM_BASE*self.intr.deproject(rgbd_image.depth)
        self.T_CAM_BASE = T_CAM_BASE
        self.col_interface.setup()
        self.col_interface.set_scancloud(self.points_3d)
        self.yk = YuMiKinematics()
        self.planner=Planner()

    def single_grasp(self,loc,grasp_dist,tcp):
        '''
        returns a Grasp object for a single grasp on the given location
        loc is a tuple of (x index,y index) into the depth image (NOT a point)
        grasp_dist is how deep to grab the cable
        '''
        #1. floodfill starting from selection to find points which are within a radius and also smoothly varying depth information
        cable_points,centroid = self.segment_cable(loc)
        #2. find the axis of the cable
        cable_ax = self.princ_axis(cable_points)
        grasp_poses = self.generate_grasps(cable_ax,centroid.vector,tcp,grasp_dist)
        grasp_poses=self.filter_unreachable(grasp_poses,tcp)
        grasp_pose  = self.select_single_grasp(grasp_poses,tcp)
        if grasp_pose is None:raise GraspException("No collision free grasps found")
        #at the end, sanity check that z axis is still facing negative
        if(grasp_pose.rotation[:,2].dot([0,0,1])>0):
            print("Warning: upward gripper grasp returned")
            #raise Exception("Grasp calculated returned a gripper orientation with the gripper pointing upwards")
        return Grasp(grasp_pose)

    def double_grasp(self,loc1,loc2,l_dist,r_dist,l_tcp,r_tcp,min_dist=.005,batch_size=200):
        #note: l_tcp and r_tcp arent used for collision detection, it just picks the more
        #restrictive one. same for l_dist and r_dist
        self.yk.set_tcp(l_tcp,r_tcp)
        grasp_dist=min(l_dist,r_dist)
        points1,c1 = self.segment_cable(loc1)
        points2,c2 = self.segment_cable(loc2)
        ax1=self.princ_axis(points1)
        ax2=self.princ_axis(points2)
        if l_tcp.translation[2]<r_tcp.translation[2]:
            tcp = l_tcp
        else:
            tcp = r_tcp
        def choose_arms(g1,g2):
            if (g1*tcp.inverse()).translation[1]>(g2*tcp.inverse()).translation[1]:
                gleft=g1
                gright=g2
            else:
                gleft=g2
                gright=g1
            if(l_dist > r_dist):
                #correct left grasp (this comes fromt the fact we picked the more conservative grasp dist for all collision checking)
                diff=RigidTransform(translation=[0,0,l_dist - r_dist],from_frame=gleft.from_frame,to_frame=gleft.from_frame)
                gleft=gleft*diff
            if(r_dist > l_dist):
                #correct right grasps
                diff=RigidTransform(translation=[0,0,r_dist - l_dist],from_frame=gright.from_frame,to_frame=gright.from_frame)
                gright=gright*diff
            return gleft.as_frames(from_frame=self.yk.l_tcp_frame),gright.as_frames(self.yk.r_tcp_frame)
        grasps1 = self.generate_grasps(ax1,c1.vector,tcp,grasp_dist)
        grasps2 = self.generate_grasps(ax2,c2.vector,tcp,grasp_dist)
        # self.col_interface.visualize_grasps(grasps1+grasps2,tcp)
        pairs = cross(grasps1,grasps2)
        if(len(pairs)==0):raise GraspException("No grasps found for one point")
        shuffle(pairs)
        chosen_pair=None
        def costfn(dist,g1,g2):
            #higher is better
            vertcost=np.dot([0,0,-1.],g1.rotation[:,2]) + np.dot([0,0,-1.],g2.rotation[:,2])
            #dist will be in the .1 range, vertcost will be in the 1 range
            return vertcost/10. + dist
        for i in range(math.ceil(len(pairs)/batch_size)):
            batch=pairs[i*batch_size:(i+1)*batch_size]
            res = self.col_interface.rate_pair(batch,tcp)
            res.sort(key=lambda tup:-tup[0])#sort by decreasing distance
            if res[0][0]<min_dist:
                print("no pairs within dist found")
                continue
            j=0
            while(j<len(res) and res[j][0]>min_dist):
                j+=1
            #find the most vertical pair among the furthest apart
            print("searching through",j,"non-colliding pairs")
            bestcost=0
            bestpair=None
            for dist,g1,g2 in res[:j]:
                cost=costfn(dist,g1,g2)
                if cost>bestcost:
                    #filter out kinematically infeasible grasps
                    gl,gr = choose_arms(g1,g2)
                    lg,rg = self.planner.find_joints(None,None,self.yk,2,gl,gr)
                    if lg is None or rg is None:
                        continue
                    bestcost=cost
                    bestpair = gl,gr
            chosen_pair = bestpair
            if chosen_pair is not None:
                break
        if chosen_pair is None:
            raise GraspException("No grasp pair found")
        gleft,gright=chosen_pair
        return Grasp(gleft),Grasp(gright)

    def generate_grasps(self,cable_ax,centroid,tcp,grasp_dist):
        '''
        returns a pose which avoids obstacles in the point cloud
        '''
        #for now just rotate the gripper to be perpendicular to the cable, and oriented so that the 
        new_y = cable_ax
        #new z axis rotates towards the direction of the axis
        new_x = np.cross(new_y,[0.,0.,-1.])#this is negative so that the gripper is pointing downwards
        new_x /= np.linalg.norm(new_x)
        new_z = np.cross(new_x,new_y)
        R = RigidTransform.rotation_from_axes(new_x,new_y,new_z)
        cable_pose = RigidTransform(rotation=R,translation=centroid,from_frame = tcp.from_frame,to_frame="base_link")
        candidate_grasps = self.generate_candidates(cable_pose,grasp_dist)
        #self.col_interface.visualize_grasps(candidate_grasps,tcp)
        print("Generated",len(candidate_grasps),"candidate grasps")
        # reachable_grasps = self.filter_unreachable(candidate_grasps,tcp)
        # print("Downsampled to",len(reachable_grasps), "reachable grasps")
        valid_grasps = self.filter_collisions(candidate_grasps,tcp,grasp_dist)
        print("Downsampled to ",len(valid_grasps),"valid grasps")
        return valid_grasps

    def generate_candidates(self,cable_pose,grasp_dist):
        '''
        generates grasps that are rotated around the cable 
        but not necessarily achievable
        '''
        ax_range   = (-1.6, 1.6) #(-1.5,1.5) #rotates around cable axis
        ax_N       = 13
        orth_range = (-np.pi/3.5,np.pi/3.5) #(-np.pi/5,np.pi/5)#rotates the gripper in the direction of cable axis
        orth_N     = 6
        grasps = []
        grasp_T = RigidTransform(translation=[0,0,grasp_dist],from_frame=cable_pose.from_frame,to_frame=cable_pose.from_frame)
        for ax_th in np.linspace(ax_range[0],ax_range[1],ax_N):
            for orth_th in np.linspace(orth_range[0],orth_range[1],orth_N):
                ax_R = RigidTransform.rotation_from_axis_angle(np.array([0.,1.,0.])*ax_th)
                orth_R = RigidTransform.rotation_from_axis_angle(np.array([1.,0.,0.])*orth_th)
                new_R = RigidTransform(rotation=ax_R.dot(orth_R),from_frame=cable_pose.from_frame,to_frame=cable_pose.from_frame)
                grasps.append(cable_pose*new_R*grasp_T)
        return grasps

    def filter_collisions(self,candidate_grasps,tcp,grasp_dist):
        '''
        goes through the list of candidate grasp poses and returns the sublist
        that are free of collisions with the cable (and table)
        '''
        new_g = []
        #from grasp frame to wrist frame
        T_WRIST_TCP = tcp.inverse()
        wrist_poses=[gr*T_WRIST_TCP for gr in candidate_grasps]
        collisions=self.col_interface.collide_gripper(wrist_poses,tcp,grasp_dist)
        for i in range(len(wrist_poses)):
            if not collisions[i]:new_g.append(candidate_grasps[i])
        return new_g

    def filter_unreachable(self,candidate_grasps,tcp):
        new_g=[]
        ls=self.yk.L_NICE_STATE
        rs=self.yk.R_NICE_STATE
        #TODO this should really only compute left or right depending on the y coordinate of the grasp
        for gr in candidate_grasps:
            g=Grasp(gr)
            preg=g.compute_pregrasp()
            l_q,r_q=self.yk.ik(preg.as_frames(from_frame=self.yk.l_tcp_frame),preg.as_frames(from_frame=self.yk.r_tcp_frame), left_qinit=ls,right_qinit=rs)
            ls=l_q if l_q is not None else ls
            rs=r_q if r_q is not None else rs
            if l_q is not None or r_q is not None:
                new_g.append(gr)
        return new_g

    def select_single_grasp(self,grasps,tcp):
        #selects the pose with furthest wrist mesh to cable mesh
        best = 0
        bestg=None
        T_WRIST_TCP = tcp.inverse()
        wrist_poses=[g*T_WRIST_TCP for g in grasps]
        distances=self.col_interface.closest_gripper(wrist_poses,tcp)
        for i in range(len(grasps)):
            dist=distances[i]
            if(dist>best):
                best=dist
                bestg=grasps[i]
        return bestg

    def segment_cable(self,loc):
        '''
        returns a PointCloud corresponding to cable points along the provided location
        inside the depth image
        '''
        q=[loc]
        pts=[]
        closepts=[]
        visited=set()
        start_point=self.ij_to_point(loc).data
        #RADIUS2 = .01**2#distance from original point before termination
        RADIUS2 = 1#distance from original point before termination
        CLOSE2 = .002**2
        #DELTA = .00075#if the depth changes by this much, stop floodfill
        DELTA = .0007
        NEIGHS = [(-1,0),(1,0),(0,1),(0,-1)]
        #carry out floodfill
        while len(q)>0:
            next_loc = q.pop()
            next_point = self.ij_to_point(next_loc).data
            visited.add(next_loc)
            diff = start_point-next_point
            dist = diff.dot(diff)
            if(dist>RADIUS2):
                continue
            pts.append(next_point)
            if(dist<CLOSE2):closepts.append(next_point)
            #add neighbors if they're within delta of current height
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0],next_loc[1]+n[1])
                if test_loc[0]>=self.depth.width or test_loc[0]<0 \
                        or test_loc[1]>=self.depth.height or test_loc[1]<0:
                    continue
                if(test_loc in visited):continue
                test_pt = self.ij_to_point(test_loc).data
                if(abs(test_pt[2]-next_point[2])<DELTA):
                    q.append(test_loc)
        return PointCloud(np.array(pts).T,"base_link"),PointCloud(np.array(closepts).T,"base_link").mean()

    def segment_channel(self,loc):
        '''
        returns a PointCloud corresponding to cable points along the provided location
        inside the depth image
        '''
        q=[loc]
        pts=[]
        closepts=[]
        visited=set()
        start_point=self.ij_to_point(loc).data
        #RADIUS2 = .01**2#distance from original point before termination
        RADIUS2 = 1#distance from original point before termination
        CLOSE2 = .002**2
        #DELTA = .00075#if the depth changes by this much, stop floodfill
        DELTA = 0.3
        NEIGHS = [(-1,0),(1,0),(0,1),(0,-1)]
        #carry out floodfill
        while len(q)>0:
            next_loc = q.pop()
            next_point = self.ij_to_point(next_loc).data
            visited.add(next_loc)
            diff = start_point-next_point
            dist = diff.dot(diff)
            if(dist>RADIUS2):
                continue
            pts.append(next_point)
            if(dist<CLOSE2):closepts.append(next_point)
            #add neighbors if they're within delta of current height
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0],next_loc[1]+n[1])
                if test_loc[0]>=self.depth.width or test_loc[0]<0 \
                        or test_loc[1]>=self.depth.height or test_loc[1]<0:
                    continue
                if(test_loc in visited):continue
                test_pt = self.ij_to_point(test_loc).data
                if(abs(test_pt[2]-next_point[2])<DELTA):
                    q.append(test_loc)
        return PointCloud(np.array(pts).T,"base_link"),PointCloud(np.array(closepts).T,"base_link").mean()

    def ij_to_point(self,loc):
        lin_ind = self.depth.width*loc[1]+loc[0]
        return self.points_3d[lin_ind]

    def princ_axis(self,points):
        '''
        returns the direction of the principle axis of the points
        points should be a 3xN array
        '''
        #construct moment matrix based on centroid and find the eigen vectors
        centroid = points.mean()
        x = points.x_coords - centroid.x
        y = points.y_coords - centroid.y
        z = points.z_coords - centroid.z
        Ixx = x.dot(x)
        Ixy = x.dot(y)
        Ixz = x.dot(z)
        Iyy = y.dot(y)
        Iyz = y.dot(z)
        Izz = z.dot(z)
        M=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
        w,v=np.linalg.eig(M)
        return v[:,np.argmax(w)]

    def close(self):
        print("Deleting grasp")

if __name__=='__main__':
    from phoxipy.phoxi_sensor import PhoXiSensor
    from matplotlib import pyplot as plt
    from yumiplanning.yumi_kinematics import YuMiKinematics as YK
    from tcps import *
    img=RgbdImage.from_color_and_depth(ColorImage.open("data2/color_1791.npy",frame='phoxi'),DepthImage.open("data2/depth_1791.npy",frame='phoxi'))
    T_CAM_BASE = RigidTransform.load("/home/jkerr/yumi/phoxipy/tools/phoxi_to_world_etch.tf").as_frames(from_frame="phoxi",to_frame="base_link")
    #T_CAM_BASE = RigidTransform.load("/nfs/diskstation/calib/phoxi/phoxi_to_world.tf").as_frames(from_frame="phoxi",to_frame="base_link")
    intr=PhoXiSensor.create_intr(img.width,img.height)
    LTCP=ABB_WHITE.as_frames(YK.l_tcp_frame,YK.l_tip_frame)
    RTCP=ABB_WHITE.as_frames(YK.r_tcp_frame,YK.r_tip_frame)
    points_3d = intr.deproject(img.depth)
    fig, ax = plt.subplots()
    ax.imshow(img.depth.data)
    left_coords,right_coords=None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        global left_coords,right_coords
        point=T_CAM_BASE*points_3d[lin_ind]
        print("Clicked point in world coords: ",point)
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # left_coords=(100,100)
    # right_coords=(200,200)
    g=GraspSelector(img,intr,T_CAM_BASE)
    # print("done initing one")
    # g=GraspSelector(img,intr,T_CAM_BASE)
    # g.single_grasp(left_coords,.01,TCP,vis=True)
    print("Beginning double grasp\n\n")
    g1,g2=g.double_grasp(left_coords,right_coords,.02,.02,LTCP,RTCP)
    print('done grasping',g1.pose.rotation,g2.pose.rotation)
    g.col_interface.visualize_grasps([g1.pose,g2.pose],RTCP)