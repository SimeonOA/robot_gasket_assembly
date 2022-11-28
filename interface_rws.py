from yumirws.yumi import YuMi
from autolab_core import RigidTransform
import numpy as np
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from yumiplanning.yumi_planner import *
from phoxipy.phoxi_sensor import PhoXiSensor
from grasp import Grasp
import time
from scipy import interpolate as inter

'''
This class invokes yumirws, phoxipy, yumiplanning to give a user-friendly interface for
sending motion commands, getting depth camera data, and etc

Consider it an abstract wrapper for the yumi+phoxi which enables some project-specific
capabilities without polluting the original code
(ie keep the yumipy,yumiplanning repo's pure to their intent)
'''

def remove_twist(traj):
    '''removes the jump in the last joint by freezing it before the jump'''
    print('\nremoving wrist twist\n')
    lastval=traj[-1]
    jumpi=None
    for i in range(1,len(traj)):
        lastval=traj[i,-1]
        if abs(traj[i,-1]-traj[i-1,-1])>.4:
            jumpi=i
            print(f"break point {traj[i,-1]} ,{traj[i-1,-1]}")
            break
    if jumpi is not None:
        traj[jumpi:,-1] = lastval
    print(f"jumpi {jumpi}")
    return traj

class Interface:
    GRIP_DOWN_R = np.diag([1,-1,-1])#orientation where the gripper is facing downwards
    L_HOME_STATE=[-0.5810662 , -1.34913424,  0.73567095,  0.55716616,  1.56402364,
        1.25265177,  2.84548536]
    R_HOME_STATE=np.array([ 0.64224786, -1.34920282, -0.82859683,  0.52531042, -1.64836569,
            1.20916355, -2.83024169])
    def __init__(self,phoxi_name,l_tcp,r_tcp,speed=(.3,2*np.pi)):
        #set up the robot
        self.L_TCP=l_tcp
        self.R_TCP=r_tcp
        self.speed=speed
        self.y=YuMi(l_tcp=self.L_TCP,r_tcp=self.R_TCP)
        self.yk = YK()
        self.yk.set_tcp(self.L_TCP,self.R_TCP)
        self.set_speed(speed)
        #set up the phoxi
        self.T_PHOXI_BASE = RigidTransform.load("../phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi",to_frame="base_link")
        self.cam = PhoXiSensor("1703005")
        self.cam.start()
        img=self.cam.read()
        self.cam.intrinsics=self.cam.create_intr(img.width,img.height)
        self.planner = Planner()

    def take_image(self):
        return self.cam.read()

    def set_speed(self,speed):
        '''
        set tcp move speed. format is tuple of (m/s,deg/s)
        '''
        self.speed=speed

    def home(self):
        try:
            self.go_config_plan(self.L_HOME_STATE,self.R_HOME_STATE)
        except PlanningException:
            self.go_configs([self.L_HOME_STATE],[self.R_HOME_STATE])
        self.sync()
        
    def __del__(self):
        self.cam.stop()

    def open_grippers(self):
        self.y.left.open_gripper()
        self.y.right.open_gripper()

    def close_grippers(self):
        self.y.left.close_gripper()
        self.y.right.close_gripper()

    #move robot to the given point
    def go_cartesian(self,l_targets=[],r_targets=[],fine=False,nwiggles=(None,None),rot=(None,None),
            removej6jump=False):
        self.sync()
        l_cur_q = self.y.left.get_joints()
        r_cur_q = self.y.right.get_joints()
        l_cur_p, r_cur_p = self.yk.fk(qleft=l_cur_q,qright=r_cur_q)
        lpts=[l_cur_p]+l_targets
        rpts=[r_cur_p]+r_targets
        #compute the actual path (THESE ARE IN URDF ORDER (see urdf_order_2_yumi for details))
        lpath,rpath = self.yk.interpolate_cartesian_waypoints(l_waypoints=lpts,l_qinit=l_cur_q,
                    r_qinit=r_cur_q,r_waypoints=rpts,N=50)
        lpath=np.array(lpath)
        rpath=np.array(rpath)
        if removej6jump:
            lpath=remove_twist(lpath)
            rpath=remove_twist(rpath)
        if nwiggles[0] is not None: lpath = self.wiggle(lpath,rot[0],nwiggles[0],5)
        if nwiggles[1] is not None: rpath = self.wiggle(rpath,rot[1],nwiggles[1],5)
        self.go_configs(l_q=lpath,r_q=rpath,together=True)
        #this "fine" motion is to account for the fact that our ik is slightly (~.5cm) wrong becase
        # the urdf file is wrong
        if fine:
            if(len(l_targets)>0):self.y.left.goto_pose(l_targets[0],speed=self.speed)
            if(len(r_targets)>0):self.y.right.goto_pose(r_targets[0],speed=self.speed)

    def go_pose_plan(self,l_target=None,r_target=None,fine=False):
        self.sync()
        l_cur = self.y.left.get_joints()
        r_cur = self.y.right.get_joints()

        l_path,r_path = self.planner.plan_to_pose(l_cur,r_cur,self.yk,l_target,r_target)
    
        self.go_configs(l_path,r_path,True)
        if fine:
            if l_target is not None:self.y.left.goto_pose(l_target,speed=self.speed)
            if r_target is not None:self.y.right.goto_pose(r_target,speed=self.speed)

    def go_config_plan(self,l_q=None,r_q=None):
        self.sync()
        l_cur = self.y.left.get_joints()
        r_cur = self.y.right.get_joints()
        l_path,r_path=self.planner.plan(l_cur,r_cur,l_q,r_q)
        self.go_configs(l_path,r_path,True)

    def go_delta(self,l_trans=None,r_trans=None,reltool=False):
        #meant for small motions
        self.sync()
        l_delta,r_delta=None,None
        if l_trans is not None:
            l_cur=self.y.left.get_pose()
            if reltool:
                l_delta=RigidTransform(translation=l_trans,from_frame=l_cur.from_frame,to_frame=l_cur.from_frame)
                l_new=l_cur*l_delta
            else:
                l_delta=RigidTransform(translation=l_trans,from_frame=l_cur.to_frame,to_frame=l_cur.to_frame)
                l_new=l_delta*l_cur
        if r_trans is not None:
            r_cur=self.y.right.get_pose()
            if reltool:
                r_delta=RigidTransform(translation=r_trans,from_frame=r_cur.from_frame,to_frame=r_cur.from_frame)
                r_new=r_cur*r_delta
            else:
                r_delta=RigidTransform(translation=r_trans,from_frame=r_cur.to_frame,to_frame=r_cur.to_frame)
                r_new=r_delta*r_cur
        if l_delta is not None:
            self.y.left.goto_pose(l_new,speed=self.speed)
        if r_delta is not None:
            self.y.right.goto_pose(r_new,speed=self.speed)

    def pull_apart(self):
        self.sync()
        lp=RigidTransform(translation=[.4,.6,.2],rotation=self.GRIP_DOWN_R,
                    from_frame=YK.l_tcp_frame,to_frame='base_link')
        rp=RigidTransform(translation=[.4,-.6,.2],rotation=self.GRIP_DOWN_R,
                    from_frame=YK.r_tcp_frame,to_frame='base_link')
        try:
            self.go_cartesian(l_targets=[lp],
                    r_targets=[rp],
                    nwiggles=(10,10),rot=(.3,.3),removej6jump=True)
        except:
            print("Couldn't compute smooth cartesian path, falling back to planner")
            self.go_pose_plan(lp,rp)

        # now move closer to center again
        lp=RigidTransform(translation=[.4,.1,.3],rotation=self.GRIP_DOWN_R,
                    from_frame=YK.l_tcp_frame,to_frame='base_link')
        rp=RigidTransform(translation=[.4,-.1,.3],rotation=self.GRIP_DOWN_R,
                    from_frame=YK.r_tcp_frame,to_frame='base_link')
        try:
            self.go_cartesian(l_targets=[lp],
                    r_targets=[rp],
                    nwiggles=(10,10),rot=(.0,.0),removej6jump=True)
        except:
            print("Couldn't compute smooth cartesian path, falling back to planner")
            self.go_pose_plan(lp,rp)

    def wiggle(self,jpath,rot,nwiggles,ji):
        '''
        applies nwiggles wiggles along the path by distributing them evenly
        '''
        def getT(jtraj):
            return np.linspace(0,1,jtraj.shape[0])
        jpath=np.array(jpath.copy())
        if len(jpath)==0:return []
        def clip(j):
            return max(min(j,self.yk.left_joint_lims[1][5]),self.yk.left_joint_lims[0][5])
        jtraj=jpath[:,ji]
        jspline=inter.CubicSpline(getT(jpath),jtraj)
        wigglespline=inter.CubicSpline(np.linspace(0,1,2*nwiggles+2),[0]+nwiggles*[-rot/2.,rot/2.]+[0])
        newjtraj=[clip(jspline(t)+wigglespline(t)) for t in np.linspace(0,1,len(jpath))]
        jpath[:,ji]=newjtraj
        return jpath

    def go_configs(self,l_q=[],r_q=[],together=False):
        '''
        moves the arms along the given joint trajectories
        l_q and r_q should be given in urdf format as a np array
        '''
        if together and len(l_q) == len(r_q):
            self.y.move_joints_sync(l_q,r_q,speed=self.speed)
        else:
            if len(l_q)>0:self.y.left.move_joints_traj(l_q,speed=self.speed,zone='z20')
            if len(r_q)>0:self.y.right.move_joints_traj(r_q,speed=self.speed,zone='z20')

    def grasp(self,l_grasp=None,r_grasp=None):
        '''
        Carries out the grasp motions on the desired end poses. responsible for doing the 
        approach to pre-grasp as well as the actual grasp itself.
        attempts
        both arguments should be a Grasp object
        '''
        l_waypoints=[]
        r_waypoints=[]
        l_pre,r_pre=None,None
        self.close_grippers()
        if l_grasp is not None:
            l_pre=l_grasp.compute_pregrasp()
            l_waypoints.append(l_pre)
        if r_grasp is not None:
            r_pre=r_grasp.compute_pregrasp()
            r_waypoints.append(r_pre)
        try:
            self.go_cartesian(l_waypoints,r_waypoints,fine=True)
        except:
            self.go_pose_plan(l_pre,r_pre,fine=True)
        self.sync()
        #move in
        if l_grasp is not None:
            self.y.left.goto_pose(l_grasp.compute_gripopen(),linear=True,speed=l_grasp.speed)
        if r_grasp is not None:
            self.y.right.goto_pose(r_grasp.compute_gripopen(),linear=True,speed=r_grasp.speed)
        self.sync()
        #put grippers in right position
        if l_grasp is not None:
            self.y.left.move_gripper(l_grasp.gripper_pos)
        if r_grasp is not None:
            self.y.right.move_gripper(r_grasp.gripper_pos)
        self.sync()
        if l_grasp is not None:
            self.y.left.goto_pose(l_grasp.pose,linear=True,speed=l_grasp.speed)
        if r_grasp is not None:
            self.y.right.goto_pose(r_grasp.pose,linear=True,speed=r_grasp.speed)
        self.sync()
        #grasp
        if l_grasp is not None:
            self.y.left.close_gripper()
        if r_grasp is not None:
            self.y.right.close_gripper()

    def sync(self):
        self.y.left.sync()
        self.y.right.sync()

    def refine_states(self,left=True,right=True,t_tol=.1,r_tol=.4):
        '''
        attempts to move the arms into a better joint configuration without deviating much at the end effector pose
        '''
        l_q = self.y.left.get_joints()
        r_q = self.y.right.get_joints()
        l_traj,r_traj = self.yk.refine_state(l_state=l_q,r_state=r_q,t_tol=t_tol,r_tol=r_tol)
        if not left:l_traj=[]
        if not right:r_traj=[]
        self.go_configs(l_q=l_traj,r_q=r_traj)

    def shake_left_J(self,rot,num_shakes,speed=(1.5,2000),ji=5):
        curj = self.y.left.get_joints()
        forj = curj.copy()
        forj[ji] = min(curj[ji]+rot/2.,self.yk.left_joint_lims[1][5])
        backj = curj.copy()
        backj[ji] = max(curj[ji]-rot/2.,self.yk.left_joint_lims[0][5])
        traj=[]
        for i in range(num_shakes):
            traj.append(backj)
            traj.append(forj)
        traj.append(curj)
        self.y.left.move_joints_traj(traj,speed=speed,zone='z20')

    def shake_right_J(self,rot,num_shakes,speed=(1.5,2000),ji=5):
        curj = self.y.right.get_joints()
        forj = curj.copy()
        forj[ji] = min(curj[ji]+rot/2.,self.yk.right_joint_lims[1][5])
        backj = curj.copy()
        backj[ji] = max(curj[ji]-rot/2.,self.yk.right_joint_lims[0][5])
        traj=[]
        for i in range(num_shakes):
            traj.append(backj)
            traj.append(forj)
        traj.append(curj)
        self.y.right.move_joints_traj(traj,speed=speed,zone='z20')

    def shake_left_R(self,rot,num_shakes,trans=[0,0,0],speed=(1.5,2000)):
        '''
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        '''
        #assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp=self.yk.l_tcp
        old_r_tcp=self.yk.r_tcp
        rot=np.array(rot)
        trans=np.array(trans)
        self.yk.set_tcp(None,None)
        cur_state = self.y.left.get_joints()
        curpose,_ = self.yk.fk(qleft=cur_state)
        R_for = RigidTransform(translation=trans/2,rotation=RigidTransform.rotation_from_axis_angle(rot/2.0),
            from_frame=self.yk.l_tcp_frame,to_frame=self.yk.l_tcp_frame)
        R_back = RigidTransform(translation=-trans/2,rotation=RigidTransform.rotation_from_axis_angle(-rot/2.0),
            from_frame=self.yk.l_tcp_frame,to_frame=self.yk.l_tcp_frame)
        target_for = curpose*R_for
        target_back = curpose*R_back
        target_for_q,_=self.yk.ik(left_qinit=cur_state,left_pose=target_for)
        target_back_q,_=self.yk.ik(left_qinit=cur_state,left_pose=target_back)
        if(np.linalg.norm(target_for_q-target_back_q)>3):
            print("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp,old_r_tcp)
            return
        traj=[]
        for i in range(num_shakes):
            traj.append(target_for_q)
            traj.append(target_back_q)
        traj.append(cur_state)
        self.y.left.move_joints_traj(traj,speed=speed,zone='z10')
        self.yk.set_tcp(old_l_tcp,old_r_tcp)
        
    def shake_right_R(self,rot,num_shakes,trans=[0,0,0],speed=(1.5,2000)):
        '''
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        '''
        #assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp=self.yk.l_tcp
        old_r_tcp=self.yk.r_tcp
        rot=np.array(rot)
        trans=np.array(trans)
        self.yk.set_tcp(None,None)
        cur_state = self.y.right.get_joints()
        _,curpose = self.yk.fk(qright=cur_state)
        R_for = RigidTransform(translation=trans/2,rotation=RigidTransform.rotation_from_axis_angle(rot/2.0),
            from_frame=self.yk.r_tcp_frame,to_frame=self.yk.r_tcp_frame)
        R_back = RigidTransform(translation=-trans/2,rotation=RigidTransform.rotation_from_axis_angle(-rot/2.0),
            from_frame=self.yk.r_tcp_frame,to_frame=self.yk.r_tcp_frame)
        target_for = curpose*R_for
        target_back = curpose*R_back
        _,target_for_q=self.yk.ik(right_qinit=cur_state,right_pose=target_for)
        _,target_back_q=self.yk.ik(right_qinit=cur_state,right_pose=target_back)
        if(np.linalg.norm(target_for_q-target_back_q)>3):
            print("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp,old_r_tcp)
            return
        traj=[]
        for i in range(num_shakes):
            traj.append(target_for_q)
            traj.append(target_back_q)
        traj.append(cur_state)
        self.y.right.move_joints_traj(traj,speed=speed,zone='z10')
        self.yk.set_tcp(old_l_tcp,old_r_tcp)
