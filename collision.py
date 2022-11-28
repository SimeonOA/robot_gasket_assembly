from autolab_core import RigidTransform,Box
from queue import Empty
import numpy as np
import trimesh
import time
import multiprocessing as mp

#this defines the type of python process that will be created. the default is ForkProcess,
#but this creates a ton of duplicated memory so we use this instead
MP_CONTEXT = mp.get_context('fork')

class CollisionChecker(MP_CONTEXT.Process):
    def __init__(self,output_queue):
        super().__init__()
        self.output_queue=output_queue
        self.input_queue=MP_CONTEXT.Queue()

    def run(self):
        '''
        requests are formatted like (key,args,index)
        '''
        while True:
            try:
                request=self.input_queue.get(timeout=1)
            except Empty:
                continue
            if request[0]=="collide_gripper":
                self.output_queue.put(("collide_gripper",self.collide_gripper(*request[1]),request[2]))
            elif request[0]=="closest_gripper":
                self.output_queue.put(("closest_gripper",self.closest_gripper(*request[1]),request[2]))
            elif request[0]=="rate_pair":
                self.output_queue.put(("rate_pair",self.rate_pair(*request[1]),request[2]))
            elif request[0]=='set_meshes':
                self.set_meshes(*request[1])

    def set_meshes(self,point_mesh,wrist_mesh):
        self.point_mesh=point_mesh
        self.wrist_mesh=wrist_mesh
        self.scan_manager=trimesh.collision.CollisionManager()
        self.scan_manager.add_object('scan',self.point_mesh)

    @staticmethod
    def setup_gripper(manager,wrist_mesh,wrist_pose,tcp,grip_dims,arm_dims,grasp_dist,z_buffer=.015):
        tr = wrist_pose.matrix
        manager.add_object('wrist',wrist_mesh,tr)
        if grip_dims is not None:
            T_Z=RigidTransform(translation=[0,0,
                    -grip_dims[2]/2. - z_buffer - grasp_dist],
                from_frame=tcp.from_frame,to_frame=tcp.from_frame)
            T_BOX = wrist_pose*tcp*T_Z
            grip_box = trimesh.creation.box(extents=grip_dims)
            manager.add_object('gripper',grip_box,T_BOX.matrix)
        if arm_dims is not None:
            T_Z = RigidTransform(translation=[0,0,-arm_dims[1]/2.],from_frame=wrist_pose.from_frame,
                    to_frame=wrist_pose.from_frame)
            T_CYL=wrist_pose*T_Z
            arm_cyl = trimesh.creation.cylinder(radius=arm_dims[0],height=arm_dims[1])
            manager.add_object('arm',arm_cyl,T_CYL.matrix)

    def collide_gripper(self,wrist_pose,tcp,grasp_dist,grip_dims,arm_dims):
        '''
        does a collision check on the gripper wrist mesh. the 
        pose frame should be the gripper base frame
        '''
        man=trimesh.collision.CollisionManager()
        CollisionChecker.setup_gripper(man,self.wrist_mesh,wrist_pose,tcp,grip_dims,arm_dims,grasp_dist)
        coll = man.in_collision_other(self.scan_manager)
        return coll

    def closest_gripper(self,wrist_pose,tcp,arm_dims):
        '''
        returns the closest distance from gripper mesh to
        pointcloud mesh
        '''
        man=trimesh.collision.CollisionManager()
        CollisionChecker.setup_gripper(man,self.wrist_mesh,wrist_pose,tcp,None,arm_dims,None)
        dist = man.min_distance_other(self.scan_manager)
        return dist
    
    def rate_pair(self,grasp1,grasp2,tcp):
        T_WRIST_TCP=tcp.inverse()
        m1=trimesh.collision.CollisionManager()
        CollisionChecker.setup_gripper(m1,self.wrist_mesh,grasp1*T_WRIST_TCP,tcp,None,(.05,.14),None)
        m2=trimesh.collision.CollisionManager()
        CollisionChecker.setup_gripper(m2,self.wrist_mesh,grasp2*T_WRIST_TCP,tcp,None,(.05,.14),None)
        return m1.min_distance_other(m2),grasp1,grasp2

    def add_work(self,name,args,index):
        self.input_queue.put((name,args,index))

class CollisionInterface:
    def __init__(self,n_procs=8,z_min=-.02):
        self.z_min=z_min
        self.n_procs=n_procs
        self.procs=[]
        
    def setup(self):
        if len(self.procs)==self.n_procs:return
        self.wrist_mesh=trimesh.load("gripper.stl")
        self.output_queue=MP_CONTEXT.Queue()
        for i in range(self.n_procs):
            self.procs.append(CollisionChecker(self.output_queue))
            self.procs[-1].start()
        
    def set_scancloud(self,pointcloud):
        bounds=Box(np.array([-10,-10.,self.z_min]),np.array([10,10,.3]),"base_link")
        pointcloud,_ = pointcloud.box_mask(bounds)
        self.tri_pc = trimesh.PointCloud(pointcloud.data.T)
        self.point_mesh = trimesh.voxel.ops.points_to_marching_cubes(self.tri_pc.vertices,pitch=.005)
        for i in range(len(self.procs)):
            self.procs[i].add_work('set_meshes',(self.point_mesh,self.wrist_mesh),0)

    def collide_gripper(self,wrist_poses,tcp,grasp_dist,grip_dims=(.015,.007,.04),arm_dims=(.05,.14)):
        for i in range(len(wrist_poses)):
            args=(wrist_poses[i],tcp,grasp_dist,grip_dims,arm_dims)
            self.procs[i%len(self.procs)].add_work("collide_gripper",args,i)
        results = [None for i in range(len(wrist_poses))]
        for i in range(len(wrist_poses)):
            name,res,index=self.output_queue.get(True)
            if name!= "collide_gripper":raise Exception("wtf")
            results[index]=res
        return results

    def closest_gripper(self,wrist_poses,tcp,arm_dims=(.05,.11)):
        for i in range(len(wrist_poses)):
            args=(wrist_poses[i],tcp,arm_dims)
            self.procs[i%len(self.procs)].add_work("closest_gripper",args,i)
        results = [None for i in range(len(wrist_poses))]
        for i in range(len(wrist_poses)):
            name,res,index=self.output_queue.get(True)
            if name!= "closest_gripper":raise Exception("wtf")
            results[index]=res
        return results

    def visualize_grasps(self,grasps,tcp):
        scene=trimesh.scene.Scene()#for visualizing the collision
        scene.add_geometry(self.point_mesh)
        T_WRIST_TCP=tcp.inverse()
        for g in grasps:
            tr = g.as_frames(from_frame=T_WRIST_TCP.to_frame)*T_WRIST_TCP
            scene.add_geometry(self.wrist_mesh,transform=tr.matrix)
        scene.show()

    def rate_pair(self,pairs,tcp):
        for i in range(len(pairs)):
            args=(pairs[i][0],pairs[i][1],tcp)
            self.procs[i%len(self.procs)].add_work("rate_pair",args,i)
        results = [0.0 for i in range(len(pairs))]
        for i in range(len(pairs)):
            name,res,index=self.output_queue.get(True)
            if name!= "rate_pair":raise Exception("wtf")
            results[index]=res
        return results

    def close(self):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.join(timeout=.1)