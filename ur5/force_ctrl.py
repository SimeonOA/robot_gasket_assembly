from ur5py import UR5Robot
import time
import pdb

robot = UR5Robot(gripper=2)
# robot.set_playload(0)
pose = robot.get_pose(False)
robot.force_mode(robot.get_pose(convert=False),[0,0,1,0,0,0],[0,0,20,0,0,0],2,[3,3,3,3,3,3],damping=0.002)
# pdb.set_trace()
# time.sleep(0.01)
pose[0] -= 0.05
# for i in range(0, 1000):
#     # robot.force_mode(robot.get_pose(convert=False),[0,0,1,0,0,0],[30,30,30,0,0,0],2,[1,1,1,1,1,1])
#     tmp_pose = robot.get_pose(convert=False)
#     tmp_pose[0] += i * 0.0005
#     robot.servo_pose(tmp_pose, 0.01,convert=False)
#     time.sleep(0.01)
time.sleep(4)
robot.move_pose(pose, convert=False, interp="tcp", vel=0.1, acc=0.5)
time.sleep(4)
robot.end_force_mode()  