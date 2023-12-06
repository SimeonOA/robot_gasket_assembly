import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import math
import sys
from autolab_core import RigidTransform, CameraIntrinsics
import random
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import cv2
from save_calibration_imgs import save_img

from autolab_core import RigidTransform, RgbdImage, DepthImage, ColorImage
# from autolab_core.transformations import rotation_matrix 


import logging
# T_CAM_BASE = RigidTransform.load("/home/mallika/triton4-lip/phoxipy/tools/phoxi_to_world_bww.tf").as_frames(from_frame="phoxi", to_frame="base_link")
TEMPLATE_HEIGHT = {'curved':0.0254, 'straight':0.015, 'trapezoid':0.02}


def click_pts(rgb_img, matched_template, camera_intrinsics):
    rope_list = []
    channel_list = []

    # Function to handle mouse click events
    def on_mouse_click(event):
        if event.button == 1:  # Left-click
            rope_list.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot where the click occurred
        elif event.button == 3:  # Right-click
            channel_list.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'bo')  # Plot a blue dot where the click occurred

        plt.draw()

    plt.imshow(rgb_img)
    plt.axis('image')  # Set aspect ratio to be equal
    plt.title('Click on four points')

    # Connect the click event to the figure
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # Show the plot and wait for user input
    plt.show()

    point_dict = {rope_point: 1 for rope_point in rope_list}
    point_dict.update({channel_point: 0 for channel_point in channel_list})

    print("points clicked",point_dict)
    # Save the points array if needed
    np.savetxt('points.txt', np.array(rope_list + channel_list))

    
    robot.close_grippers()



    while True:
        use_click_pts = True
        use_lstsq = False
        breakpoint()
        for point in point_dict:
            indicator = point_dict[point]
            transformed_end = get_world_coord_from_pixel_coord(point, camera_intrinsics)
            # hardcoded z height
            transformed_end[2] = 0.15
            print("transformed_end", transformed_end)
            # means the point is on the rope therefore we want to reach to the table
            if indicator == 1:
                # we shouldn't need to do anything here
                pass 
            # point is on the channel so we want to be right above it
            elif indicator == 0:
                transformed_end[2] += TEMPLATE_HEIGHT[matched_template]
            print('FINAL transformed_end', transformed_end)
            robot.go_to_ee_pose(transformed_end)
            # plt.scatter(x=point[0],y=point[1] , c='r')
            # plt.imshow(rgb_img)
            # plt.show()
            robot.go_home()
       



def get_world_coord_from_pixel_coord(pixel_coord, cam_intrinsics):
    '''
    pixel_coord: [x, y] in pixel coordinates
    cam_intrinsics: 3x3 camera intrinsics matrix
    '''
    DIST_TO_TABLE = 0.775
    print('cam instrics', cam_intrinsics)
    cam_intrinsics = cam_intrinsics.copy()
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = DIST_TO_TABLE * np.linalg.inv(cam_intrinsics).dot(np.r_[pixel_coord, 1.0])
    point_3d_world = T_CAM_BASE.matrix.dot(np.r_[point_3d_cam, 1.0])
    # print(T_CAM_BASE.matrix)
    point_3d_world = point_3d_world[:3]/point_3d_world[3]
    print('non-homogenous = ', point_3d_world)
    return point_3d_world


def grab_zed_mat(side_cam,runtime_parameters, image, point_cloud, depth):
    if side_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        side_cam.retrieve_image(image, sl.VIEW.LEFT)
        side_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        side_cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return image, point_cloud, depth

def click_points_simple(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    left_coords,right_coords = None,None
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        # lin_ind = int(img.depth.ij_to_linear(np.array(xind),np.array(yind)))
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords=coords
        elif(event.button==3):
            right_coords=coords
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords, right_coords

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print more debug statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()

def get_transform_matrix(calibration_matrix):
    cam_pose = calibration_matrix[:3]
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    # rot = rotation_matrix(np.pi, [0,1,0], cam_pose)[:3,:3]
    return RigidTransform(rotation_matrix, cam_pose, from_frame="camera", to_frame="world")

def transform_from_robot_frame(transformed_point, calibration_matrix):
    transformed_point = transformed_point.copy()
    transformed_point[2] -= 0.165
    cam_pose = calibration_matrix[:3]
    transformed_point = transformed_point - cam_pose
    cam_euler = calibration_matrix[3:]
    rotation_matrix = R.from_euler("xyz", cam_euler).as_matrix()
    original_point = np.dot(transformed_point, np.linalg.inv(rotation_matrix.T))
    return original_point


def L_rigid_tf(world_tf):
    place_transform = RigidTransform(
        translation=world_tf,
        rotation= np.array([np.pi, 0, 0]),
        from_frame='world',
        to_frame="base_link",
    )
    return place_transform

if __name__ == "__main__":
    # left_cam_calibration_matrix = np.array([ 5.87372635e-01,  3.48340382e-02,  7.42931818e-01,  3.08162185e+00,
    #    -1.33631679e-03,  1.69080620e+00])
    # # left_cam_calibration_matrix = np.array([-9.01009703*2.54/100, 0,  7.42931818e-01,  np.pi,
    # #    0,  np.pi/2])
    # left_cam_calibration_matrix = np.array([5.87372635e-01, 0,  7.42931818e-01,  np.pi,
    #    0,  np.pi/2])
    
    # rvecs = np.array([[-0.00562848],
    #    [ 0.03775286],
    #    [-1.56895915]]).reshape((3,))

    # # Given Euler angles (in radians) as a 3x1 array
    # euler_angles = np.array([[-0.00562848], [0.03775286], [-1.56895915]])

    # # Angle of rotation in radians
    # rotation_angle = np.pi  # π radians

    # # Compute the rotation matrix for the Z-axis rotation
    # rotation_matrix_z = np.array([
    #     [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
    #     [np.sin(rotation_angle), np.cos(rotation_angle), 0],
    #     [0, 0, 1]
    # ])

    # # Convert the Euler angles to a rotation matrix
    # # Assuming ZYX convention: Rotate Z, then Y, then X
    # rotation_matrix_euler = np.dot(np.dot(rotation_matrix_z, np.eye(3)), np.eye(3))

    # # Apply the rotation to the original Euler angles
    # new_rvecs = np.dot(rotation_matrix_euler, euler_angles)

    # print(new_rvecs)
    # # tvecs= np.array([[ -1.7528218 ],
    # #    [  2.8796733 ],
    # #    [203.03759151]]).reshape((3,))
    # tvecs = np.array([5.87372635e-01, -0.0575, 0.78])
    # # rvecs = np.array([np.pi, 0, np.pi/2])
    # left_cam_calibration_matrix = np.concatenate((tvecs, rvecs))

    # one obtained directly from r2d2 calibration
    left_cam_calibration_matrix = np.array([ 0.56468337, -0.01213499,  0.74841675,  3.11672833,  0.00908736,
        1.6351875 ])
    # hand-tuned version 
    left_cam_calibration_matrix = np.array([ 0.56968337, -0.01213499,  0.74841675,  3.11672833,  0.00908736,
        1.6351875 ])
    print("current camera extrinsics matrix is!", left_cam_calibration_matrix)
    T_CAM_BASE = get_transform_matrix(left_cam_calibration_matrix)
    # T_CAM_BASE = 
    #np.array([ 5.11421595e-01, -7.65744179e-04,  7.37325573e-01,  3.09109159e+00,
       #-9.21380307e-04,  1.84634534e+00])
       
    # matrix obtained through manual cv2 process with justin and tara 
#     camera_intrinsics = np.array([[513.40203347 ,  0.,         641.97983643],
#  [  0.,         513.51966493, 376.51538906],
#  [  0.,           0.,           1.,        ]])

    #matrix obtained using the default parameters of the 
    camera_intrinsics = np.array([[513.40203347 ,  0.,         641.97983643],
 [  0.,         513.51966493, 376.51538906],
 [  0.,           0.,           1.,        ]])

    robot = FrankaFranka()
    # breakpoint()
    robot.go_home()
    
    robot.close_grippers()
    # robot.go_to_ee_pose([0.4, 0.0, 0.14])

    # print(robot.robot.get_ee_pose())

    # Create a Camera object
    # id = int(varied_camera_2_id)
    side_cam = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.set_from_serial_number(20120598) #overhead camera
    status = side_cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Camera Failed To Open")
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.SENSING_MODE_FILL
    calibration_params = side_cam.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    focal_left_y = calibration_params.left_cam.fy
    princ_left_x = calibration_params.left_cam.cx
    princ_left_y = calibration_params.left_cam.cy
    
    cam_intrinsics_from_zed = CameraIntrinsics(frame='camera',fx=focal_left_x, fy=focal_left_y, cx=princ_left_x, cy=princ_left_y)
    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis
    # tz = calibration_params.T.z
    # Horizontal field of view of the left eye in degrees
    # h_fov = calibration_params.left_cam.h_fov

    image = sl.Mat()
    point_cloud = sl.Mat()
    depth = sl.Mat()

    image, point_cloud, depth = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)
    rgb_img = image.get_data()[:,:,:3]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    img = rgb_img.copy()
    plt.imshow(img)
    plt.show()
    
    use_chessboard = True
    if not use_chessboard:
        click_pts(img, 'straight', cam_intrinsics_from_zed._K)
    else:
        CHECKERBOARD = (6,9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 
        
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        plt.imshow(gray)
        plt.show()
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            chessboard = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        else:
            print('ret is false :(')
        plt.imshow(chessboard)
        plt.show()
        place_1 = corners[0][0]
        print(place_1)
        place_2 = corners[5][0]
        place_3 = corners[-6][0]
        place_4 = corners[-1][0]
        print(place_2)

        def plot_corners():
            plt.scatter(x=place_1[0],y=place_1[1] , c='r')
            plt.scatter(x=place_2[0],y=place_2[1] , c='g')
            plt.scatter(x=place_3[0],y=place_3[1] , c='b')
            plt.scatter(x=place_4[0],y=place_4[1] , c='y')
            plt.imshow(rgb_img)
            plt.show()
        plot_corners()
        # print(T_CAM_BASE)
        print('THIS IS CAM INTRINSICS MATRIX FROM ZED', cam_intrinsics_from_zed._K)
        place1_depthless = get_world_coord_from_pixel_coord(place_1, cam_intrinsics_from_zed._K)
        # place1_depthless = get_world_coord_from_pixel_coord(place_1, camera_intrinsics)
        place2_depthless = get_world_coord_from_pixel_coord(place_2, cam_intrinsics_from_zed._K)
        # place2_depthless = get_world_coord_from_pixel_coord(place_2, camera_intrinsics)
        place3_depthless = get_world_coord_from_pixel_coord(place_3, cam_intrinsics_from_zed._K)
        # place3_depthless = get_world_coord_from_pixel_coord(place_3, camera_intrinsics)
        place4_depthless = get_world_coord_from_pixel_coord(place_4, cam_intrinsics_from_zed._K)
        # place4_depthless = get_world_coord_from_pixel_coord(place_4, camera_intrinsics)
        places = [place1_depthless, place2_depthless, place4_depthless, place3_depthless]
        print("Depthless", place1_depthless)

        # place1_transform = L_rigid_tf(place1_depthless)
        # place2_transform = L_rigid_tf(place2_depthless)
        # place3_transform = L_rigid_tf(place3_depthless)
        # place4_transform = L_rigid_tf(place4_depthless)
        
        while True:
            breakpoint()
            for place in places:
                # breakpoint()
                print('original place', place)
                place[2] = 0.155
                robot.go_to_ee_pose(place)
                print("Goal: ", place)
                print("Actual: ", robot.robot.get_ee_pose())
            robot.go_home()
        # iface.go_cartesian(
        #     l_targets=[place1_transform, place2_transform,place4_transform, place3_transform, place1_transform], removejumps=[6])
        # iface.sync()
        # # time.sleep(1)
        # # iface.home()
        # iface.sync()
        time.sleep(5)