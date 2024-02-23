from zed_camera import ZedCamera
import pyzed.sl as sl
from utils import *
import matplotlib.pyplot as plt
import cv2


# side_cam = sl.Camera()
# # Create a InitParameters object and set configuration parameters
# init_params = sl.InitParameters()
# init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
# init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
# init_params.camera_resolution = sl.RESOLUTION.HD720
# init_params.set_from_serial_number(20120598) #overhead camera
# status = side_cam.open(init_params)
# if status != sl.ERROR_CODE.SUCCESS:
#     raise RuntimeError("Camera Failed To Open")
# runtime_parameters = sl.RuntimeParameters()
# zed = ZedCamera(side_cam)
# zed.set_calibration_mode()
# data, timestamp = zed.read_camera()
# cam_id = 20120598
# cam_id = 22008760

# zed = sl.Camera()

# # Set configuration parameters
# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD1080
# # init_params.camera_fps = 30
# # init_params.set_from_serial_number(cam_id)
# err = zed.open(init_params)
# if err != sl.ERROR_CODE.SUCCESS:
#     print('aaa')
#     exit()
# zed_serial = zed.get_camera_information().serial_number
# print("Hello! This is my serial number: {}".format(zed_serial))

cam_id = 22008760
side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(cam_id)
# image, point_cloud, depth = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)
# rgb_img = image.get_data()[:,:,:3]#
rgb_img = get_zed_img(side_cam,runtime_parameters, image, point_cloud, depth )
# rgb_img = 0
print("aaaa")
# Naming a window 
# cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
  
# Using resizeWindow()  
# cv2.imshow("Resized Window",rgb_img)
# cv2.resizeWindow("Resized_Window", 300, 700)
plt.imshow(rgb_img)
plt.show()
# plt.show()
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()