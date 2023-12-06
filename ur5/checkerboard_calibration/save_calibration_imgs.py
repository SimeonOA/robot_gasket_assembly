import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyzed.sl as sl

# # Open the video device
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# if not cap.isOpened():
#     print("Error: Couldn't open the video device!")
#     exit()
def grab_zed_mat(side_cam,runtime_parameters, image, point_cloud, depth):
    if side_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        side_cam.retrieve_image(image, sl.VIEW.LEFT)
        side_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        side_cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return image, point_cloud, depth

# # Capture a single frame
# ret, img = cap.read()

def save_img(img):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    # Save the captured frame as an image
    cv2.imwrite('/home/r2d2/R2D2/cable_insertion/calibration/'+current_time+'.jpg', img)
    print("Saved to " + '/home/r2d2/R2D2/cable_insertion/calibration/'+current_time+'.jpg')

# # Release the video device
# cap.release()

if __name__ == "__main__":
    # Initialize robot interface
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
    # cam_instrinsics = CameraIntrinsics(focal_left_x, focal_left_y, princ_left_x, princ_left_y)
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
    save_img(rgb_img)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
