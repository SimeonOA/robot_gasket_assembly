import pyzed.sl as sl

def setup_zed_camera(cam_id):
    cam = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    # init_params.camera_fps = 30
    init_params.set_from_serial_number(cam_id)
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Camera Failed To Open")
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()
    depth = sl.Mat()
    return cam, runtime_parameters, image, point_cloud, depth

def grab_zed_mat(side_cam,runtime_parameters, image, point_cloud, depth):
    if side_cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        side_cam.retrieve_image(image, sl.VIEW.LEFT)
        side_cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        side_cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return image, point_cloud, depth

def get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth):
    image, _, _ = grab_zed_mat(side_cam, runtime_parameters, image, point_cloud, depth)
    rgb_img = image.get_data()[:,:,:3]
    return rgb_img 