import pyzed.sl as sl

class Zed():
    def __init__(self, cam_id=None):
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        # init_params.camera_fps = 30
        if cam_id:
            init_params.set_from_serial_number(cam_id)
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()
        self.depth = sl.Mat()

    def get_zed_mat(self):
        if self.cam.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(self.image, sl.VIEW.LEFT)
            self.cam.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            self.cam.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        return self.image, self.point_cloud, self.depth

    def get_zed_img(self):
        image, _, _ = self.get_zed_mat()
        rgb_img = image.get_data()[:,:,:3]
        return rgb_img 