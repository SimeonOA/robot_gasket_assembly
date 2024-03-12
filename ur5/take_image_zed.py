from utils import *

overhead_cam_id = 22008760
front_eval_cam_id = 20120598 #20812520
side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id)
front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth = setup_zed_camera(front_eval_cam_id)
overhead_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
plt.imsave('overhead.png', overhead_img)
front_img = get_zed_img(front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth)
front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
plt.imsave('left_changed.png', front_img)