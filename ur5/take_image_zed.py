from utils import *

overhead_cam_id = ...
front_eval_cam_id = ...
side_cam, runtime_parameters, image, point_cloud, depth = setup_zed_camera(overhead_cam_id)
front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth = setup_zed_camera(front_eval_cam_id)
overhead_img = get_zed_img(side_cam, runtime_parameters, image, point_cloud, depth)
overhead_img = cv2.cvtColor(overhead_img, cv2.COLOR_BGR2RGB)
i=...
plt.imsave(f'/home/gasket/robot_cable_insertion/ur5/evaluation_st/overhead{i}.png', overhead_img)
front_img = get_zed_img(front_cam, front_runtime_parameters, front_image, front_point_cloud, front_depth)
front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
plt.imsave(f'/home/gasket/robot_cable_insertion/ur5/evaluation_st/front{i}.png', front_img)