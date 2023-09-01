import push
import utils
from depth_sensing import get_rgb_get_depth


depth_image, rgb_image = get_rgb_get_depth()

skeleton_img = utils.skeletonize_img()
total_length, final_endpoints = utils.find_length_and_endpoints(skeleton_img)