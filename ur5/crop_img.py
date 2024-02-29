from PIL import Image
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# def crop_transparent_edges(input_image_path, output_image_path):
#     # Open the input image
#     img = Image.open(input_image_path)
#     # Convert the image to RGBA if it is not already
#     img = img.convert("RGBA")

#     # Get the bounding box of the non-transparent pixels
#     bbox = img.getbbox()

#     # If the image is completely transparent, bbox will be None
#     if bbox:
#         # Crop the image to the bounding box
#         cropped_img = img.crop(bbox)
#         # Save the cropped image
#         cropped_img.save(output_image_path)
#     else:
#         print("The image is completely transparent.")

# # Example usage
# input_image_path = '/home/gasket/robot_cable_insertion/ur5/images/curved_channel_right_direction.png'
# output_image_path = '/home/gasket/robot_cable_insertion/ur5/images/curved_channel_right_direction_cropped.png'

# crop_transparent_edges(input_image_path, output_image_path)



# def resize_image(input_image_path, output_image_path):
#     with Image.open(input_image_path) as image:
#         size = (368, 184)
#         resized_image = image.resize(size)
#         resized_image.save(output_image_path)

# # Example usage
# input_image_path = '/home/gasket/robot_cable_insertion/ur5/template_masks/full_trapezoid_assembly_cropped_good.png'
# output_image_path = '/home/gasket/robot_cable_insertion/ur5/template_masks/full_trapezoid_assembly_cropped_good_resized.png'
# # new_size = (800, 600) # New dimensions (width, height)
# breakpoint()
# resize_image(input_image_path, output_image_path)

straight_template_mask = cv2.imread('/home/gasket/robot_cable_insertion/ur5/templates_crop_master/master_straight_channel_template.png')
channel_skeleton = skeletonize(straight_template_mask)
output_image_path = '/home/gasket/robot_cable_insertion/ur5/images/master_straight_channel_template_skeleton.png'
cv2.imwrite(output_image_path, channel_skeleton) 


