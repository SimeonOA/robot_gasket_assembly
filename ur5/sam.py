# from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np, matplotlib, time
import os
from calibration.image_robot import ImageRobot
from real_sense_modules import *

GREEN_HSV = np.array([[0, 121, 131], [179, 255, 255]])
CABLE_HSV = np.array([[0, 0, 156], [84, 119, 255]])

def post_process_mask(binary_map):
    # convert to binary by thresholding
    # breakpoint()
    # if invert:
    #     ret, binary_map = cv2.threshold(src,127,255,cv2.THRESH_BINARY_INV)
    # else:
    #     ret, binary_map = cv2.threshold(src,127,255,0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
    val = 255

    for i in range(0, nlabels - 1):
        if areas[i] >= 9000:   #keep
            result[labels == i + 1] = val
    
    plt.imshow(result, cmap='gray')
    plt.show()

    return result

# def mask_colors(colors, in_min, in_max, color_space="hsv"):
#     # colors (M, 3), in_min (N, 3), in_max (N, 3)
#     if color_space == "hsv":
#         colors = matplotlib.colors.rgb_to_hsv(colors)

#     m = (colors[:, None] >= in_min[None]) & (colors[:, None] <= in_max[None])
#     combined_mask = np.any(np.all(m, axis=-1), axis=-1)
#     return combined_mask

def mask(image, lower, upper):
    # output mask values: array([  0, 255], dtype=uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def pick_hsv(frame):
    # A required callback method that goes into the trackbar function.
    def nothing(x):
        pass

    # Create a window named trackbars.
    cv2.namedWindow("Trackbars")

    # Now create 6 trackbars that will control the lower and upper range of 
    # H,S and V channels. The Arguments are like this: Name of trackbar, 
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    while True:
        # Get the new values of the trackbar in real time as the user changes 
        # them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
        
        # Filter the image and get the binary mask, where white represents 
        # your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)
    
        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Converting the binary mask to 3 channel image, this is just so 
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3,frame,res))
        
        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
        
        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        # If the user presses `s` then print this array.
        if key == ord('s'):
            
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)
            
            # Also save this array as penval.npy
            np.save('hsv_value',thearray)
            break
    
    cv2.destroyAllWindows()

# img_path = "test_images/straight_channel.png"
# # img_path = "test_images/curved_channel_2.png"
# image = cv2.imread(img_path)[64:600,189:922]
camCal = ImageRobot()
# # Sets up the realsense and gets us an image
pipeline, colorizer, align, depth_scale = setup_rs_camera()
time.sleep(1)
image, scaled_depth_image, aligned_depth_frame = get_rs_image(pipeline, align, depth_scale, use_depth=False)


pick_hsv(image)
# masked_bg = mask_colors(image, GREEN_HSV[0], GREEN_HSV[1])
# masked_cable = mask_colors(image, CABLE_HSV[0], CABLE_HSV[1])
masked_bg = mask(image, GREEN_HSV[0], GREEN_HSV[1])
masked_cable = mask(image, CABLE_HSV[0], CABLE_HSV[1])
masked_cable = post_process_mask(masked_cable)
# plt.imshow(masked_cable, cmap='gray')
# plt.show()
# breakpoint()
masked_channel = cv2.bitwise_and(1-masked_bg, 1-masked_cable)

# plt.imshow(masked_channel, cmap='gray')
# plt.show()
# breakpoint()
res = post_process_mask(masked_channel)
# res = post_process_mask(masked_channel, invert=True)
# res = post_process_mask(masked_channel, invert=False)
# plt.imshow(post_process_mask(masked_channel, invert=True), cmap='gray')
# plt.show()

# # breakpoint()
# # convert to binary by thresholding
# # ret, binary_map = cv2.threshold(masked_channel,127,255,0)

# # do connected components processing
# nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_channel, None, None, None, 8, cv2.CV_32S)

# #get CC_STAT_AREA component as stats[label, COLUMN] 
# areas = stats[1:,cv2.CC_STAT_AREA]

# result = np.zeros((labels.shape), np.uint8)

# for i in range(0, nlabels - 1):
#     if areas[i] >= 100:   #keep
#         result[labels == i + 1] = 255

# breakpoint()



'''sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# predictor.set_image("test_images/straight_channel.png")
# masks, _, _ = predictor.predict(<input_prompts>)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

breakpoint()

print(len(masks))
print(masks[0].keys())'''