#Adpated from https://stackoverflow.com/questions/57469394/opencv-choosing-hsv-thresholds-for-color-filtering
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
# from depth_sensing import get_rgb_get_depth

def default(x):
    pass

# Create a window
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,default) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,default)
cv2.createTrackbar('VMin','image',0,255,default)
cv2.createTrackbar('HMax','image',0,179,default)
cv2.createTrackbar('SMax','image',0,255,default)
cv2.createTrackbar('VMax','image',0,255,default)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# depth_image, rgb_image = get_rgb_get_depth()
# rgb_image = rgb_image[:,:,:3]


# img = cv2.imread('/home/lawrence/Documents/bag/baggingbot/data_collection/Figure_1.png')
# img = cv2.imread('problem_segments_im/color29.png')
# img = cv2.imread(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_rope_error222.png")
# img = cv2.imread(f"/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask_channel_error.png")

# img = cv2.imread('/home/lawrence/Documents/bag/baggingbot/data_collection/data/raw_data/test/image_regular_179.png')
img_path = "/home/gasket/robot_cable_insertion/ur5/images/curved_channel_right_direction_cropped.png"
# img_path = '/home/r2d2/Robot-Cable-Insertion_TRI-Demo/problem_segments_im/rope_mask1.png'
img = cv2.imread(img_path)
output = img
waitTime = 33

while True:
    image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_cable = image.copy()[1:2, 1:2]
    image_channel = image.copy()[1:2, 1:2]
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(image_cable)
    # plt.show()
    # plt.imshow(image_channel)  
    # plt.show()
    print("cable_mean", np.mean(image_cable))
    print("channel_mean", np.mean(image_channel))
    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()