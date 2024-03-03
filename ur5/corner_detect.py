import cv2
import numpy as np

# Load your mask image
image = cv2.imread('/home/gasket/robot_cable_insertion/ur5/images/full_trapezoid_assembly_cropped_good_resized_skeleton.png')

breakpoint()
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold if your mask isn't binary
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Find corners
corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=100)

# Convert the corners to integers
corners = np.int0(corners)

print(corners)


# Loop through each corner and draw them on the original image
for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# Display the image with corners
output_image_path = '/home/gasket/robot_cable_insertion/ur5/images/trapezoid_corners.png'
cv2.imwrite(output_image_path, image)
