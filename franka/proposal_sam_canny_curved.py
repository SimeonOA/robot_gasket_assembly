import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
from matplotlib.path import Path
import json

img_path = 'imgs/curved7.png'
img = cv.imread(img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]

blur_radius, sigma = 5, 0
dst = img
dst = cv.GaussianBlur(dst,(blur_radius, blur_radius),sigma)
plt.imshow(dst, cmap='gray')
plt.show()

edges = cv.Canny(dst,100,200)
plt.imshow(edges, cmap='gray')
plt.show()
kernel = np.ones((5, 5), np.uint8)
edges = cv.dilate(edges, kernel)
plt.imshow(edges, cmap='gray')
plt.show()


dst = cv.bitwise_and(dst, dst, mask=edges)
output = np.zeros_like(dst).sum(axis=-1) + 255
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        for k in range(dst.shape[2]):
            if dst[i,j,k] <= 100:
                output[i,j] = 0
                break

plt.imshow(output, cmap='gray')
plt.show()

cnt = sorted(cv.findContours(output.astype('uint8'), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)[-2]
mask = np.zeros_like(img, dtype=np.uint8)
masked = cv.drawContours(mask, [cnt],-1, 255, -1)
mask = mask.sum(axis=-1).astype('uint8')
kernel = np.ones((5, 5), np.uint8)
mask = cv.dilate(mask, kernel)
plt.imshow(mask, cmap='gray')
plt.show()

file_name = img_path.split('.png')[0] + '_mask.png'
plt.imsave(file_name, mask)