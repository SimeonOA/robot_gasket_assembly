import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# img_path = 'imgs/straight10_cropped.png'
img_path = 'imgs/straight10.png'
# img_path = 'imgs/trapezoid6.png'
img = cv.imread(img_path) #, cv.IMREAD_GRAYSCALE
# print(img.shape)
# img = np.load('imgs/straight10_cropped.npy')
# print(img.shape)

dst = img
# dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
dst = cv.GaussianBlur(img,(5,5),0)
# # kernel = np.ones((5,5),np.float32)/25
# # dst = cv.filter2D(img,-1,kernel)
# # dst = cv.bilateralFilter(img,9,75,75)
plt.imshow(dst, cmap='gray')
plt.show()

# laplacian = cv.Laplacian(img,cv.CV_64F)
# img = dst
# sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
# plt.imshow(sobelx)
# plt.show()

# sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
# plt.imshow(sobely)
# plt.show()

edges = cv.Canny(dst,0,255)

# plt.imshow(255-edges, cmap='gray')
# plt.show()

# print(edges.min(), edges.max())
# print(np.unique(edges))

# edges = cv.GaussianBlur(edges,(3,3),0)
# edges = cv.bilateralFilter(edges,9,75,75)
plt.imshow(edges, cmap='gray')
plt.show()

# # Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
# kernel = None
edges = cv.dilate(edges, kernel)
plt.imshow(edges, cmap='gray')
plt.show()

# for x in cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]:
#     print(x.shape)

# print(cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0].shape)

# cnt = sorted(cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)
# for x in cnt:
#     print(x.shape)

cnt = sorted(cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2], key=cv.contourArea)[-3]
mask = np.zeros_like(img, dtype=np.uint8)
masked = cv.drawContours(mask, [cnt],-1, 255, -1)

# print(np.unique(mask), mask.shape, mask.dtype)
# print(img.shape)

# mask = 255-mask
# masked = 255-masked

plt.imshow(masked, cmap='gray')
plt.show()

# print(mask.sum(axis=-1).shape)

mask = mask.sum(axis=-1).astype('uint8')

dst = cv.bitwise_not(cv.bitwise_and(img, img, mask=mask))
segmented = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
plt.imshow(segmented)
plt.show()

# masked = cv.bitwise_and(img, img, mask=edges)
# plt.imshow(masked)
# plt.show()

# dst = cv.inpaint(masked,255-edges,3,cv.INPAINT_TELEA)
# plt.imshow(dst)
# plt.show()

# plt.imsave('straight10_cropped_masked_inpaint.png', dst)

# cv.cvtColor(255-edges,cv.COLOR_GRAY2RGB)

# plt.imshow(img)
# plt.show()

# mag_sobel = np.absolute(sobely)
# for i in range(mag_sobel.shape[0]):
#     for j in range(mag_sobel.shape[1]):
#         if mag_sobel[i,j] >= 255:
#             mag_sobel[i,j]=255
#         else:
#             mag_sobel[i,j]=0

# # kernel = np.ones((5,5),np.float32)/25
# # mag_sobel = cv.filter2D(mag_sobel,-1,kernel)
# # mag_sobel = cv.bilateralFilter(mag_sobel.astype('float32'),9,75,75)
# plt.imshow(mag_sobel, cmap='gray')
# plt.show()