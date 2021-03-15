import cv2
import numpy as np
from matplotlib import pyplot as plt


images= cv2.imread('source/test2.jpg')
img= cv2.fastNlMeansDenoisingColored(images,None,10,10,7,21)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)




imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray,(5,5),1)
imgcanny = cv2.Canny(imgblur,150,240)
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(imgcanny, kernel,iterations=1)
inpaint = cv2.inpaint(img,dilate,3,cv2.INPAINT_TELEA)
# cv2.imshow("canny",imgcanny)
# cv2.imshow("frame", img)
# cv2.imshow("frame", imggray)
# cv2.imshow("mask", mask)
# cv2.imshow("inpaint",inpaint)



plt.subplot(121)
plt.imshow(images)
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(inpaint)
plt.title('Result')
plt.xticks([]), plt.yticks([])
plt.show()
