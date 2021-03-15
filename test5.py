import cv2
import numpy as np
from matplotlib import pyplot as plt


images= cv2.imread('source/test5.jpg')
img= cv2.fastNlMeansDenoisingColored(images,None,10,10,7,21)

def empty(a):
    pass
# # membut atrack bar untuk mendapatkan nilai lower upper, untuk mask
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",360,280)
# cv2.createTrackbar("Hue min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue max","TrackBars",179,179,empty)
# cv2.createTrackbar("Sat min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val min","TrackBars",0,255,empty)
# cv2.createTrackbar("Val max","TrackBars",255,255,empty)

while True:
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    ##-----> track bar untuk mendapatkan nilai lower dan upper <-----#
    #tulisannya harus sama
    # h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
    # h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
    # s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
    # s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
    # v_min = cv2.getTrackbarPos("Val min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val max", "TrackBars")

    #-->resotorasi gambar bagian kiri
    lower = np.array([25,0,0])
    upper = np.array([179,255,255])
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    mask = cv2.inRange(hsv, lower, upper)
    # contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     #-1 artiny menginput semua object untuk didraw conourt
    # cv2.drawContours(img, contours,-1,(0,255,0),1)
    # capmask =cv2.bitwise_and(img,img,dst=mask)

    # print(Contours)
    ret, thresh1 = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(thresh1, kernel,iterations=1)
    inpaint = cv2.inpaint(img,dilate,1,cv2.INPAINT_NS)
    # cv2.imshow("frame", img)
    #
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
    ##----->> to exit press q
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break