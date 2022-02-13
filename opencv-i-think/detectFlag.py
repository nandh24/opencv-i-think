import cv2 as cv
from cv2 import Canny
import numpy as np

img = cv.imread('opencv-i-think/photos/9502.jpg')

#img = cv.GaussianBlur(preimg, (9,9), cv.BORDER_DEFAULT)
#cv.imshow('first', img)

#resized = cv.resize(img, (1080, 720))
#cv.imshow('resized original', resized)

hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_color_bound = np.array([30,30,30])
upper_color_bound = np.array([255, 255, 255])


mask = cv.inRange(hsvImg, lower_color_bound, upper_color_bound)
inverted_image = cv.bitwise_not(mask)
#ny = cv.Canny(inverted_image, 125, 175)

contours, _ = cv.findContours(inverted_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    approx = cv.approxPolyDP(contour, 0.0999999 * cv.arcLength(contour, True), True)
    if len(approx) == 4:
        print("Found")


res = cv.bitwise_and(img,img, mask= mask)



cv.imshow('frame',img)
#cv.imshow('mask',mask)
cv.imshow('invmask', inverted_image)
#cv.imshow('canny', canny)
cv.imshow('res',res)




#blur = cv.GaussianBlur(resized, (9,9), cv.BORDER_DEFAULT)
#cv.imshow('blur', blur)

#canny = cv.Canny(hsvImage, 125, 175)
#cv.imshow('hsv', hsvImg)


cv.waitKey(0)
