import cv2 as cv

img = cv.imread('photos/Screen Shot 2022-01-28 at 6.26.47 PM.png')

cv.imshow('Picture', img)

cv.waitKey(0)
