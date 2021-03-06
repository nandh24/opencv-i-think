import cv2 as cv

img = cv.imread('photos/Screen Shot 2022-01-28 at 6.26.47 PM.png')
#cv.imshow('photos/Screen Shot 2022-01-28 at 6.26.47 PM.png', img)

blur = cv.GaussianBlur(img, (9,9), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blur)
canny = cv.Canny(blur, 125, 175)
#cv.imshow('Edges', canny)

dilated = cv.dilate(canny, (7, 7), iterations = 3)
#cv.imshow('Dilated', dilated)

eroded = cv.erode(dilated, (7, 7), iterations=3)
#cv.imshow('Eroded', eroded)

resized = cv.resize(img, (1024, 432))
#cv.imshow('Resized', resized)

cropped = img[50:400, 50:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)