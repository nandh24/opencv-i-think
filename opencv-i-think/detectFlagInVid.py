import cv2 as cv 
from cv2 import Canny
import numpy as np

capture = cv.VideoCapture('videos/950_Video_Trim.mp4')

c = 0
               
while True:
    isTrue, img = capture.read() 

    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_color_bound = np.array([30,30,30])
    upper_color_bound = np.array([255, 255, 255])


    mask = cv.inRange(hsvImg, lower_color_bound, upper_color_bound)
    inverted_image = cv.bitwise_not(mask)

    cv.CascadeClassifier()

    contours, _ = cv.findContours(inverted_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image_cordinates = cv.detect(inverted_image)
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.0999999 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            if c==0:
                print("Found")
                for (x,y,w,h) in face_cordinates:
                    cv.rectangle(inverted_image,(x,y),(x+w,y+h),(0,255,0),2)
                    c = c+1
        else:
            c = 0



    res = cv.bitwise_and(img,img, mask= mask)



    cv.imshow('frame',img)
    cv.imshow('invmask', inverted_image)
    cv.imshow('res',res)


    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()