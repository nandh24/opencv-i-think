import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype = "uint8")
#cv.imshow('Blank', blank)

#blank[0:100, 0:200] = 0, 255, 0
#cv.imshow('Green', blank)

#cv.rectangle(blank, (0, 0), (250, 250), (255, 0, 0), cv.FILLED)
#cv.imshow('Rectangle', blank)

#cv.circle(blank, (250, 250), 40, (0, 255, 255), thickness= -1)
#cv.line(blank, (0, 0), (500, 500), (255, 255, 255), thickness=3)
#cv.imshow("Line", blank)

cv.putText(blank, "Hello", (250, 250), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), thickness=1)
cv.imshow('Text', blank)
cv.waitKey(0)