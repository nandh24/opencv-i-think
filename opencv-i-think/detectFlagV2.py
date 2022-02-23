import cv2 as cv
import numpy as np
import math
import threading
from networktables import NetworkTables

capture = cv.VideoCapture(0)

Connected_to_server = False

cond = threading.Condition()
notified = [False]
def connect():
    cond = threading.Condition()
    notified = [False]

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0] = True
            cond.notify()


    NetworkTables.initialize(server='roborio-2643-frc.local')
    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()

    return NetworkTables.getTable('vision-movement')


if Connected_to_server:
    table = connect()

Distance = table.getNumber("Distance", 0.0)
Degree = table.getNumber("Degree", 0.0)

while True:
    def mathLol(a):
        heightHub = 78.0
        startingInch = 127.5

        startingDistancePixels = ((424*startingInch)/heightHub)

        realDistance = ((heightHub*startingDistancePixels)/a)
        table.setNumber("Distance", realDistance)

        degree = (math.atan(startingDistancePixels/a)*180)/math.pi
        table.setNumber("Degree", degree)

        return (str(realDistance) + " Degree: " + str(degree))

    








    isTrue, img = capture.read()               

    img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    templates = [cv.imread('opencv-i-think/photos/68(2).jpg', 0), cv.imread('opencv-i-think/photos/68(3).jpg', 0), cv.imread('opencv-i-think/photos/74.5.jpg', 0), cv.imread('opencv-i-think/photos/84.jpg', 0), cv.imread('opencv-i-think/photos/68.jpg', 0), cv.imread('opencv-i-think/photos/115.jpg', 0), cv.imread('opencv-i-think/photos/162.jpg', 0), cv.imread('opencv-i-think/photos/209.jpg', 0), cv.imread('opencv-i-think/photos/256.jpg', 0), cv.imread('opencv-i-think/photos/303.jpg', 0)]
    
    
    ret, img3 = cv.threshold(img2, 253, 255, 0)
    cv.imshow("img", img3)
    cnt = 1
    

    highestMatch = 0

    for i in range(len(templates)):
        template = templates[i]         
        method = cv.TM_CCOEFF_NORMED 
            # Apply template Matching
        temp_res = cv.matchTemplate(img3, template, method) 
        if np.amax(temp_res) >= highestMatch:
            w, h = template.shape[::-1] 
            highestMatch = np.amax(temp_res)
            res = temp_res
        #print(str(cnt) + " " + str(np.amax(temp_res)))
        cnt = cnt+1


    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if (np.amax(highestMatch) < 0.8) and (np.amax(highestMatch) > 0.53):
        cv.rectangle(img3,top_left, bottom_right, 255, 2)

    imgH, imgW = img3.shape
    calibZeroPixel = int(0.877 * imgH)

    rectTop = top_left[1]
    rectBottom = bottom_right[1]



    detectedHeightY = int((rectTop+rectBottom)/2)
    imageXisZero = int(imgW/2)



    
    #detectedDistanceFromZero = 

    cv.line(img3, (0, calibZeroPixel), (imgW, calibZeroPixel), color=255, thickness=3)
    cv.line(img3, (0, detectedHeightY), (imgW, detectedHeightY), color=255, thickness=3)
    cv.line(img3, (imageXisZero, 0), (imageXisZero, imgH), color=255, thickness=3)

    print(np.amax(highestMatch))
    mathLol(calibZeroPixel-detectedHeightY)
    #print("detected distance pixel: ", calibZeroPixel-detectedHeightY)
    print("detected distance pixel: ", table.getNumber("Distance"))
    print("detected degree: ", table.getNumber("Degree"))
    #print("detected distance inches: ",  mathLol(calibZeroPixel-detectedHeightY))
    #print("detected distance: " + str((imgH -calibZeroPixel)*conversion))
    cv.imshow('img',img3)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

