from turtle import distance
import cv2 as cv
import numpy as np
import math
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from networktables import NetworkTables

capture = cv.VideoCapture(0, cv.CAP_V4L2)
#capture = cv.VideoCapture(0, cv.CAP_V4L2)
'''Tested:
CAP_V4L2: Fastest
CAP_V4L: As fast as V4L2
CAP_VFW: Very fast and could be the fastest
'''
countCuda = cv.cuda.getCudaEnabledDeviceCount()
print(countCuda)
print(cv.getBuildInformation())
#The next line allows us to enable and disable CUDA
#Setting the CUDA string to '-1' disables CUDA
#Remove the following line to enable CUDA

#os.environ['CUDA_VISABLE_DEVICES'] = '-1'

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


if(Connected_to_server == False):
    table = connect()
    print("Connected!")
from networktables import NetworkTables


capture.set(cv.CAP_PROP_FPS, 2)
g_img  = cv.cuda_GpuMat()

templates = [cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68.jpg', 0), cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68(2).jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/74.5.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/84.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68(3).jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/115.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/162.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/209.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/256.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/303.jpg', 0)]

method = cv.TM_CCOEFF_NORMED
matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, method)

def calculation(a):
        if(a!=0):
            heightHub = 78.0  #numbers used for calibration
            startingInch = 139.75 #numbers used for calibration
            startingDistancePixels = ((204*startingInch)/heightHub)

            realDistance = ((heightHub*startingDistancePixels)/a)
            table.putNumber("Distance", realDistance)

            degree = (math.atan(a/startingDistancePixels)*180)/math.pi
            table.putNumber("Degree", degree)

            return (str(realDistance) + " Degree: " + str(degree))
        else:
            return ("Robot too close/far to determine")


while True:

    isTrue, img = capture.read()	

    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, img3 = cv.cuda.threshold(img2, 253, 255, 0)

    g_img.upload(img3)
    # highestMatch = 0

    futures = []

    clahe_img  = cv.cuda.createCLAHE(clipLimit = 5.0, tileGridSize = (8, 8))
    dst_img = clahe_img.apply(g_img, cv.cuda_Stream.Null())
    result_img = dst_img.download()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for template in templates:
            future = executor.submit(cv.matchTemplate, result_img, template, method)
            #future = executor.submit(matcher.match(result_img, template))
            futures.append(future)
    while not all([i.done() for i in futures]):
        pass
    temp_res = [future.result() for future in futures]
    maxes = [np.amax(i) for i in temp_res]
    maximum = max(maxes)
    index = maxes.index(maximum)
    w, h = templates[index].shape[::-1]
    res = temp_res[index]

    # for i in range(len(templates)):
    #     template = templates[i]         
    #     method = cv.TM_CCOEFF_NORMED 
    #         # Apply template Matching
    #     temp_res = cv.matchTemplate(img3, template, method) 
    #     if np.amax(temp_res) >= highestMatch:
    #         w, h = template.shape[::-1] 
    #         highestMatch = np.amax(temp_res)
    #         res = temp_res
    #     #print(str(cnt) + " " + str(np.amax(temp_res)))

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if (maximum < 0.8) and (maximum > 0.42):
        cv.rectangle(img3,top_left, bottom_right, 255, 2)
        #cv.line(img3, (0, detectedHeightY), (imgW, detectedHeightY), color=255, thickness=3)

        imgH, imgW = img3.shape
        calibZeroPixel = int(0.89 * imgH)

        rectTop = top_left[1]
        rectBottom = bottom_right[1]


        rectLeft = top_left[0]
        rectRight = bottom_right[0]
        distanceFromVerticalLine = (rectLeft+rectRight)*0.5

        detectedHeightY = int((rectTop+rectBottom)/2)
        imageXisZero = int(imgW/2)



        turretPixels = (-distanceFromVerticalLine+imageXisZero)

        #calculation(a=calibZeroPixel-detectedHeightY, turretTurn=turretPixels)
        #detectedDistanceFromZero = 

        cv.line(img3, (0, calibZeroPixel), (imgW, calibZeroPixel), color=255, thickness=3)
        cv.line(img3, (imageXisZero, 0), (imageXisZero, imgH), color=255, thickness=3)

        print(maximum)
        print("detected distance pixel: ", calibZeroPixel-detectedHeightY)
        print("detected distance inches: ",  calculation(turretPixels))    #print("detected distance: " + str((imgH -calibZeroPixel)*conversion))
    else:
        print("not detecting")
    cv.imshow('img',img3)
    if cv.waitKey(1) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
