import cv2 as cv
import numpy as np
import math
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from networktables import NetworkTables

#capture = cv.VideoCapture(0, cv.CAP_V4L2)
capture = cv.cuda.VideoCapture(0, cv.CAP_V4L2)
countCuda = cv.cuda.getCudaEnabledDeviceCount()
print(countCuda)
print(cv.getBuildInformation())
#The next line allows us to enable and disable CUDA
#Setting the CUDA string to '-1' disables CUDA
#Remove the following line to enable CUDA

#os.environ['CUDA_VISABLE_DEVICES'] = '-1'

#Connected_to_server = False

#cond = threading.Condition()
#notified = [False]
#def connect():
    #cond = threading.Condition()
    #notified = [False]

    #def connectionListener(connected, info):
        #print(info, '; Connected=%s' % connected)
        #with cond:
            #notified[0] = True
            #cond.notify()
            
    #NetworkTables.initialize(server='roborio-2643-frc.local')
    #NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

    #with cond:
       #print("Waiting")
       #if not notified[0]:
           #cond.wait()

    #return NetworkTables.getTable('vision-movement')

#if(Connected_to_server == False):
    #table = connect()
    #print("Connected!")

capture.set(cv.CAP_PROP_FPS, 5)

templates = [cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68.jpg', 0), cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68(2).jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/74.5.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/84.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/68(3).jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/115.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/162.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/209.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/256.jpg', 0), 
cv.imread('/home/team2643/opencv-i-think/opencv-i-think/photos/303.jpg', 0)]

g_templates = []

def mathLol(a):
        if(a!=0):
            heightHub = 78.0
            startingInch = 139.75
            startingDistancePixels = ((204*startingInch)/heightHub)

            realDistance = ((heightHub*startingDistancePixels)/a)
            #table.putNumber("Distance", realDistance)

            degree = (math.atan(startingDistancePixels/a)*180)/math.pi
            #table.putNumber("Degree", degree)

            return (str(realDistance) + " Degree: " + str(degree))
        else:
            return ("Robot too close/far to determine")

for template in templates:
	g_template = cv.cuda_GpuMat()
	g_template.upload(template)	
	g_templates.append(g_template)

method = cv.TM_CCOEFF_NORMED
matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, method)

g_img = cv.cuda_GpuMat()

while True:
    isTrue, img = capture.read()		

    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, img3 = cv.threshold(img2, 249, 255, 0)
    
    g_img.upload(img3)

    # highestMatch = 0

    highestMatch = 0

    for i in range(len(g_templates)): 
         g_template = g_templates[i]     
        # Apply template Matching
         g_temp_res = matcher.match(g_img, g_template)
         temp_res = g_temp_res.download()
        #  res_min = 0
        #  res_max = 0
        #  cv.cuda.minMax(src=g_temp_res, minVal=res_min, maxVal=res_max)
        #  print(res_max)
         currentMatch = np.amax(temp_res)
         if currentMatch >= highestMatch:
             w, h = templates[i].shape[::-1] 
             highestMatch = currentMatch
             res = temp_res
         #print(str(cnt) + " " + str(np.amax(temp_res)))

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if (highestMatch < 0.8) and (highestMatch > 0.42):
        cv.rectangle(img3,top_left, bottom_right, 255, 2)
        #cv.line(img3, (0, detectedHeightY), (imgW, detectedHeightY), color=255, thickness=3)

    imgH, imgW = img3.shape
    calibZeroPixel = int(0.89 * imgH)

    rectTop = top_left[1]
    rectBottom = bottom_right[1]

    detectedHeightY = int((rectTop+rectBottom)/2)
    imageXisZero = int(imgW/2)
    
    #detectedDistanceFromZero = 

    cv.line(img3, (0, calibZeroPixel), (imgW, calibZeroPixel), color=255, thickness=3)
    cv.line(img3, (imageXisZero, 0), (imageXisZero, imgH), color=255, thickness=3)

    print(highestMatch)
    print("detected distance pixel: ", calibZeroPixel-detectedHeightY)
    print("detected distance inches: ",  mathLol(calibZeroPixel-detectedHeightY))    #print("detected distance: " + str((imgH -calibZeroPixel)*conversion))
    cv.imshow('img',img3)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
