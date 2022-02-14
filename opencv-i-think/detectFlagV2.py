import cv2 as cv
import numpy as np


#capture = cv.VideoCapture('opencv-i-think/videos/950_Video_Trim.mp4')
capture = cv.VideoCapture(0)

while True:
    isTrue, img = capture.read() 

    hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_color_bound = np.array([30,30,30])
    upper_color_bound = np.array([255, 255, 255])


    mask = cv.inRange(hsvImg, lower_color_bound, upper_color_bound)
    inverted_image = cv.bitwise_not(mask)

    img2 = inverted_image.copy()
    template = cv.imread('opencv-i-think/photos/matching2.jpg',0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = 'cv.TM_CCOEFF'
    img = img2.copy()
    method = eval(methods)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)

    imgH, imgW = img.shape
    calibZeroPixel = int(0.76 * imgH)

    rectTop = top_left[1]
    recBottom = bottom_right[1]



    detectedHeight = int((rectTop+recBottom)/2)


    cv.line(img, (0, calibZeroPixel), (imgW, calibZeroPixel), color=255, thickness=3)
    cv.line(img, (0, detectedHeight), (imgW, detectedHeight), color=255, thickness=3)
    print(imgW)
    cv.imshow('img',img)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
