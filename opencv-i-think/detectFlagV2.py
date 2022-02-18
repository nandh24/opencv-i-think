import cv2 as cv
import numpy as np


#capture = cv.VideoCapture('opencv-i-think/videos/950_Video_Trim.mp4')
capture = cv.VideoCapture(0)

while True:
    isTrue, img = capture.read()               

    # hsvImg = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower_color_bound = np.array([30,30,30])
    # upper_color_bound = np.array([255, 255, 255])


    # mask = cv.inRange(hsvImg, lower_color_bound, upper_color_bound)
    # inverted_image = cv.bitwise_not(mask)


    # img3 = cv.cvtColor(inverted_image, cv.COLOR_HSV2RGB)
    img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    templates = [cv.imread('opencv-i-think/photos/68.jpg', 0), cv.imread('opencv-i-think/photos/115.jpg', 0), cv.imread('opencv-i-think/photos/162.jpg', 0), cv.imread('opencv-i-think/photos/209.jpg', 0), cv.imread('opencv-i-think/photos/256.jpg', 0), cv.imread('opencv-i-think/photos/303.jpg', 0)]
    
    
    ret, img3 = cv.threshold(img2, 253, 255, 0)
    cv.imshow("img", img3)
    cnt = 1
    





    ####

    highestMatch = 0

    for i in range(len(templates)):
        template = templates[i]         
        method = cv.TM_CCOEFF_NORMED  
            # Apply template Matching
        temp_res = cv.matchTemplate(img3, template, method) # this is problem line pls help or else the continuing suffering will comence 
        if np.amax(temp_res) >= highestMatch:
            w, h = template.shape[::-1] 
            highestMatch = np.amax(temp_res)
            res = temp_res
        # else:
        #     res = cv.matchTemplate(img3, template, method)
        #print(str(cnt) + " " + str(np.amax(temp_res)))
        cnt = cnt+1

 
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    if (np.amax(highestMatch) < 0.7) and (np.amax(highestMatch) > 0.39):
        cv.rectangle(img3,top_left, bottom_right, 255, 2)

    imgH, imgW = img3.shape
    calibZeroPixel = int(0.76 * imgH)

    rectTop = top_left[1]
    rectBottom = bottom_right[1]



    detectedHeight = int((rectTop+rectBottom)/2)


    #cv.line(img, (0, calibZeroPixel), (imgW, calibZeroPixel), color=255, thickness=3)
    #cv.line(img, (0, detectedHeight), (imgW, detectedHeight), color=255, thickness=3)

    print(np.amax(highestMatch))
    cv.imshow('img',img3)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

