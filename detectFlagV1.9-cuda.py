import cv2 as cv
import numpy as np


capture = cv.VideoCapture('videos/950_Video_Trim.mp4')
# capture = cv.VideoCapture(0)

print(cv.cuda.getCudaEnabledDeviceCount())

while True:
    isTrue, img = capture.read() 

    if not isTrue:
        break

    img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    template = cv.imread('photos/162.jpg', 0)

    g_template = cv.cuda_GpuMat()
    g_template.upload(template)

    w, h = template.shape[::-1]
    method = cv.TM_CCOEFF

    ret, img3 = cv.threshold(img2, 237, 255, 0)

    g_img = cv.cuda_GpuMat()
    g_img.upload(img3)

    matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, method)

     # Apply template Matching
    # res = cv.matchTemplate(img3, template, method)
    g_res = matcher.match(g_img, g_template)
    res = g_res.download()

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    imgH, imgW = img3.shape
    calibZeroPixel = int(0.76 * imgH)

    rectTop = top_left[1]
    recBottom = bottom_right[1]

    detectedHeight = int((rectTop+recBottom)/2)

    print(calibZeroPixel-detectedHeight)
    cv.imshow('img',img3)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
