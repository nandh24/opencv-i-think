import cv2 as cv

def rescaleFrame(frame, scale = 0.5):

    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#img = cv.imread('photos/Screen Shot 2022-01-28 at 6.26.47 PM.png')
#cv.imshow('Photo', img)

#picture = rescaleFrame(img)

#cv.imshow('Resized', picture)

#cv.waitKey(0)

capture = cv.VideoCapture('https://media.tenor.co/videos/5bc3c822e7d615de68163bc14ad0d0b4/mp4')
while True:
    #ret, frame = capture.read()
    ret, frame = capture.read() 

    frame_resize = rescaleFrame(frame)
    if ret:
        cv.imshow("Image", frame)
    else:
       print('no video')
       capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()



