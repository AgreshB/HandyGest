import cv2 
import numpy as np 
import imutils 

# global background variable 
bg = None
threshold = 25

def run_avg(image , weigth):
    global bg
    if bg is None :
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image,bg,weigth)

def seperate(image):
    global bg
    global threshold
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thres = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thres = cv2.GaussianBlur(thres,(5,5),0)

    # get the contours in the thresholded image
    #(cont, _) = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    # if len(cont) == 0:
    #     return
    # else:
    #segmented = max(cont, key=cv2.contourArea)
    return (thres)