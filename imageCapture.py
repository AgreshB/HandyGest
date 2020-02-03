# Mac Dependencies
# - brew install python
# - pip install numpy
# - brew tap homebrew/science
# - brew install opencv

import cv2
import tensorflow as tf 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

import backgroundSub

IMAGE_SIZE_X = 320
IMAGE_SIZE_Y = 120

model = tf.keras.models.load_model("model/model-6-thres.h5")

def drawRect(image):
    # Start coordinate, here (5, 5) 
    # represents the top left corner of rectangle 
    start_point = (0, 0) 
    
    # Ending coordinate, here (220, 220) 
    # represents the bottom right corner of rectangle 
    end_point = (600, 900) 
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

def predictImage(image):
    X=[]
    X.append(image)
    X = np.array(X,dtype="float16")
    X = X.reshape(1,120,320,1)
    prediction = model.predict(X)
    return np.argmax(prediction)

if __name__ == "__main__":   
    cap = cv2.VideoCapture(0)
    # initial weight for running average in background sub
    weight = 0.5
    frame_num = 0
    predictionGest = 0
    top , right , bottom , left = 0  , 720 , 0 , 600 
    thres = None
    while(True):
        cv2.namedWindow('Original Frame',cv2.WINDOW_NORMAL)
        cv2.namedWindow('grey',cv2.WINDOW_NORMAL)
        cv2.namedWindow('threshold',cv2.WINDOW_NORMAL)
        cv2.namedWindow('test',cv2.WINDOW_NORMAL)

        # reading each frame
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contor = frame.copy()
        # croppped image to only shopw hand part 
        grey = rgb[top:right,bottom:left]
        grey = cv2.resize(grey, (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        if frame_num < 30:
            backgroundSub.run_avg(grey,weight)
        if frame_num % 10 == 0: # small delay
            hand = backgroundSub.seperate(grey)
            if hand is not None :
                (thres , segmented) = hand 
                cv2.drawContours(contor,[segmented + (right,top)],-1,(0,0,255))
                cv2.imshow('threshold', thres)

        cv2.rectangle(contor, (top, left), (bottom, right), (0,255,0), 2)

        # _,thres = cv2.threshold(grey,80,150,cv2.THRESH_BINARY_INV)
        # thres = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # convert to gray scale 
        # grey = cv2.cvtColor(rgb1 , cv2.COLOR_BGR2GRAY)
        # draw rectangle around the cross 
        frame_num += 1
        rgb = drawRect(rgb)
        #resive the image for CNN 
        

        #predict and put text
        if thres is not None and frame_num %10 == 0:
            pred_thres = cv2.resize(thres,(320,120))
            #pred_thres = cv2.cvtColor(pred_thres,cv2.COLOR_BGR2GRAY)
            predictionGest = predictImage(pred_thres)

        #class_names = ["down", "palm", "l", "fist", "fist_moved", "thumb", "index", "ok", "palm_moved", "c"] 
        class_names = ["c","palm", "L", "fist","index", "ok"] 
        cv2.putText(rgb,class_names[predictionGest],(10,100),cv2.FONT_HERSHEY_SIMPLEX,6,(0,0,0),2)

        cv2.imshow('Original Frame', rgb)
        cv2.imshow('grey',grey)
        #cv2.imshow('threshold', thres)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite('capture.jpg', thres)
            break

    cap.release()
    cv2.destroyAllWindows()



