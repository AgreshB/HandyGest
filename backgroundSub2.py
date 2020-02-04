import cv2 
import numpy as np 
import imutils 
import backgroundSub
import tensorflow as tf 

class backgroundSub2:

    def __init__(self):
        self.isBgCaptured = False
        # Background subtractor learning rate
        self.bgSubtractorLr = 0
        self.bgSubThreshold = 35

        self.x0 = 0
        self.y0 = 0
        self.height= 0
        self.width = 0
        self.roiColor = (255, 0, 0)

        self.oldBgCapture = False

        self.startPredict = False
        self.model = tf.keras.models.load_model("model/custModel-6-V2.h5")
        self.class_names = ["palm", "fist","ok","peace","L","index"] 

    def setupFrame(self, frame_width, frame_height):
        """self.x0 and self.y0 are top left corner coordinates
        self.width and self.height are the width and height the ROI
        """
        x, y = 0.1, 0.2
        self.x0 = int(frame_width*x)
        self.y0 = int(frame_height*y)
        self.width = 400
        self.height = 400
    
    def bgSubMasking(self, frame):
        fgmask = self.bgSubtractor.apply(frame, learningRate=self.bgSubtractorLr)
        kernel = np.ones((4, 4), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.bitwise_and(frame, frame, mask=fgmask)
    
    def predictImage(self,image):
        resImage= cv2.resize(image,(200,200))
        X=[]
        X.append(resImage)
        X = np.array(X,dtype="float16")
        X = X.reshape(1,200,200,1)
        prediction = self.model.predict(X)
        return np.argmax(prediction)

    def subtraction(self): 
        cap = cv2.VideoCapture(0)
        frame_number =0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setupFrame(frame_width, frame_height)
        while(True):
            cv2.namedWindow('Original Frame',cv2.WINDOW_NORMAL)
            cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)

            # reading each frame
            ret, frame = cap.read()
            cv2.rectangle(frame, (self.x0, self.y0), (self.x0 + self.width - 1, self.y0 + self.height - 1), self.roiColor, 2)
            roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width,:]

            if(self.isBgCaptured):
                bgSubMask = self.bgSubMasking(roi)
                grey = cv2.cvtColor(bgSubMask,cv2.COLOR_BGR2GRAY)
                thresholded = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)[1]
                cv2.imshow("threshold",thresholded)

            if(self.startPredict):
                resImage = cv2.resize(thresholded,(200,200))
                gesture =self.class_names[self.predictImage(thresholded)]
                cv2.putText(frame,gesture,(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),cv2.LINE_AA)
            
            cv2.imshow('Original Frame',frame)
            cv2.imshow('cropped',roi)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('b'):
                self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, self.bgSubThreshold,detectShadows=False)
                self.isBgCaptured = True
            elif k == ord('w'):
                self.oldBgCapture = True
                frame_number=0
            elif k ==ord('q'):
                break
            elif k ==ord('p'):
                self.startPredict = not self.startPredict
            elif k == ord('y'):
                # moving roi up 
                self.y0 -= 5
            elif k == ord('x'):
                #moving roi to left
                self.x0 -= 5
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect = backgroundSub2()
    detect.subtraction()