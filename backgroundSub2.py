import cv2 
import numpy as np 
import imutils 
import backgroundSub
import tensorflow as tf 
from execute import process

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
        self.process = process('VLC')

        self.startPredict = False
        self.model = tf.keras.models.load_model("model/custModel-7.h5")
        self.class_names = ["palm", "fist","ok","peace","L","index","None"] 
        self.startExecute = False

        self.frame_number = 0
        self.commandList=[]

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
        kernel2 = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=3)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2, iterations=3)
        return cv2.bitwise_and(frame, frame, mask=fgmask)
    
    def predictImage(self,image):
        resImage= cv2.resize(image,(200,200))
        X=[]
        X.append(resImage)
        X = np.array(X,dtype="float16")
        X = X.reshape(1,200,200,1)
        prediction = self.model.predict(X)
        return np.argmax(prediction)
    
    def checkAndRun(self,command):
        if(self.frame_number % 5 == 0):
            commandToRun = max(set(self.commandList) , key= self.commandList.count)
            self.process.runCommand(commandToRun)
            self.frame_number = 0
            self.command =[]
        else:
            self.commandList.append(command)
        
    def subtraction(self): 
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setupFrame(frame_width, frame_height)
        cv2.namedWindow('Original Frame',cv2.WINDOW_NORMAL)
        cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
        cv2.namedWindow("threshold",cv2.WINDOW_NORMAL)
        cv2.moveWindow('Original Frame' ,6,27)
        cv2.moveWindow('cropped' ,6,327)
        cv2.moveWindow('threshold' ,6,550)
        while(True):
        
            # reading each frame
            ret, frame = cap.read()
            cv2.rectangle(frame, (self.x0, self.y0), (self.x0 + self.width - 1, self.y0 + self.height - 1), self.roiColor, 2)
            roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width,:]
            self.frame_number +=1

            if(self.isBgCaptured):
                bgSubMask = self.bgSubMasking(roi)
                grey = cv2.cvtColor(bgSubMask,cv2.COLOR_BGR2GRAY)
                thresholded = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)[1]
                cv2.imshow("threshold",thresholded)

            if(self.startPredict):
                resImage = cv2.resize(thresholded,(200,200))
                gesture =self.class_names[self.predictImage(thresholded)]
                cv2.putText(frame,gesture,(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),cv2.LINE_AA)
                if (self.startExecute):
                    self.checkAndRun(gesture)
                    

            cv2.imshow('Original Frame',frame)
            cv2.imshow('cropped',roi)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('b'):
                self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, self.bgSubThreshold,detectShadows=False)
                self.isBgCaptured = True
            elif k ==ord('q'):
                cv2.imwrite('capture.jpg', frame)
                break
            elif k ==ord('p'):
                self.startPredict = not self.startPredict
            elif k ==ord('e'):
                self.frame_number = 0
                self.startExecute = not self.startExecute
            elif k == ord('w'):
                # moving roi up 
                self.y0 -= 5
            elif k ==ord('s'):
                self.y0 += 5
            elif k == ord('d'):
                self.x0 += 5
            elif k == ord('a'):
                #moving roi to left
                self.x0 -= 5
        cap.release()
        cv2.destroyAllWindows()

    def printMenu(self):
        print("-------------------------------------------")
        print("               HANDY GEST                  ")
        print("-------------------------------------------")
        print("OPTIONS:")
        print("To Caputure BAckground (or re-calib): Press b")
        print("To move the ROI                     : Use w a s d ")
        print("To TOGGLE PREDICTION                : Press p")
        print("To TOGGLE EXECUTION                 : Press e")

if __name__ == "__main__":
    detect = backgroundSub2()
    detect.printMenu()
    detect.subtraction()