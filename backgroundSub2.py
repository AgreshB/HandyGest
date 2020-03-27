import cv2 
import numpy as np 
import imutils 
import backgroundSub
import tensorflow as tf 
import PySimpleGUI as sg
from execute import process


class backgroundSub2:

    def __init__(self,name):
        self.isBgCaptured = False
        # Background subtractor learning rate
        self.bgSubtractorLr = 0
        self.bgSubThreshold = 35
        self.isHandHistCreated = False
        self.isRunning = False

        self.x0 = 0
        self.y0 = 0
        self.height= 0
        self.width = 0
        self.roiColor = (255, 0, 0)
        self.process = process(name)
        self.currentApp = name

        self.xs = [6.0/20.0, 9.0/20.0, 12.0/20.0]
        self.ys = [9.0/20.0, 10.0/20.0, 11.0/20.0]

        self.startPredict = False
        self.model = tf.keras.models.load_model("model/custModelOrgRes-9.h5")
        self.class_names = ["palm", "fist","ok","peace","L","index","None"] 
        self.startExecute = False

        self.frame_number = 0
        self.commandList=[]
        self.showStats = False
        self.showCommands = False


    def setupFrame(self, frame_width, frame_height):
        """self.x0 and self.y0 are top left corner coordinates
        self.width and self.height are the width and height the ROI
        """
        x, y = 0.1, 0.2
        self.x0 = int(frame_width*x)
        self.y0 = int(frame_height*y)
        self.width = 400
        self.height = 400
      
    def histMasking(self, frame, handHist):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], handHist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)
        # thresh = cv2.erode(thresh, kernel, iterations=5)

        thresh = cv2.merge((thresh, thresh, thresh))
        return cv2.bitwise_and(frame, thresh)

    def createHandHistogram(self, frame):
        rows, cols, _ = frame.shape
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros([180, 20, 3], dtype=hsvFrame.dtype)
        i = 0
        for x in self.xs:
            for y in self.ys:
                x0, y0 = int(x*rows), int(y*cols)
                roi[i*20:i*20 + 20, :, :] = hsvFrame[x0:x0 + 20, y0:y0 + 20, :]

                i += 1
        handHist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(handHist, handHist, 0, 255, cv2.NORM_MINMAX)

    def detectHand(self,frame,handHist):
        roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width,:]
        # Color masking
        histMask = self.histMasking(roi, handHist)
        # Background substraction
        bgSubMask = self.bgSubMasking(roi)

        if(self.showStats):
            cv2.imshow("bgMask", bgSubMask)
            cv2.imshow("histMask", histMask)
        
        # final mask take bitwise and of both
        mask = cv2.bitwise_and(histMask, bgSubMask)
        return mask

    def bgSubMasking(self, frame):
        fgmask = self.bgSubtractor.apply(frame, learningRate=self.bgSubtractorLr)
        kernel = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((3,3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2, iterations=3)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.bitwise_and(frame, frame, mask=fgmask)
    
    def predictImage(self,image):
        resImage= cv2.resize(image,(300,300))
        X=[]
        X.append(resImage)
        X = np.array(X,dtype="float16")
        X = X.reshape(1,300,300,1)
        prediction = self.model.predict(X)
        return np.argmax(prediction)
    
    def runCommand(self):
        commandToRun = max(set(self.commandList) , key= self.commandList.count)
        if(self.startExecute):
            self.process.checkAndRun(commandToRun)
        self.frame_number = 0
        self.commandList =[]
        return commandToRun

    def drawRect(self, frame):
        roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width,:]
        rows, cols, _ = roi.shape
        frame1 = frame.copy()
        for x in self.xs:
            for y in self.ys:
                x0, y0 = int(x*rows), int(y*cols)
                cv2.rectangle(frame1, (self.y0+ y0, self.x0+ x0), (self.y0+ y0 + 20, self.x0+ x0 + 20), (0, 0, 0), 1)
        return frame1 
    
    def subtraction(self): 
        self.isRunning = True
        check=0
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setupFrame(frame_width, frame_height)

        #creating a black image for options window 
        options = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        helpWin = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

        cv2.namedWindow("Status",cv2.WINDOW_NORMAL)
        cv2.namedWindow('Original Frame',cv2.WINDOW_NORMAL)
        if(self.showStats):
            cv2.namedWindow("threshold",cv2.WINDOW_NORMAL)
            cv2.namedWindow("histMask",cv2.WINDOW_NORMAL)
            cv2.namedWindow("bgMask",cv2.WINDOW_NORMAL)
        
        cv2.moveWindow('Original Frame' ,6,27)
        cv2.moveWindow('threshold' ,6,550)
        gesture ='None'
        handHist = None
        while(True):
            # reading each frame
            options = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
            ret, frame = cap.read()
            cv2.rectangle(frame, (self.x0, self.y0), (self.x0 + self.width - 1, self.y0 + self.height - 1), self.roiColor, 2)
            roi = frame[self.y0:self.y0 + self.height, self.x0:self.x0 + self.width,:]
            self.frame_number +=1


            if(self.isHandHistCreated and self.isBgCaptured):
                handMask = self.detectHand(frame,handHist)
                grey = cv2.cvtColor(handMask,cv2.COLOR_BGR2GRAY)
                thresholded = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)[1]
                
            elif not self.isHandHistCreated:
                frame = self.drawRect(frame)
            
            if(self.startPredict):
                resImage = cv2.resize(thresholded,(300,300))
                gesture1 =self.class_names[self.predictImage(thresholded)]
                self.commandList.append(gesture1)
                if (self.frame_number % 5 == 0):
                    gesture = self.runCommand()
                cv2.putText(frame,gesture,(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),cv2.LINE_AA)

            #print options window  
            #cv2.putText(options,self.printMenu(),(5,5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.putText(options,"Background Capture :{}".format(self.isBgCaptured),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.putText(options,"Hist Capture       :{}".format(self.isHandHistCreated),(10,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.putText(options,"Execution started  :{0} ({1})".format(self.startExecute,self.process.app),(10,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.putText(options,"Prediction started :{}".format(self.startPredict),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

            cv2.imshow('Original Frame',frame)
            cv2.imshow('Status',options)
        
            if(self.showStats):
                cv2.imshow("threshold",thresholded)
            elif(check!=0):
                cv2.destroyWindow("threshold")
                cv2.destroyWindow("histMask")
                cv2.destroyWindow("bgMask")
                check = 0
            
            if(self.showCommands):
                helpWin = self.printHelp(helpWin)
                cv2.imshow('Help',helpWin)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('b'):
                self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, self.bgSubThreshold,detectShadows=False)
                self.isBgCaptured = True
            elif k== ord('z'):
                self.isHandHistCreated = not self.isHandHistCreated
                handHist = self.createHandHistogram(roi)
            elif k ==ord('q'):
                break
            elif k ==ord('p'):
                self.frame_number= 0
                self.startPredict = not self.startPredict
            elif k ==ord('e'):
                self.frame_number = 0
                self.startExecute = not self.startExecute
            elif k ==ord('r'):
                self.printStatus()
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
            elif k ==ord('o'):
                self.showStats = not self.showStats
                check =1
            elif k ==ord('h'):
                self.helper()
                # self.showCommands = not self.showCommands
                # if(not self.showCommands):
                #     cv2.destroyWindow('Help')
        cap.release()
        cv2.destroyAllWindows()
        self.isRunning=False

    def printMenu(self):
        menu =[]
        menu.append("To Caputure BAckground (or re-calib): Press b\n")
        menu.append("To move the ROI                     : Use w a s d\n")
        menu.append("To TOGGLE PREDICTION                : Press p\n")
        menu.append("To TOGGLE EXECUTION                 : Press e\n")
        menu.append("To Capture HandHistogram            : Press z\n")
        menu.append("To Open Status Window               : Press r\n")
        menu.append("To Toggle Stats                     : Press o\n")
        menu.append("\nCOMMAND LIST \n")
        for pr in self.process.printMenu():
            menu.append(pr)
        return menu
        
    def run(self,app):
        if app != self.currentApp:
            self.process.resetApp(app)
            self.currentApp = app
        if(not self.isRunning):
            self.subtraction()

    def helper(self):
        layoutHelp = []
        layoutHelp.append([sg.Text('OPTIONS', size=(20, 1), font=("Helvetica", 25))])
        menu = self.printMenu()
        for string in menu:
            layoutHelp.append([sg.Text(string)])
        layoutHelp.append([sg.Quit()])
        helpW = sg.Window('Help', layoutHelp)
        event, value = helpW.read()
        if event in ('Quit', None):
            helpW.close()
    
    def printStatus(self):
        statusLayout = [
            [sg.Text('STAUTUS', size=(20, 1), font=("Helvetica", 25))],
            [sg.Text('Background Capture :{}'.format(self.isBgCaptured), size=(30, 1), font=("Helvetica", 25))],
            [sg.Text('Hist Capture       :{}'.format(self.isHandHistCreated))],
            [sg.Text('Execution started  :{0} ({1})'.format(self.startExecute,self.process.app))],
            [sg.Text('Prediction started :{}'.format(self.startPredict))],
            [sg.Radio('VLC', "RADIO1", default=True,key = 'VLC'), sg.Radio('Chrome', "RADIO1",key = 'Chrome')],
            [sg.Quit()]
       ]
        statusWindow = sg.Window('Status and options',statusLayout)
        event , value = statusWindow.read()
        apps= ['VLC','Chrome']
        currentApp = self.currentApp
        if value[self.currentApp]==False:
            apps.remove(self.currentApp)
            currentApp = apps[0]
        statusWindow.close()
        self.run(currentApp)
