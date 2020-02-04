import cv2 
import numpy as np 
import imutils 
import shutil
import os

#set to required directory
#os.getcwd()
path = "/Users/Radha/Documents/AGRESH/Manchester/study/Year3/project/dataset"
mainFold = "data"
print ("The current working directory is %s" % path)

#creating main folder
def setup():
    global path
    global mainFold
    path = os.path.join(path,mainFold)
    '''
    only need to run once
    os.mkdir(path)
    for i in range(0,5):
        foldernumber = str(i)
        subpath = os.path.join(path,foldernumber)
        os.mkdir(subpath)
    '''

def write(image,type,frame,folder):
    global path
    fpath = os.path.join(path,str(folder))
    pictureName = "frame_{0}_{1}_{2}.png".format(folder,type,frame)
    picPath = os.path.join(fpath,pictureName)
    print(picPath)
    cv2.imwrite(picPath,image)

def cleanUp():
    global path
    shutil.rmtree(path)