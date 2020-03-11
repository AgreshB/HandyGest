import pyautogui as pgui 
import time
import subprocess

class process:

    def __init__(self,appName):
        self.isPause = False
        self.appRunning = False
        self.isMute = False
        self.app = appName
        self.previous = ""
        self.exceptionGest = ['L','ok']
        self.commandList = {"L": 'Volume Down', "ok":'Volume Up' , "fist":'Mute' , "palm" : 'Unmute' , "peace":'Pause' , "index":'Play'}
        self.switchCounter=0


    def openApp(self):
        self.appRunning = True
        subprocess.call(["/usr/bin/open", "/Applications/{}.app".format(self.app)])

    def printMenu(self):
        menu= ""
        for command in self.commandList.keys():
            menu +="{:<8}: {:<8}\n".format(command,self.commandList[command])
        return menu
    
    def runCommand(self, command):
        if command =='None':
            self.previous = command
            return
        elif command =='fist':
            # mute if fist
            #if self.isMute ==False:
            pgui.hotkey("option","command","down")
            #self.isMute = True
        elif command =='palm':
            # unmute if palm
            #if self.isMute ==True:
            pgui.hotkey("option","command","down")
            #self.isMute = False
        elif command == 'peace':
            # pause if palm 
            #if self.isPause == False:
            pgui.hotkey("space")
            #self.isPause = True
        elif command =='index':
            # play if index
            #if self.isPause == True :
            pgui.hotkey("space")
            #self.isPause = False
        elif command =='ok':
            # volume up if ok
            pgui.hotkey("up")
        elif command =='L':
            # volume down if index 
            pgui.hotkey("down")
        self.previous = command
    
    def checkAndRun(self,command):
        print("Command : {0} ,  Prev : {1}".format(command,self.previous))
        if command in self.exceptionGest:
            self.runCommand(command)
            return
        if command == self.previous:
            return
        else:
            if(self.switchCounter >3):
                self.runCommand(command)
                self.switchCounter =0
            else:
                self.switchCounter+=1
        



