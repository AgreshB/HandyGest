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
        self.exceptionGest = {"VLC":['L','ok'] , "Chrome":['index','peace']}
        self.commandList = {"L": 'Volume Down', "ok":'Volume Up' , "fist":'Mute' , "palm" : 'Unmute' , "peace":'Pause' , "index":'Play'}
        self.switchCounter=0


    def openApp(self):
        self.appRunning = True
        subprocess.call(["/usr/bin/open", "/Applications/{}.app".format(self.app)])

    def printMenu(self):
        menu=[]
        for command in self.commandList.keys():
            menu.append("{:<8}: {:<8}\n".format(command,self.commandList[command]))
        return menu
    
    def runCommand(self, command):
        if command =='None':
                self.previous = command
                return
        if self.app == "VLC":
            if command =='fist':
                # mute if fist
                pgui.hotkey("option","command","down")
            elif command =='palm':
                # unmute if palm
                pgui.hotkey("option","command","down")
            elif command == 'peace':
                # pause if palm 
                pgui.hotkey("space")
            elif command =='index':
                # play if index
                pgui.hotkey("space")
            elif command =='ok':
                # volume up if ok
                pgui.hotkey("up")
            elif command =='L':
                # volume down if L 
                pgui.hotkey("down")
        else:
            if command == 'index':
                #scroll down if index
                pgui.hotkey("down")
            elif command == 'peace':
                #scroll up if peace 
                pgui.hotkey("up")
            elif command == 'ok':
                # refresh if ok
                pgui.hotkey("command","r")
            elif command == 'fist':
                # back if fist
                pgui.hotkey("command","left")
            elif command == 'palm':
                #forward if palm
                pgui.hotkey("command","right")
            elif command == 'L':
                #toggle full screen
                pgui.hotkey("control","command","f")
            
        self.previous = command
    
    def checkAndRun(self,command):
        print("Command : {0} ,  Prev : {1}".format(command,self.previous))
        exceptionList = self.exceptionGest[self.app]

        if command in exceptionList:
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
        
    def resetApp(self,name):
        self.app = name
        self.previous=""



