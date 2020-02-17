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


    def openApp(self):
        self.appRunning = True
        subprocess.call(["/usr/bin/open", "/Applications/{}.app".format(self.app)])

    def runCommand(self, command):
        if command =='None':
            return
        elif command =='fist':
            # mute if fist
            if self.isMute ==False:
                pgui.hotkey("option","command","down")
                self.isMute = True
            return
        elif command =='palm':
            # unmute if palm
            if self.isMute ==True:
                pgui.hotkey("option","command","down")
                self.isMute = False
            return
        elif command == 'peace':
            # pause if palm 
            if self.isPause == False:
                pgui.hotkey("space")
                self.isPause = True
            return
        elif command =='ok':
            # play if ok
            if self.isPause == True :
                pgui.hotkey("space")
                self.isPause = False
            return
        elif command =='L':
            # volume up if L
            pgui.hotkey("up")
            return
        elif command =='index':
            # volume down if index 
            pgui.hotkey("down")
            return
    


