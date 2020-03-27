import PySimpleGUI as sg
from backgroundSub2 import backgroundSub2 as bg

sg.ChangeLookAndFeel('DarkAmber')
detector = bg('VLC')
needReset = False
running =False

def printHelp():
    layoutHelp = []
    layoutHelp.append([sg.Text('OPTIONS', size=(20, 1), font=("Helvetica", 25))])
    menu = detector.printMenu()
    for string in menu:
        layoutHelp.append([sg.Text(string)])
    layoutHelp.append([sg.Quit()])
    helpW = sg.Window('Help', layoutHelp)
    event, value = helpW.read()
    if event in ('Quit', None):
        helpW.close()

def runApp():
    global value
    currentApp = 'VLC'
    if value[currentApp]==False:
        currentApp = 'Chrome'
    formWindow.close()
    detector.run(currentApp)


    

# Lookup dictionary that maps button to function to call
dispatch_dictionary = {'Help':printHelp, 'Submit': runApp}

#form = sg.FlexForm('Hand Gest', default_element_size=(40, 1))

layout = [
    [sg.Text('Welcome to HANDY-GEST!', size=(30, 1), font=("Helvetica", 25))],
    [sg.Text('Press Help for All comands or "h" during program run for ')],
    [sg.Text('Please select which app you want to control ?')],
    [sg.Radio('VLC', "RADIO1", default=True,key = 'VLC'), sg.Radio('Chrome', "RADIO1",key = 'Chrome')],
    [sg.Submit(), sg.Quit(),sg.Button('Help')]
     ]

formWindow = sg.Window('HandyGest', layout)
value = []

while True:
    # Read the Window
    event, value = formWindow.read()
    print(event,value)
    if event in ('Quit', None):
        break
    # Lookup event in function dictionary
    if event in dispatch_dictionary:
        func_to_call = dispatch_dictionary[event]   # get function from dispatch dictionary
        func_to_call()
    else:
        print('Event {} not in dispatch dictionary'.format(event))
formWindow.close()
