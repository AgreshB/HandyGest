# HandyGest
Third year project : Automatic hand gesture detection

My project is the implementation of Automatic hand gestures detection to operations\ perform tasks on one’s machine. This is done using deep learning with CNNs. The output of this then piped to application and perform actions such as pause, play etc. with respect to video player. The project is coded in python

Find more on the github page : https://github.com/AgreshB/HandyGest

# Project Structure:
    .
    ├── README.md                   Contains Instructions and steps to run program
    ├── __pycache__
    ├── backgroundSub2.py           This file contains the main code for detetcting the hand
    ├── createCustom.py             This is a subortinate file used to create the custom data set
    ├── execute.py                  Deals with execution of operations based on the command
    ├── helper.py                   Helper file for writing the custom dataset
    ├── main.py                     Front end of the application with GUI framework
    └── model
    │   └── custModelOrgRes-9.h5    main model used for the project
    ├── training.ipynb              Training file (ipynb version opens only in github, open in google colab)
    └── training.py                 Training file (.py version , has few bugs , open in google colab)


# HOW TO RUN THE PROJECT
    -First please install the required packages with the following command:
        'run pip install -r requirements.txt'
    -Then run 'python3 main.py'
    -Follow on screen commands for help before running the program 