from Test import AcessCamera
from Test import AdaptiveThresold1
from Test import AdaptiveThresolding
from Test import MotionDetection
from Test import LaneLineDetection
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import time
import playsound
import speech_recognition as sr


def getAudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except:
            print("Sorry. Please speak again!")
    return said

def Task():
    text = getAudio()
    if "camera" in text:
        AcessCamera()
    if "motion" in text:
        MotionDetection()
    if "line" in text:
        LaneLineDetection()
    if "image" in text:
        AdaptiveThresolding()
    if "video" in text:
        AdaptiveThresold1()
import tkinter
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
def pic():
    im = cv2.imread("blur-boy.PNG");
window = Tk()
window.geometry("650x500")
window.title("welcome")
label1 = Label(window,text="You Speak We Hear",fg="blue",relief="solid",width=20,font=("Verdana",20,"bold"))
label1.place(x=140,y=20)
photo = PhotoImage(file = r"mic.png")
label2 = Label(window,text="Click to Speak",fg="cyan",width=10,font=("Verdana",12,"bold"))
label2.place(x=180,y=300)
b1=Button(window,text="Speak!",fg='green',command=Task,image=photo)
b1.place(x=190,y=220)
window.mainloop()


