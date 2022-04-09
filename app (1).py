import tkinter as Tkinter
from tkinter import *
from tkinter import filedialog, Text
import os

top = Tkinter.Tk()

apps = []

def addFile():
    for widget in frame.winfo_children():
        widget.destroy()

    filename = filedialog.askopenfilename(initialdir="/home/dev/Documents/IET_HACKATHOn/Problem_statement-1", title='Select File',
    filetypes=(("html", "*.html"),("all files", "*.*")))
    apps.append(filename)
    
    for app in apps:
        label = Tkinter.Label(frame, text = app, bg="#2E8857", fg="#D9DDDC",bd = 2, height = 2, padx = 3, pady = 3, width = 80)
        label.pack()

flags = os.O_RDWR

def runFile():

    for app in apps:
        import pandas as pd
        import numpy as np
        df = pd.read_html(app)
        l=[]
        l1=[]
        for i in df[1]['Variables']:
            l.append(str(i))
        for i in l:
            a=i.find('.')
            if a==-1:
                a=0
            l1.append(i[a+1:])
        for i in range(len(l1)):
            df[1]['Variables'][i]=l1[i]
        sd = df[1][df[1]['Usage'] == 'shared']
        res = sd[['Variables','Tasks (Write)','Tasks (Read)','Detailed Type','Nb Read','Nb Write']]
        res_cols = ['Variables','W.T','R.T','Detailed Type','Nb Read','Nb Write']
        res.columns = res_cols
        filename1 = filedialog.asksaveasfilename(initialdir="/home/dev/Documents/IET_HACKATHOn/Problem_statement-1", title='Select File',
        filetypes=(("", ".xlsx"),("all files", "*.*")))
        res.to_excel(filename1, index = False)
        k = 0
        os.open('/home/dev/Downloads/IET_Hackathon/new1.xlsx', flags)

def delFile():
    for widget in frame.winfo_children():
        widget.destroy()

C = Tkinter.Canvas(top, bg="#D9DDDC",height = 430, width= 700, bd = 3)
C.pack()

frame = Tkinter.Frame(top, bg="#FFFFFF")
frame.place(relwidth=0.8, relheight=0.65, relx = 0.1, rely = 0.1)

uploadFile = Tkinter.Button(top ,  text = "Upload File", padx=300, pady=10,fg= "black",activebackground = "#3BB143", bg="#4CBB1C",font=('verdana', 10),width = 15, command = addFile)
uploadFile.pack()

displayResult = Tkinter.Button(top , text = "Generate Excel", padx=300, pady=10,fg= "black",activebackground = "#3BB143", bg="#4CBB1C",font=('comicsans', 10),width =  15, command= runFile)
displayResult.pack()

deleteEntry = Tkinter.Button(top , text = "Delete Entries", padx=300, pady=10,fg= "black",activebackground = "#3BB143", bg="#4CBB1C",font=('comicsans', 10),width =  15, command= lambda : apps.clear())
deleteEntry.pack()

top.mainloop()