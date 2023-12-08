import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk2

def hello():
    print("hello")

def butten_function():
    print("butten was press")

window = tk.Tk()
window.title("hello worls")
window.geometry("800x500")

text=tk.Text(master=window)
text.pack()

label=ttk.Label(master=window,text="this is label")
label.pack()


entry=ttk.Entry(master=window)
entry.pack()

exercise_label=ttk.Label(master=window,text="my label")
exercise_label.pack()

butten=ttk.Button(master=window,text="butten",command=butten_function)
butten.pack()

exercise_button=ttk.Button(master=window,text="exercise_button",command=hello)
exercise_button.pack()
exercise_button1=ttk.Button(master=window,text="exercise_button",command=lambda: print("hello"))
exercise_button1.pack()
# run 
window.mainloop()
