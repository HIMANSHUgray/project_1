import tkinter as tk
from tkinter import ttk
def button_func():
    enter_text=entry.get()
    # update label
    label.config(text=enter_text)
    # or label.configer(text="some other text")
    # or label["text"]="some other text"
    entry["state"]="disabled"
    """{'background': ('background', 'frameColor', 'FrameColor', '', ''),
      'foreground': ('foreground', 'textColor', 'TextColor', '', ''), 
      'font': ('font', 'font', 'Font', '', ''),
      'borderwidth': ('borderwidth', 'borderWidth', 'BorderWidth', '', ''), 
      'relief': ('relief', 'relief', 'Relief', '', ''), 
      'anchor': ('anchor', 'anchor', 'Anchor', '', ''),
      'justify': ('justify', 'justify', 'Justify', '', ''),
      'wraplength': ('wraplength', 'wrapLength', 'WrapLength', '', ''),
      'takefocus': ('takefocus', 'takeFocus', 'TakeFocus', '', ''), 
      'text': ('text', 'text', 'Text', '', ''), 
      'textvariable': ('textvariable', 'textVariable', 'Variable', '', ''), 
      'underline': ('underline', 'underline', 'Underline', -1, -1), 
      'width': ('width', 'width', 'Width', '', ''), 
      'image': ('image', 'image', 'Image', '', ''), 
      'compound': ('compound', 'compound', 'Compound', '', ''), 
      'padding': ('padding', 'padding', 'Pad', '', ''), 
      'state': ('state', 'state', 'State', <string object: 'normal'>, <string object: 'normal'>), 
      'cursor': ('cursor', 'cursor', 'Cursor', '', ''), 
      'style': ('style', 'style', 'Style', '', ''), 
      'class': ('class', '', '', '', '')}"""
    print(label.configure())
    
window=tk.Tk()
window.title("hello world")
window.geometry("800x500")

entry=ttk.Entry(master=window,text="wright some thing")
entry.pack()

label=ttk.Label(master=window, text="this is a label")
label.pack()

button=ttk.Button(master=window,text="this is a button",command=button_func)
button.pack()

window.mainloop()

# window2=tk.Tk()
# window2.geometry("300x50")
# label2=ttk.Label(master=window2,text="are you shour you wnat to close the window")
# label2.pack()
# window2.mainloop()