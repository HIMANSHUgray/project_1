# Import the necessary modules for creating a GUI application using tkinter.
import tkinter as tk 
# ttkbootstrap is used for themed styling in this code.
import ttkbootstrap as ttk

# Define a function to convert miles to kilometers.
def canvert():
    miles_input = entry_var.get()  # Get the value entered in the entry field.
    km_output = miles_input * 1.61  # Convert miles to kilometers.
    output_string.set(km_output)  # Update the output label with the result.

# Create a window using ttkbootstrap with the "darkly" theme.
window = ttk.Window(themename="darkly")
window.title("demo")  # Set the window title.
window.geometry("300x150")  # Set the window size.

# Create a label for the title of the application.
title_label = ttk.Label(master=window, text="miles to kilometer", font="calibri 24 bold")
title_label.pack()  # Place the title label in the window.

# Create an input frame.
input_frame = ttk.Frame(master=window)
entry_var = tk.IntVar()  # Create a variable to store the user's input.
entry = ttk.Entry(master=input_frame, textvariable=entry_var)  # Create an entry field for input.
button = ttk.Button(master=input_frame, text="convert", command=canvert)  # Create a button for conversion.
entry.pack(side="left", padx=10)  # Place the entry field to the left of the button.
button.pack()  # Place the button.
input_frame.pack(pady=10)  # Place the input frame in the window with some padding.

# Create a string variable to store the output value.
output_string = tk.StringVar()
output_label = ttk.Label(master=window, text="Output", font="calibri 24", textvariable=output_string)
output_label.pack(pady=5)  # Place the output label with some padding.

# Start the GUI application and wait for user interactions.
window.mainloop()
