from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label, Frame, filedialog
import tkinter as tk
from PIL import Image, ImageTk
import cv2

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\THONG PC\Downloads\Image-processing\build\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.geometry("1173x688")
window.configure(bg="#FFD2B1")

canvas = Canvas(
    window,
    bg="#FFD2B1",
    height=688,
    width=1173,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

# Define rectangles on the canvas
canvas.create_rectangle(60.0, 28.0, 771.0, 231.0, fill="#FFEDE1", outline="")
canvas.create_rectangle(825.0, 28.0, 1113.0, 660.0, fill="#FFEDE1", outline="")
canvas.create_rectangle(60.0, 260.0, 771.0, 660.0, fill="#FFEDE1", outline="")

# Video label where the video will be shown
video_label = Label(window)
video_label.place(x=60, y=260, width=711, height=400)  # Position and size within the rectangle

def load_and_play_video(path):
    cap = cv2.VideoCapture(path)
    target_width = 711
    target_height = 400

    def play_video():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (target_width, target_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk  
            video_label.configure(image=imgtk)
            video_label.after(20, play_video) 
        else:
            cap.release() 

    play_video()

def load_video():
    filepath = filedialog.askopenfilename()
    if filepath:
        load_and_play_video(filepath)

# Button to load and play video
button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=load_video,
    relief="flat"
)
button_1.place(x=861.0, y=99.0, width=216.0, height=61.0)

window.resizable(False, False)
window.mainloop()


