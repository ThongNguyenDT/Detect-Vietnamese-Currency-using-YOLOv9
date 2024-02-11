from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import cv2
from PIL import Image, ImageTk

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\THONG PC\Downloads\Image-processing\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1173x688")
window.configure(bg = "#FFD2B1")


video_frame = Frame(window, bg="#D9D9D9", height=450)
video_frame.pack(side="top", fill='x')
video_frame.pack_propagate(False)
video_label = Label(video_frame)
video_label.pack(expand=True)

canvas = Canvas(
    window,
    bg = "#FFD2B1",
    height = 688,
    width = 1173,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    60.0,
    28.0,
    771.0,
    231.0,
    fill="#FFEDE1",
    outline="")

canvas.create_rectangle(
    825.0,
    28.0,
    1113.0,
    660.0,
    fill="#FFEDE1",
    outline="")

canvas.create_rectangle(
    60.0,
    260.0,
    771.0,
    660.0,
    fill="#FFEDE1",
    outline="")


button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: load_video(video_label),
    relief="flat"
)
button_1.place(
    x=861.0,
    y=99.0,
    width=216.0,
    height=61.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=861.0,
    y=206.0,
    width=216.0,
    height=61.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=861.0,
    y=313.0,
    width=216.0,
    height=61.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat"
)
button_4.place(
    x=861.0,
    y=420.0,
    width=216.0,
    height=61.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_5 clicked"),
    relief="flat"
)
button_5.place(
    x=861.0,
    y=527.0,
    width=216.0,
    height=61.0
)





def load_and_play_video(path, label):
    cap = cv2.VideoCapture(path)

    def play_video():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk  
            label.configure(image=imgtk)
            label.after(20, play_video) 
        else:
            cap.release() 
    play_video()

def open_camera(video_label):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

    
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    
        video_label.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def load_video(label):
    filepath = filedialog.askopenfilename()
    if filepath:
        load_and_play_video(filepath, label)
window.resizable(False, False)
window.mainloop()
