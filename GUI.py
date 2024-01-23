import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import cv2
from PIL import Image, ImageTk

def setup_gui(window):
    window.title("Nhận diện và đếm tiền trong video")
    window.geometry("1200x650")  

    control_frame = Frame(window, bg="#6BBDE0", width=330)
    control_frame.pack(side="right", fill="y")
    control_frame.pack_propagate(False) 

    video_frame = Frame(window, bg="#D9D9D9", height=450)
    video_frame.pack(side="top", fill='x')
    video_frame.pack_propagate(False)

    info_frame = Frame(window, bg="#6B93E0", height=200)
    info_frame.pack(side="bottom", fill="x")
    info_frame.pack_propagate(False)
    
    video_label = Label(video_frame)
    video_label.pack(expand=True)
 
    load_button = Button(control_frame, text="Load Video", command=lambda: load_video(video_label), width=20, height=2)
    play_button = Button(control_frame, text="Play", command=lambda: print("Phát video"), width=20, height=2)
    stop_button = Button(control_frame, text="Pause", command=lambda: print("Dừng video"), width=20, height=2)
    exit_button = Button(control_frame, text="Exit", command=window.quit, width=20, height=2)
    openCamera_button = Button(control_frame, text="Open Camera", command=lambda: open_camera(video_label), width=20, height=2)


    load_button.pack(pady=20)
    play_button.pack(pady=20)
    stop_button.pack(pady=20)
    exit_button.pack(pady=20)
    openCamera_button.pack(pady=20)

    total_label = Label(info_frame, text="Tổng số tiền trong video: ", bg="blue", fg="white")
    total_label.pack(side="left", padx=20)  
    

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
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert frame to RGB format and then to ImageTk format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update the display
        video_label.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def load_video(label):
    filepath = filedialog.askopenfilename()
    if filepath:
        load_and_play_video(filepath, label)

if __name__ == "__main__":
    root = tk.Tk()
    setup_gui(root)
    root.attributes("-topmost", True)
    root.mainloop()
