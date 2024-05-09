import os
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Label, filedialog

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(
    r"build\\assets\\frame0")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv9 model
model = YOLO("last\\best.pt")
tracker = DeepSort(max_age=10)

classes_path = "data\\money.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")


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


def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1 + radius, y1,
              x1 + radius, y1,
              x2 - radius, y1,
              x2 - radius, y1,
              x2, y1,
              x2, y1 + radius,
              x2, y1 + radius,
              x2, y2 - radius,
              x2, y2 - radius,
              x2, y2,
              x2 - radius, y2,
              x2 - radius, y2,
              x1 + radius, y2,
              x1 + radius, y2,
              x1, y2,
              x1, y2 - radius,
              x1, y2 - radius,
              x1, y1 + radius,
              x1, y1 + radius,
              x1, y1]

    return canvas.create_polygon(points, **kwargs, smooth=True)


round_rectangle(60, 28, 771, 231, radius=45, fill="#FFEDE1", outline="")
round_rectangle(825.0, 28.0, 1113.0, 660.0, radius=45, fill="#FFEDE1", outline="")
round_rectangle(60.0, 260.0, 771.0, 660.0, radius=45, fill="#FFEDE1", outline="")

video_label = Label(window)
run_video = False

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3))


def draw_corner_rect(img, bbox, line_length=30, line_thickness=5, rect_thickness=1,
                     rect_color=(255, 0, 255), line_color=(0, 255, 0)):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    if rect_thickness != 0:
        cv2.rectangle(img, bbox, rect_color, rect_thickness)

    # Top Left  x, y
    cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
    cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

    # Top Right  x1, y
    cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
    cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

    # Bottom Left  x, y1
    cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
    cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

    # Bottom Right  x1, y1
    cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
    cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

    return img


text_results = ""
total = 0


def model_processor(frame):
    results = model(frame)
    # frame = results[0].plot()

    a = results[0].boxes.data
    a = a.detach().cpu().numpy()

    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        #        print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        id = int(row[5])

        c = float(row[4])
        if c > 0.5:
            list.append([[x1, y1, x2 - x1, y2 - y1], c, id])

    tracks = tracker.update_tracks(list, frame=frame)
    global text_results, total
    text_results = ""
    total = 0
    print("-----------------")
    for index, track in enumerate(tracks):
        if not track.is_confirmed():
            print("nothing")
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        B, G, R = map(int, color)
        class_name = class_names[class_id]
        text = f"{track_id} - {class_name}"
        text_result = f"item {index} track id: {track_id} class: {class_name}"
        total += int(class_name)
        text_results = text_results + "\n" + text_result
        print(text_results)

        # Draw the bounding box and ID on the frame
        frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3,
                                 rect_thickness=1,
                                 rect_color=(B, G, R), line_color=(R, G, B))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def get_frame_size(cap):
    # Lấy kích thước video gốc
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Xác định tỷ lệ cho khung hiển thị
    display_width = 650
    display_height = 350

    # Tính tỷ lệ giữa khung hiển thị và video gốc
    scale_width = display_width / original_width
    scale_height = display_height / original_height
    scale = min(scale_width, scale_height)

    # Tính toán kích thước mới để giữ tỷ lệ video
    target_width = int(original_width * scale)
    target_height = int(original_height * scale)

    return target_width, target_height


def load_and_play_video(path):
    global cap
    cap = cv2.VideoCapture(path)

    target_width, target_height = get_frame_size(cap)

    global video_label, run_video
    video_label.after(1, video_label.destroy())
    video_label = Label(window)
    video_label.place(x=90, y=280, width=650, height=350)  # Giữ vị trí giống như cũ
    run_video = True

    def play_video():
        ret, frame = cap.read()
        if ret:
            # detect
            frame = model_processor(frame)

            # plot
            frame = cv2.resize(frame, (target_width, target_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            if run_video:
                global text_results, total
                canvas.itemconfigure(plot, text=f"detect:{total} \n{text_results}")
                video_label.after(20, play_video)
        else:
            cap.release()

    play_video()


def open_camera():
    release()
    global video_label, run_video, video_capture
    video_label.after(1, video_label.destroy())
    video_label = Label(window)
    video_label.place(x=90, y=280, width=650, height=350)
    run_video = True

    video_capture = cv2.VideoCapture(0)
    target_width, target_height = get_frame_size(video_capture)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        # process
        frame = model_processor(frame)

        # plot
        frame = cv2.resize(frame, (target_width, target_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.update()
        global text_results, total
        canvas.itemconfigure(plot, text=f"detect:{total} \n{text_results}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def load_video():
    release()
    filepath = filedialog.askopenfilename()
    if filepath:
        load_and_play_video(filepath)


def release():
    try:
        cap.release()
    except Exception:
        pass
    try:
        video_capture.release()
        cv2.destroyAllWindows()
    except Exception:
        pass


canvas.create_text(
    235.0,
    61.0,
    anchor="nw",
    text=f"NHÓM 09: \nNguyễn Dương Tiến Thông - 21110313\nNgô Ngọc Thông - 21110312\nHồ Gia Kiệt - 21110224\n",
    fill="#000000",
    font=("Inter", 15 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    151.0,
    122.0,
    image=image_image_1
)

canvas.create_text(
    271.0,
    153.0,
    anchor="nw",
    text=f"ỨNG DỤNG NHẬN DIỆN TIỀN VIỆT",
    fill="#000000",
    font=("Inter", 24 * -1)
)

plot = canvas.create_text(
    861.0,
    510.0,
    anchor="nw",
    text=f"detect: {text_results}",
    fill="#000000",
    font=("Inter", 15 * -1)
)

def update_ip():
    try:
        canvas.itemconfigure(plot, text=f"ỨNG DỤNG NHẬN DIỆN TIỀN VIỆT\n detect: {text_results}")
        window.after(10, update_ip)
    except StopIteration:
        pass

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=load_video,
    relief="flat"
)
button_2.place(x=861.0, y=206.0, width=216.0, height=61.0)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: open_camera(),
    relief="flat"
)
button_3.place(x=861.0, y=313.0, width=216.0, height=61.0)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=window.destroy,
    relief="flat"
)
button_4.place(x=861.0, y=420.0, width=216.0, height=61.0)

window.resizable(False, False)
window.mainloop()
