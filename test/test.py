import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv9 model
model = YOLO('train\exp2\\best.pt')
tracker = DeepSort(max_age=10)

# Open the video file
video_path = "2.mp4"
cap = cv2.VideoCapture(video_path)

classes_path = "data\money.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# Store the track history
track_history = defaultdict(lambda: [])

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



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True)
        #
        # # Get the boxes and track IDs
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()
        #
        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        #
        # # Plot the tracks
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)
        #
        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

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
            print(f"item {index} track id: {track_id} class: {class_name}")

            # Draw the bounding box and ID on the frame
            frame = draw_corner_rect(frame, (x1, y1, x2 - x1, y2 - y1), line_length=15, line_thickness=3,
                                     rect_thickness=1,
                                     rect_color=(B, G, R), line_color=(R, G, B))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 10, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
