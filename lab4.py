import csv
from datetime import datetime
import cv2


static_back = None

motion_list = [None, None]


time = []
header = ["Start", "End"]


video = cv2.VideoCapture("./resources/video.mov", cv2.CAP_ANY)


w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter("./resources/output.mov", fourcc, 25, (w, h))
font = cv2.FONT_HERSHEY_DUPLEX


while video.isOpened():

    check, frame = video.read()
    if check:

        motion = 0


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)


        if static_back is None:
            static_back = gray
            continue


        diff_frame = cv2.absdiff(static_back, gray)

        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 3000:
                continue
            motion = 1
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").split(".")[0]
            frame_to_write = frame.copy()
            frame_to_write = cv2.putText(
                frame_to_write, dt, (10, 30), font, 1, (0, 0, 0), 4, cv2.LINE_8
            )
            video_writer.write(frame_to_write)

        motion_list.append(motion)

        motion_list = motion_list[-2:]


        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").split(".")[0])

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").split(".")[0])

        key = cv2.waitKey(1)

        if key == ord("q"):

            if motion == 1:
                time.append(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").split(".")[0]
                )
            break
    else:
        if motion == 1:
            time.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").split(".")[0])
        break

with open("./resources/intervals.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(0, len(time), 2):
        writer.writerow([f"{time[i]}", f"{time[i + 1]}"])

video.release()
cv2.destroyAllWindows()