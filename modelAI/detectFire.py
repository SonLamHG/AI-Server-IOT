import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

# Mô hình yolov10 phát hiện lửa
model = YOLO(f"{os.getcwd()}/modelAI/fire.pt")

# Địa chỉ IP của ESP32-CAM (ví dụ: "http://192.168.1.50:81/stream")
url = 'http://192.168.1.37:5000/video_feed'

def stream_video(url):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Không thể mở stream từ ESP32-CAM")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame, thử lại...")
            time.sleep(1)
            continue

        # Fire detection
        results = model(frame, conf=0.8, verbose=False)
        print('fsvsvsvsvsv')

        # Vẽ bounding box
        for result in results:
            boxes = result.boxes
            for box in boxes:
                print('detected fire')
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                c = box.cls
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[int(c)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("ESP32-CAM Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Viết hàm detectFire đầu vào là 1 frame đầu ra là frame đã được vẽ bounding box nếu phát hiện lửa
def DetectFire(frame, conf: float = 0.6):
    results = model(frame, conf=conf, verbose=False)
    has_fire = False
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Lấy toạ độ và nhãn
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            c = int(box.cls)
            score = None
            try:
                score = float(box.conf.tolist()[0])
            except Exception:
                pass

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = model.names[c] if hasattr(model, "names") and c in model.names else "fire"

            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label}{'' if score is None else f' {score:.2f}'}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            has_fire = True
            detections.append({"box": [x1, y1, x2, y2], "class": c, "label": label, "score": score})

    # Trả về: frame đã vẽ (nếu có), cờ có lửa, và danh sách detection
    return frame, has_fire, detections
