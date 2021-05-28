import cv2
import time
from imutils.video import FPS

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("test.mp4")

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

fps = FPS().start()

k = 0
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    # reset count for each classes
    civilian = 0
    emergency = 0
    motorcycles = 0
    
    if k % 2 == 0:
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        trackers = []
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            box_tuple = (box[0], box[1], box[2], box[3])
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame,box_tuple)
            trackers.append(tracker)
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)   
            if classid == 0:
                civilian += 1
            elif classid == 1:
                emergency += 1
            elif classid == 2:
                motorcycles += 1 
               
    
    else:
        a = 0;
        for (classid, score, tracker) in zip(classes, scores, trackers):
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if classid == 0:
                    civilian += 1
                elif classid == 1:
                    emergency += 1
                elif classid == 2:
                    motorcycles += 1 
    
    cv2.putText(frame, str(civilian) + " Cars", (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0,), 2)
    cv2.putText(frame, str(emergency) + " Emergency Vehicles", (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0,), 2)
    cv2.putText(frame, str(motorcycles) + " Motorcycles", (0, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0,), 2)
    
    k += 1
    fps.update()
    fps.stop()
    
    fps_label = "FPS: %.2f" % (fps.fps())
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)
