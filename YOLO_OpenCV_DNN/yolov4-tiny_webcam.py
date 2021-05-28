# import necessary libraries
import cv2
from threading import Thread

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=60):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Set parameters
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

# load class names
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# initialise yolov4-tiny in opencv-dnn
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# counter to alternate between tracker and detector
counter = 0

# initialise video stream
videostream = VideoStream().start()

while cv2.waitKey(1) < 1:
    frame = videostream.read() # grab current frame

    # reset count for each classes
    civilian = 0
    emergency = 0
    motorcycles = 0
    
    # detector
    if counter % 2 == 0:
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) # detect object
        trackers = []
        for (classid, score, box) in zip(classes, scores, boxes):
            box_tuple = (box[0], box[1], box[2], box[3])
            # assign tracker to each detected object
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame,box_tuple)
            trackers.append(tracker)
            if classid == 0:
                civilian += 1
            elif classid == 1:
                emergency += 1
            elif classid == 2:
                motorcycles += 1 
    
    # tracker
    else:
        for (classid, score, tracker) in zip(classes, scores, trackers):
            (success, box) = tracker.update(frame) # track objects
            if success: 
                if classid == 0:
                    civilian += 1
                elif classid == 1:
                    emergency += 1
                elif classid == 2:
                    motorcycles += 1
                  
    counter += 1
