# USAGE
# python show_face.py --detector face_detection_model
# import the necessary packages
import argparse
import ctypes
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--fullscreen", type=bool, default=True,
                help="Enter presentation mode in fullscreen")
ap.add_argument("-m", "--minimgwidth", type=int, default=80,
                help="Minimal detected face width size")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# vs.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

vs = cv2.VideoCapture(0)
# Check success
if not vs.isOpened():
    raise Exception("Could not open video device")
# Set properties. Each returns === True on success (i.e. correct resolution)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2.0)

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screenHeight = user32.GetSystemMetrics(0)
screenWidth = user32.GetSystemMetrics(1)

# start the FPS throughput estimator
fps = FPS().start()

overlay = cv2.imread('loreta/overlay.jpg')
overlay = cv2.resize(overlay, (screenHeight, screenWidth))

background = cv2.imread('loreta/background.png')
background = cv2.resize(background, (screenHeight, screenWidth))


def display_face_frame(face):
    if args["fullscreen"]:
        face = cv2.resize(face, (screenHeight, screenWidth))
        face = cv2.addWeighted(face, 1, overlay, -1, 0)
        rotated_face = imutils.rotate_bound(face, 270)
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", rotated_face)
    else:
        cv2.imshow("Debug frame", face)


def get_biggest_face_coordinates(detections, w, h):
    biggest_width = args["minimgwidth"]
    biggest_rect = None

    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detection_width = endX - startX
            if biggest_width < detection_width:
                biggest_width = detection_width
                biggest_rect = (startX, startY, endX, endY)
    return biggest_rect


def get_adapted_face(rect):
    (startX, startY, endX, endY) = rect
    nose_coordinates = (float(startX + endX) / 2, float(startY + endY) / 2)

    w = endX - startX
    h = w / 9 * 16
    startY = int(nose_coordinates[1] - float(h / 2))
    endY = int(nose_coordinates[1] + float(h / 2))

    face = frame[startY:endY, startX:endX]
    return face


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    rate, frame = vs.read()
    (h, w) = frame.shape[:2]
    print("[INFO] - img width, " + str(w))
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    rect = get_biggest_face_coordinates(detections, w, h)
    if rect is not None:

        # extract the face ROI
        face = get_adapted_face(rect)
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if (fW > 0 and fH > 0) and fW > args["minimgwidth"]:
            display_face_frame(face)

    # update the FPS counter
    fps.update()

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
