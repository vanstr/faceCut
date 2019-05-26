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
from imutils.video import VideoStream

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-f", "--fullscreen", type=bool, default=True,
                help="Enter presentation mode in fullscreen")
ap.add_argument("-m", "--minimgwidth", type=int, default=100,
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
vs = VideoStream(src=0).start()
time.sleep(2.0)

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screenHeight = user32.GetSystemMetrics(0)

# start the FPS throughput estimator
fps = FPS().start()


def display_face_frame(face):
    face = cv2.resize(face, (faceHeight * ratio, faceWidth * ratio))
    if args["fullscreen"]:
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


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    (h, w) = frame.shape[:2]

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
        (startX, startY, endX, endY) = rect

        # extract the face ROI
        faceHeightError = (endY - startY) / 7
        faceWidthError = (endX - startX) / 8
        face = frame[startY - faceHeightError:endY + faceHeightError, startX - faceWidthError:endX + faceWidthError]
        faceHeight = endY - startY + 2 * faceHeightError
        faceWidth = endX - startX + 2 * faceWidthError
        ratio = screenHeight / faceHeight
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
vs.stop()
