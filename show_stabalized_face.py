# USAGE
# python show_stabalized_face.py --shape-predictor shape_predictor_68_face_landmarks.dat
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import dlib

import ctypes

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

user32 = ctypes.windll.user32
print("[INFO] 0 = " + str(user32.GetSystemMetrics(0)) + " 1 = " + str(user32.GetSystemMetrics(1)) )
imgWidth = user32.GetSystemMetrics(0)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()

# initialize FaceAligner
print("[INFO] initialize FaceAligner")
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredLeftEye=(0.38, 0.38), desiredFaceWidth=imgWidth)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # frame = imutils.resize(frame, width=1200)

    # load the input image, resize it, and convert it to grayscale
    # image = frame
    image = imutils.resize(frame, width=imgWidth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    print("[INFO] detected " + str(len(rects)) + " faces")
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=imgWidth)
        faceAligned = fa.align(image, gray, rect)
        cut = (user32.GetSystemMetrics(0) - user32.GetSystemMetrics(1))/2
        (h2, w2) = faceAligned.shape[:2]
        faceAligned = faceAligned[0:user32.GetSystemMetrics(0), cut:(w2 - cut)]

        # display the output images
        rotatedface = rotated = imutils.rotate_bound(faceAligned, 90)
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", rotatedface)
        break

    # update the FPS counter
    fps.update()

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
