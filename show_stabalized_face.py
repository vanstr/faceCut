# USAGE
# python show_stabalized_face.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --shape-predictor shape_predictor_68_face_landmarks.dat
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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

imgWidth = 600

# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()

# initialize FaceAligner
print("[INFO] initialize FaceAligner")
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=imgWidth)

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
	(h, w) = frame.shape[:2]

	
	# load the input image, resize it, and convert it to grayscale
	image = frame
	#image = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	# cv2.imshow("Input", image)
	rects = detector(gray, 2)

	# loop over the face detections
	for rect in rects:
		# extract the ROI of the *original* face, then align the face
		# using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=imgWidth)
		faceAligned = fa.align(image, gray, rect)

		# display the output images
		cv2.imshow("Original", faceOrig)
		cv2.imshow("Frame", faceAligned)		

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