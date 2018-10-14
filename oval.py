# USAGE
# python oval.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# create a mask image of the same shape as input image, filled with 0s (black color)
mask = np.zeros_like(image)
(imgH, imgW) = image.shape[:2]
# create a white filled ellipse

tempImg[0:imgH, 0:imgW] = cv2.blur(image[0:imgH, 0:imgW], (50, 50))
mask=cv2.ellipse(mask, center=(int((imgW) / 2), int((imgH) / 2)), axes=(imgW-imgW/2,imgH-imgH/2), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
# Bitwise AND operation to black out regions outside the mask
result = np.bitwise_and(image,mask)
# Convert from BGR to RGB for displaying correctly in matplotlib
# Note that you needn't do this for displaying using OpenCV's imshow()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

cv2.imshow("image_rgb", image_rgb)
cv2.imshow("mask_rgb", mask_rgb)
cv2.imshow("result_rgb", result_rgb)


cv2.waitKey(0)