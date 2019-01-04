# USAGE
# python show_stabalized_face.py --shape-predictor shape_predictor_68_face_landmarks.dat
# import the necessary packages

import argparse
import collections
import ctypes
import time

import cv2
import dlib
import imutils
import numpy as np
import sys
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.video import FPS
from imutils.video import VideoStream

# construct the argument parser and parse the arguments
MINIMAL_RECOGNIZED_FACES_AMOUNT = 5
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

#user32 = ctypes.windll.user32
WISHED_FACE_IMG_PIXEL_WIDTH = 480 #int(user32.GetSystemMetrics(0) / 4)
WISHED_FACE_IMG_PIXEL_HEIGHT = 320 #int(user32.GetSystemMetrics(1) / 4)
print("[INFO] 0 = " + str(WISHED_FACE_IMG_PIXEL_WIDTH) + " 1 = " + str(WISHED_FACE_IMG_PIXEL_HEIGHT))
minImgSize = int(WISHED_FACE_IMG_PIXEL_WIDTH / 6)  # filter out users located too far
imgWidth = WISHED_FACE_IMG_PIXEL_WIDTH
is_full_screen_mode = False


is_shown_face_model = False

BLANK_IMAGE = np.zeros((WISHED_FACE_IMG_PIXEL_HEIGHT, WISHED_FACE_IMG_PIXEL_WIDTH, 3), np.uint8)
lastValidFaceRecognition = 0
faceRecognitionStat = collections.deque([], 15)
faceRecognitionStat.appendleft(False)

print(faceRecognitionStat)


#
# ser = serial.Serial('COM4')
#
# if ser.isOpen():
#     ser.close()
# ser.open()
# ser.isOpen()


def move_face_model(move_forward):
    # global ser
    print("[INFO] move:" + str(move_forward))
    # if move_forward:
    #     ser.write("F\n")
    # else:
    #     ser.write("B\n")


def update_face_model_state(face_recognition_stat):
    global is_shown_face_model
    successful_recognition_amount = list(face_recognition_stat).count(True)

    print("[DEBUG] successful_recognition_amount" + str(successful_recognition_amount))
    if not is_shown_face_model:
        if successful_recognition_amount > MINIMAL_RECOGNIZED_FACES_AMOUNT:
            print("[INFO] Show face model...")
            is_shown_face_model = True
            move_face_model(True)
    else:
        if True not in face_recognition_stat:
            print("[INFO] Hide face model...")
            is_shown_face_model = False
            move_face_model(False)


def get_aligned_face(image, gray, face_rect):
    face_aligned = fa.align(image, gray, face_rect)
    cut = int((WISHED_FACE_IMG_PIXEL_WIDTH - WISHED_FACE_IMG_PIXEL_HEIGHT) / 2)
    (h2, w2) = face_aligned.shape[:2]
    return face_aligned[0:WISHED_FACE_IMG_PIXEL_WIDTH, cut:(w2 - cut)]


def display_image(image):
    if is_full_screen_mode:
        rotated_face = imutils.rotate_bound(image, 270)
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", rotated_face)
    else:
        cv2.imshow("Frame", image)


def get_valid_face(gray, face_recognition_stat):
    detected_faces = detector(gray, 1)
    detected_faces_amount = len(detected_faces)
    print("[INFO] detected " + str(detected_faces_amount) + " faces")
    if detected_faces_amount > 0:
        valid_face = get_the_biggest_face(detected_faces)
        if valid_face is not None:
            face_recognition_stat.appendleft(True)
            return valid_face
    face_recognition_stat.appendleft(False)
    return None


def get_the_biggest_face(rects):
    biggest_width = minImgSize
    biggest_rect = None
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        if biggest_width < w:
            biggest_width = w
            biggest_rect = rect
    return biggest_rect


# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()

# initialize FaceAligner
print("[INFO] initialize FaceAligner")
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredLeftEye=(0.38, 0.38), desiredFaceWidth=imgWidth)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0, resolution=(WISHED_FACE_IMG_PIXEL_HEIGHT, WISHED_FACE_IMG_PIXEL_WIDTH)).start()
vs = VideoStream(src=0).start()

time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    image = vs.read()

    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    face = get_valid_face(gray, faceRecognitionStat)
    if face is not None and is_shown_face_model:
        aligned_face_image = get_aligned_face(image, gray, face)
        display_image(aligned_face_image)

    update_face_model_state(faceRecognitionStat)

    if not is_shown_face_model:
        display_image(BLANK_IMAGE)

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
sys.exit()